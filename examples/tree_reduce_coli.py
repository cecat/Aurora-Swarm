"""Tree-Reduce pattern example using TOM.COLI bacterial gene dataset.

Two-phase flow:
1. Leaf phase: Load all gene prompts from chunk files, run one scatter_gather
   over all of them (test.coli_v3 style). Collect gene responses into a flat list.
2. Reduce phase: Group gene responses by fanin (e.g. 16), run one reduce per
   group, then hierarchically reduce those outputs until a single synthesis.
   The reduce asks whether genes might function together for higher-level
   function (pathway, complex, regulation).

USAGE EXAMPLES:
---------------

Using a hostfile:
    python examples/tree_reduce_coli.py /path/to/batch_1/ \\
        --hostfile agents.txt \\
        --num-files 16 \\
        --output result.txt

Using environment variable:
    export AURORA_SWARM_HOSTFILE=/path/to/agents.txt
    python examples/tree_reduce_coli.py /path/to/batch_1/

With socket monitoring (for scaling experiments):
    python examples/tree_reduce_coli.py /path/to/batch_1/ \\
        --hostfile agents.txt \\
        --num-files 32 \\
        --monitor-sockets \\
        --socket-interval 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

# Allow importing plots.py at repo root when run as examples/tree_reduce_coli.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plots  # noqa: E402

from aurora_swarm import VLLMPool, parse_hostfile
from aurora_swarm.patterns.scatter_gather import scatter_gather


# ---------------------------------------------------------------------------
# Helpers (reused from scatter_gather_coli)
# ---------------------------------------------------------------------------


def print_with_timestamp(message: str) -> None:
    """Print message with timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr)


def discover_chunk_files(
    input_dir: Path, num_files: int, skip_files: int = 0
) -> list[Path]:
    """Discover chunk files in input directory."""
    chunk_files = sorted(input_dir.glob("chunk_*.txt"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.txt files found in {input_dir}")
    start_idx = skip_files
    end_idx = skip_files + num_files
    return chunk_files[start_idx:end_idx]


def parse_gene_line(
    line: str, line_num: int
) -> tuple[str, str, str, str] | None:
    """Parse a TSV line: genome_id, organism, gene_id, gene_description."""
    line = line.strip()
    if not line:
        return None
    parts = line.split("\t")
    if len(parts) < 4:
        print_with_timestamp(
            f"Warning: Line {line_num} has fewer than 4 fields ({len(parts)}), skipping: {line[:50]}..."
        )
        return None
    genome_id = parts[0]
    organism = parts[1]
    gene_id = parts[2]
    gene_description = "\t".join(parts[3:])
    return genome_id, organism, gene_id, gene_description


def construct_prompt(
    organism: str, gene_id: str, gene_description: str
) -> str:
    """Construct LLM prompt for bacterial gene analysis (test.coli_v3 style)."""
    gene_data = f"{gene_id}\t{gene_description}"
    prompt = (
        "Please tell me (using the knowledge you have been trained on) what you know about this bacterial gene in "
        + organism
        + " whose various IDs are given here, though they all refer to the same gene: "
        + gene_data
        + ". In particular, we want to know the following information: Is this gene well studied or is it hypothetical with unknown function? "
        "Is the gene essential for survival? Is the gene or gene product a good antibacterial drug target? What other genes does this gene interact with? "
        "Is this gene part of an operon (cluster of genes on the chromosome that work together to carry out complex functions)? "
        "Is this gene involved in transcriptional regulation? Is it known what gene regulates this gene's expression? "
        "Does this gene also occur in other bacteria? If you were starting out as a research microbiologist, what might be a hypothesis you could explore related to this protein that would have significant scientific impact? "
        "Where possible, give concise answers to these questions as well as describe the function of the gene more generally if it is known."
    )
    return prompt


def read_chunk_genes(chunk_path: Path) -> list[str]:
    """Read one chunk file and return list of gene prompts."""
    prompts = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            result = parse_gene_line(line, line_num)
            if result is None:
                continue
            genome_id, organism, gene_id, gene_description = result
            prompt = construct_prompt(organism, gene_id, gene_description)
            prompts.append(prompt)
    return prompts


def read_all_chunk_genes(chunk_files: list[Path]) -> list[str]:
    """Read all chunk files in order and return one flat list of gene prompts."""
    all_prompts = []
    for chunk_path in chunk_files:
        prompts = read_chunk_genes(chunk_path)
        if not prompts:
            print_with_timestamp(f"  Warning: No valid genes in {chunk_path.name}")
            continue
        all_prompts.extend(prompts)
    return all_prompts


# ---------------------------------------------------------------------------
# Gene response assembly and reduce input limits
# ---------------------------------------------------------------------------

MAX_RESPONSE_CHARS = 40000  # truncate each gene response when building reduce input

# Max chars for combined reduce input (prompt must fit in model context with output)
# Conservative: ~3 chars/token; 80k chars ~27k tokens. With reduce output ~69k, total ~96k < 131k context
MAX_REDUCE_INPUT_CHARS = 160000


# ---------------------------------------------------------------------------
# Reduce loop (from tree_reduce pattern, with max_tokens support)
# ---------------------------------------------------------------------------


def _has_content(text: str) -> bool:
    return bool((text or "").strip())


DEFAULT_REDUCE_PROMPT = """The following are individual gene analysis responses. Consider whether the genes described might function together to provide a higher-level function (e.g. pathway, complex, regulation). Produce a concise synthesis.

Gene responses (level {level}):

{responses}"""


def _is_connection_error(error: str | None) -> bool:
    """True if error suggests connection failure (retry with different agent may help)."""
    if not error:
        return False
    err_lower = error.lower()
    return (
        "cannot connect" in err_lower
        or "clientconnectorerror" in err_lower
        or "connection refused" in err_lower
        or "connection reset" in err_lower
        or "timeout" in err_lower
    )


async def _send_all_with_max_tokens(
    pool: VLLMPool,
    prompts: list[str],
    max_tokens: int | list[int] | None = None,
    retry_other_agents: bool = True,
) -> list:
    """Send prompts to pool, optionally with max_tokens override (int or list per prompt).
    On connection error, retries with other agents in round-robin order (up to pool.size attempts).
    """
    results: list = [None] * len(prompts)
    mt_list = max_tokens if isinstance(max_tokens, list) else [max_tokens] * len(prompts)
    pending = [
        (i, i % pool.size, 1, p, mt_list[i] if i < len(mt_list) else None)
        for i, p in enumerate(prompts)
    ]

    while pending:
        tasks = [
            pool.post(agent_idx, prompt, max_tokens=mt)
            for _, agent_idx, _, prompt, mt in pending
        ]
        responses = await asyncio.gather(*tasks)
        next_pending = []
        for (orig_i, agent_idx, attempt, prompt, mt), resp in zip(pending, responses):
            if resp.success:
                results[orig_i] = resp
            elif (
                retry_other_agents
                and _is_connection_error(resp.error)
                and pool.size > 1
                and attempt < pool.size
            ):
                next_agent = (agent_idx + 1) % pool.size
                next_pending.append((orig_i, next_agent, attempt + 1, prompt, mt))
            else:
                results[orig_i] = resp
        pending = next_pending

    return results


async def run_reduce_phase(
    pool: VLLMPool,
    items: list[str],
    reduce_prompt: str,
    fanin: int,
    max_tokens_leaves: int,
    max_tokens_contingency: int,
    stage_callback: Callable[[int], None] | None = None,
) -> tuple[list[tuple[int, int, str]], str | None]:
    """
    Run hierarchical reduce over a list of items (e.g. gene responses).
    Groups items into batches of fanin, reduces each batch, then repeats until one remains.

    Returns (all_reduce_outputs, final_text).
    all_reduce_outputs: list of (level, group_index, text) for each reduce step.
    final_text: the single final synthesis, or None if failed.
    """
    all_outputs: list[tuple[int, int, str]] = []
    current = [s for s in items if _has_content(s)]
    level = 1

    while len(current) > 1:
        if stage_callback is not None:
            stage_callback(level)
        groups = [current[i : i + fanin] for i in range(0, len(current), fanin)]
        supervisor_prompts = []
        reduce_max_tokens_list = []
        for group in groups:
            combined = "\n---\n".join(group)
            if len(combined) > MAX_REDUCE_INPUT_CHARS:
                orig_len = len(combined)
                combined = (
                    combined[: MAX_REDUCE_INPUT_CHARS - 60]
                    + "\n... [input truncated to fit model context]"
                )
                print_with_timestamp(
                    f"Reduce level {level}: truncated input from {orig_len} to {MAX_REDUCE_INPUT_CHARS} chars"
                )
            filled = reduce_prompt.replace("{responses}", combined)
            filled = filled.replace("{level}", str(level))
            supervisor_prompts.append(filled)
            # max_tokens = max_tokens_leaves * num_leaves + contingency
            reduce_max_tokens_list.append(
                max_tokens_leaves * len(group) + max_tokens_contingency
            )

        sup_responses = await _send_all_with_max_tokens(
            pool, supervisor_prompts, max_tokens=reduce_max_tokens_list
        )
        for i, r in enumerate(sup_responses):
            if not r.success:
                print_with_timestamp(
                    f"Reduce level {level} response {i} failed: {r.error}"
                )
            elif not _has_content(r.text):
                print_with_timestamp(
                    f"Reduce level {level} response {i} empty"
                )
            else:
                all_outputs.append((level, i, r.text))
        current = [
            r.text for r in sup_responses if r.success and _has_content(r.text)
        ]
        level += 1

    if not current:
        return all_outputs, None
    return all_outputs, current[0]


# ---------------------------------------------------------------------------
# Socket monitoring
# ---------------------------------------------------------------------------


async def monitor_sockets(
    interval: float,
    stop_event: asyncio.Event,
    pid: int | None = None,
) -> None:
    """Background task: periodically count and log socket FDs."""
    pid = pid or os.getpid()
    fd_dir = Path(f"/proc/{pid}/fd")

    while not stop_event.is_set():
        await asyncio.sleep(interval)
        if stop_event.is_set():
            break
        if not fd_dir.exists():
            print_with_timestamp("Socket monitor: /proc not available (non-Linux?)")
            break
        count = 0
        try:
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                    if target.startswith("socket:"):
                        count += 1
                except (OSError, FileNotFoundError):
                    pass
            print_with_timestamp(f"Sockets in use: {count}")
        except Exception as e:
            print_with_timestamp(f"Socket monitor error: {e}")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


async def run_tree_reduce_coli(
    pool: VLLMPool,
    chunk_files: list[Path],
    reduce_prompt: str,
    fanin: int,
    max_tokens_leaves: int,
    max_tokens_contingency: int,
    monitor_sockets_enabled: bool,
    socket_interval: float,
    plot_sockets: bool = False,
    plot_output: Path | None = None,
    settings: dict | None = None,
) -> tuple[str | None, dict]:
    """
    Run leaf phase (one scatter_gather over all prompts) then reduce phase.
    Returns (final_result_text, timing_stats).
    """
    timing = {
        "leaf_seconds": 0.0,
        "reduce_seconds": 0.0,
        "total_seconds": 0.0,
        "total_prompts": 0,
        "total_chunks": len(chunk_files),
    }

    stop_monitor = asyncio.Event()
    monitor_task = None
    if monitor_sockets_enabled:
        monitor_task = asyncio.create_task(
            monitor_sockets(socket_interval, stop_monitor)
        )

    recorder = None
    if plot_sockets:
        recorder = plots.SocketRecorder(pid=os.getpid(), interval=socket_interval)
        recorder.set_stage("leaf")
        recorder.start()

    def stage_callback(level: int) -> None:
        if recorder is not None:
            recorder.set_stage(f"reduce {level}")

    try:
        # Leaf phase: load all prompts from chunk files, then one scatter_gather
        all_prompts = read_all_chunk_genes(chunk_files)
        if not all_prompts:
            print_with_timestamp("Leaf phase: no valid prompts from chunk files")
            return None, timing

        print_with_timestamp(
            f"Leaf phase: scatter_gather over {len(all_prompts)} prompts from {len(chunk_files)} chunk files"
        )
        leaf_start = datetime.now()
        responses = await scatter_gather(pool, all_prompts)
        all_gene_responses = []
        for r in responses:
            if r.success and _has_content(r.text):
                text = (
                    r.text[:MAX_RESPONSE_CHARS] + "..."
                    if len(r.text) > MAX_RESPONSE_CHARS
                    else r.text
                )
                all_gene_responses.append(text)
        total_prompts = len(all_prompts)
        leaf_end = datetime.now()
        timing["leaf_seconds"] = (leaf_end - leaf_start).total_seconds()
        timing["total_prompts"] = total_prompts
        print_with_timestamp(
            f"Leaf phase complete: {len(all_gene_responses)} gene responses from {total_prompts} prompts"
        )

        if not all_gene_responses:
            return None, timing

        # Reduce phase: hierarchical reduce over groups of gene responses (fanin per group)
        print_with_timestamp("Reduce phase: starting hierarchical reduction...")
        reduce_start = datetime.now()
        all_reduce_outputs, final = await run_reduce_phase(
            pool,
            all_gene_responses,
            reduce_prompt,
            fanin,
            max_tokens_leaves,
            max_tokens_contingency,
            stage_callback=stage_callback if plot_sockets else None,
        )
        reduce_end = datetime.now()
        timing["reduce_seconds"] = (reduce_end - reduce_start).total_seconds()
        timing["total_seconds"] = (reduce_end - leaf_start).total_seconds()
        timing["all_reduce_outputs"] = all_reduce_outputs

        return final, timing

    finally:
        if monitor_task is not None:
            stop_monitor.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        if recorder is not None and plot_output is not None and settings is not None:
            await recorder.stop()
            try:
                plots.plot_socket_usage(recorder.get_samples(), settings, plot_output)
                print_with_timestamp(f"Socket usage plot saved to {plot_output}")
            except Exception as e:
                print_with_timestamp(f"Failed to save socket plot: {e}")


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing chunk_*.txt files with gene data",
    )
    parser.add_argument(
        "--hostfile",
        type=Path,
        help="Path to hostfile (default: AURORA_SWARM_HOSTFILE env var)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=16,
        help="Number of chunk files to process (default: 16)",
    )
    parser.add_argument(
        "--skip-files",
        type=int,
        default=0,
        help="Number of chunk files to skip at start (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for final result (default: stdout)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4048,
        help="Max tokens per leaf (gene) response (default: 4048)",
    )
    parser.add_argument(
        "--max-tokens-contingency",
        type=int,
        default=4096,
        help="Extra tokens added to reduce max_tokens (default: 4096). Reduce uses max_tokens * num_leaves + contingency.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-120b",
        help="Model name (default: openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Semaphore value: max in-flight requests across all vLLM agents (default: 64). Per-agent max ≈ this / pool size.",
    )
    parser.add_argument(
        "--connector-limit",
        type=int,
        default=1024,
        help="Max TCP connections in pool (default: 1024)",
    )
    parser.add_argument(
        "--batch-concurrency",
        type=int,
        default=256,
        help="vLLM max concurrent sequences for batch timeout scaling (default: 256)",
    )
    parser.add_argument(
        "--timeout-per-sequence",
        type=float,
        default=60.0,
        help="Estimated seconds per sequence for batch timeout scaling (default: 60)",
    )
    parser.add_argument(
        "--batch-timeout-cap",
        type=float,
        default=None,
        metavar="SECS",
        help="Cap batch timeout in seconds (default: no cap)",
    )
    parser.add_argument(
        "--fanin",
        type=int,
        default=16,
        help="Gene responses per reduce group (default: 16). Smaller fanin yields more reduce ops.",
    )
    parser.add_argument(
        "--reduce-prompt",
        type=str,
        default=DEFAULT_REDUCE_PROMPT,
        help="Reduce prompt template with {responses} and {level}",
    )
    parser.add_argument(
        "--monitor-sockets",
        action="store_true",
        help="Enable background socket count monitoring",
    )
    parser.add_argument(
        "--socket-interval",
        type=float,
        default=10.0,
        help="Seconds between socket samples (default: 10)",
    )
    parser.add_argument(
        "--plot-sockets",
        action="store_true",
        help="Record socket count over time/stage and save a plot",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("socket_usage.png"),
        help="Output path for socket usage plot (default: socket_usage.png)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1
    if not args.input_dir.is_dir():
        print(f"Error: Input path is not a directory", file=sys.stderr)
        return 1

    hostfile_path = args.hostfile
    if hostfile_path is None:
        hostfile_path = os.environ.get("AURORA_SWARM_HOSTFILE")
        if hostfile_path:
            hostfile_path = Path(hostfile_path)
    if hostfile_path is None or not Path(hostfile_path).exists():
        print(
            "Error: No hostfile. Use --hostfile or set AURORA_SWARM_HOSTFILE",
            file=sys.stderr,
        )
        return 1

    hostfile_path = Path(hostfile_path)

    try:
        chunk_files = discover_chunk_files(
            args.input_dir, args.num_files, args.skip_files
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print_with_timestamp("=" * 80)
    print_with_timestamp("Tree-Reduce COLI Example")
    print_with_timestamp("=" * 80)
    print_with_timestamp(f"concurrency: {args.concurrency}")
    print_with_timestamp(f"connector_limit: {args.connector_limit}")
    print_with_timestamp(f"batch_concurrency: {args.batch_concurrency}")
    print_with_timestamp(f"timeout_per_sequence: {args.timeout_per_sequence}")
    if args.batch_timeout_cap is not None:
        print_with_timestamp(f"batch_timeout_cap: {args.batch_timeout_cap}")
    print_with_timestamp(f"fanin: {args.fanin}")
    print_with_timestamp(f"num_files: {len(chunk_files)}")
    print_with_timestamp(f"max_tokens (leaves): {args.max_tokens}")
    print_with_timestamp(f"max_tokens_contingency: {args.max_tokens_contingency}")

    endpoints = parse_hostfile(hostfile_path)
    print_with_timestamp(f"Loaded {len(endpoints)} endpoints from {hostfile_path}")

    pool = VLLMPool(
        endpoints,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        concurrency=args.concurrency,
        connector_limit=args.connector_limit,
        batch_concurrency=args.batch_concurrency,
        timeout_per_sequence=args.timeout_per_sequence,
        batch_timeout_cap=args.batch_timeout_cap,
    )

    plot_settings = (
        {
            "concurrency": args.concurrency,
            "connector_limit": args.connector_limit,
            "timeout": args.timeout,
            "fanin": args.fanin,
            "max_tokens": args.max_tokens,
            "num_files": len(chunk_files),
            "pool_size": len(endpoints),
        }
        if args.plot_sockets
        else None
    )

    try:
        final, timing = await run_tree_reduce_coli(
            pool,
            chunk_files,
            args.reduce_prompt,
            args.fanin,
            args.max_tokens,
            args.max_tokens_contingency,
            args.monitor_sockets,
            args.socket_interval,
            plot_sockets=args.plot_sockets,
            plot_output=args.plot_output if args.plot_sockets else None,
            settings=plot_settings,
        )
    finally:
        await pool.close()

    print_with_timestamp("=" * 80)
    print_with_timestamp("TIMING SUMMARY")
    print_with_timestamp("=" * 80)
    print_with_timestamp(f"Leaf phase:     {timing['leaf_seconds']:.2f} s")
    print_with_timestamp(f"Reduce phase:   {timing['reduce_seconds']:.2f} s")
    print_with_timestamp(f"Total:          {timing['total_seconds']:.2f} s")
    print_with_timestamp(f"Total prompts:  {timing['total_prompts']}")
    if timing["total_seconds"] > 0:
        print_with_timestamp(
            f"Throughput:     {timing['total_prompts'] / timing['total_seconds']:.2f} prompts/s"
        )
    print_with_timestamp("=" * 80)

    if final is None:
        print_with_timestamp("Tree-reduce failed: no valid output")
        return 1

    all_reduce_outputs = timing.get("all_reduce_outputs", [])

    if args.output:
        lines: list[str] = []
        if all_reduce_outputs:
            lines.append("=" * 80)
            lines.append("REDUCE OUTPUTS (by level and group)")
            lines.append("=" * 80)
            for level, group_idx, text in all_reduce_outputs:
                lines.append("")
                lines.append("-" * 80)
                lines.append(f"Level {level}, Group {group_idx}")
                lines.append("-" * 80)
                lines.append(text)
            lines.append("")
            lines.append("=" * 80)
            lines.append("FINAL SYNTHESIS")
            lines.append("=" * 80)
            lines.append("")
        lines.append(final)
        args.output.write_text("\n".join(lines), encoding="utf-8")
        print_with_timestamp(f"Output written to {args.output} ({len(all_reduce_outputs)} reduce outputs + final)")
    else:
        if all_reduce_outputs:
            print("\nReduce outputs:")
            for level, group_idx, text in all_reduce_outputs:
                print(f"\n--- Level {level}, Group {group_idx} ---")
                print(text)
        print("\nFinal result:")
        print("-" * 40)
        print(final)
        print("-" * 40)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
