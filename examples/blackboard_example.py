"""Blackboard pattern example with prompt size control and optional logging.

Runs the Blackboard pattern (hypotheses + critiques) against role-tagged vLLM
endpoints. Supports limiting the board view sent to agents (last_n, max_chars, or
full) to keep prompts within context; uses VLLMPool's dynamic context length.
Optionally logs prompts and responses to a directory for debugging.

HOSTFILE:
  Must contain role=hypotheses and role=critiques (e.g. from
  scripts/submit_oss120b.sh with blackboard=1).

USAGE:
  python examples/blackboard_example.py --hostfile agents.txt --max-rounds 3
  python examples/blackboard_example.py --hostfile agents.txt \\
      --board-limit last_n --board-limit-value 10 --log-dir /tmp/bb_log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

from aurora_swarm import VLLMPool, parse_hostfile
from aurora_swarm.patterns.blackboard import Blackboard
from aurora_swarm.pool import Response

BoardState = dict[str, list[str]]
BoardLimitStrategy = Literal["full", "last_n", "max_chars"]


def print_with_timestamp(message: str) -> None:
    """Print message with timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr)


def board_view_for_prompt(
    board: BoardState,
    strategy: BoardLimitStrategy,
    value: int,
) -> str:
    """Build a string representation of the board for inclusion in the prompt.

    Parameters
    ----------
    board
        Current board state (section -> list of entries).
    strategy
        "full" = include all entries; "last_n" = last N per section;
        "max_chars" = truncate to value chars at line boundary.
    value
        For last_n: max entries per section. For max_chars: max character length.
        Ignored for full.

    Returns
    -------
    str
        Formatted board view for the prompt.
    """
    lines: list[str] = []
    for section, entries in board.items():
        if strategy == "last_n" and value >= 0:
            entries = entries[-value:] if value else []
        block = [f"**{section}** ({len(entries)} entries):"]
        for i, e in enumerate(entries, 1):
            block.append(f"  {i}. {e}")
        lines.append("\n".join(block))
    text = "\n\n".join(lines) if lines else "(Board is empty.)"

    if strategy == "max_chars" and value > 0 and len(text) > value:
        text = text[: value].rsplit("\n", 1)[0] if "\n" in text[:value] else text[:value]
    return text


def make_prompt_fn(
    board_limit: BoardLimitStrategy,
    board_limit_value: int,
):
    """Build prompt_fn(role, board) -> str that uses board_view_for_prompt."""

    def prompt_fn(role: str, board: BoardState) -> str:
        view = board_view_for_prompt(board, board_limit, board_limit_value)
        if role == "hypotheses":
            return (
                "You are a scientist proposing hypotheses. "
                "Current board state:\n\n"
                f"{view}\n\n"
                "Propose a single new hypothesis (one short sentence)."
            )
        else:
            return (
                "You are a critical reviewer. "
                "Current board state:\n\n"
                f"{view}\n\n"
                "Provide a brief one-sentence critique of the most recent hypothesis."
            )

    return prompt_fn


def write_log(
    log_dir: Path,
    round_num: int,
    section: str,
    prompt: str,
    responses: list[Response],
) -> None:
    """Write prompt and responses to log_dir for one round/section."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"round_{round_num}_section_{section}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"round": round_num, "section": section, "kind": "prompt", "length": len(prompt), "text": prompt},
                ensure_ascii=False,
            )
            + "\n"
        )
        for i, r in enumerate(responses):
            rec = {
                "round": round_num,
                "section": section,
                "kind": "response",
                "index": i,
                "success": r.success,
                "length": len(r.text) if r.text else 0,
                "text": r.text,
                "error": r.error,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def run_blackboard_with_logging(
    pool: VLLMPool,
    sections: list[str],
    prompt_fn: Callable[[str, BoardState], str],
    max_rounds: int,
    convergence_fn: Callable[[BoardState], bool] | None,
    log_dir: Path,
) -> tuple[BoardState, int]:
    """Run blackboard loop (same logic as Blackboard.run) and log each prompt/response.
    Returns (final_board, rounds_completed).
    """
    board: BoardState = {s: [] for s in sections}
    round_num = 0
    for _ in range(max_rounds):
        for section in sections:
            sub = pool.by_tag("role", section)
            if sub.size == 0:
                continue
            prompt = prompt_fn(section, board)
            responses = await sub.broadcast_prompt(prompt)
            write_log(log_dir, round_num + 1, section, prompt, responses)
            for r in responses:
                if r.success:
                    board[section].append(r.text)
        round_num += 1
        if convergence_fn is not None and convergence_fn(board):
            break
    return board, round_num


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blackboard example: hypotheses + critiques with prompt size control and optional logging.",
    )
    parser.add_argument(
        "--hostfile",
        type=Path,
        default=None,
        help="Path to hostfile with role=hypotheses and role=critiques (or AURORA_SWARM_HOSTFILE)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-120b",
        help="Model name (default: openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens per response (default: 1024); dynamic sizing may reduce for long prompts",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum rounds (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Max concurrent requests (default: 64)",
    )
    parser.add_argument(
        "--board-limit",
        choices=["full", "last_n", "max_chars"],
        default="last_n",
        help="How to limit board size in prompts: full, last_n, or max_chars (default: last_n)",
    )
    parser.add_argument(
        "--board-limit-value",
        type=int,
        default=10,
        help="For last_n: entries per section; for max_chars: max character length (default: 10)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="If set, log prompts and responses under this directory (for debugging)",
    )
    parser.add_argument(
        "--convergence-entries",
        type=int,
        default=None,
        help="Stop when total board entries >= N (optional)",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    hostfile_path = args.hostfile
    if hostfile_path is None:
        hostfile_path = os.environ.get("AURORA_SWARM_HOSTFILE")
        hostfile_path = Path(hostfile_path) if hostfile_path else None
    if not hostfile_path or not hostfile_path.exists():
        print("Error: No hostfile. Use --hostfile or set AURORA_SWARM_HOSTFILE.", file=sys.stderr)
        print("Hostfile must have role=hypotheses and role=critiques.", file=sys.stderr)
        return 1

    endpoints = parse_hostfile(hostfile_path)
    roles = set()
    for ep in endpoints:
        r = ep.tags.get("role")
        if r:
            roles.add(r)
    if "hypotheses" not in roles or "critiques" not in roles:
        print("Error: Hostfile must contain endpoints with role=hypotheses and role=critiques.", file=sys.stderr)
        return 1

    print_with_timestamp("Blackboard Example (prompt size control + optional logging)")
    print_with_timestamp(f"Endpoints: {len(endpoints)}, board-limit: {args.board_limit}, value: {args.board_limit_value}")
    if args.log_dir:
        print_with_timestamp(f"Logging to: {args.log_dir}")

    async with VLLMPool(
        endpoints,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        concurrency=args.concurrency,
    ) as pool:
        sections = ["hypotheses", "critiques"]
        prompt_fn = make_prompt_fn(args.board_limit, args.board_limit_value)

        convergence_fn = None
        if args.convergence_entries is not None:
            n = args.convergence_entries

            def converge(board: BoardState) -> bool:
                return sum(len(v) for v in board.values()) >= n

            convergence_fn = converge

        start = datetime.now()
        if args.log_dir:
            final_board, rounds_run = await run_blackboard_with_logging(
                pool,
                sections,
                prompt_fn,
                args.max_rounds,
                convergence_fn,
                args.log_dir,
            )
        else:
            bb = Blackboard(sections=sections, prompt_fn=prompt_fn)
            final_board = await bb.run(
                pool,
                max_rounds=args.max_rounds,
                convergence_fn=convergence_fn,
            )
            rounds_run = bb.round

        elapsed = (datetime.now() - start).total_seconds()

        print_with_timestamp("SUMMARY")
        print_with_timestamp(f"  Rounds: {rounds_run}")
        print_with_timestamp(f"  Hypotheses: {len(final_board['hypotheses'])}")
        print_with_timestamp(f"  Critiques: {len(final_board['critiques'])}")
        print_with_timestamp(f"  Elapsed: {elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
