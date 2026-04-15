"""
run_pilot.py — Entry point for the 1,000-agent SwarmSim pilot.

Usage:
    # Dry run (no LLM, validates all mechanics):
    python -m swarmsim.run_pilot --dry-run

    # Live run (requires vLLM endpoint):
    python -m swarmsim.run_pilot --endpoint http://localhost:8000

    # Custom parameters:
    python -m swarmsim.run_pilot \\
        --endpoint http://localhost:8000 \\
        --n-agents 1000 \\
        --n-days 90 \\
        --seed 42 \\
        --out-dir ./output

Options:
    --dry-run           Run without LLM (all decisions are fallback defaults).
    --endpoint URL      vLLM OpenAI-compatible endpoint base URL.
    --model NAME        Model name to pass to the endpoint (default: sim_config.MODEL).
    --n-agents N        Number of agents to simulate (default: 1000).
    --n-days N          Number of simulated days (default: 90).
    --seed N            Random seed (default: 42).
    --no-osmnx          Skip OSMnx real place geometry (faster startup).
    --out-dir DIR       Directory for output files (default: ./pilot_output).
    --log-level LEVEL   Logging verbosity: DEBUG, INFO, WARNING (default: INFO).

Output files (written to --out-dir):
    metrics.json        Per-day epidemic and behavioral metrics.
    decisions_sample.jsonl  Sample of agent decisions (1 per 100 agents per day).
    run_config.json     Snapshot of run parameters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SwarmSim 1K pilot simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run",   action="store_true",
                   help="Run without LLM (fallback decisions only)")
    p.add_argument("--endpoint",  default="",
                   help="Single vLLM endpoint as host:port (e.g. x4206c1s1b0n0:6739)")
    p.add_argument("--hostfile",  default="",
                   help="Aurora hostfile (tab-delimited host/port/tags); "
                        "used instead of --endpoint for multi-node runs. "
                        "Typically scripts/hostfile written by submit_oss120b.sh")
    p.add_argument("--model",     default="",
                   help="Model name (overrides sim_config.MODEL)")
    p.add_argument("--n-agents",  type=int, default=0,
                   help="Number of agents (0 = use sim_config.N_AGENTS)")
    p.add_argument("--n-days",    type=int, default=0,
                   help="Simulated days (0 = use sim_config.N_SIM_DAYS)")
    p.add_argument("--seed",      type=int, default=-1,
                   help="Random seed (-1 = use sim_config.RANDOM_SEED)")
    p.add_argument("--no-osmnx",  action="store_true",
                   help="Disable OSMnx place geometry (faster startup)")
    p.add_argument("--out-dir",   default="./pilot_output",
                   help="Output directory")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════════════
# Pool setup
# ═══════════════════════════════════════════════════════════════════════════════

def _build_pool(endpoint: str, hostfile: str, model: str):
    """
    Build an Aurora Swarm VLLMPool.

    Priority: hostfile > endpoint (single node).

    hostfile: path to tab-delimited Aurora hostfile written by submit_oss120b.sh
              Format per line: hostname <TAB> port [<TAB> key=value ...]
              Use parse_hostfile() from aurora_swarm.

    endpoint: single host:port string, e.g. "x4206c1s1b0n0:6739"
              Also accepts "hostname" (defaults to port 6739).
    """
    try:
        from aurora_swarm import VLLMPool, AgentEndpoint, parse_hostfile
    except ImportError:
        log.error("aurora_swarm not installed. Install it or use --dry-run.")
        sys.exit(1)

    if hostfile:
        try:
            endpoints = parse_hostfile(hostfile)
            log.info("Loaded %d endpoint(s) from hostfile: %s", len(endpoints), hostfile)
        except Exception as exc:
            log.error("Failed to parse hostfile %s: %s", hostfile, exc)
            sys.exit(1)
    else:
        # Parse single host:port
        if ":" in endpoint:
            host, port_str = endpoint.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                log.error("Could not parse port from endpoint: %s", endpoint)
                sys.exit(1)
        else:
            host = endpoint
            port = 6739    # Aurora default vLLM port
        endpoints = [AgentEndpoint(host=host, port=port)]
        log.info("Single endpoint: host=%s port=%d", host, port)

    try:
        pool = VLLMPool(endpoints=endpoints, model=model, use_batch=True)
        log.info("VLLMPool created: %d endpoint(s), model=%s", len(endpoints), model)
        return pool
    except Exception as exc:
        log.error("Failed to create VLLMPool: %s", exc)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Imports (deferred so --help works without dependencies) ───────────────
    import random
    from . import sim_config as cfg
    from .population     import generate_population
    from .schedules      import build_schedules
    from .disease_engine import seed_infections
    from .prompt         import build_agent_texts
    from .coordinator    import Coordinator, default_covid_timeline

    # ── Resolve config overrides ──────────────────────────────────────────────
    n_agents  = args.n_agents  or cfg.N_AGENTS
    n_days    = args.n_days    or cfg.N_SIM_DAYS
    seed      = args.seed if args.seed >= 0 else cfg.RANDOM_SEED
    model     = args.model     or cfg.MODEL
    use_osmnx = not args.no_osmnx

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run config snapshot ───────────────────────────────────────────────────
    run_config = {
        "timestamp":    datetime.now().isoformat(),
        "n_agents":     n_agents,
        "n_days":       n_days,
        "seed":         seed,
        "model":        model,
        "endpoint":     args.endpoint or args.hostfile or "(offline)",
        "dry_run":      args.dry_run,
        "use_osmnx":    use_osmnx,
        "sim_start":    cfg.SIM_START_DATE,
        "neighborhood": cfg.NEIGHBORHOOD,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    log.info("Run config: %s", run_config)

    # ── Population ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    log.info("Generating population (%d agents) ...", n_agents)
    agents, places = generate_population(
        n_agents=n_agents,
        seed=seed,
        use_osmnx=use_osmnx,
    )

    log.info("Building schedules ...")
    schedules = build_schedules(agents, places, seed=seed)

    log.info("Pre-rendering prompt blocks A+B ...")
    build_agent_texts(agents, places)

    log.info("Seeding infections (%d initial) ...", cfg.INITIAL_INFECTED)
    rng = random.Random(seed)
    seeded = seed_infections(agents, cfg.INITIAL_INFECTED, sim_day=0, rng=rng)
    log.info("Seeded agent IDs: %s", seeded)

    log.info("Setup complete in %.1fs", time.perf_counter() - t0)

    # ── Pool ──────────────────────────────────────────────────────────────────
    pool = None
    if not args.dry_run:
        if not args.endpoint and not args.hostfile:
            log.error("--endpoint or --hostfile is required unless --dry-run is set.")
            sys.exit(1)
        pool = _build_pool(args.endpoint, args.hostfile, model)
    else:
        log.info("DRY RUN mode: LLM calls disabled, fallback decisions used.")

    # ── Coordinator ───────────────────────────────────────────────────────────
    coord = Coordinator(
        agents=agents,
        places=places,
        schedules=schedules,
        pool=pool,
        policy_timeline=default_covid_timeline(),
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    t_run = time.perf_counter()
    log.info("Starting simulation: %d days ...", n_days)
    await coord.run(n_days=n_days, rng_seed=seed)
    elapsed = time.perf_counter() - t_run
    log.info("Simulation complete in %.1fs (%.1f sim-days/sec)",
             elapsed, n_days / elapsed if elapsed > 0 else 0)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    coord.save_metrics(metrics_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary(coord, n_agents, n_days, elapsed)


def _print_summary(coord, n_agents: int, n_days: int, elapsed: float) -> None:
    """Print a human-readable summary of the completed run."""
    from .disease_engine import epidemic_summary
    from .models import INFECTIOUS_STATES

    epi = epidemic_summary(coord.agents)

    print("\n" + "=" * 60)
    print(f"SwarmSim Pilot — Run Complete")
    print("=" * 60)
    print(f"  Agents:     {n_agents:,}")
    print(f"  Sim days:   {n_days}")
    print(f"  Wall time:  {elapsed:.1f}s")
    print()
    print("  Final disease state:")
    for state, count in sorted(epi.items(), key=lambda x: -x[1]):
        bar = "█" * (count * 40 // n_agents)
        print(f"    {state:<22} {count:>5}  {bar}")

    if coord.metrics:
        peak_inf = max(
            m.symptomatic_mild + m.symptomatic_severe +
            m.pre_symptomatic + m.asymptomatic
            for m in coord.metrics
        )
        peak_hosp = max(m.hospitalized + m.icu for m in coord.metrics)
        peak_dead = max(m.deceased for m in coord.metrics)
        print()
        print(f"  Peak infectious:    {peak_inf:>5}")
        print(f"  Peak hospitalized:  {peak_hosp:>5}")
        print(f"  Total deceased:     {peak_dead:>5}")
        print()
        total_llm      = sum(m.llm_calls    for m in coord.metrics)
        total_fallback = sum(m.llm_fallback for m in coord.metrics)
        print(f"  Total LLM calls:    {total_llm:>8,}")
        print(f"  Fallback decisions: {total_fallback:>8,}")
        if total_llm > 0:
            print(f"  Fallback rate:      {total_fallback/total_llm*100:.1f}%")

    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    """Synchronous wrapper for use as a console script entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
