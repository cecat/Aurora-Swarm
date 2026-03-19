#!/usr/bin/env python3
"""Create a VLLMPool from a hostfile (minimal example).

Run from the repo root (after ``pip install -e .``)::

    python getting_started/create_vllm_pool.py /path/to/agents.hostfile

Or::

    PYTHONPATH=. python getting_started/create_vllm_pool.py /path/to/agents.hostfile
"""

from __future__ import annotations

import argparse
import sys

from aurora_swarm import VLLMPool, parse_hostfile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse a hostfile and construct a VLLMPool."
    )
    parser.add_argument(
        "hostfile",
        help="Path to tab-separated hostfile (host<TAB>port<TAB>optional tags)",
    )
    args = parser.parse_args()

    try:
        endpoints = parse_hostfile(args.hostfile)
    except OSError as e:
        print(f"Error reading hostfile: {e}", file=sys.stderr)
        return 1

    if not endpoints:
        print("No endpoints found in hostfile.", file=sys.stderr)
        return 1

    pool = VLLMPool(endpoints)
    print(f"VLLMPool created: {pool.size} agent(s)")
    for i, ep in enumerate(pool.endpoints):
        print(f"  [{i}] {ep.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
