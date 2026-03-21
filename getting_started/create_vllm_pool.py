#!/usr/bin/env python3
"""Create a VLLMPool from a hostfile (minimal example).

Run from the repo root (after ``pip install -e .``)::

    python getting_started/create_vllm_pool.py /path/to/agents.hostfile

"""

from __future__ import annotations
import argparse
import asyncio
import sys
from aurora_swarm import VLLMPool, parse_hostfile

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse a hostfile and construct a VLLMPool."
    )
    parser.add_argument(
        "hostfile",
        help="Path to tab-separated hostfile (host<TAB>port<TAB>optional tags)",
    )
    args = parser.parse_args()

    endpoints = parse_hostfile(args.hostfile)
    if not endpoints:
        print("No endpoints found in hostfile.", file=sys.stderr)
        return 1

    pool = VLLMPool(endpoints)
    print(f"VLLMPool created: {pool.size} agent(s)")
    for i, ep in enumerate(pool.endpoints):
        print(f"  [{i}] {ep.url}")
        client = pool._openai_clients[i]
        try:
            resp = await client.models.list()
            model_ids = [m.id for m in resp.data]
            print(f"  client_index={i} {ep.url}/v1/models -> {len(model_ids)} model(s): {model_ids}")
        except Exception as e:
            print(f"  client_index={i} {ep.url}/v1/models -> error: {e}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
