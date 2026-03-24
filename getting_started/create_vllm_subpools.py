#!/usr/bin/env python3
"""Demonstrate VLLMPool subpools.

This shows how to create subpools from a parent `VLLMPool` via `slice()`,
then query each subpool's vLLM/OpenAI-compatible `/v1/models` endpoint
using the subpool's OpenAI clients.

Run from the repo root (after ``pip install -e .``)::

    python getting_started/create_vllm_subpools.py /path/to/agents.hostfile
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from aurora_swarm import VLLMPool, parse_hostfile


async def main() -> int:
    parser = argparse.ArgumentParser(description="Demo VLLMPool subpools.")
    parser.add_argument(
        "hostfile",
        help="Path to tab-separated hostfile (host<TAB>port<TAB>optional tags)",
    )
    args = parser.parse_args()

    endpoints = parse_hostfile(args.hostfile)
    if not endpoints:
        print("No endpoints found in hostfile.", file=sys.stderr)
        return 1

    async with VLLMPool(endpoints) as pool:
        print(f"parent pool size={pool.size}")
        for i, ep in enumerate(pool.endpoints):
            print(f"  parent_index=[{i}] {ep.url}")

        # Split into two halves (adjust as needed)
        mid = pool.size // 2
        subpools: list[tuple[str, VLLMPool]] = [
            ("left", pool.slice(0, mid)),
            ("right", pool.slice(mid, pool.size)),
        ]

        for sub_name, subpool in subpools:
            print(f"\n{sub_name} subpool size={subpool.size}")
            for sub_client_index, ep in enumerate(subpool.endpoints):
                # Index in the *parent* pool (just for clarity in the output)
                parent_client_index = next(
                    (
                        parent_i
                        for parent_i, parent_ep in enumerate(pool.endpoints)
                        if parent_ep == ep
                    ),
                    None,
                )

                client = subpool._openai_clients[sub_client_index]
                try:
                    resp = await client.models.list()
                    model_ids = [m.id for m in resp.data]
                    print(
                        f"  sub_client_index=[{sub_client_index}] "
                        f"parent_client_index=[{parent_client_index}] "
                        f"{ep.url}/v1/models -> {model_ids}"
                    )
                except Exception as e:
                    print(
                        f"  sub_client_index=[{sub_client_index}] "
                        f"parent_client_index=[{parent_client_index}] "
                        f"{ep.url}/v1/models -> error: {e}"
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

