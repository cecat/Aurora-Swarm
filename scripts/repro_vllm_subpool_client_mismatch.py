"""
Repro: VLLMPool sub-pool shares _openai_clients keyed by parent indices.

Expected: sub_pool.post_batch(agent_index=0, ...) should hit sub_pool._endpoints[0]
Actual (bug): it may use parent client index 0 (wrong base_url).
"""

from __future__ import annotations

import asyncio
from aiohttp import web

from aurora_swarm.hostfile import AgentEndpoint
from aurora_swarm.vllm_pool import VLLMPool


def _make_app(server_id: str) -> web.Application:
    app = web.Application()

    async def handle_completions(request: web.Request) -> web.Response:
        data = await request.json()
        prompt = data.get("prompt")
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt or [])
        choices = [
            {"text": f"{server_id} completion: {p}", "index": i, "finish_reason": "stop"}
            for i, p in enumerate(prompts)
        ]
        return web.json_response(
            {
                "id": f"mock-{server_id}",
                "object": "text_completion",
                "created": 1234567890,
                "model": "mock-model",
                "choices": choices,
            }
        )

    async def handle_models(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": "mock-model",
                        "object": "model",
                        "created": 1234567890,
                        "owned_by": "test",
                        "max_model_len": 8192,
                    }
                ],
            }
        )

    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_get("/v1/models", handle_models)
    return app


async def _start_server(server_id: str) -> tuple[web.AppRunner, int]:
    app = _make_app(server_id)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, port


async def main() -> None:
    # Start two distinct servers
    runner_a, port_a = await _start_server("SERVER_A")
    runner_b, port_b = await _start_server("SERVER_B")

    endpoints = [
        AgentEndpoint(host="127.0.0.1", port=port_a),
        AgentEndpoint(host="127.0.0.1", port=port_b),
    ]

    pool = VLLMPool(endpoints, model="mock-model", max_tokens=16, concurrency=4, connector_limit=8, use_batch=True)
    try:
        async with pool:
            sub = pool.slice(1, 2)  # child has only SERVER_B at index 0
            responses = await sub.post_batch(agent_index=0, prompts=["ping"])
            print("Sub-pool endpoint[0]:", sub.endpoints[0].url)
            print("Response text:", responses[0].text)
            print("Expected response to include: SERVER_B")
    finally:
        await runner_a.cleanup()
        await runner_b.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

