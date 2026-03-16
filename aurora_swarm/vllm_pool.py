"""VLLMPool — AgentPool subclass for vLLM OpenAI-compatible endpoints.

vLLM exposes an OpenAI-compatible chat completions API at
``/v1/chat/completions``.  This pool overrides :meth:`post` to speak
that protocol instead of the simpler ``/generate`` endpoint used by
the base :class:`AgentPool`.
"""

from __future__ import annotations

import asyncio
import math
import os
import aiohttp

from openai import AsyncOpenAI

# Retry on connection/timeout errors (transient when many concurrent connections open)
def _is_retryable_connection_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    return (
        name
        in (
            "APIConnectionError",
            "APITimeoutError",
            "ConnectionError",
            "TimeoutError",
        )
        or "connection" in msg
        or "timeout" in msg
    )

from aurora_swarm.hostfile import AgentEndpoint
from aurora_swarm.pool import AgentPool, Response


class VLLMPool(AgentPool):
    """Agent pool that communicates via vLLM's OpenAI-compatible API.

    Parameters
    ----------
    endpoints:
        Agent endpoints (host + port where vLLM is listening).
    model:
        Model identifier passed in the ``"model"`` field of every
        request (e.g. ``"openai/gpt-oss-120b"``).
    max_tokens:
        Maximum tokens to generate per request (default context).
        Can be overridden via ``AURORA_SWARM_MAX_TOKENS`` env var.
    max_tokens_aggregation:
        Maximum tokens for aggregation/reduce steps (larger prompts).
        Can be overridden via ``AURORA_SWARM_MAX_TOKENS_AGGREGATION`` env var.
        Defaults to 2 * max_tokens if not specified.
    model_max_context:
        Model's maximum context length. If None, will be fetched from
        vLLM's ``/v1/models`` endpoint on first request. Can be overridden
        via ``AURORA_SWARM_MODEL_MAX_CONTEXT`` env var.
    buffer:
        Safety margin (in tokens) for dynamic sizing to account for
        reasoning overhead. Defaults to 512.
    use_batch:
        If True, use batch prompting via the completions API for
        send_all_batched. If False, fall back to individual requests.
        Defaults to True.
    concurrency:
        Maximum number of in-flight requests.
    connector_limit:
        Maximum TCP connections in the aiohttp pool.
    timeout:
        Base per-request timeout in seconds. Single requests use this;
        batch requests use max(timeout, scaled) where scaled depends on batch size.
    batch_concurrency:
        vLLM's max concurrent sequences (waves). Used to scale batch timeout;
        default 256.
    timeout_per_sequence:
        Estimated seconds per sequence for batch timeout scaling. Can be set via
        AURORA_SWARM_TIMEOUT_PER_SEQUENCE. Default 60.0.
    batch_timeout_cap:
        If set, cap the computed batch timeout so one huge batch does not get
        an extreme value. Optional.
    """

    def __init__(
        self,
        endpoints: list[AgentEndpoint],
        model: str = "openai/gpt-oss-120b",
        max_tokens: int | None = None,
        max_tokens_aggregation: int | None = None,
        model_max_context: int | None = None,
        buffer: int = 512,
        use_batch: bool = True,
        concurrency: int = 512,
        connector_limit: int = 1024,
        timeout: float = 300.0,
        batch_concurrency: int = 256,
        timeout_per_sequence: float | None = None,
        batch_timeout_cap: float | None = None,
    ) -> None:
        super().__init__(
            endpoints,
            concurrency=concurrency,
            connector_limit=connector_limit,
            timeout=timeout,
        )
        self._model = model
        self._use_batch = use_batch
        self._batch_concurrency = batch_concurrency
        self._timeout_per_sequence = (
            timeout_per_sequence
            if timeout_per_sequence is not None
            else float(os.environ.get("AURORA_SWARM_TIMEOUT_PER_SEQUENCE", "60.0"))
        )
        self._batch_timeout_cap = batch_timeout_cap

        # Load from environment with fallbacks
        self._max_tokens = (
            max_tokens
            or int(os.environ.get("AURORA_SWARM_MAX_TOKENS", "512"))
        )
        self._max_tokens_aggregation = (
            max_tokens_aggregation
            or int(os.environ.get("AURORA_SWARM_MAX_TOKENS_AGGREGATION", str(self._max_tokens * 2)))
        )
        self._model_max_context = (
            model_max_context
            or (int(os.environ["AURORA_SWARM_MODEL_MAX_CONTEXT"]) if "AURORA_SWARM_MODEL_MAX_CONTEXT" in os.environ else None)
        )
        self._buffer = buffer
        self._model_max_context_cached: int | None = None
        
        # Create OpenAI clients for each endpoint (for batch requests)
        self._openai_clients: dict[int, AsyncOpenAI] = {}
        for i, ep in enumerate(self._endpoints):
            self._openai_clients[i] = AsyncOpenAI(
                base_url=f"{ep.url}/v1",
                api_key="EMPTY",  # vLLM convention
                timeout=timeout,
            )

    # -- model metadata -------------------------------------------------------

    async def _get_model_max_context(self) -> int:
        """Fetch the model's max context length from vLLM /v1/models endpoint.
        
        Cached after first call. Returns a sensible default if fetch fails.
        """
        # Return cached value if available
        if self._model_max_context_cached is not None:
            return self._model_max_context_cached
        
        # Return explicitly configured value
        if self._model_max_context is not None:
            self._model_max_context_cached = self._model_max_context
            return self._model_max_context
        
        # Fetch from vLLM API
        try:
            ep = self._endpoints[0]
            session = await self._get_session()
            async with session.get(
                f"{ep.url}/v1/models",
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                data = await resp.json()
                # Find our model in the list
                for model_info in data.get("data", []):
                    if model_info.get("id") == self._model:
                        max_len = model_info.get("max_model_len")
                        if max_len:
                            self._model_max_context_cached = max_len
                            return max_len
        except Exception:
            pass  # Fall back to default
        
        # Default fallback (131072 is common for many models)
        self._model_max_context_cached = 131072
        return self._model_max_context_cached

    # -- core request (OpenAI chat completions) ------------------------------

    async def post(self, agent_index: int, prompt: str, max_tokens: int | None = None) -> Response:
        """Send *prompt* via the OpenAI chat-completions API on the agent.

        The prompt is wrapped as a single ``user`` message.

        Parameters
        ----------
        agent_index:
            Index of the agent to send the prompt to.
        prompt:
            The prompt text.
        max_tokens:
            Optional override for max tokens. If None, uses dynamic sizing
            based on prompt length and model context limit.
        """
        ep = self._endpoints[agent_index]
        session = await self._get_session()
        
        # Compute max_tokens dynamically if not explicitly provided
        if max_tokens is None:
            # Get model's max context length
            model_max = await self._get_model_max_context()
            
            # Estimate prompt tokens (rough heuristic: 1 token ≈ 4 chars)
            prompt_est = len(prompt) // 4
            
            # Dynamic sizing: never exceed model capacity
            # Use default max_tokens as the preferred cap
            tokens = min(
                self._max_tokens,
                max(128, model_max - prompt_est - self._buffer)
            )
        else:
            tokens = max_tokens

        async with self._semaphore:
            try:
                async with session.post(
                    f"{ep.url}/v1/chat/completions",
                    json={
                        "model": self._model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": tokens,
                    },
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    data = await resp.json()
                    
                    # Check for error response
                    if resp.status != 200:
                        error_msg = data.get("error", {}).get("message", f"HTTP {resp.status}")
                        return Response(
                            success=False,
                            text="",
                            error=f"API error: {error_msg}",
                            agent_index=agent_index,
                        )
                    
                    # Check for expected response structure
                    if "choices" not in data or not data["choices"]:
                        return Response(
                            success=False,
                            text="",
                            error=f"Invalid response structure: {list(data.keys())}",
                            agent_index=agent_index,
                        )
                    
                    message = data["choices"][0]["message"]
                    text = message.get("content") or message.get("reasoning_content") or ""
                    return Response(
                        success=True,
                        text=text,
                        agent_index=agent_index,
                    )
            except Exception as exc:
                return Response(
                    success=False,
                    text="",
                    error=f"{type(exc).__name__}: {str(exc)}",
                    agent_index=agent_index,
                )

    async def post_batch(
        self,
        agent_index: int,
        prompts: list[str],
        max_tokens: int | None = None,
    ) -> list[Response]:
        """Send multiple prompts to one agent via the completions API.

        Uses the OpenAI completions endpoint which supports batch prompts
        (a list of strings). This reduces N HTTP requests to 1.

        Parameters
        ----------
        agent_index:
            Index of the agent to send prompts to.
        prompts:
            List of prompts to send in one batch.
        max_tokens:
            Optional override for max tokens. If None, uses dynamic sizing
            based on average prompt length.

        Returns
        -------
        list[Response]
            One Response per prompt, in the same order as the input.
        """
        if not prompts:
            return []

        ep = self._endpoints[agent_index]
        client = self._openai_clients[agent_index]

        # Compute max_tokens dynamically if not explicitly provided
        if max_tokens is None:
            # Get model's max context length
            model_max = await self._get_model_max_context()

            # Estimate tokens based on average prompt length
            avg_prompt_len = sum(len(p) for p in prompts) // len(prompts)
            prompt_est = avg_prompt_len // 4

            # Dynamic sizing: never exceed model capacity
            tokens = min(
                self._max_tokens,
                max(128, model_max - prompt_est - self._buffer)
            )
        else:
            tokens = max_tokens

        # Batch-size-dependent timeout: scale with number of waves
        n = len(prompts)
        waves = max(1, math.ceil(n / self._batch_concurrency))
        scaled = waves * self._timeout_per_sequence
        effective_timeout = max(self._timeout, scaled)
        if self._batch_timeout_cap is not None:
            effective_timeout = min(effective_timeout, self._batch_timeout_cap)

        async with self._semaphore:
            last_exc: BaseException | None = None
            for attempt in range(6):  # 6 attempts total (0..5)
                try:
                    # Call completions API with batch prompts
                    response = await client.completions.create(
                        model=self._model,
                        prompt=prompts,
                        max_tokens=tokens,
                        timeout=effective_timeout,
                    )

                    # Map choices to Response objects
                    results: list[Response] = []
                    for i, choice in enumerate(response.choices):
                        results.append(
                            Response(
                                success=True,
                                text=choice.text,
                                agent_index=agent_index,
                            )
                        )
                    return results

                except Exception as exc:
                    last_exc = exc
                    if attempt < 5 and _is_retryable_connection_error(exc):
                        await asyncio.sleep(2**attempt)
                        continue
                    # On error, return failed Response for each prompt
                    return [
                        Response(
                            success=False,
                            text="",
                            error=f"{type(exc).__name__}: {str(exc)}",
                            agent_index=agent_index,
                        )
                        for _ in prompts
                    ]
            # Should not reach here; if we do, treat last_exc as final failure
            assert last_exc is not None
            return [
                Response(
                    success=False,
                    text="",
                    error=f"{type(last_exc).__name__}: {str(last_exc)}",
                    agent_index=agent_index,
                )
                for _ in prompts
            ]

    async def send_all_batched(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
    ) -> list[Response]:
        """Send prompts using batch API, grouping by target agent.

        Groups prompts by their target agent (round-robin based on index),
        then sends one batched request per agent. Reconstructs results in
        input order.

        Parameters
        ----------
        prompts:
            List of prompts to send.
        max_tokens:
            Optional max tokens override.

        Returns
        -------
        list[Response]
            Responses in the same order as input prompts.
        """
        if not self._use_batch or not prompts:
            # Fall back to non-batched send_all
            return await self.send_all(prompts)

        # Group prompts by target agent
        groups: dict[int, list[tuple[int, str]]] = {}
        for i, prompt in enumerate(prompts):
            agent_idx = i % self.size
            if agent_idx not in groups:
                groups[agent_idx] = []
            groups[agent_idx].append((i, prompt))

        # Send batched requests concurrently
        async def send_group(agent_idx: int, items: list[tuple[int, str]]) -> list[tuple[int, Response]]:
            """Send batch to one agent, return (original_index, response) pairs."""
            group_prompts = [prompt for _, prompt in items]
            responses = await self.post_batch(agent_idx, group_prompts, max_tokens)
            return [(items[j][0], responses[j]) for j in range(len(items))]

        tasks = [send_group(agent_idx, items) for agent_idx, items in groups.items()]
        all_results = await asyncio.gather(*tasks)

        # Flatten and sort by original index
        indexed_responses: list[tuple[int, Response]] = []
        for result_group in all_results:
            indexed_responses.extend(result_group)
        indexed_responses.sort(key=lambda x: x[0])

        # Extract responses in order
        return [resp for _, resp in indexed_responses]

    # -- sub-pool override ---------------------------------------------------

    def _sub_pool(self, endpoints: list[AgentEndpoint]) -> "VLLMPool":
        """Create a child VLLMPool sharing concurrency settings."""
        child = VLLMPool.__new__(VLLMPool)
        child._endpoints = endpoints
        child._concurrency = self._concurrency
        child._connector_limit = self._connector_limit
        child._timeout = self._timeout
        child._batch_concurrency = self._batch_concurrency
        child._timeout_per_sequence = self._timeout_per_sequence
        child._batch_timeout_cap = self._batch_timeout_cap
        child._semaphore = self._semaphore
        child._session = self._session
        child._model = self._model
        child._use_batch = self._use_batch
        child._max_tokens = self._max_tokens
        child._max_tokens_aggregation = self._max_tokens_aggregation
        child._model_max_context = self._model_max_context
        child._buffer = self._buffer
        child._model_max_context_cached = self._model_max_context_cached
        child._openai_clients = self._openai_clients  # Share clients
        return child
