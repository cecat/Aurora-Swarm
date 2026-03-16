"""AgentPool — async connection pool for Aurora agent endpoints.

Provides semaphore-throttled, pooled HTTP access to 1000–4000 LLM agent
instances.  Every public pattern function in this package takes an
``AgentPool`` as its first argument.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Sequence

import aiohttp

from aurora_swarm.hostfile import AgentEndpoint


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class Response:
    """Result of a single agent call."""

    success: bool
    text: str
    error: str | None = None
    agent_index: int = -1


# ---------------------------------------------------------------------------
# AgentPool
# ---------------------------------------------------------------------------

class AgentPool:
    """Async pool of agent HTTP endpoints with concurrency control.

    Parameters
    ----------
    endpoints:
        Agent endpoints — either :class:`AgentEndpoint` objects or plain
        ``(host, port)`` tuples (tags will be empty).
    concurrency:
        Maximum number of in-flight requests (asyncio semaphore size).
    connector_limit:
        Maximum number of TCP connections in the aiohttp pool.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        endpoints: Sequence[AgentEndpoint | tuple[str, int]],
        concurrency: int = 512,
        connector_limit: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self._endpoints: list[AgentEndpoint] = []
        for ep in endpoints:
            if isinstance(ep, AgentEndpoint):
                self._endpoints.append(ep)
            else:
                host, port = ep
                self._endpoints.append(AgentEndpoint(host=host, port=port))

        self._concurrency = concurrency
        self._connector_limit = connector_limit
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(concurrency)
        self._session: aiohttp.ClientSession | None = None

    # -- lifecycle -----------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=self._connector_limit)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "AgentPool":
        await self._get_session()  # Eagerly create session so parent owns it; sub-pools share it and close() on exit closes it
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- properties ----------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of agents in the pool."""
        return len(self._endpoints)

    @property
    def timeout(self) -> float:
        """Base per-request timeout in seconds."""
        return self._timeout

    @property
    def endpoints(self) -> list[AgentEndpoint]:
        return list(self._endpoints)

    # -- selectors -----------------------------------------------------------

    def by_tag(self, key: str, value: str) -> "AgentPool":
        """Return a sub-pool of agents whose tag *key* equals *value*."""
        filtered = [ep for ep in self._endpoints if ep.tags.get(key) == value]
        return self._sub_pool(filtered)

    def sample(self, n: int) -> "AgentPool":
        """Return a sub-pool of *n* randomly chosen agents."""
        chosen = random.sample(self._endpoints, min(n, len(self._endpoints)))
        return self._sub_pool(chosen)

    def select(self, indices: Sequence[int]) -> "AgentPool":
        """Return a sub-pool with agents at the given indices."""
        selected = [self._endpoints[i] for i in indices]
        return self._sub_pool(selected)

    def slice(self, start: int, stop: int) -> "AgentPool":
        """Return a sub-pool from index *start* to *stop*."""
        return self._sub_pool(self._endpoints[start:stop])

    def _sub_pool(self, endpoints: list[AgentEndpoint]) -> "AgentPool":
        """Create a child pool sharing concurrency settings."""
        child = AgentPool.__new__(AgentPool)
        child._endpoints = endpoints
        child._concurrency = self._concurrency
        child._connector_limit = self._connector_limit
        child._timeout = self._timeout
        child._semaphore = self._semaphore        # share parent semaphore
        child._session = self._session             # share parent session
        return child

    # -- core request --------------------------------------------------------

    async def post(self, agent_index: int, prompt: str, max_tokens: int | None = None) -> Response:
        """Send *prompt* to the agent at *agent_index* and return its response.

        The call is throttled by the pool-wide semaphore so that at most
        ``concurrency`` requests are in flight at once.

        Parameters
        ----------
        agent_index:
            Index of the agent to send the prompt to.
        prompt:
            The prompt text.
        max_tokens:
            Optional maximum tokens to generate. Ignored by base AgentPool
            (only used by VLLMPool and subclasses).
        """
        ep = self._endpoints[agent_index]
        session = await self._get_session()
        async with self._semaphore:
            try:
                async with session.post(
                    f"{ep.url}/generate",
                    json={"prompt": prompt},
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    data = await resp.json()
                    return Response(
                        success=True,
                        text=data.get("response", data.get("text", "")),
                        agent_index=agent_index,
                    )
            except Exception as exc:
                return Response(
                    success=False,
                    text="",
                    error=str(exc),
                    agent_index=agent_index,
                )

    async def send_all(self, prompts: list[str]) -> list[Response]:
        """Send ``prompts[i]`` to ``agent[i % size]`` concurrently.

        Returns responses in *input* order (i.e. ``results[i]``
        corresponds to ``prompts[i]``).
        """
        tasks = [
            self.post(i % self.size, prompt)
            for i, prompt in enumerate(prompts)
        ]
        return list(await asyncio.gather(*tasks))

    async def send_all_batched(self, prompts: list[str], max_tokens: int | None = None) -> list[Response]:
        """Send prompts with batching if supported, otherwise use send_all.

        Default implementation for base AgentPool — just delegates to send_all.
        VLLMPool overrides this to use batch API for efficiency.

        Parameters
        ----------
        prompts:
            List of prompts to send.
        max_tokens:
            Optional max tokens override (ignored in base implementation).

        Returns
        -------
        list[Response]
            Responses in the same order as input prompts.
        """
        return await self.send_all(prompts)

    async def broadcast_prompt(self, prompt: str) -> list[Response]:
        """Send the same *prompt* to every agent in the pool."""
        tasks = [self.post(i, prompt) for i in range(self.size)]
        return list(await asyncio.gather(*tasks))
