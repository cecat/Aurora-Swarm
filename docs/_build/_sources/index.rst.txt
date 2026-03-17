Aurora Swarm
============

Communication patterns for orchestrating large-scale LLM agent swarms on Aurora.

Aurora Swarm provides an async Python library for coordinating thousands of LLM agent endpoints using common distributed communication patterns — **broadcast**, **scatter-gather**, **tree-reduce**, **blackboard**, and **multi-stage pipelines**. It manages pooled HTTP connections with semaphore-based concurrency control so you can safely drive 1,000–4,000+ agents from a single orchestrator process.

Quick start
-----------

.. code-block:: python

   import asyncio
   from aurora_swarm import AgentPool, parse_hostfile
   from aurora_swarm.patterns.broadcast import broadcast_and_reduce

   async def main():
       endpoints = parse_hostfile("agents.hostfile")
       async with AgentPool(endpoints) as pool:
           result = await broadcast_and_reduce(
               pool,
               prompt="Propose a hypothesis for why X happens.",
               reduce_prompt="Summarize these hypotheses:\n{responses}",
           )
           print(result.text)

   asyncio.run(main())

Key Features
------------

**Batch Prompting for High Throughput**

VLLMPool supports batch prompting to dramatically reduce HTTP overhead:

- 10,000 prompts with 100 agents = 100 requests instead of 10,000
- 100× reduction in request count improves throughput significantly
- Enabled by default, transparent to existing code
- See :doc:`batch_prompting` for details

**Aggregators** — Combine broadcast (or other) responses with :doc:`aggregators` (e.g. ``majority_vote``, ``concat``). See the runnable example ``examples/broadcast_aggregators.py``.

**Tree-Reduce** — See the runnable example ``examples/tree_reduce_example.py``.

**Embedding scatter-gather** — Use :class:`~aurora_swarm.embedding_pool.EmbeddingPool` with :func:`~aurora_swarm.patterns.embedding.scatter_gather_embeddings` for scatter-gather over ``/v1/embeddings`` endpoints. Same hostfile and ``by_tag`` usage as LLM pools. See :doc:`api` (Embedding pool, Scatter-Gather (embeddings)).

Communication patterns
----------------------

.. list-table:: Communication patterns
   :header-rows: 1
   :widths: 20 80

   * - Pattern
     - Description
   * - **Broadcast**
     - Send the same prompt to every agent and collect all responses.
   * - **Scatter-Gather**
     - Distribute different prompts across agents round-robin and gather results in input order.
   * - **Tree-Reduce**
     - Leaf agents produce answers; supervisors recursively summarize groups.
   * - **Blackboard**
     - Agents collaborate through a shared mutable workspace in iterative rounds until convergence.
   * - **Pipeline**
     - Multi-stage DAG where the output of one stage feeds the next.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents

   batch_prompting
   context_length
   pools_and_connections
   aggregators
   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
