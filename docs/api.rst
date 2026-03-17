API Reference
=============

Core
----

Package (aurora_swarm)
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aurora_swarm
   :members: AgentEndpoint, AgentPool, Response, VLLMPool, EmbeddingPool, EmbeddingResponse, parse_hostfile
   :undoc-members:
   :show-inheritance:

Hostfile
~~~~~~~~

.. autoclass:: aurora_swarm.hostfile.AgentEndpoint
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: aurora_swarm.hostfile.parse_hostfile

Pool
~~~~

.. autoclass:: aurora_swarm.pool.Response
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aurora_swarm.pool.AgentPool
   :members:
   :undoc-members:
   :show-inheritance:

VLLM pool
~~~~~~~~~

:class:`VLLMPool` is an :class:`~aurora_swarm.pool.AgentPool` subclass for vLLM OpenAI-compatible endpoints. It supports:

- Config-based and dynamic context length (see :doc:`context_length`)
- Batch prompting for high-throughput inference (see :doc:`batch_prompting`)
- Both ``/v1/completions`` (batch mode) and ``/v1/chat/completions`` (non-batch) endpoints

.. autoclass:: aurora_swarm.vllm_pool.VLLMPool
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _get_model_max_context, _sub_pool

Embedding pool
~~~~~~~~~~~~~~

:class:`EmbeddingPool` provides scatter-gather over OpenAI-compatible ``/v1/embeddings`` endpoints. It uses the same hostfile/endpoint model as :class:`~aurora_swarm.pool.AgentPool` (e.g. :func:`parse_hostfile` and :meth:`~aurora_swarm.embedding_pool.EmbeddingPool.by_tag` for role-based filtering). Use it with :func:`~aurora_swarm.patterns.embedding.scatter_gather_embeddings` for the same "pool + pattern" style as LLM scatter-gather.

.. autoclass:: aurora_swarm.embedding_pool.EmbeddingResponse
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aurora_swarm.embedding_pool.EmbeddingPool
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _get_clients, _sub_pool

Aggregators
~~~~~~~~~~~

See :doc:`aggregators` for usage guide and examples.

.. automodule:: aurora_swarm.aggregators
   :members: majority_vote, concat, best_of, top_k, structured_merge, statistics, failure_report
   :undoc-members:
   :show-inheritance:

Communication patterns
----------------------

Broadcast
~~~~~~~~~

.. autofunction:: aurora_swarm.patterns.broadcast.broadcast
.. autofunction:: aurora_swarm.patterns.broadcast.broadcast_and_reduce

Scatter-Gather
~~~~~~~~~~~~~~

.. autofunction:: aurora_swarm.patterns.scatter_gather.scatter_gather
.. autofunction:: aurora_swarm.patterns.scatter_gather.map_gather

Scatter-Gather (embeddings)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: aurora_swarm.patterns.embedding.scatter_gather_embeddings

Tree-Reduce
~~~~~~~~~~~

.. autofunction:: aurora_swarm.patterns.tree_reduce.tree_reduce

Blackboard
~~~~~~~~~~

.. autoclass:: aurora_swarm.patterns.blackboard.Blackboard
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline
~~~~~~~~

.. autoclass:: aurora_swarm.patterns.pipeline.Stage
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: name, prompt_template, n_agents, output_transform, output_filter

.. autofunction:: aurora_swarm.patterns.pipeline.run_pipeline
