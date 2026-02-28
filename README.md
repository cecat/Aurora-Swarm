# Aurora Swarm


Communication patterns for orchestrating large-scale LLM agent swarms on Aurora.

Aurora Swarm provides an async Python library for coordinating thousands of LLM agent endpoints using common distributed communication patterns — broadcast, scatter-gather, tree-reduce, blackboard, and multi-stage pipelines. It manages pooled HTTP connections with semaphore-based concurrency control so you can safely drive 1,000–4,000+ agents from a single orchestrator process.

**API reference (GitHub Pages):** [https://brettin.github.io/Aurora-Swarm/](https://brettin.github.io/Aurora-Swarm/)

---

## Getting Started

### Prerequisites

- Access to a system with [Conda](https://docs.conda.io/) (Miniconda or Anaconda)
- Python 3.11 or later
- Git

### 1. Clone the repository

```bash
git clone https://github.com/your-org/Aurora-Swarm.git
cd Aurora-Swarm
```

### 2. Create and activate a Conda environment

**On HPC (e.g. Aurora):** load the frameworks module, create a conda env, and activate the conda env before any `pip install`:

```bash
module load frameworks
conda create -p /lus/flare/projects/ModCon/brettin/conda_envs/swarm python=3
conda activate /lus/flare/projects/ModCon/brettin/conda_envs/swarm
```

Update the Aurora-Swarm/env.sh file to include the new conda environment that you just created.

```bash
cat env.sh
```

### 3. Install the package

Install in editable (development) mode so that local changes are picked up immediately:

```bash
pip install -e .
```

This pulls in the core runtime dependencies (`aiohttp>=3.9`, `openai>=1.0`).

### 4. Install development / test dependencies

```bash
pip install -e ".[dev]"
```

This adds `pytest`, `pytest-asyncio`, and everything needed to run the test suite.

### 5. Verify the installation

```bash
python -c "import aurora_swarm; print('aurora_swarm imported successfully')"
```

### 6. Run the tests

```bash
pytest -v
```

The tests spin up lightweight mock HTTP servers on localhost so no external agents are needed.

### 7. Build and view the documentation (optional)

Published docs are at **https://brettin.github.io/Aurora-Swarm/** (built and deployed from `main` via GitHub Actions). In repo **Settings → Pages**, set "Build and deployment" source to **GitHub Actions** if it is not already.

To build and view locally (on HPC, load the environment first):

```bash
module load frameworks
conda activate /lus/flare/projects/ModCon/brettin/conda_envs/swarm
pip install -e ".[docs]"
cd docs && make html && cd ..   # output in docs/_build/
# then open docs/_build/index.html in a browser
```

---

## Project Structure

```
Aurora-Swarm/
├── aurora_swarm/
│   ├── __init__.py            # Public API re-exports
│   ├── hostfile.py            # Hostfile parser (AgentEndpoint)
│   ├── pool.py                # AgentPool — async connection pool
│   ├── vllm_pool.py           # VLLMPool — OpenAI-compatible batch prompting
│   ├── aggregators.py         # Response aggregation strategies
│   └── patterns/
│       ├── __init__.py
│       ├── broadcast.py       # Pattern 1 — Broadcast
│       ├── scatter_gather.py  # Pattern 2 — Scatter-Gather
│       ├── tree_reduce.py     # Pattern 3 — Tree-Reduce
│       ├── blackboard.py      # Pattern 4 — Blackboard (shared-state)
│       └── pipeline.py        # Pattern 5 — Pipeline (multi-stage DAG)
├── examples/
│   ├── broadcast_aggregators.py   # Broadcast to all-but-one + majority_vote/concat
│   ├── broadcast_aggregators.sh   # Launcher for broadcast_aggregators.py
│   ├── scatter_gather_coli.py     # TOM.COLI gene analysis example
│   ├── scatter_gather_coli.sh     # Launcher for scatter_gather_coli.py
│   ├── tree_reduce_example.py     # Hierarchical tree-reduce summarization
│   ├── tree_reduce_example.sh     # Launcher for tree_reduce_example.py
│   ├── blackboard_example.py      # Blackboard pattern with prompt size control
│   ├── blackboard.sh              # Launcher for blackboard_example.py
│   └── context_length_demo.py     # Context length configuration demo
├── scripts/
│   └── wait_for_vllm_servers.py # Wait for hostfile + healthy vLLM; write filtered hostfile
├── tests/
│   ├── conftest.py            # Shared fixtures (mock servers, pools)
│   ├── test_broadcast.py
│   ├── test_scatter_gather.py
│   ├── test_tree_reduce.py
│   ├── test_blackboard.py
│   ├── test_pipeline.py
│   ├── test_vllm_pool.py      # VLLMPool batch prompting tests
│   ├── test_vllm_pool_config.py
│   └── integration/           # Integration tests (require running vLLM)
│       ├── conftest.py        # Hostfile resolution, vLLM fixtures
│       ├── test_broadcast.py
│       ├── test_scatter_gather.py
│       ├── test_tree_reduce.py
│       ├── test_blackboard.py
│       └── test_pipeline.py
├── docs/
│   ├── conf.py                # Sphinx configuration
│   ├── index.rst              # Landing page
│   ├── batch_prompting.rst    # Batch prompting guide
│   ├── context_length.rst     # Context length configuration
│   ├── api.rst                # API reference (autodoc)
│   ├── Makefile               # make html
│   └── _build/                # make html output
├── extra/                     # Additional documentation (markdown)
│   ├── BATCH_PROMPTING.md     # Batch prompting implementation guide
│   ├── BATCH_PROMPTING_QUICKREF.md  # Quick reference for batch API usage
│   ├── IMPLEMENTATION_SUMMARY.md    # Batch implementation summary and performance results
│   └── CONTEXT_LENGTH.md      # Context length config and dynamic sizing
├── test_batch_integration.py  # Standalone batch vs non-batch comparison
├── pyproject.toml
└── README.md                  # This file
```

---

## Quick Tour

### Hostfile

Agents are listed in a plain-text hostfile, one per line. Columns are **tab-delimited**: host, port (optional, default 8000), then optional `key=value` tags.

```
host1	8000	node=aurora-0001	role=worker
host2	8000	node=aurora-0002	role=critic
# blank lines and comments are ignored
```

Parse it with:

```python
from aurora_swarm import parse_hostfile

endpoints = parse_hostfile("agents.hostfile")
```

### Waiting for vLLM servers (PBS)

When vLLM servers are started by a PBS job, the hostfile is created only after the job starts (which can be hours or days after submit). Use the helper script to wait for the hostfile to appear and for servers to become healthy, then write a filtered hostfile that omits any nodes where vLLM failed to start:

```bash
python scripts/wait_for_vllm_servers.py --hostfile /path/to/job_hostfile.txt \
    --health-timeout 1800 --output /path/to/ready_hostfile.txt
```

The health-phase timeout starts only after the first healthy node appears. Nodes that do not become healthy in time are logged and omitted from the output hostfile. Use the output path with `AURORA_SWARM_HOSTFILE` or `--hostfile` when running `scatter_gather_coli.py` or other clients.

### AgentPool

`AgentPool` wraps a list of endpoints with pooled HTTP sessions and semaphore-based concurrency control:

```python
from aurora_swarm import AgentPool

async with AgentPool(endpoints, concurrency=512) as pool:
    response = await pool.post(0, "Hello, agent!")
    print(response.text)
```

### Pools and connections

Understanding how connections are managed helps when scaling to many agents:

- **Connection** — A single TCP link from your process to one agent (one host:port). Many vLLM servers mean many such connections; the library reuses them via a connection pool.
- **Session and connector** — There is **one** `aiohttp.ClientSession` (and one `TCPConnector`) per **pool tree**: the pool you create with `async with VLLMPool(...) as pool` plus any sub-pools you get from `pool.by_tag()`, `pool.sample()`, etc. Sub-pools share that single session; they do not create their own.
- **connector_limit** — The connector is a *connection pool*: it can open up to `connector_limit` TCP connections (default 1024) to your many agents. So you have one logical “client” (session + connector) per pool tree, but many actual TCP connections to the many servers. Set via the constructor only (e.g. `VLLMPool(..., connector_limit=1024)`); there is no property getter.
- **Lifecycle** — The **parent** pool creates the session when you enter the `async with` block. All sub-pools use that same session. When the block exits, only the parent’s `close()` runs, which closes the shared session and connector. Always use the pool as a context manager so the session is closed and you avoid “Unclosed connector” warnings.

See the [Pools and connections](https://brettin.github.io/Aurora-Swarm/pools_and_connections.html) documentation for a diagram and more detail.

### Communication Patterns

| Pattern | Description |
|---------|-------------|
| **Broadcast** | Send the same prompt to every agent and collect all responses. |
| **Scatter-Gather** | Distribute different prompts across agents round-robin and gather results in input order. |
| **Tree-Reduce** | Hierarchical map-reduce — leaf agents produce answers, supervisors recursively summarize groups. |
| **Blackboard** | Agents collaborate through a shared mutable workspace in iterative rounds until convergence. |
| **Pipeline** | Multi-stage DAG where the output of one stage feeds the next. |

### Batch Prompting for High Throughput

**VLLMPool** supports batch prompting via the OpenAI completions API, dramatically reducing HTTP request overhead:

- **Without batching:** 10,000 prompts = 10,000 HTTP requests (100 per agent with 100 agents)
- **With batching:** 10,000 prompts = 100 HTTP requests (1 per agent, each with 100 prompts)

This 100× reduction in request count significantly improves throughput for scatter-gather patterns. Batch mode is enabled by default in VLLMPool and automatically used by `scatter_gather()` and `map_gather()`.

**Usage:**

```python
from aurora_swarm import VLLMPool, parse_hostfile
from aurora_swarm.patterns.scatter_gather import scatter_gather

endpoints = parse_hostfile("agents.hostfile")
async with VLLMPool(endpoints, model="meta-llama/Llama-3.1-70B-Instruct", use_batch=True) as pool:
    prompts = [f"Analyze gene {i}" for i in range(10000)]
    responses = await scatter_gather(pool, prompts)  # Uses batch API automatically
```

**Note:** Batch mode uses the `/v1/completions` endpoint. For instruction-tuned models that expect chat formatting, prompts are sent as-is. If your model requires specific chat templates, you may need to format prompts accordingly before sending.

### Aggregators

Built-in aggregation helpers for collected responses:

- `majority_vote` — categorical consensus with confidence score
- `concat` — join response texts
- `best_of` / `top_k` — quality-ranked selection via a custom score function
- `structured_merge` — parse JSON responses and merge into a flat list
- `statistics` — numeric summary (mean, std, median, min, max)
- `failure_report` — diagnostic summary of successes and failures

See the [Aggregators documentation](https://brettin.github.io/Aurora-Swarm/aggregators.html) for details and the broadcast example.

---

## Example: Broadcast and Reduce

```python
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
```

**See also:** [examples/broadcast_aggregators.py](examples/broadcast_aggregators.py) broadcasts the same prompt to all hosts except one and demonstrates `majority_vote` and `concat` aggregators; run with `examples/broadcast_aggregators.sh <hostfile>`. The full aggregators guide is in the [documentation](https://brettin.github.io/Aurora-Swarm/aggregators.html).

---

## Example: Tree-Reduce

```python
import asyncio
from aurora_swarm import VLLMPool, parse_hostfile
from aurora_swarm.patterns.tree_reduce import tree_reduce

async def main():
    endpoints = parse_hostfile("agents.hostfile")
    async with VLLMPool(endpoints, model="meta-llama/Llama-3.1-70B-Instruct") as pool:
        result = await tree_reduce(
            pool,
            prompt="Name one benefit of renewable energy. Keep it to one sentence.",
            reduce_prompt="Below are several responses. Merge them into a concise summary (level {level}):\n\n{responses}",
            fanin=4,
        )
        print(result.text)

asyncio.run(main())
```

**See also:** [examples/tree_reduce_example.py](examples/tree_reduce_example.py) supports broadcast mode (same prompt to all) and items mode (scatter different tasks with `{item}` placeholder); run with `examples/tree_reduce_example.sh <hostfile>`.

---

## Configuration

### Context Length Management

Aurora Swarm intelligently manages `max_tokens` for different types of operations:

**Dynamic Sizing** — Automatically adjusts `max_tokens` based on prompt length to prevent truncation:
- Estimates prompt tokens (using a chars÷4 heuristic)
- Queries vLLM's `/v1/models` endpoint to get the model's max context length
- Computes: `max_tokens = min(cap, model_max - prompt_tokens - buffer)`

**Aggregation Presets** — Uses larger token budgets for reduce/aggregation steps where prompts grow:
- `max_tokens` (default): for simple broadcasts and leaf prompts
- `max_tokens_aggregation`: for reduce steps (defaults to 2× `max_tokens`)

### Environment Variables

Configure context length via environment variables:

```bash
export AURORA_SWARM_MAX_TOKENS=1024              # Default max tokens
export AURORA_SWARM_MAX_TOKENS_AGGREGATION=2048  # Aggregation/reduce steps
export AURORA_SWARM_MODEL_MAX_CONTEXT=131072     # Model's max context (optional)
```

### VLLMPool Parameters

```python
from aurora_swarm import VLLMPool, AgentEndpoint

pool = VLLMPool(
    endpoints=[AgentEndpoint("host1", 8000), AgentEndpoint("host2", 8000)],
    model="openai/gpt-oss-120b",
    max_tokens=1024,                    # Default for simple prompts
    max_tokens_aggregation=2048,        # For aggregation steps
    model_max_context=131072,           # Optional: skip API query
    buffer=512,                         # Safety margin for reasoning overhead
    concurrency=512,
    connector_limit=1024,
    timeout=300.0,
)
```

**Per-request override:**

```python
# Explicitly set max_tokens for a specific call
response = await pool.post(agent_index=0, prompt="...", max_tokens=512)
```

---

## Environment Summary

| Item | Value |
|------|-------|
| Python | >= 3.11 |
| Core dependencies | `aiohttp >= 3.9`, `openai >= 1.0` |
| Dev dependencies | `pytest >= 8.0`, `pytest-asyncio >= 0.23` |
| Docs (optional) | `pip install -e ".[docs]"` — Sphinx API reference |
| Build backend | `setuptools >= 68.0` |

---

## Additional Documentation

- **[extra/BATCH_PROMPTING.md](extra/BATCH_PROMPTING.md)** — Batch prompting implementation guide
- **[extra/BATCH_PROMPTING_QUICKREF.md](extra/BATCH_PROMPTING_QUICKREF.md)** — Batch prompting quick reference
- **[extra/IMPLEMENTATION_SUMMARY.md](extra/IMPLEMENTATION_SUMMARY.md)** — Batch prompting implementation summary
- **[extra/CONTEXT_LENGTH.md](extra/CONTEXT_LENGTH.md)** — Context length configuration (full guide)
- **[API Reference](https://brettin.github.io/Aurora-Swarm/)** — Sphinx API docs and built-in [context length](https://brettin.github.io/Aurora-Swarm/context_length.html) page
- **Integration Tests** — See `tests/integration/` for real-world usage examples
