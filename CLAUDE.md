# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aurora Swarm is an async Python library for orchestrating large-scale LLM agent swarms (1,000–4,000+ agents) using pooled HTTP connections and semaphore-based concurrency control. The repo also contains **SwarmSim**, an LLM-driven epidemic simulation (1K-agent pilot of Logan Square, Chicago) that replaces deterministic agents with LLM-powered behavioral decision-makers.

API Reference: https://brettin.github.io/Aurora-Swarm/

## Setup & Installation

```bash
pip install -e .                  # Core only
pip install -e ".[dev]"           # Add pytest, pytest-asyncio
pip install -e ".[docs]"          # Sphinx docs
pip install -e ".[uq]"            # Uncertainty quantification (numpy, scipy)
pip install -e ".[plot]"          # Plotting (matplotlib)
```

On HPC (Aurora supercomputer): source `env.sh` to load `module load frameworks` and activate the conda env.

## Commands

```bash
# Tests
pytest -v                          # All unit tests
pytest -v tests/                   # Unit tests only
pytest -v tests/integration/       # Integration tests — require real vLLM servers + --hostfile

# Docs
cd docs && make html               # Build Sphinx docs → docs/_build/

# SwarmSim pilot
python -m swarmsim.run_pilot --endpoint http://host:port --n-agents 1000 --n-days 90
```

## Architecture

### Aurora Swarm Library (`aurora_swarm/`)

**Five communication patterns** in `aurora_swarm/patterns/`:
- **Broadcast** — Send identical prompt to all agents, collect all responses
- **Scatter-Gather** — Distribute different prompts round-robin, gather results in input order
- **Tree-Reduce** — Hierarchical map-reduce: leaf agents answer, supervisors summarize groups recursively
- **Blackboard** — Iterative multi-round collaboration through a shared mutable workspace
- **Pipeline** — Multi-stage DAG where each stage's output feeds the next

**Core pool classes:**
- `AgentPool` (`pool.py`) — Base async HTTP pool; one `aiohttp.ClientSession` + `TCPConnector` shared across the entire pool tree (sub-pools do NOT create new sessions)
- `VLLMPool` (`vllm_pool.py`) — OpenAI-compatible vLLM client with batch prompting (100× HTTP reduction) and dynamic context length management
- `EmbeddingPool` (`embedding_pool.py`) — Targets `/v1/embeddings` endpoint

**Batch prompting** (enabled by default in `scatter_gather` and `map_gather`): groups prompts by target agent and sends each agent's entire batch in one `/v1/completions` call. Per-batch timeout scales with batch size.

**Dynamic context length** (`vllm_pool.py`): queries `/v1/models` on first request to get model's max context, then computes `max_tokens = min(cap, model_max - prompt_tokens - buffer)` using a chars÷4 heuristic for prompt token estimation.

**Hostfile format** (`hostfile.py`): `host<tab>port<tab>key=value` pairs parsed into `AgentEndpoint` objects. Sub-pools are created via `pool.by_tag("tagname")`.

**Public API** (`aurora_swarm/__init__.py`) exports: `AgentEndpoint`, `AgentPool`, `VLLMPool`, `EmbeddingPool`, `Response`, `EmbeddingResponse`, `parse_hostfile`.

### SwarmSim Subsystem (`swarmsim/`)

A bipartite agent/place graph simulation with two interleaved layers:

**Disease layer** (deterministic, adapted from CityCOVID):
- Hourly exposure ticks computed from co-occupant counts, ventilation, masks, infectivity state
- Daily state machine: SUSCEPTIBLE → EXPOSED → PRE/ASYMPTOMATIC → SYMPTOMATIC-MILD → SEVERE → HOSPITALIZED → ICU → RECOVERED/DECEASED
- Implemented in `disease_engine.py`

**Behavioral layer** (LLM-driven via Aurora Swarm `scatter_gather`):
- Each agent has a **5-scalar behavioral state vector** (`fear_level`, `compliance_fatigue`, `financial_pressure`, `perceived_risk`, `trust_in_news`) updated deterministically by the coordinator — this solves the stateless inference problem without long context
- **Episodic memory**: every 7 sim days, an LLM call compresses experiences into a ≤75-token natural-language memory string (`memory.py`)
- **Three-layer communication**: co-location context, social-network inbox, place event log (`communication.py`)

**Prompt structure** (`prompt.py`): 4 blocks (~750 tokens total) ordered for vLLM's Automatic Prefix Caching (APC) — static blocks first (system prompt, agent profile), dynamic blocks last (situational context, task schema). Effective per-tick payload is ~250 tokens after KV cache hits.

**Orchestration:**
- `coordinator.py` — Main tick loop; owns master clock, disease engine, state barrier, policy timeline
- `worker.py` — Per-shard LLM decision dispatcher (wraps `scatter_gather`)
- `sim_config.py` — All tuneable parameters: `N_AGENTS`, disease rates, behavioral state update scalars, occupation baselines, neighborhood config

**Population generation** (`population.py`, `schedules.py`): ATUS-calibrated activity schedules with 8 occupation types and 168-slot (hourly) weekly schedules.

**Run outputs** (`run_pilot.py`): `metrics.json` (epidemic curves + behavioral metrics), `decisions_sample.jsonl` (agent decision log), `run_config.json` (parameter snapshot).

### Uncertainty Quantification (`aurora_swarm/uq/`)

Optional module (requires `.[uq]`). Implements semantic entropy and kernel language entropy for measuring LLM response uncertainty across swarm outputs. See `examples/lab3_semantic_uncertainty.py`.

## Key Files

| Purpose | Path |
|---------|------|
| Public API | `aurora_swarm/__init__.py` |
| Connection pooling | `pool.py`, `vllm_pool.py`, `embedding_pool.py` |
| Hostfile parsing | `hostfile.py` |
| Response aggregation | `aggregators.py` |
| Disease state machine | `swarmsim/disease_engine.py` |
| Behavioral state updates | `swarmsim/behavioral_state.py` |
| Simulation tick loop | `swarmsim/coordinator.py` |
| LLM decision dispatch | `swarmsim/worker.py` |
| 4-block prompt builder | `swarmsim/prompt.py` |
| All sim parameters | `swarmsim/sim_config.py` |
| Test fixtures & mocks | `tests/conftest.py` |
| vLLM server health check | `scripts/wait_for_vllm_servers.py` |

## Testing Notes

Unit tests use mock HTTP servers (no real LLM required). Integration tests in `tests/integration/` require live vLLM endpoints specified via `--hostfile`. The `pytest-asyncio` mode is set to `auto` in `pyproject.toml`.
