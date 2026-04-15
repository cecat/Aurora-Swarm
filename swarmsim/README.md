# SwarmSim

SwarmSim is an experimental LLM-driven epidemic simulation built to evaluate whether Aurora Swarm can support large-scale agent-based modeling (ABM) workloads, and to explore the design space for replacing deterministic behavioral rules with LLM-powered agents.

The pilot deliberately operates at epidemiologically modest scale — 1,000 agents modeling Chicago's Logan Square neighborhood over 90 simulated days (March–June 2020). That scale is not intended to produce scientifically meaningful results; it is chosen to keep iteration cycles short while stress-testing Aurora Swarm's concurrency and throughput under realistic prompt loads, and to work through the core design problems (behavioral continuity, prompt token budget, memory compression, agent communication) before committing to the infrastructure required for a full-scale run. A production LLM-based model of Logan Square alone would require on the order of 73,000 agents; city-scale modeling at the level of ChiSim/CityCOVID (2.7M agents) remains a longer-term target.

The simulation uses ACS-derived demographics, ATUS-calibrated activity schedules, and a disease engine calibrated from CityCOVID (Ozik et al. 2021).

## How it works

Two layers run in lockstep each simulated hour:

- **Disease layer** — deterministic SEIR+ state machine adapted from ChiSim/CityCOVID; hourly exposure ticks computed from co-occupant counts, ventilation, mask-wearing, and infectivity state.
- **Behavioral layer** — each agent's movement and protective decisions are made by an LLM (via Aurora Swarm's `scatter_gather`). Behavioral continuity across stateless inference calls is maintained by a five-scalar behavioral state vector, 7-day episodic memory compression, and a three-layer communication system (co-location context, social-network inbox, place event log).

Prompt structure follows a four-block layout ordered for vLLM's Automatic Prefix Caching — static blocks (system prompt, agent profile) first, dynamic blocks last — reducing the effective per-tick token payload from ~750 to ~250 tokens.

## Running the pilot

```bash
python -m swarmsim.run_pilot \
    --endpoint http://host:port \
    --n-agents 1000 \
    --n-days 90
```

Outputs written to the run directory:
- `metrics.json` — epidemic curves and behavioral metrics
- `decisions_sample.jsonl` — agent decision log
- `run_config.json` — parameter snapshot

All tuneable parameters are in [`sim_config.py`](sim_config.py).

## Key files

| File | Purpose |
|------|---------|
| `coordinator.py` | Main tick loop; owns master clock, disease engine, state barrier, policy timeline |
| `worker.py` | Per-shard LLM decision dispatcher (wraps `scatter_gather`) |
| `disease_engine.py` | Hourly exposure tick and daily state machine |
| `behavioral_state.py` | Five-scalar behavioral state vector updates |
| `prompt.py` | Four-block prompt builder |
| `memory.py` | Episodic memory compression (every 7 sim days) |
| `communication.py` | Three-layer agent communication system |
| `population.py` / `schedules.py` | ATUS-calibrated population and activity schedule generation |
| `sim_config.py` | All simulation parameters |
| `run_pilot.py` | Entry point and output writing |

## Documentation

- [Overview](../docs/SwarmSim/SwarmSim-Overview.md) — what SwarmSim is, architecture summary, comparison with ChiSim/CityCOVID
- [Design](../docs/SwarmSim/SwarmSim-Design.md) — ChiSim/CityCOVID reference map and design rationale
- [Architecture](../docs/SwarmSim/SwarmSim-Architecture.md) — detailed component and data-flow documentation
- [Scaling](../docs/SwarmSim/SwarmSim-Scaling.md) — throughput analysis and feasibility at 1K–60K agents
- [Ensembles](../docs/SwarmSim/SwarmSim-Ensembles.md) — ensemble strategy and how SwarmSim's behavioral uncertainty differs from ChiSim's parameter uncertainty
