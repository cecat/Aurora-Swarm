# SwarmSim: Design Overview

## What We Are Building

SwarmSim is a COVID-19 epidemic simulation in which each synthetic resident of Chicago's Logan Square neighborhood is represented by a large language model rather than a rule-based agent.  The simulation runs for 90 simulated days (March–June 2020) at one-hour timesteps.  At each tick, every active agent receives a structured prompt describing their personal situation, local disease conditions, social network signals, and current public health policy, and responds with a JSON decision about where to go, whether to wear a mask, and whether to isolate.  Those decisions feed back into the disease engine the same way behavioral choices would in a classical ABM.

The 1,000-agent pilot targets Logan Square (Community Area 22, ~73K real population) using ACS-derived demographic distributions, ATUS-calibrated activity schedules, and a disease model calibrated from Ozik et al. (2021) / CityCOVID.

---

## Architecture

The simulation is structured as two interacting layers.

**Simulation layer (deterministic).**  Modeled directly on ChiSim/CityCOVID's validated disease engine.  A bipartite graph of agents and places, a 1-hour disease exposure tick, and a daily state machine (SUSCEPTIBLE → EXPOSED → PRE-SYMPTOMATIC / ASYMPTOMATIC → SYMPTOMATIC-MILD → SEVERE → HOSPITALIZED → ICU → DECEASED / RECOVERED).  Transmission probability is computed from co-occupant counts, place ventilation, mask-wearing, and infectivity state.  This layer uses the same logic and calibration as ChiSim/CityCOVID and produces the epidemic curve.

**Behavioral layer (LLM-driven).**  Each agent's movement and protective decisions are made by an LLM rather than coded rules.  Prompts are structured in four blocks: a static system prompt (Block A, ~300 tokens), a static agent profile (Block B, ~200 tokens), a dynamic situational context (Block C, ~150 tokens), and a fixed task/schema (Block D, ~100 tokens).  Blocks A and B are pre-rendered at simulation start and never change; vLLM's Automatic Prefix Caching (APC) keeps them in the KV cache across ticks, so the effective per-tick payload sent to the model is ~250 tokens rather than ~750.

**Infrastructure.**  Aurora Swarm's `scatter_gather` pattern dispatches one prompt per agent per tick in a single batch.  The coordinator holds all simulation state in memory (Python dicts for the 1K pilot; Redis + Postgres at scale).  vLLM endpoints are stateless inference servers with no knowledge of simulation state.

**Behavioral continuity** (the problem stateless inference would otherwise create) is maintained by three mechanisms:
- A **five-scalar behavioral state vector** (`fear_level`, `compliance_fatigue`, `financial_pressure`, `perceived_risk`, `trust_in_news`) updated deterministically by the coordinator each tick from observable events — no LLM call required.
- **Episodic memory compression**: every 7 simulated days, a brief LLM call summarizes the agent's notable experiences into a ≤75-token natural-language memory injected into Block B.
- **Three-layer communication**: (1) co-location context from the previous tick's occupancy, (2) an anonymous social-network inbox delivering disease state change notifications from contacts, (3) a rolling place event log flagging locations with notable illness clusters.

---

## Comparison with ChiSim

| Dimension | ChiSim / CityCOVID | SwarmSim |
|---|---|---|
| **Agent count** | 2.7M synthetic Chicago residents | 1K pilot; ~60K feasible at scale |
| **Places** | 1.2M geolocated places | ~500 (pilot); scales with agents |
| **Agent behavior** | Deterministic coded rules (mobility, compliance) | LLM prompt-response each tick |
| **Behavioral heterogeneity** | Parameterized by age, occupation, demographics | Emergent from LLM reasoning over agent profile |
| **Simulation architecture** | Distributed MPI / Repast HPC; agents migrate between processes | Centralized state; stateless LLM inference servers |
| **Compute substrate** | HPC MPI ranks (CPU-heavy) | GPU inference endpoints via vLLM + Aurora Swarm |
| **Timestep** | 1 hour | 1 hour (identical) |
| **Disease model** | CityCOVID SEIR+ state machine | Same state machine, same calibration (Ozik et al. 2021) |
| **Transmission model** | Co-location exposure, ventilation, density | Identical |
| **Activity schedules** | ATUS-calibrated, place-type resolved | Same archetype logic (8 occupation types, 168-slot weeks) |
| **Agent memory** | Stateful in-process across ticks | Behavioral state vector + compressed episodic memory |
| **Agent-to-agent communication** | None (co-location only, no messaging) | Three-layer system (co-location, inbox, place event log) |
| **Reproducibility** | Deterministic given seed | Stochastic; different answer each run |
| **Run time (90 days)** | Minutes on 1K+ MPI ranks | ~6 hours for 60K agents on 3K GPU endpoints |
| **What we borrow** | Bipartite architecture, disease engine, transmission model, schedules, demographic distributions | ← all of this |
| **What we add** | — | LLM behavioral decision-making, behavioral state vector, episodic memory, social inbox |

---

## Technical Trade-offs

**Stateless inference vs. distributed state.**  ChiSim agents are stateful objects that live in memory across ticks on MPI processes; communication happens implicitly through shared process state.  Our agents are stateless LLM calls — there is no persistent "agent process."  The benefit is radical simplicity: the coordinator is a single Python process, scaling is just adding more vLLM endpoints, and there are no MPI communication patterns to reason about.  The cost is that continuity must be engineered explicitly.  The behavioral state vector and episodic memory are direct responses to this constraint.  They are lightweight (five floats plus ≤75 tokens) but they are an approximation — the LLM cannot access its own prior outputs directly.

**Prompt token budget.**  Early designs put full agent history into every prompt, producing ~1,000-token payloads per tick.  At 1,000 agents × 24 ticks × 90 days = 2.16M calls, this is untenable.  The four-block structure with APC reduces the effective per-tick cost to ~250 tokens.  Episodic compression adds ~143 calls per 7-day period per 1,000 agents — under 2% overhead.

**Centralized state.**  The pilot holds all agent and place state in Python dicts on the coordinator.  This is a single point of failure and does not scale beyond ~10K agents on a single machine.  The architecture doc specifies Redis (behavioral state) and Postgres (agent profiles, metrics) for the 60K-agent scale run, with the coordinator becoming a thin orchestration layer.

**Throughput ceiling.**  At ~2 prompts/sec/endpoint and a 10-second per-tick wall-clock budget, 3,000 endpoints support ~60,000 agents.  Scaling to 1M agents would require either a much larger endpoint fleet, longer wall-clock budgets per tick, or a hybrid approach in which only a fraction of agents receive LLM decisions each tick.

---

## Scientific Trade-offs

**The core tension: realism vs. reproducibility.**

ChiSim is a deterministic system.  Given a fixed random seed and parameter set, it produces the same epidemic curve every run.  This property is scientifically valuable: to measure the effect of a single policy change (say, closing schools on day 18), you run the model twice, change only that one variable, and the difference in outcomes is attributable entirely to that change.  The model is a controlled experiment.

SwarmSim is stochastic at its core.  LLM sampling introduces irreducible variance in agent decisions.  Two runs with identical parameters will produce different epidemic curves.  This has several consequences:

- **Causal attribution is harder.**  When outcomes differ between a mask-mandate run and a no-mandate run, some of the difference is the policy effect and some is sampling noise.  Isolating the policy signal requires ensemble runs and statistical analysis — the same discipline applied to stochastic ABMs like NetLogo, but more expensive per run.

- **Parameter sensitivity analysis is noisier.**  In ChiSim, varying `BASE_TRANSMISSION_PROB` by 10% produces a precise, repeatable shift in the epidemic curve.  In SwarmSim, the signal may be obscured by run-to-run behavioral variance.

- **The variance may itself be informative.**  Human behavioral responses to a pandemic are genuinely stochastic.  The spread of outcomes across ensemble runs could represent plausible uncertainty in behavioral response, not just model error.

**What we gain.**  The LLM brings behavioral heterogeneity that is difficult to parameterize explicitly.  An essential worker with high financial pressure, a symptomatic household member, and an inbox message about a hospitalized colleague will reason differently than the same worker without those stressors — not because we coded that interaction, but because the model has internalized it from training data.  Emergent compliance dynamics (fatigue accumulating under prolonged lockdown, trust eroding with policy reversals) arise from the prompt context rather than requiring explicit behavioral equations.

**The pragmatic position.**  For the pilot, we are primarily testing plumbing rather than science.  The relevant validation question is whether the epidemic curve produced by the simulation is qualitatively plausible — whether agents with high fear and nearby illness events do in fact choose to isolate more, whether essential workers show earlier infection, whether behavioral fatigue appears under extended lockdown.  Quantitative calibration against ChiSim or real Chicago data is a second-phase concern once the infrastructure is validated.
