# SwarmSim-Scaling.md
**Aurora Swarm / SwarmSim Scaling Feasibility Analysis**
*April 2026*

---

## 1. The Original Claim and the Math Problem

The SwarmSim-Design document (Section 5.2) states that running LLMs for all 2.7M agents is "computationally infeasible at present" and proposes Pilot A (100–500 LLM agents) as a starting point. This document examines whether there is a middle ground — a scale large enough to be epidemiologically meaningful, yet small enough that each simulated hour completes in a tractable wall-clock time.

The user's starting hypothesis is:

> "With 1,000 nodes × 3 LLM endpoints = 3,000 LLMs, and queuing 1,000 prompts per LLM, one could drive 3,000,000 agent-timesteps per batch. But the time required per timestep would make the simulation intractable."

Let us verify this carefully.

---

## 2. Aurora Swarm Throughput: What Is Actually Achievable

### 2.1 Measured throughput

From the Aurora Swarm batch-prompting benchmark (BATCH_PROMPTING.md, tested on `openai/gpt-oss-120b`):

| Mode | Prompts | Endpoints | Time | Throughput |
|---|---|---|---|---|
| Batch | 20 | 4 | 2.76 s | 7.24 prompts/sec |
| Non-batch | 20 | 4 | 3.09 s | 6.46 prompts/sec |

That is **~1.8 prompts/sec/endpoint** in the batch-mode measurement.

The SwarmSim agent prompt (Section 4 of SwarmSim-Design.md) is long — approximately 800–1,200 tokens of system context plus the agent profile plus the task — with an expected response of ~100–200 tokens (structured JSON). Call it **~1,000 tokens input / 150 tokens output** per agent-timestep.

On `gpt-oss-120b` (a 120B-parameter reasoning model), throughput in batch mode is heavily input-bound. The 7.24 prompts/sec figure above is with a small test — at production scale with long prompts and a concurrency-limited single endpoint, **~1–3 prompts/sec per endpoint** is a realistic range. Larger, purpose-built models will be slower; smaller models will be faster.

### 2.2 Scaling to 3,000 endpoints

With 1,000 nodes × 3 endpoints each:

```
Throughput = 3,000 endpoints × 2 prompts/sec/endpoint
           = 6,000 agent-timesteps/sec (sustained, batch mode)
```

This is the key number. Call it **T = 6,000 agent-timesteps/sec** as the realistic ceiling for the hardware described.

(At 1 prompt/sec/endpoint, T = 3,000; at 3 prompts/sec, T = 9,000. The range is 3K–9K. We use 6K as a central estimate.)

---

## 3. The Time Budget: What Is a "Tractable" Simulation?

ChiSim runs with a **1-hour simulated timestep**. A meaningful COVID-19 scenario requires at minimum **70–90 simulated days** (the operational range used in CityCOVID). That means:

```
Total timesteps = 90 days × 24 hours/day = 2,160 timesteps
```

For the simulation to complete in a tolerable wall-clock time, we need to bound how long each simulated hour takes. This is not a simulated-time constraint — it is a research-time constraint. Call `W` the wall-clock seconds allowed per simulated hour.

| Wall-clock budget per sim hour | Total wall-clock for 90 sim days |
|---|---|
| W = 1 s | ~36 minutes total — ideal |
| W = 10 s | ~6 hours total — acceptable for overnight runs |
| W = 60 s | ~36 hours total — marginal (one run per 1.5 days) |
| W = 600 s | ~15 days total — intractable for a research program |

**The user's concern is correct:** if each simulated hour takes many minutes of wall-clock time, a 90-day scenario takes months of compute time, making parameter sweeps and calibration completely infeasible.

---

## 4. The Core Equation: How Many Agents Can We Afford?

At throughput T and wall-clock budget W per simulated hour, the maximum number of LLM agents is:

```
N_agents ≤ T × W
```

For the 3,000-endpoint cluster:

| W (wall-clock per sim hour) | Max LLM agents | Practical budget for 90-day run |
|---|---|---|
| 1 s | 6,000 | ~36 min — very fast |
| 10 s | 60,000 | ~6 hours |
| 60 s | 360,000 | ~36 hours |
| 600 s | 3,600,000 | ~15 days — too slow |

### The User's Arithmetic Revisited

The user described "queuing 1,000 prompts for 1,000 agents." With 3,000 endpoints and 1,000 prompts per endpoint per batch:

```
One batch = 3,000 endpoints × 1,000 prompts = 3,000,000 agent-steps
Time for one batch = 1,000 prompts / (2 prompts/sec/endpoint) = 500 seconds
```

So one batch of 3M agent-steps takes **~500 seconds (≈ 8 minutes)**. If each simulated hour maps to one batch, then:

```
Wall-clock per simulated hour = 500 seconds
Wall-clock for 90 sim days = 500 × 2,160 = 1,080,000 seconds ≈ 12.5 days
```

**The user's conclusion is correct.** At this operating point (3M agents, 1,000 prompts/endpoint), the simulation takes ~12 days of wall-clock time per 90-day scenario. That is intractable for research: you cannot run calibration sweeps (which require dozens to hundreds of scenario executions), you cannot do sensitivity analyses, and you cannot iterate on the model design.

However, the user asked: *is there a meaningful smaller scale?*

---

## 5. What Scale Is Actually Feasible?

### 5.1 The Epidemiologically Meaningful Minimum

The critical question is what population size preserves the phenomena that make LLM agents scientifically interesting — specifically, the **behavioral heterogeneity** and **contact network structure** that differ from a deterministic ABM.

ChiSim's epidemiological validity depends on:
1. **Spatial co-location contact networks** — agents must co-locate in shared places (households, workplaces, schools) to generate transmission events. This requires enough agents to realistically populate each place type.
2. **Demographic diversity** — the model's behavioral heterogeneity emerges from variation in age, occupation, income, comorbidities, and compliance. This requires sufficient population breadth.
3. **Stochastic outbreak dynamics** — small populations produce high-variance trajectories. Epidemic models are typically considered statistically stable above ~5,000–10,000 agents, though contact-network models need more.

A practical minimum for a self-contained neighborhood-scale simulation is **~10,000–50,000 agents**, corresponding to one or a few Chicago community areas. Below this, transmission networks become too sparse and the "epidemic" too dominated by random extinction events.

### 5.2 The Feasible Operating Point

Working backward from the 6-hour wall-clock budget (W = 10 s/sim-hour):

```
N_agents ≤ 6,000 × 10 = 60,000 agents
```

**60,000 LLM agents on the 3,000-endpoint cluster with a 10-second-per-simulated-hour budget is feasible and epidemiologically meaningful.**

This corresponds roughly to:
- 2–3 Chicago community areas (e.g., Logan Square + Humboldt Park, ~60K residents)
- Or a synthetic town of similar size

A 90-day scenario would complete in **~6 hours** of wall-clock time, allowing ~4 runs per day. That is sufficient for:
- Lightweight calibration (ABC with tens of scenarios)
- Policy comparison studies (5–10 scenarios)
- Behavioral sensitivity analyses

For a smaller, faster design point:

```
N_agents = 10,000 → W = 1.67 s/sim-hour → 90-day run in ~1 hour
```

This is a single Chicago neighborhood (~10K residents) and runs extremely fast — multiple ensemble members can be run in a single day.

### 5.3 Summary Table

| N agents | W (wall-clock/sim-hour) | 90-day wall-clock | Notes |
|---|---|---|---|
| 3,000 | 0.5 s | 18 min | Too small; sparse contact networks |
| 10,000 | 1.7 s | ~1 hour | Minimum meaningful; neighborhood scale |
| 30,000 | 5 s | ~3 hours | Community-area scale; good for calibration |
| **60,000** | **10 s** | **~6 hours** | **Sweet spot; 2–3 community areas** |
| 180,000 | 30 s | ~18 hours | Subdistrict scale; overnight run |
| 360,000 | 60 s | ~36 hours | Large district; marginal |
| 3,000,000 | 500 s | ~12 days | Full-city aspirational; intractable for sweeps |

---

## 6. A Revised Approach: The Neighborhood LLM-ABM

Given the feasibility analysis, the recommended reframing is to **abandon the goal of simulating all of Chicago** in the LLM-agent mode and instead simulate a **self-contained neighborhood or community-area sub-population** with full LLM agency.

### 6.1 Why Neighborhoods Are Natural Units

ChiSim already models agents at geolocated places. A neighborhood sub-simulation is not a gross approximation — it is an epidemiologically coherent unit because:
- Most daily activity (home, school, essential retail, social visits) is local
- Workplaces and hospitals are the main cross-neighborhood links
- The disease state engine is purely local to each agent
- The contact network is dominated by household and workplace co-location

The only approximation required is to model **cross-neighborhood exposure** as a stochastic boundary condition: agents who travel to workplaces outside the neighborhood bring back an externally-modeled exposure risk. This is a standard approach in spatially-structured epidemic models.

### 6.2 The Revised Pilot Plan

| Pilot | N LLM agents | Population basis | Wall-clock/90-day run | Purpose |
|---|---|---|---|---|
| **Nano** | 1,000–5,000 | Single census tract | 10–30 min | Behavioral validation; prompt debugging |
| **Neighborhood** | 10,000–20,000 | One community area | 1–2 hours | First science result; demographic heterogeneity study |
| **District** | 50,000–60,000 | 2–3 community areas | 5–6 hours | Policy comparison; calibration; publication target |
| **Sub-city** | 180,000 | Chicago South Side or North Side | ~18 hours | Large-scale study; overnight runs |

### 6.3 What This Still Cannot Do

To be honest about limitations:

1. **City-scale emergent behavior** — phenomena that only emerge from interactions between distant parts of Chicago (e.g., hospital surge propagating across districts) cannot be captured in a 60K-agent neighborhood model.

2. **Full calibration against city-wide data** — the original CityCOVID was calibrated against daily hospitalizations and deaths for all of Chicago. A neighborhood model must use local calibration targets (zip-code-level data) or accept that it is a behavioral laboratory, not a calibrated predictive tool.

3. **Ensemble size for ABC** — CityCOVID used hundreds of ensemble runs for Approximate Bayesian Computation. At 6 hours/run, you can do ~4 runs/day, which is adequate for lightweight ABC (20–50 ensemble members) but not the full calibration workflow.

---

## 7. What the LLM Agents Add at This Scale

The scientific value proposition remains strong even at neighborhood scale, because the research question shifts from **"how many get sick?"** (which the deterministic ABM already answers well) to **"how does behavioral heterogeneity shape the epidemic trajectory?"**

At 60K agents with full LLM behavioral decision-making, you can study:

| Question | Why LLM > Deterministic ABM |
|---|---|
| How does compliance heterogeneity affect peak hospitalization timing? | LLMs produce a realistic distribution of compliance reasons, not a single scalar p(comply) |
| Do essential workers' risk calculations change as they observe neighbors fall ill? | LLMs update risk perception dynamically; ABM has fixed parameters |
| How does the epidemic differ between high-income and low-income neighborhoods with the same baseline transmission? | LLM agents reason about financial pressure ("I can't miss work"), ABM agents use a scalar income modifier |
| What fraction of transmission happens despite agents making individually rational decisions? | Novel question; deterministic ABM cannot answer it |
| How do qualitatively different policy framings (mandate vs. recommendation) change behavior? | Natural language policy instructions to LLM agents; binary switches in ABM |

These are the questions that justify LLM agents. They do not require 2.7M agents.

---

## 8. Implementation Recommendations for Aurora Swarm

### 8.1 Execution Model

Each simulated hour maps to one Aurora Swarm `scatter_gather` call:

```python
# One simulated timestep
prompts = [build_agent_prompt(agent) for agent in active_agents]  # N_agents prompts
responses = await scatter_gather(pool, prompts)                    # All in parallel
decisions = [parse_response(r) for r in responses]                # JSON parse
update_simulation_state(decisions)                                 # Disease engine tick
```

At 60,000 agents across 3,000 endpoints, each endpoint receives 20 prompts per batch — well within vLLM's batch capacity and Aurora Swarm's batch-prompting implementation.

### 8.2 Prompt Caching Opportunity

Across timesteps, the **system prompt + agent profile** portion is fixed for each agent (~70% of the prompt). This is an excellent candidate for a prefix-caching mechanism if the vLLM deployment supports it (via `AURORA_SWARM_MODEL_MAX_CONTEXT` tuning). This could reduce effective token throughput by 50–60%, potentially doubling effective agent count.

### 8.3 Parallelizing Across Ensemble Members

For ABC calibration, the most important optimization is **running multiple ensemble members in parallel** across disjoint subsets of the 3,000 endpoints:

```
3,000 endpoints total
├── Run A: 1,000 endpoints → 20,000 agents, parameter set θ_1
├── Run B: 1,000 endpoints → 20,000 agents, parameter set θ_2
└── Run C: 1,000 endpoints → 20,000 agents, parameter set θ_3
```

Three parallel ensemble members × 90 days × 24 timesteps = 3 calibration data points per 6-hour wall-clock window. This is viable for lightweight ABC.

### 8.4 Separating LLM and Deterministic Agents

The hybrid design from SwarmSim-Design Section 5.1 still applies. Even in a 60K-agent neighborhood simulation, not all agents need full LLM decision-making. A natural partitioning:

- **Full LLM agency (~10–20%)**: Behaviorally interesting demographics — essential workers, elderly with comorbidities, parents of school-age children, non-compliant young adults. These are the agents whose decisions are hardest to parameterize deterministically.
- **Deterministic agents (~80–90%)**: Agents whose behavior is highly predictable — hospitalized agents, deceased agents, fully compliant retirees, infants. These cost zero LLM calls.

At 60K total agents with 15% LLM share: **9,000 LLM calls per simulated hour**, completing in **1.5 seconds wall-clock** with the 3,000-endpoint cluster. This enables a full 90-day simulation in **~54 minutes** — fast enough for real ABC calibration workflows.

---

## 9. Revised Feasibility Verdict

| Question | Answer |
|---|---|
| Is 3M LLM agents feasible? | No — ~12 days per 90-day scenario; intractable for research |
| Is 60K LLM agents feasible? | **Yes** — ~6 hours per scenario on 3,000 endpoints |
| Is 10K LLM agents feasible? | **Yes** — ~1 hour per scenario; allows rapid iteration |
| Is there a scientifically meaningful question? | **Yes** — behavioral heterogeneity studies at neighborhood scale are novel and publishable |
| Does Aurora Swarm support this? | **Yes** — `scatter_gather` with batch mode handles this directly |
| What is the right framing shift? | Replace "simulate Chicago" with "simulate a Chicago neighborhood with high-fidelity behavioral agents" |

The user's math is correct: naive full-city LLM simulation is intractable. But the 60K-agent neighborhood simulation is both feasible on the described hardware and scientifically meaningful — and can answer questions the deterministic ABM structurally cannot.

---

## 10. Key Assumptions and Uncertainties

| Assumption | Value Used | Sensitivity |
|---|---|---|
| Throughput per endpoint | 2 prompts/sec | ±3× depending on model size, prompt length, GPU generation |
| Endpoints available | 3,000 (1,000 nodes × 3) | Scales linearly with endpoint count |
| Prompt length | ~1,000 input tokens, ~150 output | Shorter prompts → proportionally higher throughput |
| Minimum meaningful population | 10,000 agents | Domain-dependent; could be 5,000 for some questions |
| Simulated duration | 90 days × 24 h = 2,160 timesteps | Shorter scenario → proportionally faster |
| Prefix cache hit rate | 0% (conservative) | 50% cache hits → 2× effective throughput |

The largest uncertainty is per-endpoint throughput, which varies by:
- Model size (120B vs 7B is ~15–20× throughput difference)
- GPU memory bandwidth (MI250X on Aurora vs A100 vs H100)
- Prompt length and batch size

If smaller models (7B–13B) are used with careful prompt compression, 10–20 prompts/sec per endpoint is achievable, which shifts the feasible range by ~5–10×, enabling 300K–600K agent simulations within the same wall-clock budget.

---

*Analysis prepared April 2026. Based on Aurora Swarm measured benchmarks, published ChiSim/CityCOVID papers, and SwarmSim-Design.md.*
