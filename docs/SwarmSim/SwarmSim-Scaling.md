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

> **⚠ Critical caveat: this benchmark does not reflect SwarmSim operating conditions.**
>
> The measurement used 20 short prompts across 4 endpoints — 5 prompts per endpoint. At that concurrency the GPU is barely utilized; the 12% speedup over non-batch mode measures HTTP overhead reduction, not GPU throughput under load. The number tells us nothing reliable about performance at the concurrency levels (64–256 prompts/endpoint), prompt lengths (~750 tokens input, ~150 tokens output), and endpoint counts (12,000–24,000) that SwarmSim actually requires.
>
> **All scaling conclusions in this document that depend on a prompts/sec figure are unvalidated until the benchmark described in Section 12 is run.** The 1–3 prompts/sec estimate below, the 50-second/256-prompt hypothesis, and the derived agent ceilings are informed estimates, not measured results.

The SwarmSim agent prompt (Section 4 of SwarmSim-Design.md) is long — approximately 800–1,200 tokens of system context plus the agent profile plus the task — with an expected response of ~100–200 tokens (structured JSON). Call it **~1,000 tokens input / 150 tokens output** per agent-timestep.

On `gpt-oss-120b` (a 120B-parameter reasoning model), throughput in batch mode is heavily input-bound. The 7.24 prompts/sec figure above is with a small test — at production scale with long prompts and a concurrency-limited single endpoint, **~1–3 prompts/sec per endpoint** is a plausible range, but this has not been measured under realistic conditions. Larger, purpose-built models will be slower; smaller models will be faster.

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
| Is 3M LLM agents feasible on 3,000 endpoints with a 120B model? | No — ~12 days per 90-day scenario; intractable for research |
| Is 3M LLM agents feasible on different infrastructure? | **Yes** — see Section 11 |
| Is 60K LLM agents feasible? | **Yes** — ~6 hours per scenario on 3,000 endpoints |
| Is 10K LLM agents feasible? | **Yes** — ~1 hour per scenario; allows rapid iteration |
| Is there a scientifically meaningful question? | **Yes** — behavioral heterogeneity studies at neighborhood scale are novel and publishable |
| Does Aurora Swarm support this? | **Yes** — `scatter_gather` with batch mode handles this directly |
| What is the right framing shift? | Replace "simulate Chicago" with "simulate a Chicago neighborhood with high-fidelity behavioral agents" |

The 60K-agent neighborhood simulation is both feasible on the described hardware and scientifically meaningful — and can answer questions the deterministic ABM structurally cannot. 3M agents is not intractable in principle; it requires a different infrastructure operating point, described in Section 11.

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

---

## 11. The Path to 3M LLM Agents

The analysis above treats 3,000 endpoints and a 120B-parameter model as fixed constraints. They are not. Scaling to 3M LLM agents is achievable by pulling two independent levers — endpoint count and inference speed — and by a third technique, model distillation, that improves both simultaneously.

### 11.1 The Two Independent Levers

The core equation is:

```
N_agents = N_endpoints × prompts_per_sec_per_endpoint × wall_clock_budget_per_tick
```

The two variables are independent. Either alone can close a large fraction of the gap to 3M agents.

**Lever 1 — More endpoints.** Aurora Swarm is designed to scale horizontally; adding endpoints requires no code changes. With 150,000 endpoints at current model throughput (2 prompts/sec), the 10-second tick budget supports 3M agents. This is an infrastructure question, not an architecture question.

**Lever 2 — Faster inference.** Smaller models have dramatically higher throughput. A 7B–13B model achieves 10–20 prompts/sec per endpoint — a 5–10× improvement. At 20 prompts/sec, only 15,000 endpoints are needed for 3M agents within a 10-second tick budget.

| Scenario | Endpoints | Prompts/sec/endpoint | Agent ceiling (10s budget) | 90-day run time |
|---|---|---|---|---|
| Current (120B model) | 3,000 | 2 | 60K | ~6 hours |
| Current model, scaled endpoints | 150,000 | 2 | 3M | ~6 hours |
| Small model (7B–13B), current endpoints | 3,000 | 20 | 600K | ~6 hours |
| Small model + modest scale-up | 15,000 | 20 | 3M | ~6 hours |
| 100K endpoints, 0.1s latency (10 prompts/sec) | 100,000 | 10 | 1M agent-ticks/sec → 3s/tick | ~1.8 hours |

The 15,000-endpoint + small-model scenario is the most practical near-term path: it requires a plausible HPC allocation and avoids the infrastructure required for 150K endpoints.

### 11.2 Model Distillation

The most powerful path to both speed and scale is **behavioral model distillation**: use the 1K pilot's outputs to fine-tune a small (7B–13B parameter) model that mimics the behavioral decisions of the full-scale LLM.

The 1K pilot generates 2.16M labeled training examples (agent context → behavioral decision JSON) over a 90-day run. That is sufficient data to fine-tune a small model to reproduce the behavioral patterns — compliance heterogeneity, fatigue dynamics, demographic variation — that the large model generates. The distilled model:

- Runs at 10–20 prompts/sec vs. ~2 prompts/sec for the 120B model (5–10× throughput gain)
- Requires fewer GPU resources per endpoint (7B fits on a single GPU; 120B requires multiple)
- Retains the behavioral richness derived from the large model's reasoning, now encoded in weights rather than generated at inference time

This means the 1K pilot has a secondary output beyond scientific validation: it produces the training corpus for a distilled behavioral model that makes city-scale LLM simulation tractable. The distilled model is then what runs in production.

### 11.3 Prompt Compression as a Third Lever

Inference time scales with token count. The current SwarmSim prompt is ~750 tokens (250 effective after APC cache hits). Compressing the dynamic context blocks by 2× yields roughly 2× throughput from the same hardware — equivalent to doubling the endpoint count at no infrastructure cost. Prompt compression and distillation are complementary: a distilled model can be fine-tuned on compressed prompts, compounding both gains.

### 11.4 Revised Summary

3M LLM agents is not an aspirational ceiling — it is an engineering target reachable through a defined sequence:

1. **1K pilot** — validate behavioral realism; generate distillation training corpus
2. **Distilled small model** — fine-tune on pilot outputs; achieve 10–20 prompts/sec per endpoint
3. **15K–30K endpoint allocation** — plausible on a major HPC system of the near future
4. **Prompt compression** — reduce effective token count by 2×

Steps 1 and 2 are software and research work. Steps 3 and 4 follow from them. The architectural work is already done.

---

---

## 12. Required Benchmark: Throughput Under SwarmSim Conditions

All scaling conclusions in this document rest on a throughput figure (prompts/sec per endpoint) that has not been measured under conditions representative of SwarmSim. The existing benchmark (20 short prompts, 4 endpoints, one run) measures HTTP overhead, not GPU throughput. The following benchmark must be run before any scaling claim can be treated as validated.

### 12.1 What to Measure

**Target conditions:**
- Model: `openai/gpt-oss-120b` (production model)
- Prompt length: ~750 tokens input (SwarmSim agent prompt, not a short test string)
- Response length: ~150 tokens (structured JSON behavioral decision)
- Concurrency levels: 16, 32, 64, 128, 256 prompts per endpoint
- Endpoints: 1 endpoint (single-node characterization), then 4, then 16
- Sustained load: run for at least 5 minutes at each concurrency level, not a single batch

**Metrics to record at each concurrency level:**
- Prompts completed per second (throughput)
- Time to complete one full batch of N prompts (latency)
- GPU utilization (is the endpoint the bottleneck, or the network?)
- Whether throughput scales linearly with concurrency or saturates

### 12.2 Why This Matters

The difference between plausible throughput estimates spans nearly an order of magnitude:

| Assumption | Source | 4K-node fleet throughput | 4K-node sweet-spot agent count |
|---|---|---|---|
| 1.8 prompts/sec/endpoint | Extrapolated from batch benchmark | 21,600/sec | ~216K agents |
| 2 prompts/sec/endpoint | Conservative estimate in docs | 24,000/sec | ~240K agents |
| 5.12 prompts/sec/endpoint | 256 prompts in 50s (hypothesis) | 61,440/sec | ~600K agents |
| 10 prompts/sec/endpoint | Optimistic small-model estimate | 120,000/sec | ~1.2M agents |

The agent ceiling for a 6-hour 90-day run ranges from 216K to 1.2M depending on which number is correct. That difference determines whether SwarmSim at 4K nodes is a neighborhood-scale tool or a district-scale tool — and whether the ChiSim integration strategy (Section 9 of SwarmSim-ChiSim-Integration.md) is necessary or optional.

### 12.3 Benchmark Script

The benchmark should use Aurora Swarm's existing `VLLMPool` with a realistic SwarmSim prompt:

```python
import asyncio, time
from aurora_swarm import VLLMPool, parse_hostfile
from aurora_swarm.patterns.scatter_gather import scatter_gather

# Use a real SwarmSim-length prompt, not a short test string
BENCHMARK_PROMPT = """[~750 token SwarmSim agent prompt here]"""

async def benchmark(hostfile: str, n_prompts: int):
    endpoints = parse_hostfile(hostfile)
    async with VLLMPool(endpoints, model="openai/gpt-oss-120b", use_batch=True) as pool:
        prompts = [BENCHMARK_PROMPT] * n_prompts
        t0 = time.perf_counter()
        responses = await scatter_gather(pool, prompts)
        elapsed = time.perf_counter() - t0
        throughput = n_prompts / elapsed
        print(f"n={n_prompts}, endpoints={len(endpoints)}, "
              f"time={elapsed:.2f}s, throughput={throughput:.2f} prompts/sec")

for n in [16, 32, 64, 128, 256]:
    asyncio.run(benchmark("hostfile.txt", n))
```

### 12.4 Update This Document

Once the benchmark is run, replace the estimate in Section 2.1 with the measured value and regenerate the scaling tables in Sections 5 and 11 using the actual throughput. The tables as written use placeholder estimates — they should not be cited in publications or proposals until validated.

---

*Analysis prepared April 2026. Based on Aurora Swarm measured benchmarks, published ChiSim/CityCOVID papers, and SwarmSim-Design.md. See Section 12 for benchmark validation requirements.*
