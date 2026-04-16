# SwarmSim Scaling: Slide Deck

---

## Slide 0 — ChiSim / CityCOVID: The Flagship System

Developed by Argonne National Laboratory and the University of Chicago, ChiSim (Chicago Social Interaction Model) is a city-scale agent-based model used operationally during the COVID-19 pandemic to advise the City of Chicago, Cook County, and the Illinois Governor's COVID-19 Modeling Task Force. CityCOVID is the COVID-19 application built on the ChiSim framework; it ran on Argonne Leadership Computing Facility (ALCF) resources (originally Theta, Cray XC40, 11.7 PFlops) and completes a 90-day scenario in minutes on 1,000+ MPI ranks.

**Scale**

| | Count |
|---|---|
| Synthetic agents | 2.7 million (full Chicago population) |
| Geolocated places | 1.2 million (households, schools, workplaces, hospitals, community venues) |
| Timestep | 1 hour |
| Typical scenario | 90 simulated days |
| Ensemble size | Hundreds of runs for ABC calibration; tens of runs for policy comparison scenarios |
| Compute | Distributed MPI / Repast HPC (C++); agents migrate between MPI ranks each tick |

**What each agent carries**

| Category | Key attributes |
|---|---|
| Identity (static) | Age, sex, race/ethnicity, zip code, household ID, income bracket |
| Role (static) | Occupation type, can-work-from-home, uses-transit, workplace ID, school ID |
| Health baseline (static) | Comorbidities (diabetes, hypertension, obesity, COPD…), healthcare access |
| Location (dynamic) | Current place ID, schedule index |
| Disease (dynamic) | State (see below), days in state, cumulative exposure dose |
| Behavior (dynamic) | Isolation compliance, mask wearing, distancing compliance, testing seeking |

**Disease state machine (changes daily, driven by hourly exposure)**

```
SUSCEPTIBLE → EXPOSED → PRE-SYMPTOMATIC ─┐
                       → ASYMPTOMATIC ───┤
                                         ↓
                              SYMPTOMATIC-MILD → SYMPTOMATIC-SEVERE
                                                       ↓
                                              HOSPITALIZED → ICU
                                                    ↓           ↓
                                               RECOVERED    DECEASED
```

**What happens at each 1-hour tick (per agent)**

| Step | What the agent does | Governed by |
|---|---|---|
| **Move** | Look up schedule → get target place; apply policy filter (is this place closed?); apply compliance draw (do I actually stay home?) | ATUS activity schedule + NPI policy state + `isolation_compliance` |
| **Interact** | At the destination, compute infectious exposure from every co-located infectious agent | Co-occupant count, `base_transmission_prob`, density scaling, mask factors |
| **Transition** | Probabilistically advance disease state based on exposure dose and days in current state | Age/comorbidity-weighted transition probabilities (calibrated via ABC) |
| **Behavior update** | If newly symptomatic: boost `isolation_compliance`; if policy changed: update schedule filters | `isolation_onset_boost` scalar; active NPI policy set |

**How ChiSim is calibrated and used**

Transmission parameters (base infection probability, density scaling, asymptomatic infectivity multiplier, age-stratified severity) are estimated via Approximate Bayesian Computation (ABC) using EMEWS/Swift-T on ALCF — hundreds of ensemble runs compared against observed daily hospitalizations and deaths. Once calibrated, policy scenarios (school closures, mask mandates, stay-at-home orders) are run as controlled experiments: two runs identical except for one policy change, difference in epidemic curve attributed to that policy.

**The limitation SwarmSim addresses**

All behavioral parameters — `isolation_compliance`, `mask_wearing`, `distancing_compliance`, `testing_seeking`, `isolation_onset_boost` — are scalar constants, uniform across the population. A retired 70-year-old and a 25-year-old essential worker with rent due have the same compliance rate. Compliance does not erode under prolonged lockdown. These are the five parameters SwarmSim replaces with demographically-stratified, time-varying lookup tables.

---

---

## Slide 1 — Today's Capability (1K nodes, 3 endpoints, 256 concurrent prompts, 50s/batch)

**Fleet:** 1,000 nodes × 3 endpoints × (256 prompts / 50s) = **15,360 agent-ticks/sec**

At current Aurora allocation, SwarmSim can run neighborhood-scale LLM simulations but cannot reach full-city scale. The practical use is to run several calibrated SwarmSim studies across Chicago neighborhood archetypes and extract lookup tables that replace five hard-coded scalar parameters in ChiSim — a simple code change (scalar constant → table lookup keyed on agent attributes already tracked by ChiSim).

| N agents | Time per tick | 90-day run | Viability |
|---|---|---|---|
| 60K | 3.9s | ~2.3 hours | Good — fast iteration |
| **150K** | **9.8s** | **~5.9 hours** | **Sweet spot — 4 runs/day** |
| 256K | 16.7s | ~10 hours | Overnight |
| 750K | 48.8s | ~29 hours | One run/day — marginal |
| 2.7M (full Chicago) | 175s | ~4.4 days | Intractable for sweeps |

**Conclusion:** Use SwarmSim to calibrate the five ChiSim behavioral parameters:

| ChiSim parameter | Current form | SwarmSim contribution |
|---|---|---|
| `isolation_compliance` | Scalar constant | Table by occupation × income × policy regime |
| `mask_wearing` | Scalar constant | Table by occupation × age × location type × policy |
| `distancing_compliance` | Scalar constant | Table by occupation × income × policy regime |
| `testing_seeking` | Scalar constant | Table by occupation × income × disease state |
| `isolation_onset_boost` | Scalar constant | **Time-decay curve** by demographic (compliance erodes over weeks — currently unmodeled in ChiSim) |

ChiSim team change required: replace five scalar constants with five table lookups keyed on fields already stored per-agent. SwarmSim produces the tables as CSV; ChiSim loads them at initialization. No architectural change required.

---

## Slide 2 — Near-Term Target (8K nodes, 3 endpoints, 256 concurrent prompts, 20s/batch)

**Fleet:** 8,000 nodes × 3 endpoints × (256 prompts / 20s) = **307,200 agent-ticks/sec**

At 8K nodes with prompts tuned to complete in 20 seconds, SwarmSim can simulate the full 2.7M Chicago population directly — making it a standalone city-scale behavioral simulator rather than a calibration tool. The hardware exists on Aurora today; the open question is prompt optimization to reach the 20s target.

| N agents | Time per tick | 90-day run | Viability |
|---|---|---|---|
| 256K | 0.8s | ~30 min | Very fast sweeps |
| 750K | 2.4s | ~1.5 hours | Excellent for calibration |
| 1M | 3.3s | ~2 hours | Excellent |
| **2.7M (full Chicago)** | **8.8s** | **~5.3 hours** | **Research viable — 4 runs/day** |
| 3M | 9.8s | ~5.9 hours | Research viable |

**Conclusion:** This target is achievable within 6 months — the Aurora hardware is in place, the Aurora Swarm infrastructure is built, and the primary engineering work is reducing per-batch latency from 50s to 20s. We should build toward this operating point now: a system that runs full Chicago in a single working day changes the research question from *"how do we approximate ChiSim?"* to *"what can we learn that ChiSim structurally cannot tell us?"*

> *The 20s/batch figure is an engineering target, not a measured result — current benchmarks show ~50s at 256 concurrent prompts using `gpt-oss-120b`. Three levers can close this gap, likely in combination: (1) **smaller model** — a distilled 7B–13B behavioral model fine-tuned on SwarmSim outputs could run 5–10× faster per prompt; (2) **prompt compression** — SwarmSim's ~750-token agent prompts can be reduced by eliminating redundant context and tightening the JSON schema, with each 2× reduction in token count yielding roughly proportional throughput gains; (3) **vLLM tuning** — adjusting batch size, KV-cache configuration, and tensor parallelism for the specific Aurora GPU architecture (Intel Ponte Vecchio) may recover significant latency at no modeling cost. The 1K pilot provides the data needed for (1); (2) and (3) are engineering tasks independent of scientific validation.*
