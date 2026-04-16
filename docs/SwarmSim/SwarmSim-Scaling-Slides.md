# SwarmSim Scaling: Two Scenarios

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

**Conclusion:** This target is achievable within 6 months — the Aurora hardware is in place, the Aurora Swarm infrastructure is built, and the primary engineering work is prompt compression to reduce per-batch latency from 50s to 20s. We should build toward this operating point now: a system that runs full Chicago in a single working day changes the research question from *"how do we approximate ChiSim?"* to *"what can we learn that ChiSim structurally cannot tell us?"*
