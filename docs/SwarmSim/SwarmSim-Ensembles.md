# SwarmSim Ensemble Strategy

## The single-run problem

Both ChiSim and SwarmSim share a fundamental limitation: a single run produces one epidemic trajectory, and there is no way to determine from that run alone whether the answer is correct. The determinism of ChiSim does not help here — a deterministic model gives you the same wrong answer every time if its parameters or behavioral assumptions are off.

The standard response to this in epidemiological modeling is ensemble runs, and both systems use them — but they are measuring fundamentally different things.

---

## ChiSim ensembles: parameter uncertainty

ChiSim is deterministic given a random seed. A single run is fully reproducible, which makes it easy to isolate the effect of any single parameter change: run twice, change one thing, the difference in outcomes is attributable entirely to that change.

The uncertainty ChiSim ensembles address is **parameter uncertainty** — the modeler does not know the true value of inputs like base transmission probability, mask efficacy, compliance rates, or hospitalization thresholds. Groups using CityCOVID/ChiSim run parameter sweep ensembles (using tools like EMEWS — Extreme-scale Model Exploration with Swift, an Argonne HPC workflow tool built for this purpose) varying these inputs across hundreds or thousands of runs. The spread of outcomes across the parameter space is the uncertainty estimate.

What ChiSim ensembles cannot address: the behavioral rules themselves are fixed. A compliant agent is always compliant at the parameterized rate. There is no run-to-run variance in how agents reason about their situation — that variance has been encoded away into scalar compliance probabilities.

---

## SwarmSim ensembles: behavioral uncertainty

SwarmSim's stochasticity is structural, not parametric. It lives in LLM sampling. Two runs with identical parameters — same agents, same disease seed, same policy timeline — produce different epidemic curves because agents make different hourly decisions. This is not a flaw to be eliminated; it represents something real: **the irreducible variance in how a population of people would actually reason and behave under a pandemic**.

The weather forecast analogy is precise. Numerical weather prediction ensembles (e.g., ECMWF runs 51 members) exist because the atmosphere is chaotic — small differences in initial conditions or model stochasticity propagate into large forecast divergence. The ensemble spread *is* the forecast uncertainty. A tight ensemble means the outcome is robust to that stochasticity; a wide one means it is sensitive to factors that cannot be pinned down.

SwarmSim ensemble spread tells you: **how sensitive are epidemic outcomes to the irreducible variance in human behavioral reasoning?** This is a question ChiSim parameter sweeps cannot answer, because they hold behavioral reasoning fixed by construction.

---

## The two ensembles are complementary

| | ChiSim ensemble | SwarmSim ensemble |
|---|---|---|
| **Source of variance** | Parameter uncertainty (transmission rates, compliance scalars) | Behavioral reasoning stochasticity (LLM sampling) |
| **What spread tells you** | Sensitivity to unknowable model parameters | Sensitivity to irreducible human behavioral variance |
| **What is held fixed** | Behavioral rules | Parameters |
| **What varies** | Parameter values across runs | Agent decisions across runs |
| **Interpretability** | Clean — any outcome difference traces to a specific parameter | Harder — differences arise from emergent reasoning chains |
| **Reproducibility** | Deterministic given seed | Stochastic; different trajectory each run |

The systems are not competing — they characterize different sources of uncertainty. Running both at matched parameters enables a useful cross-check:

- If the **ensemble means converge**, that is a form of validation: SwarmSim's LLM behavioral layer is producing aggregate dynamics consistent with ChiSim's calibrated rules. The two systems agree on the expected trajectory even though they reach it through entirely different mechanisms.

- If the **SwarmSim spread is wide**, that is a scientific finding: epidemic outcomes under this scenario are highly sensitive to behavioral variance — sensitivity that ChiSim's clean single answer was hiding by design. The parameterized compliance scalars in ChiSim were suppressing real uncertainty.

- If the **SwarmSim mean diverges from ChiSim's answer**, that points to a systematic difference in the behavioral layer — either ChiSim's compliance parameterization is mis-calibrated, or LLM reasoning introduces a systematic bias worth investigating.

---

## Practical implications for running SwarmSim

Because each SwarmSim run is stochastic, a single run should not be treated as the answer. The minimum useful output is an ensemble, and the ensemble spread is a first-class scientific output alongside the mean trajectory.

**Causal attribution becomes statistical.** To measure the effect of a policy intervention (e.g., mask mandate on day 18), run an ensemble with the mandate and an ensemble without. Some of the outcome difference between any two individual runs is the policy effect; some is behavioral sampling noise. The signal-to-noise ratio improves with ensemble size. This is the same discipline applied to stochastic ABMs like NetLogo, but more expensive per run given the LLM inference cost.

**Parameter sweeps can be layered on top.** There is no reason to choose between parameter uncertainty and behavioral uncertainty. A full design-of-experiments would vary parameters across ensemble replicates — characterizing both sources of variance simultaneously. At scale this is expensive, but the architecture supports it: each run is independent, and the coordinator holds no cross-run state.

**The SOUL.md tradeoff.** Proposals to have agents update their own persistent characteristics (analogous to OpenClaw's SOUL.md pattern) would increase emergent behavioral heterogeneity across and within runs. This makes agents more realistic in the sense of being less predictable, but it widens the ensemble spread and makes causal attribution harder. The more agents diverge from their initial profiles through self-authored evolution, the more difficult it becomes to attribute outcome differences to specific scenario parameters versus accumulated character drift. Whether this tradeoff is acceptable depends on whether emergent behavioral heterogeneity is a scientific goal or a confound in a given study.

---

## The UQ module

The `aurora_swarm/uq/` module implements semantic entropy and kernel language entropy — measures of how variable LLM responses are across the swarm at a given tick. High semantic entropy at a tick means agents are producing diverse, disagreeing decisions. Low entropy means the swarm is behaviorally coordinating.

This per-tick, per-agent variance signal has no ChiSim equivalent. It enables questions like:

- Under which policy conditions do agents disagree most in their behavioral responses?
- Does behavioral variance peak before or after epidemic peaks?
- Are certain occupation types or demographic groups more behaviorally variable than others?

The ensemble spread in epidemic outcomes and the within-tick semantic entropy of decisions are complementary diagnostics: one characterizes run-to-run variance in aggregate outcomes, the other characterizes decision-level variance within a single run.
