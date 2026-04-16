# SwarmSim → ChiSim Integration Strategy

**Argonne National Laboratory & University of Chicago**
*April 2026*

---

## 1. Strategic Framing

SwarmSim's value to disease modeling is not as a standalone epidemic simulator — it cannot scale to the 2.7M-agent populations required for city-level policy analysis. ChiSim already does that, is validated against real Chicago data, and was used operationally during the COVID-19 pandemic.

SwarmSim's specific contribution is **behavioral realism**. ChiSim's behavioral parameters — compliance rates, mask-wearing, testing-seeking — are currently population-level scalar constants, hand-tuned to fit observed data. They do not vary by demographic profile, do not erode over time under prolonged policy, and do not capture the heterogeneous reasoning that drives real compliance behavior.

SwarmSim replaces that hand-tuning with LLM-derived behavioral dynamics: agents reason from their occupation, income, household obligations, disease state, and policy environment to produce decisions. Those decisions, aggregated across ensemble runs, become demographically-stratified lookup tables that ChiSim can consume in place of its scalar constants.

**SwarmSim is a behavioral calibration tool for ChiSim, not a replacement or companion at runtime.**

---

## 2. Why Runtime Integration Does Not Work

The natural first instinct is to run SwarmSim alongside ChiSim, with LLM inference calls made for a subset of agents at each tick. This does not work for a fundamental reason:

ChiSim's speed — minutes on ALCF hardware for a 90-day run across 2.7M agents — comes from tight, deterministic MPI ticks. LLM inference is slow and variable. At realistic throughput (~2 prompts/sec/endpoint), processing 100K agents per tick requires approximately 50 seconds of wall-clock time. Inserting that pause at every tick of a 2,160-tick simulation produces a ~30-hour runtime — the same ceiling SwarmSim faces, now imposed on ChiSim as well. The LLM latency that limits SwarmSim's scale would make ChiSim equally non-scalable.

The integration must therefore be **temporal decoupling**: SwarmSim runs offline, produces outputs, and those outputs are loaded by ChiSim at initialization. No LLM calls occur during a ChiSim run.

---

## 3. What SwarmSim Produces

Each SwarmSim agent, at each tick, emits a structured JSON decision:

```json
{
  "agent_id": "...",
  "sim_day": 14,
  "hour": 9,
  "disease_state": "SYMPTOMATIC-MILD",
  "location": "WORKPLACE",
  "mask_wearing": true,
  "distancing": false,
  "health_seeking": false,
  "behavioral_deviation": "Went to work despite mild symptoms; rent due Friday"
}
```

Across an ensemble of runs (20–50 recommended), these per-agent, per-tick decisions are aggregated into **behavioral rate tables**: the empirical probability of each behavioral choice, conditioned on the agent's observable attributes and situation.

---

## 4. The Five ChiSim Parameters and Their SwarmSim Replacements

ChiSim currently encodes agent behavior through five scalar parameters defined in `SwarmSim-Design.md` Section 3.3. These are the integration targets:

| ChiSim Parameter | Current Form | SwarmSim-Derived Replacement | Primary Index Dimensions |
|---|---|---|---|
| `isolation_compliance` | Scalar float [0,1] | P(stay home \| ...) lookup table | occupation × income\_bracket × policy\_regime × days\_since\_policy |
| `mask_wearing` | Scalar float or binary | P(mask \| ...) lookup table | occupation × age\_group × location\_type × policy\_regime |
| `distancing_compliance` | Scalar float [0,1] | P(distance \| ...) lookup table | occupation × income\_bracket × policy\_regime |
| `testing_seeking` | Scalar f(severity, access) | P(seek\_care \| ...) lookup table | occupation × income\_bracket × disease\_state |
| `isolation_onset_boost` | Fixed scalar | Compliance curve over time | days\_since\_symptom\_onset |

All index dimensions — occupation type, income bracket, disease state, policy regime — are attributes ChiSim already stores for every agent. No new data collection is required.

### 4.1 Table Sizes

Using ChiSim's existing schema:
- 8 occupation types
- 4 income brackets
- 4 policy regimes (baseline / mild NPI / moderate NPI / strict NPI)
- 90 simulation days (for time-varying parameters)

A fully-stratified table has at most `8 × 4 × 4 × 90 = 11,520` entries per parameter. Five parameters total roughly 57,000 floats — trivial to store, load, and index.

### 4.2 The Most Valuable Parameter: Compliance Fatigue

The `isolation_compliance` time dimension (`days_since_policy_onset`) is the single parameter where SwarmSim's contribution is most differentiated. ChiSim currently has **no time-varying compliance** — the scalar applies identically on day 1 and day 60 of a lockdown. SwarmSim's `compliance_fatigue` behavioral state captures well-documented erosion of compliance under prolonged restriction. Replacing the scalar with a decay curve for each demographic group is likely the change most likely to improve ChiSim's fit to observed 2020 data and, more importantly, to improve its generalization to novel policy scenarios.

---

## 5. The Integration Workflow

### Phase 1: SwarmSim Calibration Runs

1. Run SwarmSim for Logan Square using 1K LLM agents with 2020 pandemic parameters
2. Run 20–50 ensemble iterations to characterize behavioral variance
3. Calibrate the ensemble mean epidemic curve to 2020 Logan Square hospitalization data
4. Repeat for 2–3 additional Chicago neighborhoods with distinct demographic profiles (e.g., a higher-income, remote-capable neighborhood; a lower-income, essential-worker-dense neighborhood)

A full demographic sweep is not required: behavioral variation is driven by occupation, income, and household structure, not geography per se. Three or four demographically-distinct neighborhoods are likely sufficient to span the behavioral parameter space.

### Phase 2: Table Extraction

From each ensemble run, aggregate per-agent, per-tick decisions:

```
For each (occupation, income_bracket, policy_regime, sim_day):
    isolation_compliance = mean(agent.went_home / agent.was_instructed_to_stay_home)
    mask_wearing         = mean(agent.wore_mask / agent.was_in_public_space)
    distancing           = mean(agent.distanced / agent.was_in_shared_space)
    testing_seeking      = mean(agent.sought_care / agent.was_symptomatic)
```

Average across ensemble runs for point estimates; retain standard deviation as uncertainty bounds. Produce one CSV per parameter.

### Phase 3: ChiSim Modification

ChiSim reads the five CSV tables at initialization. The behavioral parameter lookups in the agent tick logic change from scalar constants to table dereferences:

**Before:**
```cpp
double compliance = 0.70;  // hand-tuned constant
if (uniform_random() < compliance) { stay_home(); }
```

**After:**
```cpp
double compliance = isolation_table[agent.occupation]
                                   [agent.income_bracket]
                                   [current_policy_regime]
                                   [days_since_policy_onset];
if (uniform_random() < compliance) { stay_home(); }
```

The agent attributes used as index keys (`occupation`, `income_bracket`) are already stored per-agent in ChiSim. The policy regime and day counter are already tracked by the simulation coordinator. **No new data structures or agent attributes are required.**

### Phase 4: Validation

Run ChiSim with SwarmSim-derived tables against the 2020 held-out data:
- Primary test: does ChiSim's fit to observed hospitalizations and deaths improve?
- Secondary test: do the SwarmSim-derived tables produce more realistic compliance trajectories than the hand-tuned scalars (i.e., do they show fatigue, demographic stratification matching survey data)?
- Out-of-sample test: do the tables generalize better than the scalars to a policy scenario not included in the SwarmSim calibration runs?

---

## 6. What ChiSim Would Need to Change

The required ChiSim modification is bounded and does not require architectural change:

1. **Table loader** — read five CSV files at initialization and populate in-memory lookup structures (a 4D array per parameter)
2. **Parameter lookup replacement** — five locations in the agent tick logic where scalar constants are replaced with table dereferences
3. **Policy regime tracking** — ChiSim already tracks active policy; it needs to expose the current regime as an enum matching the four SwarmSim policy states (`baseline`, `mild_npi`, `moderate_npi`, `strict_npi`) and the count of days since the current regime became active

The interface between teams is a **file format agreement**: SwarmSim produces CSVs with standardized column names; ChiSim reads them. Neither team needs to understand the other's codebase.

---

## 7. What This Does Not Require

- ChiSim does not need to run LLM inference
- ChiSim does not need to know about Aurora Swarm or vLLM
- SwarmSim does not need to run at ChiSim's scale
- The ChiSim team does not need to change their simulation architecture, MPI layout, or disease engine
- SwarmSim runs do not need to reproduce full Chicago — neighborhood-scale pilots are sufficient

---

## 8. The Proposition to the ChiSim Team

SwarmSim produces five CSV lookup tables. ChiSim replaces five scalar constants with five table lookups keyed on fields it already tracks. The ChiSim simulation runs exactly as before at full speed on ALCF hardware.

The testable question: do LLM-derived behavioral tables produce a better fit to 2020 observed data than hand-tuned scalars, and do they generalize better to novel policy scenarios?

If yes, the scientific contribution is clear: LLM-based behavioral sampling provides a principled, data-driven method for setting ChiSim's behavioral parameters, replacing expert judgment with emergent agent reasoning. For a future pandemic, this means behavioral parameters can be derived from early observed mobility and compliance data — rather than waiting for epidemiological outcomes to calibrate against — potentially improving forecast lead time.

---

## 9. Open Questions

1. **Generalization across panhogens.** The SwarmSim calibration uses COVID-19-specific behavioral context. How much of the extracted behavioral parameter structure transfers to a novel pathogen with different severity, transmissibility, or public visibility? This is a research question, not an engineering one.

2. **Feedback from ChiSim to SwarmSim.** The current design is one-directional: SwarmSim → tables → ChiSim. An iterative loop — where ChiSim epidemic curves inform the local case context injected into SwarmSim prompts, producing better-calibrated behavioral parameters — is architecturally possible but adds complexity. Worth considering once the one-directional pipeline is validated.

3. **Sufficient neighborhood coverage.** Three or four SwarmSim neighborhood studies is an assumption. Empirical validation that the extracted tables generalize across Chicago's demographic range is required before claiming the approach works city-wide.

4. **Compliance survey validation.** Before inserting SwarmSim-derived tables into ChiSim, the tables should be compared against available behavioral survey data (Google Mobility Reports, SafeGraph, COVID-19 community surveys) to confirm the LLM behavioral rates are empirically grounded, not just internally consistent.
