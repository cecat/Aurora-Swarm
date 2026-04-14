# ChiSIM / CityCOVID → LLM Agent Design
**Argonne National Laboratory & University of Chicago**
*Prepared for TPC / OpenClaw Research — April 2026*

---

## 1. Project Overview

The ChiSIM/CityCOVID project is a city-scale agent-based model (ABM) developed at Argonne National Laboratory to simulate the spread of infectious disease — including COVID-19 — through the full population of Chicago. It was used operationally during the pandemic to advise the City of Chicago, Cook County, and the Illinois Governor's COVID-19 Modeling Task Force.

The project has two layers:
- **ChiSIM** (Chicago Social Interaction Model) — the general-purpose ABM framework for urban social interaction simulation
- **CityCOVID** — the COVID-19-specific application built on top of ChiSIM

**Key personnel:** Charles "Chick" Macal and Jonathan Ozik (Argonne DIS Division), Nicholson Collier, Eric Tatara, Chaitanya Kaligotla, Justin Wozniak, and collaborators from UChicago Medicine (Anna Hotton, Aditya Khanna, Harold Pollack, John Schneider).

**Scale:** 2.7 million synthetic agents representing Chicago residents; 1.2 million geolocated places (households, schools, workplaces, hospitals, group quarters, community venues). Runs on Argonne Leadership Computing Facility (ALCF) resources — originally on Theta (Cray XC40, 11.7 PFlops).

---

## 2. Reference Map

Read these in order for increasing technical depth.

### 2.1 Primary Framework Paper
**Macal, C.M., Collier, N.T., Ozik, J., Tatara, E.R., and Murphy, J.T. (2018).**
"ChiSIM: An Agent-Based Simulation Model of Social Interactions in a Large Urban Area."
*Proceedings of the 2018 Winter Simulation Conference*, pp. 810–820.
DOI: [10.1109/WSC.2018.8632409](https://doi.org/10.1109/WSC.2018.8632409)
PDF: http://simulation.su/uploads/files/default/2018-macal-collier-ozik-tatara-murphy.pdf
> **Start here.** Describes the full agent/place/schedule architecture of ChiSIM and the behavioral ontology.

### 2.2 CityCOVID Application Paper (WSC 2020)
**Macal, C.M., Ozik, J., Collier, N.T., Kaligotla, C., MacDonell, M.M., Wang, C., LePoire, D.J., Chang, Y., and Martinez-Moyano, I.J. (2020).**
"CityCOVID: A Computer Simulation of COVID-19 Spread in a Large Urban Area."
*Proceedings of the 2020 Winter Simulation Conference.*
Full text: https://informs-sim.org/wsc20papers/087.pdf
> Describes CityCOVID model design, disease transmission mechanics, and policy scenario methodology.

### 2.3 Population Data-Driven Workflow Paper (IJHPCA 2021)
**Ozik, J., Wozniak, J.M., Collier, N., Macal, C.M., and Binois, M. (2021).**
"A population data-driven workflow for COVID-19 modeling and learning."
*International Journal of High Performance Computing Applications* 35(5).
DOI: [10.1177/10943420211035164](https://doi.org/10.1177/10943420211035164)
OSTI: https://www.osti.gov/biblio/1819798
ResearchGate: https://www.researchgate.net/publication/354524611
> The most technically detailed paper. Contains the calibration parameter table (Table 1), disease state transitions, transmission probabilities, ML/ABC calibration workflow, and HPC architecture.

### 2.4 ChiSIM Software Repository (OSTI)
**Ozik, J., Collier, N.T., and Macal, C.M.**
*Chicago Social Interaction Model (ChiSIM) Software.*
OSTI: https://www.osti.gov/biblio/1822421

### 2.5 ChiSIM GitHub Repository (C++ Source)
https://github.com/Repast/chiSIM
> Open source, C++, MIT license. Implements the distributed agent/place framework on Repast HPC (MPI). **This is where the actual agent class definitions live.**

### 2.6 CommunityRx / CRx Model Paper (earlier ChiSIM application)
**Kaligotla, C., Ozik, J., Collier, N., Macal, C.M., Boyd, K., Makelarski, J., Huang, E.S., and Lindau, S.T. (2020).**
"Model Exploration of an Information-Based Healthcare Intervention Using Parallelization and Active Learning."
*Journal of Artificial Societies and Social Simulation* 23(4).
DOI: [10.18564/jasss.4379](https://doi.org/10.18564/jasss.4379)
> Clearest description of per-agent mechanics in a ChiSIM model; useful as a less COVID-specific reference for agent coding.

### 2.7 CA-MRSA Precursor Model (original ChiSIM ancestor)
**Macal, C.M., North, M.J., Collier, N., Dukic, V.M., et al. (2014).**
"Modeling the transmission of community-associated methicillin-resistant Staphylococcus aureus: a dynamic agent-based simulation."
*Journal of Translational Medicine* 12(1): 124.
> Historical context: ChiSIM is a generalization of this single-process C++ model.

### 2.8 Repast HPC Parallelization
**Collier, N.T., Ozik, J., and Macal, C.M. (2015).**
"Large-Scale Agent-Based Modeling with Repast HPC: A Case Study in Parallelizing an Agent-Based Model."
*Euro-Par 2015: Parallel Processing Workshops.* LNCS vol. 9523. Springer.
> Documents the MPI-distributed agent movement architecture.

### 2.9 IllinoisCOVID Consortium
https://illinoiscovid.org/about.html
> Context for CityCOVID's role alongside Northwestern, UChicago, and UIUC SEIR/compartmental models in advising IDPH.

### 2.10 Argonne CityCOVID Model Page
https://www.anl.gov/dis/citycovid-about-the-model
> High-level narrative description of the model for public audiences.

---

## 3. The ChiSIM/CityCOVID Agent Coding Schema

Each agent in ChiSIM/CityCOVID is a C++ object (in Repast HPC) with the following logical structure. This is reconstructed from the papers and GitHub source.

### 3.1 Three Inputs to Every ChiSIM Model

Every model built on the ChiSIM framework takes exactly three categories of input:

| Input | Description | Source for CityCOVID |
|---|---|---|
| **Synthetic People** | Agent attribute vectors for all 2.7M Chicago residents | Extended from RTI synthetic population (Cajka et al. 2010); statistically matches Chicago demographics |
| **Places** | 1.2M geolocated parcels with type labels | GIS data; geocoded land use records |
| **Activity Schedules** | Hourly schedule of which place each agent occupies | American Time Use Survey (ATUS) + Panel Study of Income Dynamics (PSID), assigned by demographic profile |

### 3.2 Agent Attributes (Static / Slow-changing)

These are assigned at initialization and do not change during a simulation run (or change only under specific policy interventions):

#### Sociodemographic
| Attribute | Type | Notes |
|---|---|---|
| `agent_id` | integer | Unique identifier |
| `age` | integer | Years; critical for disease progression probabilities |
| `sex` | enum {M, F} | |
| `race_ethnicity` | enum | Used for health disparity modeling |
| `household_id` | integer | Links agent to home place |
| `household_size` | integer | Number of co-residents |
| `zip_code` | string | Neighborhood of residence |
| `income_bracket` | enum/float | Household income; affects healthcare access |

#### Occupation / Social Role
| Attribute | Type | Notes |
|---|---|---|
| `occupation_type` | enum | Essential worker / remote-capable / student / school-age child / retired / unemployed / healthcare worker / nursing home resident |
| `workplace_id` | integer | Links agent to assigned workplace place |
| `school_id` | integer | For students and school-age children |
| `can_work_from_home` | boolean | Key NPI behavioral flag |
| `uses_public_transit` | boolean | Affects exposure during transit hours |

#### Health Baseline
| Attribute | Type | Notes |
|---|---|---|
| `comorbidities` | bitmask/list | Diabetes, hypertension, obesity, immunocompromised, COPD, cardiovascular disease |
| `healthcare_access` | float [0,1] | Probability of seeking care when symptomatic |
| `age_risk_multiplier` | float | Pre-computed from age; scales transition probabilities |

### 3.3 Agent Dynamic State

These change at each simulation timestep (1 hour):

#### Location State
| Attribute | Type | Notes |
|---|---|---|
| `current_place_id` | integer | Which of the 1.2M places the agent occupies this hour |
| `current_process_rank` | integer (MPI) | Which compute node holds the agent this timestep |
| `schedule_index` | integer | Current position in the hourly activity schedule |

#### Disease State Machine

The disease state is the primary dynamic variable. CityCOVID uses an extended SEIR model with the following states:

```
SUSCEPTIBLE
    │
    ▼ (exposure event: co-location with infectious agent)
EXPOSED  ← incubation period; not yet infectious
    │
    ├──────────────────────────┐
    ▼                          ▼
PRE-SYMPTOMATIC           (→ ASYMPTOMATIC INFECTIOUS)
(infectious, no symptoms)         │
    │                          │
    ▼                          │
SYMPTOMATIC-MILD               │
    │                          │
    ├──────────────────────────┘
    │                          
    ▼ (age/comorbidity-dependent probability)
SYMPTOMATIC-SEVERE
    │
    ▼
HOSPITALIZED
    │
    ├──────────────┐
    ▼              ▼
  ICU          RECOVERED
    │
    ├──────────────┐
    ▼              ▼
DECEASED       RECOVERED
```

| Disease State | Infectious? | Agent aware? | Behavior modified? |
|---|---|---|---|
| SUSCEPTIBLE | No | Yes | No (baseline schedule) |
| EXPOSED | No | No | No |
| PRE-SYMPTOMATIC | Yes | No | No (continues normal schedule) |
| ASYMPTOMATIC | Yes | No | No (continues normal schedule) |
| SYMPTOMATIC-MILD | Yes | Yes | Yes (partial isolation) |
| SYMPTOMATIC-SEVERE | Yes | Yes | Yes (strong isolation / seeks care) |
| HOSPITALIZED | Managed | Yes | Yes (schedule overridden to HOSPITAL) |
| ICU | Managed | Yes | Yes (schedule overridden to HOSPITAL) |
| RECOVERED | No | Yes | Optional relaxation of precautions |
| DECEASED | No | N/A | Schedule nullified |

#### Disease State Timing Attributes
| Attribute | Type | Notes |
|---|---|---|
| `disease_state` | enum | Current state from table above |
| `days_in_state` | integer | Counter; used for transition timing |
| `exposure_count` | float | Cumulative infectious exposure (dose proxy) |
| `symptom_onset_day` | integer | Set when transitioning to symptomatic |
| `hospitalization_day` | integer | Set when hospitalized |

#### Behavioral Compliance Attributes
| Attribute | Type | Notes |
|---|---|---|
| `isolation_compliance` | float [0,1] | Probability of following stay-at-home order |
| `mask_wearing` | boolean / float | Binary or probabilistic mask usage |
| `distancing_compliance` | float [0,1] | Social distancing behavior |
| `testing_seeking` | float [0,1] | Probability of seeking test when symptomatic |

### 3.4 Key Calibrated Transmission Parameters (from Ozik et al. 2021, Table 1)

These are model-level (not per-agent) parameters estimated via Approximate Bayesian Computation:

| Parameter | Prior Distribution | Meaning |
|---|---|---|
| Initial exposed agents | U(60, 190) | Seed infections at simulation start |
| Base hourly transmission probability | U(0.03, 0.1) | P(transmission) per infectious-susceptible co-location per hour |
| Co-location density scaling factor | U(0.5, 1.0) | Adjusts transmission for ratio of infectious to susceptible at a location |
| Asymptomatic infectivity multiplier | estimated | Relative infectiousness of asymptomatic vs. symptomatic |
| Age-specific severity probabilities | age-stratified | Probability of mild→severe, severe→hospitalized, hospitalized→ICU, ICU→deceased |

### 3.5 Per-Timestep Agent Logic (the "tick")

At each 1-hour timestep, each agent executes the following logic (deterministic ABM version):

```
1. MOVE:
   Check schedule[current_hour] → get target place_id
   Apply policy filter: if place_type is CLOSED under current NPI, redirect to HOME
   Apply compliance filter: if isolation_compliance draw succeeds, redirect to HOME
   Move to target place (possibly crossing MPI process boundaries)

2. INTERACT:
   Get list of co-located agents at current place
   For each co-located infectious agent:
     compute exposure += base_transmission_prob
                       × density_scaling(n_infectious / n_total_at_place)
                       × mask_factor(self.mask, other.mask)
                       × time_factor (1 hour)

3. TRANSITION (disease state machine):
   If SUSCEPTIBLE and exposure > threshold:
     draw random → transition to EXPOSED with probability p(exposure)
   If EXPOSED and days_in_state >= incubation_period_sample():
     draw random → transition to PRE-SYMPTOMATIC or ASYMPTOMATIC
   If PRE-SYMPTOMATIC and days_in_state >= pre_symptomatic_period:
     transition to SYMPTOMATIC-MILD
   If SYMPTOMATIC-MILD and days_in_state >= mild_period:
     draw random (age/comorbidity-weighted) → RECOVERED or SYMPTOMATIC-SEVERE
   [... etc. through full state machine]

4. BEHAVIOR UPDATE:
   If newly SYMPTOMATIC: set isolation_compliance += isolation_onset_boost
   If policy changed: update schedule filters
```

---

## 4. LLM Agent System Prompt

This section provides the agent prompt that replaces the deterministic ABM logic with LLM reasoning. The LLM governs **behavior and location choice**; a separate deterministic disease state engine governs **infection and state transitions** (see Section 5).

### 4.1 System Prompt Template

```
SYSTEM PROMPT — CityCOVID LLM Agent v1.0

You are simulating a single resident of Chicago during the COVID-19 pandemic.
You must reason and respond as this specific person would, given your personal 
profile and current health state. You are NOT an AI assistant — you ARE this person.
Do not break character. Do not mention being an AI.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR PERSONAL PROFILE (fixed for this simulation run)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name:               {{SYNTHETIC_NAME}}
Age:                {{AGE}}
Sex:                {{SEX}}
Race/ethnicity:     {{RACE_ETHNICITY}}
Neighborhood:       {{NEIGHBORHOOD}}, Chicago (zip {{ZIP_CODE}})

Household:
  - Household ID: {{HOUSEHOLD_ID}}
  - Members: {{HOUSEHOLD_MEMBERS_DESCRIPTION}}
    (e.g., "self + spouse (age 44) + 2 children (ages 8 and 11)")

Occupation:         {{OCCUPATION_TYPE}}
  (e.g., essential retail worker / remote-capable office worker / 
   school teacher / retired / student / healthcare worker)
Workplace:          {{WORKPLACE_DESCRIPTION}}
  (e.g., "grocery store, enclosed, ~25 coworkers, zip 60617")
Can work from home: {{CAN_WFH}}
Uses public transit: {{USES_TRANSIT}}

Underlying health conditions: {{COMORBIDITIES}}
  (e.g., "Type 2 diabetes, BMI 34" or "none")
Healthcare access:  {{HEALTHCARE_ACCESS}}
  (e.g., "insured, nearest clinic 1.2 miles" or "uninsured")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR CURRENT DISEASE STATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Disease state:      {{DISEASE_STATE}}
  Options: SUSCEPTIBLE | EXPOSED | PRE-SYMPTOMATIC | ASYMPTOMATIC |
           SYMPTOMATIC-MILD | SYMPTOMATIC-SEVERE | HOSPITALIZED | 
           RECOVERED | DECEASED

Days in state:      {{DAYS_IN_STATE}}
Symptom awareness:  {{SYMPTOM_AWARE}}
  (PRE-SYMPTOMATIC and ASYMPTOMATIC agents are NOT aware they are infectious.
   EXPOSED agents are NOT aware they are infected.
   Only SYMPTOMATIC states produce noticeable symptoms.)

If SYMPTOMATIC, your current symptoms are:
  {{SYMPTOM_DESCRIPTION}}
  (e.g., "mild fatigue and low-grade fever since yesterday" or
         "significant shortness of breath, fever 102°F")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT SIMULATION CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Simulation day:     Day {{SIM_DAY}} (calendar date: {{CALENDAR_DATE}})
Hour of day:        {{HOUR}}:00 ({{HOUR_LABEL}}, e.g., "Tuesday morning")
Day of week:        {{DAY_OF_WEEK}}

Active policy environment:
  {{POLICY_LIST}}
  Examples:
  - "Illinois stay-at-home order in effect"
  - "Schools closed (remote learning)"
  - "Restaurants/bars: closed for indoor dining, takeout only"
  - "Essential retail: open with capacity limits"
  - "Mask mandate: required in all indoor public spaces"
  - "No current restrictions"

Local epidemic context (what you would know from news):
  {{LOCAL_NEWS_SUMMARY}}
  (e.g., "Chicago hospitals are reporting significant strain. 
   Daily deaths are at peak. Your neighborhood has had several 
   known cases.")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR BASELINE ACTIVITY SCHEDULE (pre-pandemic typical day)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is what you would normally do on a {{DAY_OF_WEEK}}:
{{BASELINE_SCHEDULE}}
Example:
  00:00–06:00  HOME (sleep)
  07:00        HOME (get ready, make breakfast)
  08:00–16:00  WORKPLACE (grocery store)
  17:00        GROCERY RUN (different store on way home)
  18:00–21:00  HOME (dinner, TV, family time)
  22:00–23:00  HOME (sleep)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK THIS TIMESTEP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Given everything above, decide what you do this hour.

Answer the following questions as this person:

1. LOCATION — Where are you going (or staying) this hour?
   Choose the single best option:
   - HOME
   - WORKPLACE
   - SCHOOL (students / school-age children only)
   - ESSENTIAL_RETAIL (grocery, pharmacy, hardware)
   - HOSPITAL_OR_CLINIC
   - SOCIAL_VISIT (another household — friend, family)
   - RESTAURANT_BAR (only if currently permitted by policy)
   - PARK_OR_OUTDOOR
   - TRANSIT (commuting between locations)
   - OTHER: [describe]

2. REASONING — Why? 1–2 sentences from your perspective as this person.
   Consider: your job, your family obligations, your health state,
   your compliance personality, your financial pressures, 
   and the current policy environment.

3. MASK — Are you wearing a mask at this location?
   true / false / uncertain

4. DISTANCING — Are you actively maintaining 6-foot distance?
   true / false / uncertain

5. HEALTH_SEEKING — Are you seeking medical care or a COVID test this hour?
   true / false
   (Only relevant if symptomatic or in a state of uncertainty about health)

6. BEHAVIORAL_DEVIATION — Are you deviating from your baseline schedule?
   Describe briefly, or null if following baseline.

Respond ONLY with valid JSON, no commentary:

{
  "agent_id": "{{AGENT_ID}}",
  "sim_day": {{SIM_DAY}},
  "hour": {{HOUR}},
  "disease_state": "{{DISEASE_STATE}}",
  "location": "<LOCATION_TYPE>",
  "place_id": "<place_id or null>",
  "reasoning": "<1-2 sentences>",
  "mask_wearing": <true/false/null>,
  "distancing": <true/false/null>,
  "health_seeking": <true/false>,
  "behavioral_deviation": "<description or null>"
}
```

---

## 5. Architectural Design: LLM-Hybrid ABM

### 5.1 Separation of Concerns

The critical design principle is that **the LLM governs behavior; a deterministic engine governs disease physics**:

```
┌─────────────────────────────────────────────────────────┐
│                  SIMULATION ORCHESTRATOR                │
│  (Python / OpenClaw agent infrastructure)               │
└──────────────┬───────────────────────┬──────────────────┘
               │                       │
               ▼                       ▼
┌──────────────────────┐   ┌───────────────────────────┐
│   LLM BEHAVIOR       │   │  DETERMINISTIC DISEASE     │
│   ENGINE             │   │  STATE ENGINE              │
│                      │   │                            │
│  Input:              │   │  Input:                    │
│   - Agent profile    │   │   - Agent disease state    │
│   - Disease state    │   │   - Co-location exposure   │
│   - Policy context   │   │   - Age/comorbidity        │
│   - Schedule         │   │   - Calibrated params      │
│                      │   │                            │
│  Output:             │   │  Output:                   │
│   - Location choice  │   │   - State transitions      │
│   - Mask/distancing  │   │   - New disease state      │
│   - Health seeking   │   │   - Exposure dose          │
└──────────────────────┘   └───────────────────────────┘
               │                       │
               └───────────┬───────────┘
                           ▼
              ┌─────────────────────────┐
              │   CO-LOCATION NETWORK   │
              │  (who is where, when)   │
              │  → transmission events  │
              └─────────────────────────┘
```

### 5.2 Recommended Scale for Pilot

Running LLM inference for all 2.7M agents is computationally infeasible at present. Recommended approach for a research prototype:

| Configuration | LLM Agents | Deterministic Agents | Purpose |
|---|---|---|---|
| **Pilot A** | 100–500 | 0 | Behavioral validation; check if LLM decisions are epidemiologically plausible |
| **Pilot B** | 1,000 | ~10,000 | Mixed hybrid; study behavioral heterogeneity effects |
| **Pilot C** | ~5% (135K) | ~2.6M | Population-scale hybrid; research contribution |
| **Full (future)** | 2.7M | 0 | Aspirational; requires inference optimization |

### 5.3 Policy Intervention Encoding

The LLM excels at nuanced policy response. Encode policy interventions as structured context additions to the system prompt:

```python
POLICY_STATES = {
    "baseline": {
        "restaurants_open": True,
        "schools_open": True,
        "stay_at_home": False,
        "mask_mandate": False,
        "essential_retail_capacity": 1.0
    },
    "mild_npi": {
        "restaurants_open": True,  # reduced capacity
        "schools_open": True,
        "stay_at_home": False,
        "mask_mandate": False,
        "essential_retail_capacity": 0.5
    },
    "moderate_npi": {
        "restaurants_open": False,  # takeout only
        "schools_open": False,
        "stay_at_home": False,
        "mask_mandate": True,
        "essential_retail_capacity": 0.5
    },
    "strict_npi": {
        "restaurants_open": False,
        "schools_open": False,
        "stay_at_home": True,
        "mask_mandate": True,
        "essential_retail_capacity": 0.25
    }
}
```

### 5.4 Where LLM Agents Add Scientific Value Over Deterministic Agents

The deterministic ABM treats policy compliance as a scalar multiplier applied uniformly. LLM agents capture:

| Behavioral Phenomenon | ABM Treatment | LLM Agent Treatment |
|---|---|---|
| Stay-at-home compliance | p(comply) = 0.7 for all | Varies by age, occupation, income, household obligation, fear level |
| Symptomatic behavior | If symptomatic → reduce schedule by X% | "I feel sick but I can't miss work, we have rent due" |
| Information response | Uniform update on policy change day | Lagged, heterogeneous; some agents skeptical |
| Essential worker risk | Fixed transmission multiplier | Worker reasons about whether to call in; weighs risk vs. pay |
| Mask wearing | Binary compliance rate | Varies by location type, social context, personal risk perception |
| Healthcare seeking | p(seek) = f(severity, access) | Agent weighs symptoms, insurance status, fear of hospital |
| Social visit behavior | Zeroed during stay-at-home | "I'm checking on my elderly mother; I'll try to be careful" |

### 5.5 Calibration Strategy for LLM-Hybrid

The original CityCOVID used Approximate Bayesian Computation (ABC) via EMEWS/Swift-T on ALCF. For a hybrid model, a two-level calibration is needed:

**Level 1 — Behavioral calibration (LLM-specific):**
- Validate LLM agent location choices against mobility data (SafeGraph, Google Mobility Reports)
- Tune prompt framing to match observed compliance rates by demographic group
- Check that mask-wearing rates in simulation match survey data

**Level 2 — Epidemiological calibration (same as original CityCOVID):**
- Calibrate base transmission probability to match observed hospitalizations and deaths
- Use Bayesian/ABC methods; the LLM behavioral outputs feed into the co-location network which feeds into transmission
- Key calibration targets: daily hospital admissions, ICU census, death curves by zip code

### 5.6 Place Descriptor Format for LLM Context

Replace numeric place_id references with short natural language descriptors for LLM context:

```python
PLACE_DESCRIPTORS = {
    "household": "your home ({size}-person household, {type})",
    "grocery_store": "grocery store (enclosed, ~{workers} workers present, moderate ventilation)",
    "elementary_school": "elementary school (indoor, ~{students} students per classroom)",
    "office_wfh": "your home office (private, no exposure risk)",
    "hospital_er": "hospital emergency room (high risk, PPE available)",
    "restaurant_indoor": "restaurant (indoor dining, ~{tables} occupied tables)",
    "park": "outdoor park (low transmission risk, ventilated)",
    "nursing_home": "long-term care facility (high-risk population, restricted visitor access)",
    "transit_bus": "public bus (enclosed, ~{riders} riders)"
}
```

---

## 6. Data Sources for Synthetic Population

For anyone implementing this, the original CityCOVID synthetic population drew from:

| Data Source | Usage |
|---|---|
| RTI Synthesized Population Database (Cajka et al. 2010; Wheaton et al. 2009) | Base synthetic population; demographic match |
| American Time Use Survey (ATUS) | Hourly activity schedules by demographic profile |
| Panel Study of Income Dynamics (PSID) | Income, housing, employment attributes |
| Chicago land use / GIS records | Geolocated place inventory |
| Illinois vital statistics | Age-specific mortality rates |
| CDC COVID-NET / IL DPH hospitalization data | Calibration targets |
| Google / Apple Mobility Reports | Behavioral compliance validation |
| SafeGraph POI data | Points of interest, visit patterns |

---

## 7. Key Limitations and Open Questions

### 7.1 Limitations of LLM Agents for Epidemiological Simulation
1. **Stochastic inconsistency** — LLM responses are not precisely repeatable; ensemble runs needed
2. **Behavioral drift** — LLM may not maintain consistent character across 1000+ timesteps without careful state injection
3. **Calibration difficulty** — behavioral plausibility harder to verify than deterministic compliance rates
4. **Inference cost** — even 1,000 LLM agents × 24 hours × 90 simulation days = 2.16M inference calls per scenario
5. **No ground truth** — LLM behavioral decisions cannot be directly validated the way a deterministic rule can

### 7.2 Open Research Questions
1. What fraction of LLM agents is required to capture emergent behavioral heterogeneity not expressible in rule-based models?
2. Do LLM agents produce qualitatively different epidemic trajectories under policy shocks vs. deterministic agents?
3. Can LLM reasoning about compliance heterogeneity improve the fit of NPI impact predictions?
4. How sensitive are population-level outcomes to the specific LLM model used (GPT-4 vs. Llama vs. Qwen)?
5. Can LLM agents be distilled post-hoc into new deterministic rules that outperform the original ABM rules?

---

## 8. Suggested Next Steps

1. **Fetch the C++ source** from https://github.com/Repast/chiSIM to extract the exact field names and class hierarchy of the `Person` and `Place` C++ classes. This will make the prompt schema precisely match the original encoding.

2. **Download the 2018 WSC ChiSIM paper** (the simulation.su PDF) and the 2020 WSC CityCOVID paper (the informs-sim.org PDF) — both are open access.

3. **Request the OSTI software** at https://www.osti.gov/biblio/1822421 to access the full ChiSIM framework source.

4. **Run Pilot A** (100–500 LLM agents, no deterministic agents) as a behavioral validation study. Prompt agents with the system prompt above and spot-check their location choices against expected patterns.

5. **Connect to OpenClaw** — the LLM agent prompt above maps naturally to an OpenClaw agent with: system prompt (above) as its identity, a JSON-structured memory holding agent profile + current state, and a per-timestep task message. The disease state engine runs as a separate Python process feeding state updates into OpenClaw agent context.

---

*Document prepared April 2026. Based on published papers, open-source software, and Argonne/UChicago public communications. All citations are to publicly available sources.*
