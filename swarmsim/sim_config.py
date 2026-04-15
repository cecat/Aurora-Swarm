"""
sim_config.py — Simulation constants and configuration.

All tuneable parameters in one place. Override via environment variables
or by importing and mutating before calling generate_population() or run().
"""

# ── Target neighborhood ───────────────────────────────────────────────────────
NEIGHBORHOOD        = "Logan Square"        # Chicago community area
NEIGHBORHOOD_ZIP    = "60647"               # primary zip (area spans 60618 too)
SIM_START_DATE      = "2020-03-15"          # calendar anchor (day 0)

# ── Scale ─────────────────────────────────────────────────────────────────────
N_AGENTS            = 1_000
N_WORKERS           = 1                     # orchestrator shards (1 = pilot mode)
RANDOM_SEED         = 42

# ── Simulation time ───────────────────────────────────────────────────────────
N_SIM_DAYS          = 90
HOURS_PER_TICK      = 1                     # 1-hour tick (standard ABM timestep)

# ── LLM / Aurora Swarm ────────────────────────────────────────────────────────
MODEL               = "openai/gpt-oss-120b"
MAX_TOKENS_RESPONSE = 200                   # agent decision JSON ≈ 150 tokens
MAX_TOKENS_COMPRESS = 1024                  # episodic memory compression calls
USE_BATCH           = True

# ── Disease engine ────────────────────────────────────────────────────────────
# Calibrated from Ozik et al. 2021 (Table 1) / CityCOVID
BASE_TRANSMISSION_PROB      = 0.06          # per infectious-susceptible co-location/hour
ASYMPTOMATIC_FRACTION       = 0.35          # fraction exposed → asymptomatic
ASYMPTOMATIC_INFECTIVITY    = 0.50          # relative to symptomatic
DENSITY_SCALE_K             = 0.75
INITIAL_INFECTED            = 5            # seed infections at sim start

# Period distributions (in days; sampled per agent at transition)
INCUBATION_MEAN_DAYS        = 5.1
INCUBATION_SD_DAYS          = 3.0
PRE_SYMPTOMATIC_DAYS        = 2
ASYMPTOMATIC_DURATION_DAYS  = 7
MILD_DURATION_DAYS          = 4
SEVERE_DURATION_DAYS        = 3
HOSP_DURATION_DAYS          = 8
ICU_DURATION_DAYS           = 10

# Age-stratified severity multipliers (relative to baseline 30-yr-old = 1.0)
AGE_RISK_MULT = {
    (0,  17):  0.10,
    (18, 29):  0.20,
    (30, 39):  0.35,
    (40, 49):  0.55,
    (50, 59):  0.90,
    (60, 69):  1.60,
    (70, 79):  2.80,
    (80, 120): 4.50,
}

# Transition probabilities (conditioned on reaching each state)
PROB_SEVERE_GIVEN_MILD      = 0.15          # × age_risk_mult
PROB_HOSP_GIVEN_SEVERE      = 0.40          # × age_risk_mult
PROB_ICU_GIVEN_HOSP         = 0.25
PROB_DEATH_GIVEN_ICU        = 0.35

# Ventilation transmission factors
VENTILATION_FACTOR = {
    "outdoor":  0.10,
    "high":     0.50,
    "medium":   1.00,
    "low":      1.50,
}

# Mask reduction factors
MASK_WEARER_SOURCE_FACTOR   = 0.50          # infectious agent wearing mask
MASK_WEARER_DEST_FACTOR     = 0.50          # susceptible agent wearing mask

# ── Behavioral state vector ───────────────────────────────────────────────────
FEAR_DECAY_PER_TICK                     = 0.005
FEAR_RISE_HOUSEHOLD_SYMPTOMATIC         = 0.15
FEAR_RISE_SOCIAL_CONTACT_SICK           = 0.08
FEAR_RISE_SOCIAL_CONTACT_DIED           = 0.20
COMPLIANCE_FATIGUE_RISE_PER_ISO_DAY     = 0.03
COMPLIANCE_FATIGUE_DECAY_PER_ACTIVE_DAY = 0.05
FINANCIAL_PRESSURE_RISE_PER_MISSED_DAY  = 0.04
TRUST_DECAY_ON_POLICY_CHANGE            = 0.05
TRUST_SLOW_DECAY_PER_TICK               = 0.001

# Financial pressure baselines by occupation
FINANCIAL_PRESSURE_BASELINE = {
    "essential_worker":     0.70,
    "remote_capable":       0.20,
    "unemployed":           0.90,
    "retired":              0.10,
    "student":              0.25,
    "school_age":           0.00,
    "healthcare_worker":    0.50,
    "nursing_home_resident":0.30,
}

# ── Agent memory ─────────────────────────────────────────────────────────────
EPISODIC_COMPRESSION_INTERVAL_DAYS = 7
MAX_INBOX_MESSAGES                 = 5
MAX_PLACE_LOG_ENTRIES              = 3

# ── Social network ────────────────────────────────────────────────────────────
SOCIAL_CONTACTS_MIN     = 4
SOCIAL_CONTACTS_MAX     = 10
SOCIAL_CONTACTS_TARGET  = 7                 # mean contacts per agent

# ── Initial policy state (pre-lockdown baseline) ─────────────────────────────
INITIAL_POLICY = {
    "schools_open":               True,
    "restaurants_open":           True,
    "stay_at_home":               False,
    "mask_mandate":               False,
    "essential_retail_capacity":  1.0,
}
