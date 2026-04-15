"""
models.py — Agent and Place dataclasses.

These mirror the DB schema in SwarmSim-Architecture.md Section 3.
For the 1K pilot, the simulation holds these as in-memory objects.
For scale, each field maps 1:1 to a column in the agents/places tables.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ── Disease states ────────────────────────────────────────────────────────────

DISEASE_STATES = (
    "SUSCEPTIBLE",
    "EXPOSED",
    "PRE-SYMPTOMATIC",
    "ASYMPTOMATIC",
    "SYMPTOMATIC-MILD",
    "SYMPTOMATIC-SEVERE",
    "HOSPITALIZED",
    "ICU",
    "RECOVERED",
    "DECEASED",
)

INFECTIOUS_STATES = {"PRE-SYMPTOMATIC", "ASYMPTOMATIC", "SYMPTOMATIC-MILD",
                     "SYMPTOMATIC-SEVERE"}
AWARE_STATES      = {"SYMPTOMATIC-MILD", "SYMPTOMATIC-SEVERE",
                     "HOSPITALIZED", "ICU", "RECOVERED"}

# ── Occupation types ──────────────────────────────────────────────────────────

OCCUPATION_TYPES = (
    "essential_worker",
    "remote_capable",
    "student",
    "school_age",
    "retired",
    "unemployed",
    "healthcare_worker",
    "nursing_home_resident",
)

# ── Place types ───────────────────────────────────────────────────────────────

PLACE_TYPES = (
    "household",
    "workplace",
    "school",
    "essential_retail",
    "hospital",
    "restaurant",
    "park",
    "transit",
    "nursing_home",
    "community_venue",
)


@dataclass
class Place:
    """A geolocated venue. Passive data container; never runs code."""

    place_id:   int
    place_type: str          # one of PLACE_TYPES
    label:      str          # natural language descriptor for LLM injection
    zip_code:   str
    capacity:   int
    ventilation: str         # "outdoor" | "high" | "medium" | "low"
    lat:        Optional[float] = None
    lon:        Optional[float] = None

    # Rolling event log (last N notable events); serialized as JSON list of strings
    event_log: list[str] = field(default_factory=list)

    def is_open_under_policy(self, policy: dict) -> bool:
        """Return False if this place type is closed under the current policy."""
        if self.place_type == "school" and not policy.get("schools_open", True):
            return False
        if self.place_type == "restaurant" and not policy.get("restaurants_open", True):
            return False
        return True


@dataclass
class Agent:
    """
    A synthetic Chicago resident.

    Layer 1 (static profile) — set at init, never changed.
    Layer 2 (behavioral state vector) — updated deterministically each tick.
    Layer 3 (episodic memory) — updated by LLM compression every 7 sim-days.
    Disease state — updated by the disease engine each tick.
    """

    # ── Layer 1: Static profile ───────────────────────────────────────────────
    agent_id:           int
    synthetic_name:     str
    age:                int
    sex:                str               # "M" | "F"
    race_ethnicity:     str
    zip_code:           str
    neighborhood:       str

    household_id:       int
    household_size:     int

    occupation_type:    str               # one of OCCUPATION_TYPES
    workplace_id:       Optional[int]     # None for school_age, retired, etc.
    school_id:          Optional[int]     # for school_age and student
    can_wfh:            bool
    uses_transit:       bool

    comorbidities:      list[str]         # e.g. ["diabetes", "hypertension"]
    healthcare_access:  float             # 0.0–1.0
    age_risk_mult:      float             # pre-computed from age

    # Social network
    social_contacts:    list[int] = field(default_factory=list)   # agent_ids

    # Prompt text cached at init (Blocks A and B — the prefix-cached portion)
    system_text:        str = ""
    profile_text:       str = ""

    # ── Disease state ─────────────────────────────────────────────────────────
    disease_state:      str   = "SUSCEPTIBLE"
    days_in_state:      int   = 0
    state_duration:     int   = 9999      # sampled once on state entry; days
    exposure_count:     float = 0.0       # accumulated across current day; reset at midnight
    symptom_onset_day:  Optional[int] = None
    hosp_day:           Optional[int] = None

    # ── Layer 2: Behavioral state vector ──────────────────────────────────────
    fear_level:         float = 0.10      # 0–1; rises with illness events nearby
    compliance_fatigue: float = 0.00      # 0–1; rises with consecutive isolation days
    financial_pressure: float = 0.30      # 0–1; occupation-dependent baseline
    perceived_risk:     float = 0.10      # 0–1; tracks local case rate
    trust_in_news:      float = 0.60      # 0–1; erodes with policy inconsistency

    # ── Layer 3: Episodic memory ──────────────────────────────────────────────
    episodic_memory:         str = ""     # ≤75 tokens; compressed natural language
    last_compression_day:    int = 0

    # ── Bookkeeping ───────────────────────────────────────────────────────────
    current_place_id:   Optional[int] = None
    worker_shard:       int = 0

    # Per-tick tracking (not persisted between ticks)
    mask_wearing:       bool = False
    days_isolated:      int  = 0          # consecutive isolation days
    missed_work_days:   int  = 0          # cumulative

    @property
    def is_infectious(self) -> bool:
        return self.disease_state in INFECTIOUS_STATES

    @property
    def is_symptomatic_aware(self) -> bool:
        return self.disease_state in AWARE_STATES

    @property
    def is_active(self) -> bool:
        """False for DECEASED agents; they are skipped each tick."""
        return self.disease_state != "DECEASED"

    @property
    def needs_llm_decision(self) -> bool:
        """
        HOSPITALIZED and ICU agents have schedule overridden to hospital — no LLM needed.
        DECEASED agents are inactive.
        All others get an LLM decision each tick.
        """
        return self.disease_state not in ("HOSPITALIZED", "ICU", "DECEASED")
