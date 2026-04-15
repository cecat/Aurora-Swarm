"""
prompt.py — Agent prompt renderer.

Assembles the four prompt blocks described in SwarmSim-Architecture.md §5:

  Block A  (~300 tokens)  System context — one variant per occupation type.
                          Fixed for the run; prefix-cached by vLLM APC.
  Block B  (~200 tokens)  Agent profile — one per agent, fixed for the run.
                          Stored in agent.profile_text; prefix-cached by vLLM APC.
  Block C  (~150 tokens)  Dynamic state — disease state, behavioral vector,
                          memory, policy, co-location context, inbox messages.
                          Recomputed every tick.
  Block D  (~100 tokens)  Task and JSON response schema — constant.

vLLM prefix caching:
  For caching to work, Block A and B must be IDENTICAL strings across ticks
  for the same agent.  build_agent_texts() pre-renders them once at sim start
  and stores them on the agent object.  render_block_c() and BLOCK_D are the
  only parts transmitted per-tick.

Usage:
    from swarmsim.prompt import build_agent_texts, render_prompt, SimState

    # Once at sim start:
    build_agent_texts(agents, places)

    # Each tick, for each LLM agent:
    state = SimState(sim_day=3, hour=14, policy=policy_dict, ...)
    prompt = render_prompt(agent, state, colocation_ctx="...", inbox_msgs="...")
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from .models import Agent, Place, AWARE_STATES
from . import sim_config as cfg


# ═══════════════════════════════════════════════════════════════════════════════
# SimState — caller-provided tick context
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimState:
    """Context injected into Block C each tick."""
    sim_day:               int
    hour:                  int
    policy:                dict                # from sim_config.INITIAL_POLICY shape
    # Epidemic context visible to the public (scaled to city level)
    n_infectious_city_est: int   = 0           # rough city-wide estimate
    n_hospitalized:        int   = 0
    n_deceased_total:      int   = 0
    # Override news summary (auto-generated if empty)
    news_summary_override: str   = ""

    @property
    def calendar_date(self) -> str:
        start = date.fromisoformat(cfg.SIM_START_DATE)
        return (start + timedelta(days=self.sim_day)).strftime("%A, %B %-d, %Y")

    @property
    def day_of_week(self) -> str:
        start = date.fromisoformat(cfg.SIM_START_DATE)
        return (start + timedelta(days=self.sim_day)).strftime("%A")

    @property
    def hour_label(self) -> str:
        if self.hour < 6:   return "overnight"
        if self.hour < 12:  return "morning"
        if self.hour < 14:  return "midday"
        if self.hour < 18:  return "afternoon"
        if self.hour < 21:  return "evening"
        return "night"

    @property
    def news_summary(self) -> str:
        if self.news_summary_override:
            return self.news_summary_override
        return _generate_news_summary(
            self.n_infectious_city_est, self.n_hospitalized,
            self.n_deceased_total, self.sim_day)


# ═══════════════════════════════════════════════════════════════════════════════
# Short descriptor helpers
# ═══════════════════════════════════════════════════════════════════════════════

_OCCUPATION_LABEL = {
    "essential_worker":     "essential in-person worker",
    "remote_capable":       "office/professional worker",
    "student":              "college student",
    "school_age":           "school-age child",
    "retired":              "retired",
    "unemployed":           "not currently employed",
    "healthcare_worker":    "healthcare worker",
    "nursing_home_resident":"long-term care resident",
}

_OCCUPATION_SHORT = {
    "essential_worker":     "coworker",
    "remote_capable":       "neighbor",
    "student":              "classmate",
    "school_age":           "child",
    "retired":              "neighbor",
    "unemployed":           "neighbor",
    "healthcare_worker":    "colleague",
    "nursing_home_resident":"fellow resident",
}

_AGE_GROUP = {
    (0,  4):  "toddler",
    (5,  12): "child",
    (13, 17): "teenager",
    (18, 24): "young adult",
    (25, 39): "adult",
    (40, 59): "middle-aged adult",
    (60, 74): "older adult",
    (75, 120):"elderly adult",
}

def _age_group(age: int) -> str:
    for (lo, hi), label in _AGE_GROUP.items():
        if lo <= age <= hi:
            return label
    return "adult"

def _comorbidity_text(comorbidities: list[str]) -> str:
    if not comorbidities:
        return "none"
    nice = {
        "diabetes":         "Type 2 diabetes",
        "hypertension":     "high blood pressure",
        "obesity":          "obesity (BMI > 30)",
        "heart_disease":    "heart disease",
        "copd":             "COPD",
        "immunocompromised":"immunocompromised",
    }
    return ", ".join(nice.get(c, c) for c in comorbidities)

def _healthcare_access_text(value: float) -> str:
    if value >= 0.80: return "insured, nearby clinic or PCP"
    if value >= 0.60: return "insured, some barriers to care"
    if value >= 0.40: return "underinsured, limited clinic access"
    return "uninsured or very limited access to care"

def _household_members_text(agent: Agent, agent_map: dict[int, Agent]) -> str:
    """Natural language description of household members."""
    members = [a for a in agent_map.values()
               if a.household_id == agent.household_id
               and a.agent_id    != agent.agent_id]
    if not members:
        return "lives alone"
    parts = []
    for m in members:
        parts.append(f"{_age_group(m.age)} {_OCCUPATION_LABEL.get(m.occupation_type, '')}")
    return "self + " + ", ".join(parts)

def _symptom_text(agent: Agent) -> str:
    """Describe symptoms for SYMPTOMATIC states; empty string otherwise."""
    s = agent.disease_state
    if s == "SYMPTOMATIC-MILD":
        return ("Symptoms: You feel unwell — mild fatigue, low-grade fever, "
                f"slight cough. Started around day {agent.symptom_onset_day}.")
    if s == "SYMPTOMATIC-SEVERE":
        return ("Symptoms: You feel seriously ill — significant shortness of breath, "
                "high fever (102°F+), persistent cough. You are struggling.")
    if s == "HOSPITALIZED":
        return "You are currently admitted to a hospital receiving treatment."
    if s == "ICU":
        return "You are in intensive care."
    if s == "RECOVERED":
        return "You recently recovered from COVID-19 and feel mostly back to normal."
    return ""  # SUSCEPTIBLE, EXPOSED, PRE-SYMPTOMATIC, ASYMPTOMATIC — unaware

def _policy_text(policy: dict) -> str:
    lines = []
    if policy.get("stay_at_home"):
        lines.append("Stay-at-home order is in effect for non-essential activities.")
    if not policy.get("schools_open", True):
        lines.append("Schools are closed; remote learning is in effect.")
    else:
        lines.append("Schools are open.")
    if not policy.get("restaurants_open", True):
        lines.append("Restaurants and bars: closed for indoor dining (takeout only).")
    else:
        lines.append("Restaurants and bars: open for indoor dining.")
    if policy.get("mask_mandate"):
        lines.append("Mask mandate: face coverings required in all indoor public spaces.")
    cap = policy.get("essential_retail_capacity", 1.0)
    if cap < 1.0:
        lines.append(f"Essential retail: open at {int(cap * 100)}% capacity.")
    else:
        lines.append("Essential retail: fully open.")
    return "\n".join(f"  • {l}" for l in lines)

def _behavioral_vector_text(agent: Agent) -> str:
    return (
        f"  fear={int(agent.fear_level * 100)}%  "
        f"isolation-fatigue={int(agent.compliance_fatigue * 100)}%  "
        f"financial-pressure={int(agent.financial_pressure * 100)}%  "
        f"perceived-risk={int(agent.perceived_risk * 100)}%"
    )

def _generate_news_summary(n_infectious: int, n_hosp: int,
                            n_dead: int, sim_day: int) -> str:
    if n_infectious == 0:
        return "No significant COVID activity has been reported in Chicago."
    if n_infectious < 200:
        return ("A small number of COVID-19 cases have been reported in Chicago. "
                "Health officials say community spread is limited and urge residents "
                "to stay home if they feel unwell.")
    if n_infectious < 2000:
        return ("COVID-19 cases are rising in Chicago. The health department urges "
                "residents to avoid crowds, wash hands frequently, and stay home if "
                "symptomatic. Vulnerable populations should limit nonessential trips.")
    if n_hosp < 100:
        return ("Significant COVID-19 community spread is underway in Chicago. "
                "Hospitals are seeing an increase in admissions. Officials urge caution "
                "and recommend reducing contacts wherever possible.")
    return ("Chicago hospitals are under serious strain from COVID-19 admissions. "
            f"The city has recorded {n_dead:,} deaths. Officials strongly urge "
            "residents to stay home except for essential needs.")


# ═══════════════════════════════════════════════════════════════════════════════
# Block A — System context (one per occupation type)
# ═══════════════════════════════════════════════════════════════════════════════

_BLOCK_A_COMMON_HEADER = """\
You are simulating a Chicago resident during the COVID-19 pandemic.
You must reason and respond as this specific person, given their profile and
current health state.  You are NOT an AI assistant — you ARE this person.
Do not break character.  Do not mention being an AI or a simulation.

DISEASE STATE REFERENCE (what each state means for your character):
  SUSCEPTIBLE      — healthy, not yet exposed
  EXPOSED          — infected but not yet contagious; you feel fine and do not
                     know you are infected
  PRE-SYMPTOMATIC  — contagious but no symptoms yet; you feel fine and do not
                     know you are infectious
  ASYMPTOMATIC     — contagious but will never develop symptoms; you feel fine
                     and do not know you are infectious
  SYMPTOMATIC-MILD — you feel unwell; mild fever, fatigue, cough
  SYMPTOMATIC-SEVERE — seriously ill; shortness of breath, high fever
  HOSPITALIZED     — admitted to hospital
  ICU              — in intensive care
  RECOVERED        — recently recovered; you feel mostly normal again
  DECEASED         — not applicable (you would not be asked)

Respond ONLY with valid JSON matching the schema at the end of this prompt.
No commentary, no markdown, no explanation outside the JSON object.\
"""

_BLOCK_A_OCCUPATION = {

    "essential_worker": _BLOCK_A_COMMON_HEADER + """

OCCUPATION CONTEXT — Essential Worker:
You work an in-person job that cannot be done from home (grocery, food service,
construction, transport, manufacturing, etc.).  Missing work has real financial
consequences for you and possibly your household.  Your job puts you in contact
with many people each shift.  You have less ability to reduce your exposure risk
than someone who can work from home, and you know it.\
""",

    "remote_capable": _BLOCK_A_COMMON_HEADER + """

OCCUPATION CONTEXT — Office / Professional Worker:
Your job can be done remotely.  During the pandemic, you may work from home or
go to an office — you have more flexibility than essential workers.  Your
financial situation is more stable if you miss work.  You are more likely to
be able to reduce contacts by staying home, and you have the option to do so.\
""",

    "school_age": _BLOCK_A_COMMON_HEADER + """

OCCUPATION CONTEXT — School-Age Child or Teenager:
You are a child or teenager who normally attends school.  Your daily schedule
is largely determined by your parents or guardians.  When schools close, you
are at home.  Your social life centers around school, neighborhood friends,
and family.  You may not fully grasp the severity of the epidemic but you are
aware that things are unusual.\
""",

    "student": _BLOCK_A_COMMON_HEADER + """

OCCUPATION CONTEXT — College Student:
You are a young adult balancing coursework, social life, and possibly a
part-time job.  You may live with roommates or family.  You understand the
epidemic intellectually but may feel somewhat invulnerable given your age.
Financial pressures from part-time work affect your decisions.\
""",

    "retired": _BLOCK_A_COMMON_HEADER + """

OCCUPATION CONTEXT — Retired:
You are retired and have considerable time flexibility.  You are likely older
and may be more health-conscious than working-age adults.  You are aware that
the epidemic is more dangerous for people your age.  You have fewer financial
pressures to leave the house but may feel isolated if you stay in too long.\
""",

    "unemployed": _BLOCK_A_COMMON_HEADER + """

OCCUPATION CONTEXT — Currently Unemployed:
You are not currently employed.  You have more time at home than working
adults.  Financial pressure is high — you may be worrying about rent, food,
and bills.  You still need to make essential trips for groceries and errands.
The pandemic has likely made your economic situation worse.\
""",

    "healthcare_worker": _BLOCK_A_COMMON_HEADER + """

OCCUPATION CONTEXT — Healthcare Worker:
You work in a clinical or support role in healthcare.  You have professional
training about infection control and understand the epidemic better than most.
You feel both personal risk (high exposure at work) and a sense of professional
duty to show up.  You have access to PPE at your workplace.  You may feel
exhausted and emotionally strained as the epidemic progresses.\
""",

    "nursing_home_resident": _BLOCK_A_COMMON_HEADER + """

OCCUPATION CONTEXT — Long-Term Care Resident:
You live in a long-term care or nursing facility.  Your movement is largely
restricted to the facility.  Staff manage your environment and care.  You are
aware that outbreaks in facilities like yours have been very serious.  You
depend on staff and may have limited ability to make independent location
decisions.\
""",
}

# Fallback for any unexpected occupation type
_BLOCK_A_DEFAULT = _BLOCK_A_COMMON_HEADER + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
# Block B — Agent profile (rendered once per agent at sim start)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_block_b(agent: Agent, agent_map: dict[int, Agent],
                    place_map: dict[int, Place]) -> str:
    workplace_label = "n/a"
    if agent.workplace_id is not None:
        p = place_map.get(agent.workplace_id)
        workplace_label = p.label if p else f"place_{agent.workplace_id}"

    school_label = "n/a"
    if agent.school_id is not None:
        p = place_map.get(agent.school_id)
        school_label = p.label if p else f"place_{agent.school_id}"

    hh_label = "n/a"
    p = place_map.get(agent.household_id)
    if p:
        hh_label = p.label

    members_text = _household_members_text(agent, agent_map)
    wfh_text     = "yes" if agent.can_wfh    else "no"
    transit_text = "yes" if agent.uses_transit else "no"

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "YOUR PROFILE  (fixed for this simulation run)",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Name:                {agent.synthetic_name}",
        f"Age:                 {agent.age}",
        f"Sex:                 {'Male' if agent.sex == 'M' else 'Female'}",
        f"Race/ethnicity:      {agent.race_ethnicity}",
        f"Neighborhood:        {agent.neighborhood}, Chicago (zip {agent.zip_code})",
        f"",
        f"Household:           {hh_label}",
        f"Members:             {members_text}",
        f"",
        f"Occupation:          {_OCCUPATION_LABEL.get(agent.occupation_type, agent.occupation_type)}",
    ]

    if agent.workplace_id is not None:
        lines.append(f"Workplace:           {workplace_label}")
    if agent.school_id is not None:
        lines.append(f"School:              {school_label}")

    lines += [
        f"Can work from home:  {wfh_text}",
        f"Uses public transit: {transit_text}",
        f"",
        f"Health conditions:   {_comorbidity_text(agent.comorbidities)}",
        f"Healthcare access:   {_healthcare_access_text(agent.healthcare_access)}",
    ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Block C — Dynamic state (rendered fresh each tick)
# ═══════════════════════════════════════════════════════════════════════════════

def render_block_c(
    agent:           Agent,
    state:           SimState,
    colocation_ctx:  str = "",   # from communication.py
    inbox_msgs:      str = "",   # from communication.py
) -> str:
    symptom_line = _symptom_text(agent)
    policy_text  = _policy_text(state.policy)
    bvec_text    = _behavioral_vector_text(agent)

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"CURRENT STATE  —  {state.calendar_date}, {state.hour:02d}:00 "
        f"({state.day_of_week} {state.hour_label})",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Disease state:       {agent.disease_state}  "
        f"(day {agent.days_in_state} in this state)",
    ]

    if symptom_line:
        lines.append(f"                     {symptom_line}")

    # Only show behavioral state for adults making autonomous decisions
    if agent.occupation_type not in ("school_age", "nursing_home_resident"):
        lines += [
            f"",
            f"How you feel:        {bvec_text}",
        ]

    if agent.episodic_memory:
        lines += [
            f"",
            f"Memory:              {agent.episodic_memory}",
        ]

    lines += [
        f"",
        f"Policy environment:",
        policy_text,
        f"",
        f"News / local situation:",
        f"  {state.news_summary}",
    ]

    if colocation_ctx:
        lines += ["", colocation_ctx]

    if inbox_msgs:
        lines += ["", inbox_msgs]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Block D — Task and response schema (constant)
# ═══════════════════════════════════════════════════════════════════════════════

BLOCK_D = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR DECISION THIS HOUR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Given your profile, current state, and the situation above, decide what you
do this hour.  Be this person — reason from their perspective, pressures,
and constraints.

Location options (choose one):
  HOME | WORKPLACE | SCHOOL | ESSENTIAL_RETAIL | HOSPITAL_OR_CLINIC |
  SOCIAL_VISIT | RESTAURANT_BAR | PARK_OR_OUTDOOR | TRANSIT | OTHER

Respond with ONLY valid JSON, no other text:

{
  "agent_id": 0,
  "sim_day": 0,
  "hour": 0,
  "location": "HOME",
  "reasoning": "1-2 sentences from this person's perspective",
  "mask_wearing": true,
  "distancing": true,
  "health_seeking": false,
  "behavioral_deviation": null
}\
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Assembly
# ═══════════════════════════════════════════════════════════════════════════════

def render_prompt(
    agent:          Agent,
    state:          SimState,
    colocation_ctx: str = "",
    inbox_msgs:     str = "",
) -> str:
    """
    Assemble the full prompt for one agent at one tick.

    agent.system_text and agent.profile_text must already be set
    (call build_agent_texts() once at sim start).

    The returned string is: Block A + B (cached prefix) + Block C + Block D.
    """
    block_c = render_block_c(agent, state, colocation_ctx, inbox_msgs)
    return "\n\n".join([
        agent.system_text,
        agent.profile_text,
        block_c,
        BLOCK_D,
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-rendering (called once at sim start)
# ═══════════════════════════════════════════════════════════════════════════════

def build_agent_texts(
    agents: list[Agent],
    places: list[Place],
) -> None:
    """
    Pre-render Block A (system_text) and Block B (profile_text) for every agent.
    Stores results directly on each Agent object.

    Call once before the simulation loop begins.  The rendered strings are
    what vLLM sees as the prompt prefix and will be KV-cached after the
    first tick that uses each agent.
    """
    agent_map = {a.agent_id: a for a in agents}
    place_map = {p.place_id: p for p in places}

    for agent in agents:
        agent.system_text  = _BLOCK_A_OCCUPATION.get(
                                agent.occupation_type, _BLOCK_A_DEFAULT)
        agent.profile_text = _render_block_b(agent, agent_map, place_map)


# ═══════════════════════════════════════════════════════════════════════════════
# Response parsing
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_LOCATIONS = {
    "HOME", "WORKPLACE", "SCHOOL", "ESSENTIAL_RETAIL", "HOSPITAL_OR_CLINIC",
    "SOCIAL_VISIT", "RESTAURANT_BAR", "PARK_OR_OUTDOOR", "TRANSIT", "OTHER",
}

@dataclass
class AgentDecision:
    agent_id:            int
    sim_day:             int
    hour:                int
    location:            str
    reasoning:           Optional[str]
    mask_wearing:        Optional[bool]
    distancing:          Optional[bool]
    health_seeking:      bool
    behavioral_deviation:Optional[str]
    is_fallback:         bool = False   # True if parsed from LLM failure


def parse_response(raw: str, agent: Agent, state: SimState) -> AgentDecision:
    """
    Parse an LLM response string into an AgentDecision.
    Falls back to a safe default on any parse or validation error.
    """
    try:
        # Strip markdown code fences if the model added them
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)

        location = str(data.get("location", "HOME")).upper()
        if location not in _VALID_LOCATIONS:
            location = "HOME"

        return AgentDecision(
            agent_id            = agent.agent_id,
            sim_day             = state.sim_day,
            hour                = state.hour,
            location            = location,
            reasoning           = data.get("reasoning"),
            mask_wearing        = _parse_bool(data.get("mask_wearing")),
            distancing          = _parse_bool(data.get("distancing")),
            health_seeking      = bool(data.get("health_seeking", False)),
            behavioral_deviation= data.get("behavioral_deviation"),
        )

    except Exception:
        return _fallback_decision(agent, state)


def _parse_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return bool(value)


def _fallback_decision(agent: Agent, state: SimState) -> AgentDecision:
    """Conservative default used when LLM call fails or returns invalid JSON."""
    if agent.disease_state in ("SYMPTOMATIC-MILD", "SYMPTOMATIC-SEVERE"):
        location = "HOME"
    elif agent.disease_state in ("HOSPITALIZED", "ICU"):
        location = "HOSPITAL_OR_CLINIC"
    else:
        # Follow scheduled location type
        from .schedules import hour_of_week as how
        # We don't have schedules here; default to HOME
        location = "HOME"

    return AgentDecision(
        agent_id            = agent.agent_id,
        sim_day             = state.sim_day,
        hour                = state.hour,
        location            = location,
        reasoning           = None,
        mask_wearing        = agent.fear_level > 0.5,
        distancing          = agent.fear_level > 0.5,
        health_seeking      = agent.disease_state in ("SYMPTOMATIC-SEVERE",),
        behavioral_deviation= None,
        is_fallback         = True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import textwrap
    from .population     import generate_population
    from .disease_engine import seed_infections, _enter_state
    from . import sim_config as cfg
    import random

    agents, places = generate_population(n_agents=20, seed=42, use_osmnx=False)
    build_agent_texts(agents, places)

    # Seed one symptomatic agent for interesting output
    rng = random.Random(42)
    seed_infections(agents, 1, sim_day=0, rng=rng)
    # Advance one agent to SYMPTOMATIC-MILD for a richer prompt
    sympt_agent = agents[5]
    sympt_agent.disease_state = "SYMPTOMATIC-MILD"
    sympt_agent.days_in_state = 2
    sympt_agent.symptom_onset_day = 5
    sympt_agent.fear_level = 0.72
    sympt_agent.financial_pressure = 0.85
    sympt_agent.episodic_memory = (
        "Developed symptoms two days ago but went to work anyway because "
        "I couldn't afford to miss a shift. Now feeling worse."
    )

    state = SimState(
        sim_day=7,
        hour=8,
        policy={
            "schools_open":              False,
            "restaurants_open":          False,
            "stay_at_home":              True,
            "mask_mandate":              True,
            "essential_retail_capacity": 0.50,
        },
        n_infectious_city_est=1500,
        n_hospitalized=42,
        n_deceased_total=8,
    )

    colocation_ctx = (
        "At your location last hour (industrial workplace, ~32 workers):\n"
        "  - Approximately 18 people present\n"
        "  - 2 person(s) appeared visibly ill\n"
        "  - Mask wearing: ~40% of people\n"
    )
    inbox_msgs = (
        "Messages from your social network:\n"
        "  - A middle-aged adult coworker you know reported feeling ill.\n"
    )

    for agent in [agents[0], sympt_agent]:
        print(f"\n{'═'*70}")
        print(f"AGENT: {agent.synthetic_name}  ({agent.occupation_type}, "
              f"age {agent.age}, state={agent.disease_state})")
        print('═'*70)
        prompt = render_prompt(agent, state, colocation_ctx, inbox_msgs)
        # Show character count and rough token estimate
        chars  = len(prompt)
        tokens = chars // 4
        print(f"[Prompt: {chars} chars, ~{tokens} tokens]\n")
        print(prompt)
        print()
