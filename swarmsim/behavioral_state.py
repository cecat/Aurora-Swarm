"""
behavioral_state.py — Deterministic behavioral state vector updates.

Implements SwarmSim-Architecture.md §5.  All five scalars are updated
by the orchestrator at tick boundaries — no LLM calls required.

State vector fields (all float, 0.0–1.0):
    fear_level          — rises with nearby illness events; slow decay
    compliance_fatigue  — rises each day agent isolates; resets on non-isolation
    financial_pressure  — occupation-dependent baseline; rises with missed work
    perceived_risk      — tracks local case prevalence estimate
    trust_in_news       — erodes with policy inconsistency; very slow recovery

Update schedule:
    update_hourly()      — call each tick; updates perceived_risk
    update_midnight()    — call once per sim-day (hour==0); updates all others

Design notes:
    - All formulas are first-order exponential approach or linear accumulation,
      so the scalar can never exceed 1.0 or drop below 0.0.
    - Financial pressure is the only scalar with an occupation-specific baseline.
      It rises above that baseline with missed work days and falls back toward it
      when the agent works.
    - Fear and perceived_risk are driven by observable events in the simulation
      (case counts, social network messages, nearby illness) — not LLM output.
    - Compliance fatigue is a simple counter-based mechanism: isolation days
      accumulate fatigue; any non-isolation day resets the counter (not the
      scalar — scalar decays gradually).
    - trust_in_news is updated externally (call nudge_trust()) when a policy
      reversal is detected by the coordinator.

Usage:
    from swarmsim.behavioral_state import update_hourly, update_midnight, nudge_fear, nudge_trust

    # Each tick:
    update_hourly(agent, n_infectious_city_est, n_agents_city)

    # At midnight (hour == 0):
    update_midnight(agent, worked_today, isolated_today, sim_day)

    # When a social contact transitions to a notable state:
    nudge_fear(agent, event="contact_hospitalized")

    # When coordinator detects a policy reversal:
    nudge_trust(agent, delta=-0.05)
"""

from __future__ import annotations

from .models import Agent
from . import sim_config as cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Fear decay: each day without an illness event, fear decays toward the agent's
# baseline (10%).  Half-life ≈ 5 days.
_FEAR_DECAY_PER_DAY       = 0.13        # multiplicative: new = old * (1 - decay)
_FEAR_BASELINE            = 0.10        # floor that decay approaches

# Fear nudges by event type
_FEAR_NUDGE = {
    "contact_symptomatic":    0.08,
    "contact_hospitalized":   0.15,
    "contact_deceased":       0.25,
    "colocation_ill_1":       0.04,     # 1 visibly ill person at same place
    "colocation_ill_3plus":   0.10,     # 3+ visibly ill at same place
    "self_symptomatic":       0.30,     # agent themselves becomes symptomatic
}

# Compliance fatigue: rises with consecutive isolated days
_FATIGUE_PER_ISOLATION_DAY = 0.04      # additive per isolated day
_FATIGUE_DECAY_PER_DAY     = 0.08      # multiplicative decay per non-isolation day

# Financial pressure: rises with missed work days; decays when working
_FP_RISE_PER_MISSED_DAY   = 0.05       # additive per missed work day
_FP_DECAY_PER_WORK_DAY    = 0.03       # additive decay toward baseline per work day

# Perceived risk: exponential approach toward city prevalence estimate
_RISK_SMOOTHING_ALPHA     = 0.15       # blending weight toward new observation

# Trust in news: very slow decay with no policy inconsistencies;
# nudged externally on policy reversals
_TRUST_PASSIVE_DECAY      = 0.002      # per day, always-on erosion
_TRUST_FLOOR              = 0.20       # minimum trust level


# ═══════════════════════════════════════════════════════════════════════════════
# Per-tick update (every simulated hour)
# ═══════════════════════════════════════════════════════════════════════════════

def update_hourly(
    agent:                 Agent,
    n_infectious_city_est: int,
    n_agents_city:         int,
) -> None:
    """
    Update perceived_risk each hour toward the current city prevalence.

    n_infectious_city_est: estimated number of infectious people in the city
                           (simulation sample × scale-up factor)
    n_agents_city:         total simulated population (used as denominator)
    """
    if n_agents_city <= 0:
        return

    prevalence = min(1.0, n_infectious_city_est / max(n_agents_city, 1))

    # Exponential smoothing: blend old perceived_risk toward true prevalence
    agent.perceived_risk = _clamp(
        agent.perceived_risk + _RISK_SMOOTHING_ALPHA * (prevalence - agent.perceived_risk)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Midnight update (once per sim-day)
# ═══════════════════════════════════════════════════════════════════════════════

def update_midnight(
    agent:          Agent,
    worked_today:   bool,
    isolated_today: bool,
    sim_day:        int,
) -> None:
    """
    End-of-day behavioral state update.  Call once per agent after the
    midnight disease tick, before building next-tick prompts.

    worked_today:   True if agent was at their workplace/school during work hours
    isolated_today: True if agent stayed home all day (policy, fear, or LLM decision)
    sim_day:        current simulation day (used for logging/diagnostics if needed)
    """
    _update_fear(agent)
    _update_compliance_fatigue(agent, isolated_today)
    _update_financial_pressure(agent, worked_today)
    _update_trust(agent)


def _update_fear(agent: Agent) -> None:
    """Decay fear toward baseline.  Nudges are applied separately."""
    if agent.fear_level > _FEAR_BASELINE:
        gap   = agent.fear_level - _FEAR_BASELINE
        decay = gap * _FEAR_DECAY_PER_DAY
        agent.fear_level = _clamp(agent.fear_level - decay)


def _update_compliance_fatigue(agent: Agent, isolated_today: bool) -> None:
    if isolated_today:
        agent.days_isolated += 1
        agent.compliance_fatigue = _clamp(
            agent.compliance_fatigue + _FATIGUE_PER_ISOLATION_DAY
        )
    else:
        agent.days_isolated = 0
        # Decay — fatigue doesn't reset instantly
        agent.compliance_fatigue = _clamp(
            agent.compliance_fatigue * (1.0 - _FATIGUE_DECAY_PER_DAY)
        )


def _update_financial_pressure(agent: Agent, worked_today: bool) -> None:
    baseline = cfg.FINANCIAL_PRESSURE_BASELINE.get(agent.occupation_type, 0.30)

    if not worked_today and agent.disease_state not in ("HOSPITALIZED", "ICU", "DECEASED"):
        # Missing work (by choice or policy) increases pressure
        agent.financial_pressure = _clamp(
            agent.financial_pressure + _FP_RISE_PER_MISSED_DAY
        )
        agent.missed_work_days += 1
    else:
        # Working: pressure decays toward occupation baseline
        target = baseline
        if agent.financial_pressure > target:
            agent.financial_pressure = _clamp(
                agent.financial_pressure - _FP_DECAY_PER_WORK_DAY
            )
        # Don't push below baseline (financial security has a floor for each type)


def _update_trust(agent: Agent) -> None:
    """Passive daily trust erosion.  Policy reversals applied via nudge_trust()."""
    agent.trust_in_news = _clamp(
        max(_TRUST_FLOOR, agent.trust_in_news - _TRUST_PASSIVE_DECAY)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Event-driven nudges (called by coordinator on specific observations)
# ═══════════════════════════════════════════════════════════════════════════════

def nudge_fear(agent: Agent, event: str) -> None:
    """
    Apply a fear boost for a specific social or environmental event.

    Typical events and callers:
        "contact_symptomatic"  — write_state_change_to_inboxes detected notable state
        "contact_hospitalized" — same
        "contact_deceased"     — same
        "colocation_ill_1"     — build_colocation_context saw 1 visibly ill person
        "colocation_ill_3plus" — build_colocation_context saw 3+ visibly ill people
        "self_symptomatic"     — disease engine transitioned agent to SYMPTOMATIC
    """
    delta = _FEAR_NUDGE.get(event, 0.0)
    if delta:
        agent.fear_level = _clamp(agent.fear_level + delta)


def nudge_trust(agent: Agent, delta: float) -> None:
    """
    Adjust trust_in_news by delta (negative = erosion, positive = recovery).
    Enforces [TRUST_FLOOR, 1.0] bounds.

    Call with delta=-0.05 on policy reversal (e.g., schools close then reopen).
    Call with delta=+0.01 on consistent policy week (coordinator optional).
    """
    agent.trust_in_news = _clamp(
        max(_TRUST_FLOOR, agent.trust_in_news + delta)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Batch helpers (coordinator calls these across all agents each tick)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_fear_nudges_from_state_changes(
    state_changes: list,            # list[StateChange] from disease_engine
    agent_map:     dict[int, Agent],
) -> None:
    """
    For each notable state change, nudge the agent's own fear (not their contacts —
    contacts receive inbox messages via communication.py; this updates the agent who
    changed state when they become symptomatic).
    """
    for change in state_changes:
        agent = agent_map.get(change.agent_id)
        if agent is None:
            continue
        if change.new_state in ("SYMPTOMATIC-MILD", "SYMPTOMATIC-SEVERE"):
            nudge_fear(agent, "self_symptomatic")


def apply_fear_nudges_from_colocation(
    agent:           Agent,
    n_visibly_ill:   int,
) -> None:
    """
    Nudge an agent's fear based on what they observed at their location last tick.
    Call after build_colocation_context() has been evaluated.
    """
    if n_visibly_ill == 0:
        return
    if n_visibly_ill >= 3:
        nudge_fear(agent, "colocation_ill_3plus")
    else:
        nudge_fear(agent, "colocation_ill_1")


def apply_contact_fear_nudges(
    state_changes: list,            # list[StateChange]
    agent_map:     dict[int, Agent],
) -> None:
    """
    For each notable state change, apply a fear nudge to every social contact
    of the agent who changed state.

    communication.py handles inbox messages (the narrative text).
    This function handles the numeric fear update that happens in parallel.
    """
    _STATE_TO_FEAR_EVENT = {
        "SYMPTOMATIC-MILD":   "contact_symptomatic",
        "SYMPTOMATIC-SEVERE": "contact_symptomatic",
        "HOSPITALIZED":       "contact_hospitalized",
        "DECEASED":           "contact_deceased",
    }

    for change in state_changes:
        event = _STATE_TO_FEAR_EVENT.get(change.new_state)
        if event is None:
            continue

        agent = agent_map.get(change.agent_id)
        if agent is None:
            continue

        for contact_id in agent.social_contacts:
            contact = agent_map.get(contact_id)
            if contact is not None:
                nudge_fear(contact, event)


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def behavioral_state_summary(agents: list[Agent]) -> dict[str, float]:
    """Return mean behavioral state scalars across active agents.  For monitoring."""
    active = [a for a in agents if a.is_active]
    if not active:
        return {}
    n = len(active)
    return {
        "mean_fear":              sum(a.fear_level          for a in active) / n,
        "mean_fatigue":           sum(a.compliance_fatigue  for a in active) / n,
        "mean_financial_pressure":sum(a.financial_pressure  for a in active) / n,
        "mean_perceived_risk":    sum(a.perceived_risk       for a in active) / n,
        "mean_trust":             sum(a.trust_in_news        for a in active) / n,
        "n_active":               float(n),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    from .population     import generate_population
    from .schedules      import build_schedules, get_place_at, hour_of_week
    from .disease_engine import (compute_occupancy, run_exposure_tick,
                                  run_midnight_tick, seed_infections)
    from . import sim_config as cfg

    N      = 300
    N_DAYS = 30
    rng    = random.Random(cfg.RANDOM_SEED)

    agents, places = generate_population(n_agents=N, seed=cfg.RANDOM_SEED,
                                         use_osmnx=False)
    schedules = build_schedules(agents, places, seed=cfg.RANDOM_SEED)
    seed_infections(agents, cfg.INITIAL_INFECTED, sim_day=0, rng=rng)

    agent_map = {a.agent_id: a for a in agents}

    print(f"[behavioral_state] Running {N_DAYS}-day test ({N} agents) ...\n")
    print(f"{'Day':>4}  {'Fear':>6}  {'Fatigue':>7}  {'Fin.Press':>9}  "
          f"{'Perc.Risk':>9}  {'Trust':>6}  {'Infectious':>10}")
    print("─" * 65)

    for tick in range(N_DAYS * 24):
        sim_day = tick // 24
        hour    = tick % 24

        # Move agents
        how = hour_of_week(sim_day, hour)
        for agent in agents:
            if agent.is_active:
                if agent.disease_state in ("HOSPITALIZED", "ICU"):
                    hosp = next((p for p in places if p.place_type == "hospital"), None)
                    agent.current_place_id = hosp.place_id if hosp else agent.household_id
                else:
                    agent.current_place_id = get_place_at(schedules, agent.agent_id, how)

        # Exposure
        occupancy = compute_occupancy(agents)
        run_exposure_tick(agents, places, occupancy)

        # Hourly perceived_risk update
        n_infectious = sum(1 for a in agents if a.is_infectious)
        for agent in agents:
            if agent.is_active:
                update_hourly(agent, n_infectious * 2700, N * 2700)

        if hour == 0:
            midnight = run_midnight_tick(agents, sim_day, rng)

            # Self-fear nudges for newly symptomatic
            apply_fear_nudges_from_state_changes(midnight.state_changes, agent_map)

            # Contact fear nudges
            apply_contact_fear_nudges(midnight.state_changes, agent_map)

            # Midnight behavioral update — simplified: assume "worked" if at
            # workplace during work hours (hour 8), isolated if stayed home all day
            for agent in agents:
                if not agent.is_active:
                    continue
                # Rough heuristic for smoke test: non-HOME placement at hour 8
                h8_place = get_place_at(schedules, agent.agent_id,
                                        hour_of_week(sim_day, 8))
                worked_today   = h8_place != agent.household_id
                isolated_today = (agent.current_place_id == agent.household_id
                                  and not worked_today)
                update_midnight(agent, worked_today, isolated_today, sim_day)

            stats = behavioral_state_summary(agents)
            print(f"{sim_day:>4}  "
                  f"{stats['mean_fear']:>6.3f}  "
                  f"{stats['mean_fatigue']:>7.3f}  "
                  f"{stats['mean_financial_pressure']:>9.3f}  "
                  f"{stats['mean_perceived_risk']:>9.4f}  "
                  f"{stats['mean_trust']:>6.3f}  "
                  f"{n_infectious:>10}")

    print("\n[behavioral_state] Done.")
