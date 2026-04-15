"""
disease_engine.py — Deterministic disease state machine and exposure engine.

This module has no dependency on LLMs or Aurora Swarm.  It runs once per
simulation tick, after all agents have been assigned their current location
(agent.current_place_id is set for this tick by the worker / coordinator).

Execution model:
  Every tick (hourly):
    1. compute_occupancy()   — build place → [agent_ids] from current locations
    2. compute_exposure()    — for each occupied place, accumulate exposure_count
                               on susceptible agents from co-located infectious ones
  At midnight only (hour == 0):
    3. advance_states()      — run each agent's state machine forward one day
                               exposure_count resets to 0 after this step

Disease progression is modelled at DAY resolution even though the simulation
tick is hourly.  Incubation periods, recovery durations, etc. are in days.
Exposure accumulates across all 24 hourly ticks of each day.

State transitions are stochastic (seeded RNG) except for the EXPOSED →
PRE-SYMPTOMATIC / ASYMPTOMATIC branch and severity escalations, which depend
on agent age_risk_mult and calibrated base probabilities from Ozik et al. 2021.

Reference: SwarmSim-Architecture.md §8
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .models import Agent, Place, INFECTIOUS_STATES
from . import sim_config as cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Return types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StateChange:
    """Records a single disease-state transition for downstream processing."""
    agent_id:  int
    old_state: str
    new_state: str
    sim_day:   int


@dataclass
class TickResult:
    """Aggregate output of one full disease-engine tick."""
    state_changes:       list[StateChange] = field(default_factory=list)
    n_newly_exposed:     int = 0
    n_newly_symptomatic: int = 0    # MILD + SEVERE combined
    n_newly_hospitalized:int = 0
    n_newly_recovered:   int = 0
    n_newly_deceased:    int = 0
    n_exposure_events:   int = 0    # susceptible-infectious co-location pairs
    n_infectious:        int = 0    # snapshot at tick start
    n_susceptible:       int = 0    # snapshot at tick start

    def log_line(self, tick: int) -> str:
        day  = tick // 24
        hour = tick % 24
        return (
            f"[tick {tick:5d}  day {day:3d} h{hour:02d}] "
            f"inf={self.n_infectious:4d}  sus={self.n_susceptible:5d}  "
            f"exposure_pairs={self.n_exposure_events:4d}"
            + (f"  → +{self.n_newly_exposed} exposed"
               if self.n_newly_exposed else "")
            + (f"  +{self.n_newly_symptomatic} sympt"
               if self.n_newly_symptomatic else "")
            + (f"  +{self.n_newly_hospitalized} hosp"
               if self.n_newly_hospitalized else "")
            + (f"  +{self.n_newly_recovered} recov"
               if self.n_newly_recovered else "")
            + (f"  +{self.n_newly_deceased} dead"
               if self.n_newly_deceased else "")
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Occupancy
# ═══════════════════════════════════════════════════════════════════════════════

def compute_occupancy(agents: list[Agent]) -> dict[int, list[int]]:
    """
    Build place_id → [agent_id, ...] from agents' current_place_id.
    Called after the worker has set current_place_id for all active agents.
    DECEASED agents are skipped (they have no location).
    """
    occupancy: dict[int, list[int]] = defaultdict(list)
    for agent in agents:
        if agent.is_active and agent.current_place_id is not None:
            occupancy[agent.current_place_id].append(agent.agent_id)
    return dict(occupancy)


# ═══════════════════════════════════════════════════════════════════════════════
# Exposure calculation
# ═══════════════════════════════════════════════════════════════════════════════

def _mask_factor(source: Agent, target: Agent) -> float:
    """Transmission reduction from mask wearing on source and/or target."""
    factor = 1.0
    if source.mask_wearing:
        factor *= cfg.MASK_WEARER_SOURCE_FACTOR
    if target.mask_wearing:
        factor *= cfg.MASK_WEARER_DEST_FACTOR
    return factor


def _asymptomatic_factor(agent: Agent) -> float:
    """Relative infectivity of pre-symptomatic / asymptomatic vs. symptomatic."""
    if agent.disease_state in ("PRE-SYMPTOMATIC", "ASYMPTOMATIC"):
        return cfg.ASYMPTOMATIC_INFECTIVITY
    return 1.0


def compute_exposure(
    occupancy:   dict[int, list[int]],
    agent_map:   dict[int, Agent],
    place_map:   dict[int, Place],
    result:      TickResult,
) -> None:
    """
    For each occupied place, accumulate exposure_count on susceptible agents
    from co-located infectious agents.  Mutates agents in place.

    Exposure formula (per infectious-susceptible pair per hour):
        exposure += BASE_TRANSMISSION_PROB
                  × density_scaling(n_inf / n_total)
                  × ventilation_factor(place)
                  × asymptomatic_factor(infectious_agent)
                  × mask_factor(infectious, susceptible)

    exposure_count accumulates across all 24 ticks of the day and is used
    at midnight to determine SUSCEPTIBLE → EXPOSED transitions.
    """
    for place_id, occupant_ids in occupancy.items():
        place     = place_map.get(place_id)
        vent      = cfg.VENTILATION_FACTOR.get(
                        place.ventilation if place else "medium", 1.0)
        occupants = [agent_map[aid] for aid in occupant_ids]

        infectious  = [a for a in occupants if a.is_infectious]
        susceptible = [a for a in occupants if a.disease_state == "SUSCEPTIBLE"]

        if not infectious or not susceptible:
            continue

        n_total   = max(1, len(occupants))
        n_inf     = len(infectious)
        density   = cfg.DENSITY_SCALE_K * (n_inf / n_total)

        for s in susceptible:
            for i in infectious:
                exposure = (cfg.BASE_TRANSMISSION_PROB
                            * density
                            * vent
                            * _asymptomatic_factor(i)
                            * _mask_factor(i, s))
                s.exposure_count += exposure
                result.n_exposure_events += 1


# ═══════════════════════════════════════════════════════════════════════════════
# State machine
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_incubation(rng: random.Random) -> int:
    """
    Sample incubation period in days from a lognormal distribution.
    Mean ≈ 5.1 days, SD ≈ 3.0 days (COVID-19 literature; Lauer et al. 2020).
    """
    mu    = math.log(cfg.INCUBATION_MEAN_DAYS)
    sigma = cfg.INCUBATION_SD_DAYS / cfg.INCUBATION_MEAN_DAYS   # approx
    days  = rng.lognormvariate(mu, sigma)
    return max(1, round(days))


def _enter_state(
    agent:   Agent,
    new_state: str,
    sim_day: int,
    rng:     random.Random,
) -> None:
    """
    Transition agent into new_state.  Samples state duration (in days),
    resets days_in_state to 0, and sets any state-entry bookkeeping fields.
    """
    agent.disease_state = new_state
    agent.days_in_state = 0

    if new_state == "EXPOSED":
        agent.state_duration = _sample_incubation(rng)

    elif new_state in ("PRE-SYMPTOMATIC",):
        agent.state_duration = cfg.PRE_SYMPTOMATIC_DAYS

    elif new_state == "ASYMPTOMATIC":
        agent.state_duration = cfg.ASYMPTOMATIC_DURATION_DAYS

    elif new_state == "SYMPTOMATIC-MILD":
        agent.state_duration   = cfg.MILD_DURATION_DAYS
        agent.symptom_onset_day = sim_day

    elif new_state == "SYMPTOMATIC-SEVERE":
        agent.state_duration = cfg.SEVERE_DURATION_DAYS

    elif new_state == "HOSPITALIZED":
        agent.state_duration = cfg.HOSP_DURATION_DAYS
        agent.hosp_day       = sim_day

    elif new_state == "ICU":
        agent.state_duration = cfg.ICU_DURATION_DAYS

    else:   # RECOVERED, DECEASED — terminal; duration irrelevant
        agent.state_duration = 9999


def _severity_prob(base: float, age_risk_mult: float, cap: float = 0.95) -> float:
    """Scale a base transition probability by age_risk_mult, capped at `cap`."""
    return min(cap, base * age_risk_mult)


def _advance_one(
    agent:   Agent,
    sim_day: int,
    rng:     random.Random,
) -> Optional[str]:
    """
    Advance one agent's disease state machine by one day.
    Returns the new disease_state string if a transition occurred, else None.

    Called only at hour == 0 (midnight) of each simulated day.
    exposure_count is read here for SUSCEPTIBLE → EXPOSED and then reset.
    """
    s = agent.disease_state
    r = agent.age_risk_mult

    # ── Terminal / passive states ──────────────────────────────────────────────
    if s in ("RECOVERED", "DECEASED"):
        return None

    # ── SUSCEPTIBLE: test for infection from accumulated daily exposure ────────
    if s == "SUSCEPTIBLE":
        if agent.exposure_count > 0:
            # Dose-response: P(infection) = 1 − exp(−exposure_count)
            prob = 1.0 - math.exp(-agent.exposure_count)
            if rng.random() < prob:
                agent.exposure_count = 0
                _enter_state(agent, "EXPOSED", sim_day, rng)
                return "EXPOSED"
        agent.exposure_count = 0
        agent.days_in_state += 1
        return None

    # For all other states: exposure is irrelevant
    agent.exposure_count = 0
    agent.days_in_state += 1

    # Not yet due for a transition
    if agent.days_in_state < agent.state_duration:
        return None

    # ── EXPOSED → PRE-SYMPTOMATIC or ASYMPTOMATIC ─────────────────────────────
    if s == "EXPOSED":
        if rng.random() < cfg.ASYMPTOMATIC_FRACTION:
            _enter_state(agent, "ASYMPTOMATIC", sim_day, rng)
            return "ASYMPTOMATIC"
        _enter_state(agent, "PRE-SYMPTOMATIC", sim_day, rng)
        return "PRE-SYMPTOMATIC"

    # ── PRE-SYMPTOMATIC → SYMPTOMATIC-MILD ───────────────────────────────────
    if s == "PRE-SYMPTOMATIC":
        _enter_state(agent, "SYMPTOMATIC-MILD", sim_day, rng)
        return "SYMPTOMATIC-MILD"

    # ── ASYMPTOMATIC → RECOVERED ──────────────────────────────────────────────
    if s == "ASYMPTOMATIC":
        _enter_state(agent, "RECOVERED", sim_day, rng)
        return "RECOVERED"

    # ── SYMPTOMATIC-MILD → SEVERE or RECOVERED ────────────────────────────────
    if s == "SYMPTOMATIC-MILD":
        if rng.random() < _severity_prob(cfg.PROB_SEVERE_GIVEN_MILD, r):
            _enter_state(agent, "SYMPTOMATIC-SEVERE", sim_day, rng)
            return "SYMPTOMATIC-SEVERE"
        _enter_state(agent, "RECOVERED", sim_day, rng)
        return "RECOVERED"

    # ── SYMPTOMATIC-SEVERE → HOSPITALIZED or RECOVERED ───────────────────────
    if s == "SYMPTOMATIC-SEVERE":
        if rng.random() < _severity_prob(cfg.PROB_HOSP_GIVEN_SEVERE, r):
            _enter_state(agent, "HOSPITALIZED", sim_day, rng)
            return "HOSPITALIZED"
        _enter_state(agent, "RECOVERED", sim_day, rng)
        return "RECOVERED"

    # ── HOSPITALIZED → ICU or RECOVERED ──────────────────────────────────────
    if s == "HOSPITALIZED":
        if rng.random() < cfg.PROB_ICU_GIVEN_HOSP:
            _enter_state(agent, "ICU", sim_day, rng)
            return "ICU"
        _enter_state(agent, "RECOVERED", sim_day, rng)
        return "RECOVERED"

    # ── ICU → DECEASED or RECOVERED ──────────────────────────────────────────
    if s == "ICU":
        if rng.random() < cfg.PROB_DEATH_GIVEN_ICU:
            _enter_state(agent, "DECEASED", sim_day, rng)
            return "DECEASED"
        _enter_state(agent, "RECOVERED", sim_day, rng)
        return "RECOVERED"

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main tick entry points
# ═══════════════════════════════════════════════════════════════════════════════

def run_exposure_tick(
    agents:    list[Agent],
    places:    list[Place],
    occupancy: dict[int, list[int]],
) -> TickResult:
    """
    Run the exposure phase for one hourly tick.
    Call this every tick (including hour 0).

    Returns a TickResult with exposure counts populated.
    State changes will be empty — those come from run_midnight_tick().
    """
    agent_map = {a.agent_id: a for a in agents}
    place_map = {p.place_id: p for p in places}

    result = TickResult(
        n_infectious  = sum(1 for a in agents if a.is_infectious),
        n_susceptible = sum(1 for a in agents if a.disease_state == "SUSCEPTIBLE"),
    )

    compute_exposure(occupancy, agent_map, place_map, result)
    return result


def run_midnight_tick(
    agents:  list[Agent],
    sim_day: int,
    rng:     random.Random,
) -> TickResult:
    """
    Run the disease state machine for one simulated day.
    Call this once per day at hour == 0, AFTER run_exposure_tick() for that tick.

    Advances all active agents' state machines, resets exposure_count,
    and returns a TickResult with state_changes populated.
    """
    result = TickResult()

    for agent in agents:
        if not agent.is_active:
            continue

        old_state = agent.disease_state
        new_state = _advance_one(agent, sim_day, rng)

        if new_state is not None:
            change = StateChange(
                agent_id  = agent.agent_id,
                old_state = old_state,
                new_state = new_state,
                sim_day   = sim_day,
            )
            result.state_changes.append(change)

            # Aggregate counters
            if new_state == "EXPOSED":
                result.n_newly_exposed += 1
            elif new_state in ("SYMPTOMATIC-MILD", "SYMPTOMATIC-SEVERE"):
                result.n_newly_symptomatic += 1
            elif new_state == "HOSPITALIZED":
                result.n_newly_hospitalized += 1
            elif new_state == "RECOVERED":
                result.n_newly_recovered += 1
            elif new_state == "DECEASED":
                result.n_newly_deceased += 1

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Initialisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def seed_infections(
    agents:    list[Agent],
    n_infected: int,
    sim_day:   int,
    rng:       random.Random,
) -> list[int]:
    """
    Set n_infected agents to EXPOSED at simulation start.
    Prefers working-age adults (ages 18–65) as index cases.
    Returns list of seeded agent_ids.
    """
    candidates = [a for a in agents
                  if a.disease_state == "SUSCEPTIBLE" and 18 <= a.age <= 65]
    if len(candidates) < n_infected:
        candidates = [a for a in agents if a.disease_state == "SUSCEPTIBLE"]

    seeded = rng.sample(candidates, min(n_infected, len(candidates)))
    for agent in seeded:
        _enter_state(agent, "EXPOSED", sim_day, rng)

    return [a.agent_id for a in seeded]


def epidemic_summary(agents: list[Agent]) -> dict[str, int]:
    """Return a count of agents in each disease state."""
    counts: dict[str, int] = defaultdict(int)
    for agent in agents:
        counts[agent.disease_state] += 1
    return dict(counts)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI smoke test — runs a headless 30-day simulation (no LLM, schedule-driven)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from .population import generate_population
    from .schedules  import build_schedules, get_place_at, hour_of_week
    from . import sim_config as cfg

    N      = 500
    N_DAYS = 30
    rng    = random.Random(cfg.RANDOM_SEED)

    print(f"[disease_engine] headless {N_DAYS}-day test with {N} agents ...")

    agents, places = generate_population(n_agents=N, seed=cfg.RANDOM_SEED,
                                         use_osmnx=False)
    schedules      = build_schedules(agents, places, seed=cfg.RANDOM_SEED)
    seeded         = seed_infections(agents, cfg.INITIAL_INFECTED, sim_day=0, rng=rng)
    print(f"[disease_engine] Seeded EXPOSED: agents {seeded}")

    # Headless tick loop — agents follow schedule exactly (no LLM)
    agent_map = {a.agent_id: a for a in agents}

    print(f"\n{'Day':>4} {'Suscept':>7} {'Exposed':>7} {'Infect':>7} "
          f"{'Sympt':>7} {'Hosp':>6} {'Recov':>7} {'Dead':>5}")
    print("─" * 60)

    for tick in range(N_DAYS * 24):
        sim_day = tick // 24
        hour    = tick % 24

        # Move agents to their scheduled place
        how = hour_of_week(sim_day, hour)
        for agent in agents:
            if agent.is_active:
                if agent.disease_state in ("HOSPITALIZED", "ICU"):
                    # Override: hospitalized agents stay at hospital
                    hosp = next((p for p in places if p.place_type == "hospital"),
                                None)
                    agent.current_place_id = hosp.place_id if hosp else agent.household_id
                else:
                    agent.current_place_id = get_place_at(schedules, agent.agent_id, how)

        # Exposure phase (every tick)
        occ    = compute_occupancy(agents)
        result = run_exposure_tick(agents, places, occ)

        # State machine (midnight only)
        if hour == 0:
            midnight = run_midnight_tick(agents, sim_day, rng)
            summary  = epidemic_summary(agents)

            susc  = summary.get("SUSCEPTIBLE", 0)
            exp   = summary.get("EXPOSED", 0)
            inf   = (summary.get("PRE-SYMPTOMATIC", 0) +
                     summary.get("ASYMPTOMATIC", 0))
            symp  = (summary.get("SYMPTOMATIC-MILD", 0) +
                     summary.get("SYMPTOMATIC-SEVERE", 0))
            hosp  = summary.get("HOSPITALIZED", 0) + summary.get("ICU", 0)
            recov = summary.get("RECOVERED", 0)
            dead  = summary.get("DECEASED", 0)

            print(f"{sim_day:>4} {susc:>7} {exp:>7} {inf:>7} "
                  f"{symp:>7} {hosp:>6} {recov:>7} {dead:>5}")

            if midnight.state_changes:
                for sc in midnight.state_changes[:3]:   # show first 3
                    a = agent_map[sc.agent_id]
                    print(f"     ↳ agent {sc.agent_id:4d} "
                          f"({a.occupation_type}, age {a.age}) "
                          f"{sc.old_state} → {sc.new_state}")

    print("\n[disease_engine] Done.")
