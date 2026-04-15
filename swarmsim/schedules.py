"""
schedules.py — Hourly activity schedule generator.

Produces a weekly schedule for each agent: 168 place_ids (one per hour,
Monday 00:00 through Sunday 23:00).

Design:
  - Six occupation archetypes define the baseline "pre-pandemic typical week."
  - Each archetype has a weekday template and a weekend template, expressed
    as (start_hour, place_type_key) breakpoints that are expanded to 24 slots.
  - Per-agent variation is introduced via a random time-shift (−1, 0, or +1 hour)
    applied to all work/school transition points.
  - Place type keys are resolved to actual place_ids for each agent.
  - Transit is NOT modelled as a separate place; commute slots map directly to
    the destination. The `uses_transit` flag on Agent is used by the prompt
    builder and disease engine for risk adjustment.
  - Policy filter (closures, stay-at-home) is applied at tick-time in worker.py,
    not stored here.

Schedule index convention:
  hour_of_week = day_of_week * 24 + hour_of_day
  day_of_week:  0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun

Usage:
    from swarmsim.schedules import build_schedules, get_place_at
    schedules = build_schedules(agents, places)
    place_id  = get_place_at(schedules, agent_id=42, hour_of_week=37)
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

from .models import Agent, Place


# ═══════════════════════════════════════════════════════════════════════════════
# Place-type key constants
# These are resolved to actual place_ids per agent in _resolve_key().
# ═══════════════════════════════════════════════════════════════════════════════

HOME             = "HOME"
WORKPLACE        = "WORKPLACE"        # agent.workplace_id; falls back to HOME
WORK_OR_HOME     = "WORK_OR_HOME"     # WORKPLACE if not can_wfh, else HOME
SCHOOL           = "SCHOOL"           # agent.school_id; falls back to HOME
ESSENTIAL_RETAIL = "ESSENTIAL_RETAIL" # random essential_retail place
PARK             = "PARK"             # random park
RESTAURANT       = "RESTAURANT"       # random restaurant
COMMUNITY_VENUE  = "COMMUNITY_VENUE"  # random community_venue
NURSING_HOME     = "NURSING_HOME"     # agent.workplace_id (which IS nursing home)


# ═══════════════════════════════════════════════════════════════════════════════
# Archetype day templates
#
# Format: list of (start_hour, place_type_key)
# Each entry means "from start_hour until the next entry, occupy this place."
# The last entry runs through hour 23.
# Templates must start at hour 0.
# ═══════════════════════════════════════════════════════════════════════════════

# Time-shift buckets applied per agent to work/school transition hours.
# Keys are original hours; values are the shifted hour given a delta of ±1.
# Non-work hours (sleep, home) are never shifted.
_SHIFTABLE_HOURS = {7, 8, 9, 14, 15, 16, 17, 18, 19}


# Archetype weekday schedules
_WEEKDAY: dict[str, list[tuple[int, str]]] = {

    "essential_worker": [
        # Many shift workers start early; 8-hour shift
        (0,  HOME),              # sleep
        (5,  HOME),              # get ready
        (7,  WORKPLACE),         # shift starts
        (15, HOME),              # shift ends; commute + decompress
        (19, ESSENTIAL_RETAIL),  # errands most evenings
        (20, HOME),
        (22, HOME),              # wind down / sleep
    ],

    "remote_capable": [
        (0,  HOME),              # sleep
        (7,  HOME),              # morning routine
        (8,  WORK_OR_HOME),      # work block (WFH or in-office)
        (12, HOME),              # lunch
        (13, WORK_OR_HOME),      # afternoon work
        (17, HOME),              # end of workday
        (19, PARK),              # evening walk/exercise (2–3 days)
        (20, HOME),
        (23, HOME),              # sleep
    ],

    "school_age": [
        (0,  HOME),              # sleep
        (6,  HOME),              # breakfast, get ready
        (8,  SCHOOL),            # school day
        (15, HOME),              # after school
        (21, HOME),              # bed
    ],

    "student": [
        (0,  HOME),              # sleep
        (7,  HOME),              # morning
        (9,  SCHOOL),            # classes (late start vs. K-12)
        (14, HOME),              # afternoon home/study
        (18, WORKPLACE),         # part-time work evening shift (if has workplace)
        (22, HOME),              # late night home
    ],

    "retired": [
        (0,  HOME),              # sleep
        (8,  HOME),              # slow morning
        (10, ESSENTIAL_RETAIL),  # errands (most days)
        (11, PARK),              # walk / outdoor time
        (13, HOME),              # home for the afternoon
        (17, HOME),              # dinner, TV
        (22, HOME),              # sleep
    ],

    "unemployed": [
        (0,  HOME),              # sleep (often stays up late)
        (9,  HOME),              # late wake-up
        (11, ESSENTIAL_RETAIL),  # necessary errands
        (13, HOME),              # home most of the day
        (19, HOME),
        (23, HOME),
    ],

    "healthcare_worker": [
        # 12-hour hospital/clinic shift (7am–7pm is common)
        (0,  HOME),              # sleep
        (5,  HOME),              # get ready
        (7,  WORKPLACE),         # shift starts
        (19, HOME),              # shift ends; commute
        (21, HOME),              # quick wind-down
        (22, HOME),              # sleep early (early shift next day)
    ],

    "nursing_home_resident": [
        # Always at the nursing home (their workplace_id IS the facility)
        (0,  NURSING_HOME),
    ],
}

# Archetype weekend schedules
_WEEKEND: dict[str, list[tuple[int, str]]] = {

    "essential_worker": [
        # ~40% work weekends; otherwise errands + rest
        (0,  HOME),
        (8,  HOME),              # sleep in
        (9,  ESSENTIAL_RETAIL),  # grocery run
        (11, HOME),
        (18, RESTAURANT),        # dinner out occasionally
        (20, HOME),
        (23, HOME),
    ],

    "remote_capable": [
        (0,  HOME),
        (8,  HOME),              # sleep in
        (10, PARK),              # outdoor activity
        (12, HOME),              # lunch
        (14, ESSENTIAL_RETAIL),  # weekly shopping
        (16, HOME),
        (19, RESTAURANT),        # dinner out
        (21, HOME),
        (23, HOME),
    ],

    "school_age": [
        (0,  HOME),
        (9,  HOME),              # sleep in
        (11, PARK),              # play / outdoor
        (13, HOME),
        (16, COMMUNITY_VENUE),   # sports / activities
        (18, HOME),
        (21, HOME),              # bed
    ],

    "student": [
        (0,  HOME),
        (10, HOME),              # sleep in (classic student)
        (12, ESSENTIAL_RETAIL),  # supplies / food
        (14, HOME),
        (18, RESTAURANT),        # social life
        (22, HOME),
    ],

    "retired": [
        (0,  HOME),
        (8,  HOME),
        (10, COMMUNITY_VENUE),   # church / social / library
        (12, HOME),
        (14, PARK),              # afternoon walk
        (16, HOME),
        (21, HOME),
    ],

    "unemployed": [
        (0,  HOME),
        (10, HOME),
        (12, ESSENTIAL_RETAIL),
        (14, HOME),
        (20, HOME),
    ],

    "healthcare_worker": [
        # Weekend shifts are common in healthcare; simplified as day off
        (0,  HOME),
        (9,  HOME),
        (11, ESSENTIAL_RETAIL),
        (13, HOME),
        (18, PARK),
        (20, HOME),
    ],

    "nursing_home_resident": [
        (0,  NURSING_HOME),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Template expansion
# ═══════════════════════════════════════════════════════════════════════════════

def _expand_template(
    template: list[tuple[int, str]],
    time_shift: int = 0,
) -> list[str]:
    """
    Expand a breakpoint template to a 24-element list of place_type_keys.

    time_shift: −1, 0, or +1 — shifts work/school transition hours by this
                amount to create per-agent variation.
    """
    # Apply time shift to shiftable hours
    shifted: list[tuple[int, str]] = []
    for hour, key in template:
        if hour in _SHIFTABLE_HOURS and time_shift != 0:
            hour = max(0, min(23, hour + time_shift))
        shifted.append((hour, key))

    # Sort by hour (shift could reorder adjacent breakpoints)
    shifted.sort(key=lambda x: x[0])

    # Expand to 24 slots
    day: list[str] = [""] * 24
    for i, (start, key) in enumerate(shifted):
        end = shifted[i + 1][0] if i + 1 < len(shifted) else 24
        for h in range(start, end):
            day[h] = key

    # Fill any gaps (should not occur with well-formed templates)
    last = HOME
    for h in range(24):
        if not day[h]:
            day[h] = last
        last = day[h]

    return day


# ═══════════════════════════════════════════════════════════════════════════════
# Place-key resolution
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_key(
    key: str,
    agent: Agent,
    places_by_type: dict[str, list[Place]],
    rng: random.Random,
    # Cache random picks per agent for ESSENTIAL_RETAIL / PARK etc.
    # so the same agent always goes to the same store (realistic).
    _cache: dict,
) -> int:
    """Resolve a place_type_key to a concrete place_id for this agent."""

    if key == HOME:
        return agent.household_id

    if key in (WORKPLACE, WORK_OR_HOME):
        if key == WORK_OR_HOME and agent.can_wfh:
            return agent.household_id
        if agent.workplace_id is not None:
            return agent.workplace_id
        return agent.household_id          # fallback: work from home

    if key == SCHOOL:
        if agent.school_id is not None:
            return agent.school_id
        return agent.household_id          # no school assigned → home

    if key == NURSING_HOME:
        if agent.workplace_id is not None:
            return agent.workplace_id      # nursing home IS their workplace
        return agent.household_id

    # Pooled place types: pick once per agent and cache for the run
    cache_key = (agent.agent_id, key)
    if cache_key not in _cache:
        options = places_by_type.get(_PLACE_TYPE_MAP.get(key, ""), [])
        if options:
            _cache[cache_key] = rng.choice(options).place_id
        else:
            _cache[cache_key] = agent.household_id   # fallback

    return _cache[cache_key]


# Map schedule key → models.PLACE_TYPES value
_PLACE_TYPE_MAP = {
    ESSENTIAL_RETAIL: "essential_retail",
    PARK:             "park",
    RESTAURANT:       "restaurant",
    COMMUNITY_VENUE:  "community_venue",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Weekly schedule builder
# ═══════════════════════════════════════════════════════════════════════════════

# Days 0–4 are weekdays; 5–6 are weekend.
_WEEKDAY_DAYS  = {0, 1, 2, 3, 4}
_WEEKEND_DAYS  = {5, 6}


def build_schedules(
    agents: list[Agent],
    places: list[Place],
    seed: Optional[int] = None,
) -> dict[int, list[int]]:
    """
    Build a 168-slot weekly schedule for every agent.

    Returns:
        schedules: dict mapping agent_id → list[int] of length 168,
                   where schedules[agent_id][hour_of_week] = place_id.
    """
    rng = random.Random(seed)

    # Index places by type for fast lookup
    places_by_type: dict[str, list[Place]] = defaultdict(list)
    for p in places:
        places_by_type[p.place_type].append(p)

    # Resolution cache: (agent_id, key) → place_id
    # Ensures each agent always visits the same store / park across ticks.
    resolution_cache: dict = {}

    schedules: dict[int, list[int]] = {}

    for agent in agents:
        occ = agent.occupation_type

        # Per-agent time shift for natural variation
        time_shift = rng.choice([-1, 0, 0, 1])   # slight bias toward no shift

        weekday_keys = _expand_template(
            _WEEKDAY.get(occ, _WEEKDAY["unemployed"]), time_shift)
        weekend_keys = _expand_template(
            _WEEKEND.get(occ, _WEEKEND["unemployed"]), time_shift)

        week: list[int] = []
        for day in range(7):
            day_keys = weekday_keys if day in _WEEKDAY_DAYS else weekend_keys
            for hour_key in day_keys:
                pid = _resolve_key(
                    hour_key, agent, places_by_type, rng, resolution_cache)
                week.append(pid)

        schedules[agent.agent_id] = week

    return schedules


# ═══════════════════════════════════════════════════════════════════════════════
# Accessors
# ═══════════════════════════════════════════════════════════════════════════════

def get_place_at(
    schedules: dict[int, list[int]],
    agent_id: int,
    hour_of_week: int,
) -> int:
    """Return the scheduled place_id for agent at hour_of_week (0–167)."""
    return schedules[agent_id][hour_of_week % 168]


def get_day_schedule(
    schedules: dict[int, list[int]],
    agent_id: int,
    day_of_week: int,
) -> list[int]:
    """Return the 24 scheduled place_ids for a given day (0=Mon … 6=Sun)."""
    start = day_of_week * 24
    return schedules[agent_id][start: start + 24]


def hour_of_week(sim_day: int, hour: int) -> int:
    """Convert (sim_day, hour) to a schedule index (0–167)."""
    return (sim_day % 7) * 24 + hour


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def print_schedule_summary(
    schedules: dict[int, list[int]],
    agents: list[Agent],
    places: list[Place],
    sample_n: int = 3,
) -> None:
    """Print a human-readable week schedule for a sample of agents."""
    place_map = {p.place_id: p for p in places}
    place_type = {p.place_id: p.place_type for p in places}

    # Distribution of place types across all agent-hours
    type_counts: dict[str, int] = defaultdict(int)
    for week in schedules.values():
        for pid in week:
            type_counts[place_type.get(pid, "unknown")] += 1
    total = sum(type_counts.values())

    print("\n── Schedule type distribution (all agents, full week) ────────")
    for ptype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ptype:<20} {count:>7,}  ({count/total*100:4.1f}%)")

    print(f"\n── Sample agent schedules (Monday, hours 6–22) ─────────────")
    sample_agents = agents[:sample_n]
    for agent in sample_agents:
        print(f"\n  {agent.synthetic_name} ({agent.occupation_type}, "
              f"can_wfh={agent.can_wfh}):")
        for h in range(6, 23):
            pid   = get_place_at(schedules, agent.agent_id, h)
            ptype = place_type.get(pid, "?")
            label = place_map[pid].label if pid in place_map else f"place_{pid}"
            print(f"    Mon {h:02d}:00  [{ptype:<18}] {label[:60]}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from .population import generate_population
    from . import sim_config as cfg

    agents, places = generate_population(n_agents=200, seed=42, use_osmnx=False)
    schedules = build_schedules(agents, places, seed=cfg.RANDOM_SEED)

    print_schedule_summary(schedules, agents, places, sample_n=3)

    # Verify all 168 slots are filled for every agent
    errors = 0
    for agent in agents:
        week = schedules[agent.agent_id]
        if len(week) != 168:
            print(f"ERROR: agent {agent.agent_id} has {len(week)} slots (expected 168)")
            errors += 1
        if any(pid is None for pid in week):
            print(f"ERROR: agent {agent.agent_id} has None slots")
            errors += 1
    if errors == 0:
        print(f"✓ All {len(agents)} agents have complete 168-slot schedules.")
