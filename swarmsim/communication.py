"""
communication.py — Three-layer agent-to-agent communication.

Implements SwarmSim-Architecture.md §7.  All three layers are mediated by
the orchestrator at tick boundaries — no direct agent-to-agent messaging.

Layer 1 — Co-location context
    An agent observes what was happening at their location during the
    PREVIOUS tick: how many people were present, how many appeared visibly
    ill, mask-wearing prevalence, and any notable place events.
    Built fresh each tick from prev_occupancy and injected into Block C.

Layer 2 — Social network inbox
    When a social contact's disease state changes to a notable state
    (symptomatic, hospitalized, recovered, deceased), a brief anonymous
    message is queued in the recipient's inbox.  Inboxes are drained and
    injected into Block C at the next tick.  Max 5 unread per agent.

Layer 3 — Place event log
    Notable co-location patterns (e.g. multiple symptomatic people at one
    place) are recorded as short text entries on the Place object.
    Included in Layer 1 output.  Rolling window of 3 entries per place.

Timing note:
    Co-location context for tick N uses prev_occupancy (tick N-1 occupancy).
    agent.current_place_id is still the tick N-1 location when prompts are
    built for tick N (it has not yet been updated for tick N).
    Inbox messages written at tick N midnight are read at tick N+1.

Usage:
    from swarmsim.communication import (
        make_inboxes, build_colocation_context,
        build_inbox_messages, write_state_change_to_inboxes,
        update_place_event_logs,
    )

    inboxes = make_inboxes(agents)           # once at sim start

    # Each tick, before building prompts:
    colocation_ctx = build_colocation_context(agent, prev_occupancy,
                                              agent_map, place_map)
    inbox_msgs     = build_inbox_messages(agent, inboxes)

    # After run_midnight_tick():
    write_state_change_to_inboxes(midnight_result.state_changes,
                                  agent_map, inboxes)
    update_place_event_logs(occupancy, agent_map, place_map)
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from .models import Agent, Place
from .disease_engine import StateChange
from . import sim_config as cfg


# ── Type alias ────────────────────────────────────────────────────────────────
# Inboxes: agent_id → bounded deque of unread message strings
Inboxes = dict[int, deque]


# ═══════════════════════════════════════════════════════════════════════════════
# Initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def make_inboxes(agents: list[Agent]) -> Inboxes:
    """Create empty inboxes for every agent.  Call once at sim start."""
    return {
        a.agent_id: deque(maxlen=cfg.MAX_INBOX_MESSAGES)
        for a in agents
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Descriptor helpers (shared with prompt.py logic but kept local to avoid
# circular imports)
# ═══════════════════════════════════════════════════════════════════════════════

_AGE_GROUP_SHORT = [
    (0,   4,  "toddler"),
    (5,  12,  "child"),
    (13, 17,  "teenager"),
    (18, 29,  "young adult"),
    (30, 59,  "adult"),
    (60, 74,  "older adult"),
    (75, 120, "elderly person"),
]

_OCCUPATION_SHORT = {
    "essential_worker":      "coworker",
    "remote_capable":        "neighbor",
    "student":               "classmate",
    "school_age":            "child",
    "retired":               "neighbor",
    "unemployed":            "neighbor",
    "healthcare_worker":     "colleague",
    "nursing_home_resident": "fellow resident",
}

_NOTABLE_INBOX_STATES = {
    "SYMPTOMATIC-MILD",
    "SYMPTOMATIC-SEVERE",
    "HOSPITALIZED",
    "RECOVERED",
    "DECEASED",
}


def _age_group_short(age: int) -> str:
    for lo, hi, label in _AGE_GROUP_SHORT:
        if lo <= age <= hi:
            return label
    return "adult"


def _format_state_message(agent: Agent, new_state: str) -> str:
    """
    One-line anonymous message describing a state change, written to social
    contacts' inboxes.  Identifies by age group and occupation role only —
    no names, no agent IDs.
    """
    age_g = _age_group_short(agent.age)
    role  = _OCCUPATION_SHORT.get(agent.occupation_type, "person you know")

    templates = {
        "SYMPTOMATIC-MILD":   f"A {age_g} {role} you know is feeling ill with COVID symptoms.",
        "SYMPTOMATIC-SEVERE": f"A {age_g} {role} you know is seriously ill with COVID.",
        "HOSPITALIZED":       f"A {age_g} {role} you know has been hospitalized with COVID.",
        "RECOVERED":          f"A {age_g} {role} you know has recovered from COVID.",
        "DECEASED":           f"Someone in your social network has died.",
    }
    return templates.get(new_state, "")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1 — Co-location context
# ═══════════════════════════════════════════════════════════════════════════════

def build_colocation_context(
    agent:          Agent,
    prev_occupancy: dict[int, list[int]],
    agent_map:      dict[int, Agent],
    place_map:      dict[int, Place],
) -> str:
    """
    Build the co-location context string for agent's Block C prompt.

    Describes what the agent observed at their location LAST tick:
    - number of people present
    - number visibly symptomatic (SYMPTOMATIC states only — asymptomatic
      infectious agents are undetectable by other agents)
    - approximate mask-wearing rate among others
    - any entries in the place event log

    Returns empty string for tick 0 (no previous occupancy) or if the
    agent's previous location had no other occupants.
    """
    place_id = agent.current_place_id   # still last tick's value when called
    if place_id is None:
        return ""

    occupant_ids = prev_occupancy.get(place_id, [])
    if not occupant_ids:
        return ""

    place     = place_map.get(place_id)
    place_lbl = place.label if place else f"location {place_id}"

    # Everyone at the place last tick (including the agent themselves)
    others = [
        agent_map[aid]
        for aid in occupant_ids
        if aid in agent_map and aid != agent.agent_id
    ]
    n_total   = len(occupant_ids)     # include self for density sense
    n_others  = len(others)

    if n_others == 0:
        return ""

    # Visibly ill: only SYMPTOMATIC states are observable
    n_visibly_ill = sum(
        1 for a in others
        if a.disease_state in ("SYMPTOMATIC-MILD", "SYMPTOMATIC-SEVERE")
    )

    # Mask wearing among others
    n_masking = sum(1 for a in others if a.mask_wearing)
    mask_pct  = int(n_masking / n_others * 100) if n_others else 0

    lines = [f"At your location last hour ({place_lbl}):"]
    lines.append(f"  - Approximately {n_total} people present")

    if n_visibly_ill == 0:
        lines.append("  - No one appeared visibly unwell")
    elif n_visibly_ill == 1:
        lines.append("  - 1 person appeared visibly ill")
    else:
        lines.append(f"  - {n_visibly_ill} people appeared visibly ill")

    # Mask prevalence
    if mask_pct == 0:
        lines.append("  - Mask wearing: nobody wearing masks")
    elif mask_pct < 30:
        lines.append(f"  - Mask wearing: very few people (~{mask_pct}%)")
    elif mask_pct < 70:
        lines.append(f"  - Mask wearing: about half of people (~{mask_pct}%)")
    else:
        lines.append(f"  - Mask wearing: most people (~{mask_pct}%)")

    # Place event log (last 2 entries)
    if place and place.event_log:
        for entry in place.event_log[-2:]:
            lines.append(f"  - Note: {entry}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 — Social network inbox
# ═══════════════════════════════════════════════════════════════════════════════

def write_state_change_to_inboxes(
    state_changes: list[StateChange],
    agent_map:     dict[int, Agent],
    inboxes:       Inboxes,
) -> None:
    """
    For each notable state change, write an anonymous message to every
    social contact's inbox.

    Called by the coordinator after run_midnight_tick() returns.
    Messages are picked up by build_inbox_messages() on the next tick.
    """
    for change in state_changes:
        if change.new_state not in _NOTABLE_INBOX_STATES:
            continue

        agent = agent_map.get(change.agent_id)
        if agent is None:
            continue

        message = _format_state_message(agent, change.new_state)
        if not message:
            continue

        for contact_id in agent.social_contacts:
            inbox = inboxes.get(contact_id)
            if inbox is not None:
                inbox.append(message)


def build_inbox_messages(agent: Agent, inboxes: Inboxes) -> str:
    """
    Drain an agent's inbox and return a formatted string for Block C.
    Returns empty string if the inbox is empty.
    The inbox is cleared after reading (one-time delivery).
    """
    inbox = inboxes.get(agent.agent_id)
    if not inbox:
        return ""

    messages = list(inbox)
    inbox.clear()

    if not messages:
        return ""

    lines = ["Messages from your social network (since last hour):"]
    for msg in messages:
        lines.append(f"  - {msg}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 — Place event log
# ═══════════════════════════════════════════════════════════════════════════════

def update_place_event_logs(
    occupancy: dict[int, list[int]],
    agent_map: dict[int, Agent],
    place_map: dict[int, Place],
) -> None:
    """
    Scan occupied places for notable patterns and append to their event logs.
    Called by the coordinator after occupancy is computed for the tick.

    Thresholds for a notable event:
    - 3+ visibly symptomatic agents at one place
    - >25% of occupants visibly ill (for places with >5 people)
    - Hospital occupancy >70% of stated capacity
    """
    for place_id, occupant_ids in occupancy.items():
        place = place_map.get(place_id)
        if place is None:
            continue

        occupants = [
            agent_map[aid] for aid in occupant_ids if aid in agent_map
        ]
        n_total     = len(occupants)
        n_symptomatic = sum(
            1 for a in occupants
            if a.disease_state in ("SYMPTOMATIC-MILD", "SYMPTOMATIC-SEVERE")
        )

        new_event: Optional[str] = None

        if n_symptomatic >= 3:
            new_event = "Multiple visibly ill people have been here recently."
        elif n_total > 5 and n_symptomatic / n_total > 0.25:
            new_event = (f"A notable proportion of visitors here "
                         f"({int(n_symptomatic/n_total*100)}%) appeared unwell.")

        if place.place_type == "hospital" and n_total > place.capacity * 0.70:
            new_event = "This facility appears to be under significant strain."

        if new_event and new_event not in place.event_log:
            place.event_log.append(new_event)
            # Rolling window
            place.event_log = place.event_log[-cfg.MAX_PLACE_LOG_ENTRIES:]


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def inbox_summary(inboxes: Inboxes) -> dict[str, int]:
    """Return {total_messages, agents_with_messages} for monitoring."""
    total    = sum(len(q) for q in inboxes.values())
    nonempty = sum(1 for q in inboxes.values() if q)
    return {"total_messages": total, "agents_with_messages": nonempty}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI smoke test — runs a 14-day headless sim and shows communication output
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    from .population     import generate_population
    from .schedules      import build_schedules, get_place_at, hour_of_week
    from .disease_engine import (compute_occupancy, run_exposure_tick,
                                  run_midnight_tick, seed_infections)
    from .prompt         import build_agent_texts, render_block_c, SimState
    from . import sim_config as cfg

    N      = 300
    N_DAYS = 14
    rng    = random.Random(cfg.RANDOM_SEED)

    agents, places = generate_population(n_agents=N, seed=cfg.RANDOM_SEED,
                                         use_osmnx=False)
    schedules  = build_schedules(agents, places, seed=cfg.RANDOM_SEED)
    build_agent_texts(agents, places)
    inboxes    = make_inboxes(agents)

    agent_map  = {a.agent_id: a for a in agents}
    place_map  = {p.place_id: p for p in places}

    seed_infections(agents, cfg.INITIAL_INFECTED, sim_day=0, rng=rng)

    prev_occupancy: dict[int, list[int]] = {}

    print(f"[communication] Running {N_DAYS}-day headless test ({N} agents) ...\n")

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

        # Midnight: state machine + communication writes
        if hour == 0:
            midnight = run_midnight_tick(agents, sim_day, rng)

            if midnight.state_changes:
                write_state_change_to_inboxes(
                    midnight.state_changes, agent_map, inboxes)

            update_place_event_logs(occupancy, agent_map, place_map)

            summary = inbox_summary(inboxes)
            if summary["total_messages"] > 0:
                print(f"Day {sim_day:2d}: "
                      f"{summary['agents_with_messages']} agents have inbox "
                      f"messages ({summary['total_messages']} total)")

                # Show one example prompt with co-location + inbox
                target = next(
                    (a for a in agents
                     if inboxes.get(a.agent_id) and a.is_active),
                    None
                )
                if target and sim_day in (3, 7, 12):
                    ctx = build_colocation_context(
                        target, prev_occupancy, agent_map, place_map)
                    msgs = build_inbox_messages(target, inboxes)
                    state = SimState(
                        sim_day=sim_day, hour=0,
                        policy=cfg.INITIAL_POLICY,
                        n_infectious_city_est=sum(
                            1 for a in agents if a.is_infectious) * 2700,
                        n_hospitalized=sum(
                            1 for a in agents
                            if a.disease_state in ("HOSPITALIZED", "ICU")),
                    )
                    block_c = render_block_c(target, state, ctx, msgs)
                    print(f"\n── Block C sample (agent {target.agent_id}, "
                          f"{target.occupation_type}, "
                          f"state={target.disease_state}) ──")
                    print(block_c)
                    print()

        prev_occupancy = occupancy

    # Show place event logs at end
    logged = [(p.place_id, p.place_type, p.event_log)
              for p in places if p.event_log]
    if logged:
        print(f"\n── Place event logs ({len(logged)} places with events) ──")
        for pid, ptype, log in logged:
            p = place_map[pid]
            print(f"  [{ptype}] {p.label[:50]}")
            for entry in log:
                print(f"    • {entry}")

    print("\n[communication] Done.")
