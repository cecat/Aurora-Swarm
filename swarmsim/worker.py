"""
worker.py — Per-tick LLM decision loop for a shard of agents.

Implements SwarmSim-Architecture.md §8.  For the 1K pilot, the coordinator
runs a single worker shard covering all agents.  At scale, multiple workers
would each cover a non-overlapping subset of agents (worker_shard partition).

Per-tick responsibilities:
    1. Move each agent to their scheduled location (policy filter applied).
    2. Build Block C for every agent that needs an LLM decision this tick.
    3. Dispatch all prompts via scatter_gather (Aurora Swarm) in one batch.
    4. Parse responses and update agent behavioral state (mask_wearing,
       and behavioral_deviation flag used by disease engine).
    5. At midnight: trigger episodic memory compression for agents due.
    6. Update hourly behavioral state scalars (perceived_risk).
    7. Return a TickDecisions object for the coordinator to consume.

Agent movement logic:
    - HOSPITALIZED / ICU: routed to hospital place; no LLM call.
    - DECEASED: skipped entirely.
    - Policy filter: if place type is closed (e.g., school), reroute to HOME.
    - All others: scheduled place → LLM decision → mask/distancing state.

Behavioral interpretation:
    The LLM returns a JSON decision with fields:
        location          — one of: scheduled | home | hospital
        mask_wearing      — bool
        distancing        — bool
        health_seeking    — bool
        behavioral_deviation — bool  (broad "acting unusual" flag)

    location=="home" means the agent stays home instead of their scheduled place.
    location=="hospital" means voluntary health-seeking; routed to nearest hospital.
    The disease engine reads agent.mask_wearing directly each tick.

Async:
    tick_decisions() is a coroutine.  The coordinator awaits it.
    scatter_gather is the only blocking async call; everything else is sync.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from .models       import Agent, Place
from .schedules    import get_place_at, hour_of_week
from .prompt       import (render_block_c, render_prompt, parse_response,
                            SimState, AgentDecision)
from .communication import (build_colocation_context, build_inbox_messages,
                             Inboxes)
from .behavioral_state import update_hourly
from .memory       import (should_compress, build_compression_prompt,
                            parse_compression_response)
from . import sim_config as cfg

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Return type
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TickDecisions:
    """Results returned by tick_decisions() to the coordinator."""
    sim_day:    int
    hour:       int
    decisions:  list[AgentDecision] = field(default_factory=list)
    n_fallback: int = 0             # agents where LLM parse failed → used default
    n_skipped:  int = 0             # HOSPITALIZED/ICU/DECEASED skipped
    n_llm:      int = 0             # prompts actually sent to LLM

    # Memory compression results: agent_id → new memory string
    compression_results: dict[int, str] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Agent movement
# ═══════════════════════════════════════════════════════════════════════════════

def apply_schedule(
    agent:      Agent,
    schedules:  dict[int, list[int]],
    place_map:  dict[int, Place],
    policy:     dict,
    sim_day:    int,
    hour:       int,
    hospital_id: Optional[int],
) -> None:
    """
    Move agent to their scheduled location for this tick, applying policy
    overrides.  Mutates agent.current_place_id.

    Called for all active agents before building prompts.
    """
    if not agent.is_active:
        return

    # HOSPITALIZED / ICU → hospital (no choice)
    if agent.disease_state in ("HOSPITALIZED", "ICU"):
        agent.current_place_id = hospital_id or agent.household_id
        return

    # Get base scheduled place
    how = hour_of_week(sim_day, hour)
    scheduled_place_id = get_place_at(schedules, agent.agent_id, how)

    place = place_map.get(scheduled_place_id)
    if place is None:
        agent.current_place_id = agent.household_id
        return

    # Policy filter: closed place → home
    if not place.is_open_under_policy(policy):
        agent.current_place_id = agent.household_id
        return

    # Stay-at-home order: essential workers and healthcare workers exempt
    if policy.get("stay_at_home", False):
        exempt = agent.occupation_type in ("essential_worker", "healthcare_worker",
                                            "nursing_home_resident")
        if not exempt and place.place_type not in ("household", "hospital"):
            agent.current_place_id = agent.household_id
            return

    agent.current_place_id = scheduled_place_id


def apply_llm_location_override(
    agent:       Agent,
    decision:    AgentDecision,
    place_map:   dict[int, Place],
    hospital_id: Optional[int],
) -> None:
    """
    After receiving the LLM decision, override current_place_id if the agent
    chose to stay home or seek care.  Mutates agent.current_place_id.
    """
    if decision.location == "home":
        agent.current_place_id = agent.household_id
    elif decision.location == "hospital" and hospital_id is not None:
        agent.current_place_id = hospital_id


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt assembly helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sim_state(
    sim_day:    int,
    hour:       int,
    policy:     dict,
    agents:     list[Agent],
) -> SimState:
    n_infectious = sum(1 for a in agents if a.is_infectious)
    n_hosp       = sum(1 for a in agents
                       if a.disease_state in ("HOSPITALIZED", "ICU"))
    n_deceased   = sum(1 for a in agents if a.disease_state == "DECEASED")

    return SimState(
        sim_day=sim_day,
        hour=hour,
        policy=policy,
        n_infectious_city_est=n_infectious * 2700,   # scale-up for city estimate
        n_hospitalized=n_hosp,
        n_deceased_total=n_deceased,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main async tick function
# ═══════════════════════════════════════════════════════════════════════════════

async def tick_decisions(
    agents:         list[Agent],          # this worker's shard
    schedules:      dict[int, list[int]],
    agent_map:      dict[int, Agent],
    place_map:      dict[int, Place],
    inboxes:        Inboxes,
    prev_occupancy: dict[int, list[int]],
    policy:         dict,
    sim_day:        int,
    hour:           int,
    pool,                                 # aurora_swarm VLLMPool or AgentPool
    event_logs:     dict[int, list[str]], # for episodic memory compression
) -> TickDecisions:
    """
    Execute one simulation tick for this worker's agent shard.

    Steps:
        1. Move agents to scheduled locations (policy filter).
        2. For agents needing LLM decisions: build prompts.
        3. Batch-dispatch via scatter_gather.
        4. Parse responses; apply location overrides.
        5. Update hourly behavioral state scalars.
        6. If midnight: run episodic memory compression for due agents.

    Returns TickDecisions for the coordinator.
    """
    result = TickDecisions(sim_day=sim_day, hour=hour)

    # Find nearest hospital for routing
    hospital = next((p for p in place_map.values() if p.place_type == "hospital"),
                    None)
    hospital_id = hospital.place_id if hospital else None

    # ── Step 1: Move all agents to scheduled locations ─────────────────────────
    for agent in agents:
        apply_schedule(agent, schedules, place_map, policy,
                       sim_day, hour, hospital_id)

    # ── Step 2: Build prompts for agents needing LLM decisions ────────────────
    state = _build_sim_state(sim_day, hour, policy, list(agent_map.values()))

    llm_agents: list[Agent] = []
    prompts:    list[str]   = []

    for agent in agents:
        if not agent.is_active:
            result.n_skipped += 1
            continue
        if not agent.needs_llm_decision:
            result.n_skipped += 1
            continue

        colocation_ctx = build_colocation_context(agent, prev_occupancy,
                                                  agent_map, place_map)
        inbox_msgs     = build_inbox_messages(agent, inboxes)

        prompt = render_prompt(agent, state, colocation_ctx, inbox_msgs)
        llm_agents.append(agent)
        prompts.append(prompt)

    result.n_llm = len(prompts)

    # ── Step 3: Scatter-gather (or dry-run if no pool) ─────────────────────────
    if prompts and pool is not None:
        try:
            from aurora_swarm.patterns.scatter_gather import scatter_gather
            responses = await scatter_gather(pool=pool, prompts=prompts)
            raw_responses = [
                r.text if r.success else ""
                for r in responses
            ]
        except Exception as exc:
            log.error("scatter_gather failed at tick (%d, %d): %s",
                      sim_day, hour, exc)
            raw_responses = [""] * len(prompts)
    else:
        # Dry-run / offline mode: return empty strings → fallback decisions
        raw_responses = [""] * len(prompts)

    # ── Step 4: Parse responses and apply overrides ────────────────────────────
    for agent, raw in zip(llm_agents, raw_responses):
        decision = parse_response(raw, agent, state)

        if decision.is_fallback:
            result.n_fallback += 1

        # Location override (home or hospital if LLM chose so)
        apply_llm_location_override(agent, decision, place_map, hospital_id)

        # Behavioral state updates from decision
        agent.mask_wearing = decision.mask_wearing

        result.decisions.append(decision)

    # ── Step 5: Hourly perceived_risk update ──────────────────────────────────
    n_infectious = sum(1 for a in agent_map.values() if a.is_infectious)
    n_total      = sum(1 for a in agent_map.values() if a.is_active)

    for agent in agents:
        if agent.is_active:
            update_hourly(agent, n_infectious * 2700, n_total * 2700)

    # ── Step 6: Episodic memory compression (midnight only) ───────────────────
    if hour == 0:
        due = [a for a in agents if should_compress(a, sim_day)]
        if due and pool is not None:
            compression_results = await _run_compression(
                due, event_logs, pool, sim_day)
            result.compression_results = compression_results
        elif due:
            # Offline: leave memory unchanged
            log.debug("Skipping compression for %d agents (offline mode)", len(due))

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Episodic compression helper
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_compression(
    agents:     list[Agent],
    event_logs: dict[int, list[str]],
    pool,
    sim_day:    int,
) -> dict[int, str]:
    """Run memory compression for a batch of agents; return agent_id → memory."""
    from aurora_swarm.patterns.scatter_gather import scatter_gather

    prompts        = []
    ordered_agents = []

    for agent in agents:
        log_entries = event_logs.get(agent.agent_id, [])
        prompt = build_compression_prompt(agent, log_entries, sim_day)
        prompts.append(prompt)
        ordered_agents.append(agent)

    try:
        responses = await scatter_gather(pool=pool, prompts=prompts)
        raw_responses = [r.text if r.success else "" for r in responses]
    except Exception as exc:
        log.error("Compression scatter_gather failed at day %d: %s", sim_day, exc)
        return {}

    results: dict[int, str] = {}
    for agent, raw in zip(ordered_agents, raw_responses):
        memory = parse_compression_response(raw, agent)
        results[agent.agent_id] = memory

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI smoke test (offline / dry-run — no LLM connection required)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    from .population     import generate_population
    from .schedules      import build_schedules
    from .disease_engine import (compute_occupancy, run_exposure_tick,
                                  run_midnight_tick, seed_infections)
    from .communication  import (make_inboxes, write_state_change_to_inboxes,
                                  update_place_event_logs, inbox_summary)
    from .behavioral_state import (update_midnight, apply_fear_nudges_from_state_changes,
                                    apply_contact_fear_nudges, behavioral_state_summary)
    from .prompt         import build_agent_texts

    N      = 200
    N_DAYS = 5
    rng    = random.Random(cfg.RANDOM_SEED)

    agents, places = generate_population(n_agents=N, seed=cfg.RANDOM_SEED,
                                         use_osmnx=False)
    schedules = build_schedules(agents, places, seed=cfg.RANDOM_SEED)
    build_agent_texts(agents, places)
    inboxes   = make_inboxes(agents)
    seed_infections(agents, cfg.INITIAL_INFECTED, sim_day=0, rng=rng)

    agent_map  = {a.agent_id: a for a in agents}
    place_map  = {p.place_id: p for p in places}
    event_logs: dict[int, list[str]] = {a.agent_id: [] for a in agents}

    prev_occupancy: dict[int, list[int]] = {}
    policy = cfg.INITIAL_POLICY.copy()

    print(f"[worker] Dry-run {N_DAYS}-day test ({N} agents, offline — no LLM)\n")

    async def run():
        total_llm = 0
        total_fallback = 0

        for tick in range(N_DAYS * 24):
            sim_day = tick // 24
            hour    = tick % 24

            td = await tick_decisions(
                agents=agents,
                schedules=schedules,
                agent_map=agent_map,
                place_map=place_map,
                inboxes=inboxes,
                prev_occupancy=prev_occupancy,
                policy=policy,
                sim_day=sim_day,
                hour=hour,
                pool=None,           # offline
                event_logs=event_logs,
            )

            total_llm      += td.n_llm
            total_fallback += td.n_fallback

            # Disease tick after worker has moved agents
            occupancy = compute_occupancy(agents)
            run_exposure_tick(agents, places, occupancy)

            if hour == 0 and tick > 0:
                midnight = run_midnight_tick(agents, sim_day, rng)

                write_state_change_to_inboxes(
                    midnight.state_changes, agent_map, inboxes)
                update_place_event_logs(occupancy, agent_map, place_map)

                apply_fear_nudges_from_state_changes(
                    midnight.state_changes, agent_map)
                apply_contact_fear_nudges(midnight.state_changes, agent_map)

                # Midnight behavioral updates
                for agent in agents:
                    if not agent.is_active:
                        continue
                    how8 = hour_of_week(sim_day, 8)
                    h8_place = get_place_at(schedules, agent.agent_id, how8)
                    worked   = h8_place != agent.household_id
                    isolated = (agent.current_place_id == agent.household_id
                                and not worked)
                    update_midnight(agent, worked, isolated, sim_day)

                from .disease_engine import epidemic_summary
                from .models import INFECTIOUS_STATES
                epi  = epidemic_summary(agents)
                beh  = behavioral_state_summary(agents)
                inbx = inbox_summary(inboxes)
                n_inf  = sum(epi.get(s, 0) for s in INFECTIOUS_STATES)
                n_hosp = epi.get("HOSPITALIZED", 0) + epi.get("ICU", 0)
                n_dead = epi.get("DECEASED", 0)
                print(f"Day {sim_day:2d}  "
                      f"inf={n_inf:3d}  "
                      f"hosp={n_hosp:2d}  "
                      f"dead={n_dead:2d}  "
                      f"fear={beh.get('mean_fear', 0):.3f}  "
                      f"inbox={inbx['total_messages']:3d}  "
                      f"llm_calls={total_llm:5d}  "
                      f"fallback={total_fallback:5d}")
                total_llm = total_fallback = 0

            prev_occupancy.clear()
            prev_occupancy.update(occupancy)

        print("\n[worker] Done.")

    asyncio.run(run())
