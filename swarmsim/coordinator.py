"""
coordinator.py — Tick barrier and simulation orchestration.

Implements SwarmSim-Architecture.md §9.  The coordinator owns the master
clock, the disease engine, and all inter-tick state management.  It delegates
per-tick LLM decisions to the worker (one shard in the 1K pilot).

Tick loop (one simulated hour):
    1. Move agents via worker (schedule + policy filter).
    2. Compute occupancy.
    3. Run disease exposure tick.
    4. Dispatch LLM prompts via worker.tick_decisions().
    5. Collect decisions; apply location overrides.
    6. At midnight: run disease state machine, communication writes,
                    behavioral state midnight updates, metrics recording.
    7. Update prev_occupancy.
    8. Record per-tick metrics.

Policy management:
    The coordinator holds the current policy dict and a timeline of policy
    changes (list of (sim_day, new_policy) tuples, sorted ascending).
    At each midnight, it checks for scheduled changes and applies them,
    calling nudge_trust() for all agents if a change occurs.

Metrics:
    Written to a list of dicts (one per sim-day at midnight).  The caller
    (run_pilot.py) can dump these to JSON or CSV at the end.

Usage:
    coord = Coordinator(agents, places, schedules, pool, policy_timeline)
    await coord.run(n_days=90)
    coord.save_metrics("metrics.json")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models       import Agent, Place
from .disease_engine import (compute_occupancy, run_exposure_tick,
                              run_midnight_tick, epidemic_summary)
from .communication  import (make_inboxes, write_state_change_to_inboxes,
                              update_place_event_logs, inbox_summary)
from .behavioral_state import (update_midnight as beh_update_midnight,
                                apply_fear_nudges_from_state_changes,
                                apply_contact_fear_nudges,
                                behavioral_state_summary, nudge_trust)
from .memory       import (should_compress, record_disease_event,
                            record_isolation_event, record_contact_event)
from .prompt       import build_agent_texts
from .schedules    import build_schedules, get_place_at, hour_of_week
from .worker       import tick_decisions, TickDecisions
from .models       import INFECTIOUS_STATES
from . import sim_config as cfg

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Policy timeline helper
# ═══════════════════════════════════════════════════════════════════════════════

PolicyTimeline = list[tuple[int, dict]]   # [(sim_day, policy_dict), ...]


def default_covid_timeline() -> PolicyTimeline:
    """
    Chicago-approximate policy timeline for March–June 2020.
    sim_day 0 = March 15, 2020.
    """
    return [
        (0,  cfg.INITIAL_POLICY.copy()),
        (6,  {**cfg.INITIAL_POLICY,                         # March 21: bars/restaurants close
              "restaurants_open": False,
              "essential_retail_capacity": 0.50}),
        (10, {**cfg.INITIAL_POLICY,                         # March 25: stay-at-home
              "restaurants_open": False,
              "stay_at_home": True,
              "essential_retail_capacity": 0.50}),
        (18, {**cfg.INITIAL_POLICY,                         # April 2: schools closed for year
              "restaurants_open": False,
              "schools_open": False,
              "stay_at_home": True,
              "essential_retail_capacity": 0.50}),
        (37, {**cfg.INITIAL_POLICY,                         # April 21: mask mandate
              "restaurants_open": False,
              "schools_open": False,
              "stay_at_home": True,
              "mask_mandate": True,
              "essential_retail_capacity": 0.50}),
        (77, {**cfg.INITIAL_POLICY,                         # June 1: Phase 3 reopening
              "restaurants_open": False,
              "schools_open": False,
              "stay_at_home": False,
              "mask_mandate": True,
              "essential_retail_capacity": 0.75}),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics record
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DayMetrics:
    sim_day:            int
    calendar_date:      str
    susceptible:        int = 0
    exposed:            int = 0
    pre_symptomatic:    int = 0
    asymptomatic:       int = 0
    symptomatic_mild:   int = 0
    symptomatic_severe: int = 0
    hospitalized:       int = 0
    icu:                int = 0
    recovered:          int = 0
    deceased:           int = 0
    # Behavioral
    mean_fear:          float = 0.0
    mean_fatigue:       float = 0.0
    mean_financial_pressure: float = 0.0
    mean_perceived_risk: float = 0.0
    mean_trust:         float = 0.0
    # Communication
    inbox_messages:     int = 0
    agents_with_inbox:  int = 0
    # Performance
    tick_wall_seconds:  float = 0.0
    llm_calls:          int = 0
    llm_fallback:       int = 0
    # Policy
    policy_label:       str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Coordinator
# ═══════════════════════════════════════════════════════════════════════════════

class Coordinator:
    """
    Master simulation coordinator for the 1K pilot.

    Parameters
    ----------
    agents:          list of Agent objects (1K for pilot)
    places:          list of Place objects
    schedules:       pre-built weekly schedule dict (agent_id → 168 place_ids)
    pool:            Aurora Swarm VLLMPool or AgentPool (None for offline/dry-run)
    policy_timeline: sorted list of (sim_day, policy_dict) tuples
    """

    def __init__(
        self,
        agents:          list[Agent],
        places:          list[Place],
        schedules:       dict[int, list[int]],
        pool             = None,
        policy_timeline: Optional[PolicyTimeline] = None,
    ):
        self.agents    = agents
        self.places    = places
        self.schedules = schedules
        self.pool      = pool

        self.agent_map: dict[int, Agent] = {a.agent_id: a for a in agents}
        self.place_map: dict[int, Place] = {p.place_id: p for p in places}

        self.policy_timeline: PolicyTimeline = (
            policy_timeline if policy_timeline is not None
            else default_covid_timeline()
        )
        self._policy_idx = 0    # pointer into timeline
        self.current_policy: dict = self.policy_timeline[0][1].copy()

        self.inboxes      = make_inboxes(agents)
        self.event_logs:  dict[int, list[str]] = {a.agent_id: [] for a in agents}

        self.prev_occupancy: dict[int, list[int]] = {}
        self.metrics: list[DayMetrics] = []

        # For work/isolation tracking (midnight update)
        # Tracks whether each agent was at their scheduled workplace/school today
        self._worked_today:   dict[int, bool] = {}
        self._isolated_today: dict[int, bool] = {}

    # ── Policy management ──────────────────────────────────────────────────────

    def _check_policy_change(self, sim_day: int) -> bool:
        """Advance policy_idx if a new policy applies today.  Returns True on change."""
        changed = False
        while (self._policy_idx + 1 < len(self.policy_timeline) and
               self.policy_timeline[self._policy_idx + 1][0] <= sim_day):
            self._policy_idx += 1
            new_policy = self.policy_timeline[self._policy_idx][1]

            if new_policy != self.current_policy:
                log.info("Day %d: policy changed → %s", sim_day,
                         {k: v for k, v in new_policy.items()
                          if v != self.current_policy.get(k)})
                self.current_policy = new_policy.copy()
                changed = True

        return changed

    def _apply_mask_mandate(self) -> None:
        """If mask mandate is active, set mask_wearing=True for all active agents."""
        if self.current_policy.get("mask_mandate", False):
            for agent in self.agents:
                if agent.is_active and not agent.mask_wearing:
                    agent.mask_wearing = True

    # ── Work / isolation tracking (reset each midnight) ───────────────────────

    def _reset_daily_tracking(self) -> None:
        for agent in self.agents:
            self._worked_today[agent.agent_id]   = False
            self._isolated_today[agent.agent_id] = True   # default: isolated

    def _record_agent_location(self, agent: Agent, hour: int) -> None:
        """Mark worked/isolated based on daytime placement (hours 8–17)."""
        if hour < 8 or hour > 17:
            return
        place = self.place_map.get(agent.current_place_id)
        if place is None:
            return
        if place.place_type in ("workplace", "school", "hospital", "nursing_home"):
            self._worked_today[agent.agent_id]   = True
            self._isolated_today[agent.agent_id] = False
        elif place.place_type != "household":
            # Out of home but not at work — not isolated
            self._isolated_today[agent.agent_id] = False

    # ── Calendar helper ───────────────────────────────────────────────────────

    @staticmethod
    def _calendar_date(sim_day: int) -> str:
        from datetime import date, timedelta
        start = date(2020, 3, 15)
        return (start + timedelta(days=sim_day)).isoformat()

    @staticmethod
    def _policy_label(policy: dict) -> str:
        parts = []
        if policy.get("stay_at_home"):
            parts.append("SAH")
        if not policy.get("restaurants_open", True):
            parts.append("restaurants-closed")
        if not policy.get("schools_open", True):
            parts.append("schools-closed")
        if policy.get("mask_mandate"):
            parts.append("masks")
        return "+".join(parts) if parts else "baseline"

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _record_day_metrics(
        self,
        sim_day:    int,
        td:         TickDecisions,
        wall_secs:  float,
    ) -> None:
        epi  = epidemic_summary(self.agents)
        beh  = behavioral_state_summary(self.agents)
        inbx = inbox_summary(self.inboxes)

        m = DayMetrics(
            sim_day=sim_day,
            calendar_date=self._calendar_date(sim_day),
            susceptible=       epi.get("SUSCEPTIBLE", 0),
            exposed=           epi.get("EXPOSED", 0),
            pre_symptomatic=   epi.get("PRE-SYMPTOMATIC", 0),
            asymptomatic=      epi.get("ASYMPTOMATIC", 0),
            symptomatic_mild=  epi.get("SYMPTOMATIC-MILD", 0),
            symptomatic_severe=epi.get("SYMPTOMATIC-SEVERE", 0),
            hospitalized=      epi.get("HOSPITALIZED", 0),
            icu=               epi.get("ICU", 0),
            recovered=         epi.get("RECOVERED", 0),
            deceased=          epi.get("DECEASED", 0),
            mean_fear=               beh.get("mean_fear", 0.0),
            mean_fatigue=            beh.get("mean_fatigue", 0.0),
            mean_financial_pressure= beh.get("mean_financial_pressure", 0.0),
            mean_perceived_risk=     beh.get("mean_perceived_risk", 0.0),
            mean_trust=              beh.get("mean_trust", 0.0),
            inbox_messages=    inbx["total_messages"],
            agents_with_inbox= inbx["agents_with_messages"],
            tick_wall_seconds= wall_secs,
            llm_calls=         td.n_llm,
            llm_fallback=      td.n_fallback,
            policy_label=      self._policy_label(self.current_policy),
        )
        self.metrics.append(m)

    def save_metrics(self, path: str | Path) -> None:
        """Write daily metrics to a JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump([m.to_dict() for m in self.metrics], f, indent=2)
        log.info("Metrics saved to %s (%d days)", path, len(self.metrics))

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(
        self,
        n_days: int = cfg.N_SIM_DAYS,
        rng_seed: int = cfg.RANDOM_SEED,
    ) -> None:
        """Run the full simulation for n_days simulated days."""
        import random
        rng = random.Random(rng_seed)

        log.info("Coordinator starting: %d agents, %d days", len(self.agents), n_days)

        day_wall_start = time.perf_counter()
        last_td: Optional[TickDecisions] = None

        for tick in range(n_days * 24):
            sim_day = tick // 24
            hour    = tick % 24

            if hour == 0:
                self._reset_daily_tracking()
                policy_changed = self._check_policy_change(sim_day)

                if policy_changed:
                    # Trust erosion on policy reversal
                    for agent in self.agents:
                        if agent.is_active:
                            nudge_trust(agent, -cfg.TRUST_DECAY_ON_POLICY_CHANGE)

            # Apply mask mandate to all agents (if active, overrides LLM choice)
            self._apply_mask_mandate()

            # ── Worker tick (move + LLM) ──────────────────────────────────────
            td = await tick_decisions(
                agents=self.agents,
                schedules=self.schedules,
                agent_map=self.agent_map,
                place_map=self.place_map,
                inboxes=self.inboxes,
                prev_occupancy=self.prev_occupancy,
                policy=self.current_policy,
                sim_day=sim_day,
                hour=hour,
                pool=self.pool,
                event_logs=self.event_logs,
            )
            last_td = td

            # ── Record daytime movement for work/isolation tracking ────────────
            for agent in self.agents:
                if agent.is_active:
                    self._record_agent_location(agent, hour)

            # ── Disease exposure tick ─────────────────────────────────────────
            occupancy = compute_occupancy(self.agents)
            run_exposure_tick(self.agents, self.places, occupancy)

            # ── Midnight: disease state machine + communication ───────────────
            if hour == 0 and tick > 0:
                midnight = run_midnight_tick(self.agents, sim_day, rng)

                # Record disease events to episodic logs
                for change in midnight.state_changes:
                    log_entry = self.event_logs.get(change.agent_id)
                    agent = self.agent_map.get(change.agent_id)
                    if log_entry is not None and agent is not None:
                        record_disease_event(log_entry, agent,
                                             change.new_state, sim_day)

                # Communication: inbox writes + place event log
                write_state_change_to_inboxes(
                    midnight.state_changes, self.agent_map, self.inboxes)
                update_place_event_logs(occupancy, self.agent_map, self.place_map)

                # Behavioral state: fear nudges from disease events
                apply_fear_nudges_from_state_changes(
                    midnight.state_changes, self.agent_map)
                apply_contact_fear_nudges(
                    midnight.state_changes, self.agent_map)

                # Behavioral state: midnight scalar updates
                for agent in self.agents:
                    if not agent.is_active:
                        continue
                    worked   = self._worked_today.get(agent.agent_id, False)
                    isolated = self._isolated_today.get(agent.agent_id, True)
                    beh_update_midnight(agent, worked, isolated, sim_day)

                # Apply memory compression results
                for agent_id, new_memory in td.compression_results.items():
                    agent = self.agent_map.get(agent_id)
                    if agent is not None:
                        agent.episodic_memory = new_memory
                        agent.last_compression_day = sim_day

                # Clear weekly event logs for agents just compressed
                for agent in self.agents:
                    if should_compress(agent, sim_day):
                        self.event_logs[agent.agent_id] = []

                # Metrics
                wall_secs = time.perf_counter() - day_wall_start
                self._record_day_metrics(sim_day, last_td, wall_secs)

                epi = epidemic_summary(self.agents)
                n_inf  = sum(epi.get(s, 0) for s in INFECTIOUS_STATES)
                n_hosp = epi.get("HOSPITALIZED", 0) + epi.get("ICU", 0)
                n_dead = epi.get("DECEASED", 0)
                log.info(
                    "Day %3d (%s)  inf=%3d  hosp=%2d  dead=%2d  "
                    "llm=%4d  fallback=%4d  wall=%.1fs  policy=%s",
                    sim_day, self._calendar_date(sim_day),
                    n_inf, n_hosp, n_dead,
                    last_td.n_llm, last_td.n_fallback,
                    wall_secs, self._policy_label(self.current_policy),
                )

                day_wall_start = time.perf_counter()

            # ── Roll over occupancy ───────────────────────────────────────────
            self.prev_occupancy.clear()
            self.prev_occupancy.update(occupancy)

        log.info("Coordinator finished %d sim-days.", n_days)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI smoke test (offline dry-run)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    import sys
    from .population     import generate_population
    from .disease_engine import seed_infections

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    N      = 200
    N_DAYS = 14
    rng    = random.Random(cfg.RANDOM_SEED)

    print(f"[coordinator] Offline dry-run: {N} agents, {N_DAYS} days ...\n")

    agents, places = generate_population(n_agents=N, seed=cfg.RANDOM_SEED,
                                         use_osmnx=False)
    schedules = build_schedules(agents, places, seed=cfg.RANDOM_SEED)
    build_agent_texts(agents, places)
    seed_infections(agents, cfg.INITIAL_INFECTED, sim_day=0, rng=rng)

    coord = Coordinator(
        agents=agents,
        places=places,
        schedules=schedules,
        pool=None,   # offline
    )

    asyncio.run(coord.run(n_days=N_DAYS))

    print("\n── Final epidemic state ──")
    epi = epidemic_summary(agents)
    for state, count in sorted(epi.items()):
        print(f"  {state:<22} {count:>4}")

    print("\n── Metrics summary (first 5 days) ──")
    for m in coord.metrics[:5]:
        print(f"  Day {m.sim_day}: inf={m.symptomatic_mild + m.symptomatic_severe + m.pre_symptomatic + m.asymptomatic}  "
              f"hosp={m.hospitalized + m.icu}  dead={m.deceased}  "
              f"fear={m.mean_fear:.3f}  trust={m.mean_trust:.3f}  "
              f"policy={m.policy_label}")

    out = Path("coordinator_test_metrics.json")
    coord.save_metrics(out)
    print(f"\nMetrics written to {out}")
    print("[coordinator] Done.")
