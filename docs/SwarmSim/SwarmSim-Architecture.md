# SwarmSim-Architecture.md
**LLM-Agent Epidemic Simulation: Implementation Architecture**
*Aurora Swarm / SwarmSim Project — April 2026*

---

## 1. Design Principles

This document specifies the implementation architecture for SwarmSim — replacing ChiSim's deterministic C++/Repast HPC agents with LLM agents driven by Aurora Swarm, while preserving the bipartite agent/place structure and the contact-network disease engine.

Four governing decisions, each discussed in prior analysis documents:

1. **Centralized state, distributed inference.** All simulation state (agents, places, occupancy, inboxes) lives in a shared database. LLM endpoints are stateless inference servers. Workers pull state, build prompts, fire inference, write decisions back. This is the inverse of Repast HPC's distributed-state model, and it is the correct choice at the scales we target (1K–1M agents).

2. **Prefix-cached prompts.** Agent prompts are structured so the fixed portion (system context + agent profile, ~500 tokens) is always the same string, enabling vLLM's automatic prefix caching (APC). Only the dynamic suffix (~150 tokens) is computed fresh each tick. This restores prompt-size parity with the ChiSim binary serialized state.

3. **Deterministic behavioral state vector.** Persistent agent attributes that accumulate across ticks (fear, compliance fatigue, financial pressure, perceived risk) are maintained as explicit scalars by the orchestrator and injected into the dynamic suffix. The LLM does not maintain memory across calls; the orchestrator maintains it for the LLM.

4. **Three-layer communication.** Agent-to-agent interaction is restored via: (a) observable co-location context injected each tick, (b) an asynchronous social-network inbox written by the orchestrator, and (c) a place-level event log. All communication is mediated by the orchestrator at tick boundaries.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SHARED DATABASE                          │
│  agents | places | schedules | occupancy | inboxes | event_log  │
│  (SQLite for pilot; Redis hot + Postgres full for 100K+)        │
└──────────┬──────────────────────────┬───────────────────────────┘
           │  read agent shard        │  write decisions
           ▼                          │
┌──────────────────┐                  │
│   WORKER 0       │  ←───────────────┘
│   agents 0–N/W   │
│                  │
│  1. query state  │         ┌──────────────────────────────────┐
│  2. build suffix │────────►│       AURORA SWARM POOL          │
│  3. scatter_gath │◄────────│  VLLMPool (batch + prefix cache) │
│  4. parse JSON   │         │  vLLM endpoint 0 … endpoint K    │
│  5. write DB     │         └──────────────────────────────────┘
└─────────┬────────┘
          │  (Workers 1…W-1 identical; all share same LLM pool)
          ▼
    barrier sync
    (all workers done with tick N)
          │
          ▼
┌─────────────────────┐
│   DISEASE ENGINE    │
│   (batch, central)  │
│                     │
│  compute occupancy  │
│  compute exposure   │
│  advance states     │
│  write transitions  │
└─────────┬───────────┘
          │
          ▼
      tick N+1
```

---

## 3. Data Layer

### 3.1 Database Strategy by Scale

| Agent count | State store | Hot cache |
|---|---|---|
| 1K (pilot) | Python dicts / SQLite | None needed |
| 10K–100K | SQLite or Postgres | None needed |
| 100K–1M | Postgres | Redis (occupancy + inboxes) |
| 1M+ | Postgres (sharded) | Redis cluster |

All schema definitions below use generic SQL. For the pilot, implement as Python dataclasses backed by dicts; the field names are identical.

### 3.2 Agent Table (`agents`)

```sql
CREATE TABLE agents (
    -- Identity (immutable)
    agent_id          INTEGER PRIMARY KEY,
    synthetic_name    TEXT,
    age               INTEGER,
    sex               CHAR(1),          -- 'M' | 'F'
    race_ethnicity    TEXT,
    zip_code          TEXT,
    neighborhood      TEXT,

    -- Household
    household_id      INTEGER,
    household_size    INTEGER,

    -- Occupation
    occupation_type   TEXT,             -- 'essential_worker' | 'remote_capable' |
                                        -- 'student' | 'school_age' | 'retired' |
                                        -- 'unemployed' | 'healthcare_worker' |
                                        -- 'nursing_home_resident'
    workplace_id      INTEGER,
    school_id         INTEGER,
    can_wfh           BOOLEAN,
    uses_transit      BOOLEAN,

    -- Health baseline
    comorbidities     TEXT,             -- JSON array: ['diabetes','hypertension',...]
    healthcare_access REAL,             -- 0.0–1.0
    age_risk_mult     REAL,             -- precomputed from age

    -- Disease state (mutable each tick)
    disease_state     TEXT NOT NULL DEFAULT 'SUSCEPTIBLE',
    days_in_state     INTEGER DEFAULT 0,
    exposure_count    REAL DEFAULT 0.0,
    symptom_onset_day INTEGER,
    hosp_day          INTEGER,

    -- Behavioral state vector (mutable each tick, deterministic)
    fear_level        REAL DEFAULT 0.1, -- 0.0–1.0
    compliance_fatigue REAL DEFAULT 0.0,-- 0.0–1.0; rises with isolation days
    financial_pressure REAL DEFAULT 0.3,-- 0.0–1.0; occupation-dependent baseline
    perceived_risk    REAL DEFAULT 0.1, -- 0.0–1.0; updated from local case counts
    trust_in_news     REAL DEFAULT 0.6, -- 0.0–1.0; erodes with policy inconsistency

    -- Memory (mutable, updated every 7 sim days)
    episodic_memory   TEXT DEFAULT '',  -- compressed natural language, ≤75 tokens
    last_compression_day INTEGER DEFAULT 0,

    -- Simulation bookkeeping
    current_place_id  INTEGER,
    worker_shard      INTEGER DEFAULT 0
);
```

### 3.3 Place Table (`places`)

```sql
CREATE TABLE places (
    place_id          INTEGER PRIMARY KEY,
    place_type        TEXT NOT NULL,    -- 'household' | 'workplace' | 'school' |
                                        -- 'essential_retail' | 'hospital' |
                                        -- 'restaurant' | 'park' | 'transit' |
                                        -- 'nursing_home' | 'community_venue'
    label             TEXT,             -- natural language descriptor for LLM
    zip_code          TEXT,
    capacity          INTEGER,
    ventilation       TEXT,             -- 'high' | 'medium' | 'low' | 'outdoor'
    -- event log: rolling window, last 3 notable events at this place
    event_log         TEXT DEFAULT '[]' -- JSON array of strings
);
```

### 3.4 Schedule Table (`schedules`)

```sql
CREATE TABLE schedules (
    agent_id  INTEGER,
    hour_of_week INTEGER,               -- 0–167 (0 = Monday 00:00)
    place_id  INTEGER,
    PRIMARY KEY (agent_id, hour_of_week)
);
```

Pre-computed from archetype profiles (see Section 8). Static for the run; policy overrides applied at query time by worker, not stored here.

### 3.5 Occupancy Table (`occupancy`)

Recomputed at the start of each tick by the disease engine after all location decisions are written.

```sql
CREATE TABLE occupancy (
    tick      INTEGER,
    place_id  INTEGER,
    agent_id  INTEGER,
    PRIMARY KEY (tick, place_id, agent_id)
);
```

For the pilot, this is a Python `defaultdict(list)` rebuilt each tick.

### 3.6 Inbox Table (`inboxes`)

```sql
CREATE TABLE inboxes (
    inbox_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id   INTEGER NOT NULL,        -- recipient
    tick_sent  INTEGER NOT NULL,
    message    TEXT NOT NULL,           -- ≤ 25 tokens
    read       BOOLEAN DEFAULT FALSE
);
-- Retain only last 5 unread messages per agent; older messages dropped.
```

### 3.7 Simulation Config Table (`sim_config`)

```sql
CREATE TABLE sim_config (
    key   TEXT PRIMARY KEY,
    value TEXT
);
-- Keys: current_tick, sim_start_date, policy_state, local_news_summary,
--       daily_case_count, daily_death_count, hospital_capacity_pct
```

---

## 4. Agent State: The Three Layers

Each agent's full state at any tick is the combination of three layers:

```
┌──────────────────────────────────────────┐
│  LAYER 1: STATIC PROFILE                │  Written once at init.
│  age, occupation, household, zip, etc.  │  Never changes.
│  ~200 tokens when rendered as text.     │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│  LAYER 2: BEHAVIORAL STATE VECTOR       │  Updated deterministically
│  fear, fatigue, pressure, risk,         │  by orchestrator after each tick.
│  trust — 5 scalar floats               │  ~15 tokens when rendered.
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│  LAYER 3: EPISODIC MEMORY               │  Updated by LLM compression call
│  Short natural-language summary of      │  every 7 simulated days.
│  notable recent events.                 │  Bounded at ≤75 tokens.
└──────────────────────────────────────────┘
```

### 4.1 Behavioral State Vector Update Rules

Applied by the orchestrator (no LLM call) after each tick's decisions are written:

```python
def update_behavioral_state(agent, events, sim_state):
    """
    events: dict of what happened to this agent this tick
      - 'household_member_symptomatic': bool
      - 'social_contact_sick': bool
      - 'social_contact_died': bool
      - 'days_isolated': int (consecutive)
      - 'missed_work_days': int (cumulative)
      - 'policy_changed': bool
    sim_state: dict with 'local_case_rate', 'local_death_rate'
    """
    FEAR_RISE_HOUSEHOLD    = 0.15
    FEAR_RISE_CONTACT_SICK = 0.08
    FEAR_RISE_CONTACT_DIED = 0.20
    FEAR_DECAY_PER_TICK    = 0.005   # ~1.0 → 0.0 over 200 ticks

    FATIGUE_RISE_PER_ISOLATED_DAY = 0.03
    FATIGUE_DECAY_PER_ACTIVE_DAY  = 0.05

    PRESSURE_BASELINE = {            # occupation-specific starting point
        'essential_worker': 0.7,
        'remote_capable':   0.2,
        'unemployed':       0.9,
        'retired':          0.1,
        'student':          0.2,
        'healthcare_worker':0.5,
    }
    PRESSURE_RISE_PER_MISSED_DAY = 0.04
    PRESSURE_CAP = 1.0

    TRUST_DECAY_ON_POLICY_CHANGE = 0.05
    TRUST_SLOW_DECAY_PER_TICK    = 0.001

    a = agent

    # Fear
    if events.get('household_member_symptomatic'):
        a.fear_level = min(1.0, a.fear_level + FEAR_RISE_HOUSEHOLD)
    if events.get('social_contact_sick'):
        a.fear_level = min(1.0, a.fear_level + FEAR_RISE_CONTACT_SICK)
    if events.get('social_contact_died'):
        a.fear_level = min(1.0, a.fear_level + FEAR_RISE_CONTACT_DIED)
    a.fear_level = max(0.0, a.fear_level - FEAR_DECAY_PER_TICK)
    # Also track local epidemiological context
    a.perceived_risk = min(1.0, sim_state['local_case_rate'] * 10 +
                                sim_state['local_death_rate'] * 50)

    # Compliance fatigue
    if events.get('days_isolated', 0) > 0:
        a.compliance_fatigue = min(1.0,
            a.compliance_fatigue + FATIGUE_RISE_PER_ISOLATED_DAY)
    else:
        a.compliance_fatigue = max(0.0,
            a.compliance_fatigue - FATIGUE_DECAY_PER_ACTIVE_DAY)

    # Financial pressure
    base = PRESSURE_BASELINE.get(a.occupation_type, 0.4)
    missed = events.get('missed_work_days', 0)
    a.financial_pressure = min(PRESSURE_CAP,
        base + missed * PRESSURE_RISE_PER_MISSED_DAY)

    # Trust in news
    if events.get('policy_changed'):
        a.trust_in_news = max(0.0, a.trust_in_news - TRUST_DECAY_ON_POLICY_CHANGE)
    a.trust_in_news = max(0.1, a.trust_in_news - TRUST_SLOW_DECAY_PER_TICK)
```

### 4.2 Episodic Memory Compression

Triggered when `sim_day - agent.last_compression_day >= 7`.

```python
COMPRESSION_PROMPT = """
You are maintaining a memory summary for a Chicago resident in a disease simulation.
Below is their previous memory summary and the key events of the past 7 days.
Write a new summary in 2–3 sentences (≤75 tokens) in first person, present tense.
Keep: events that would shape future decisions. Drop: routine uneventful days.

Previous memory:
{previous_memory}

Events this week:
{event_log}

New summary (2–3 sentences, first person):
"""
```

This is a single `pool.post()` call per agent per 7 sim-days — roughly 1.4% of total inference volume.

---

## 5. Prompt Architecture

### 5.1 Structure and Token Budget

```
┌────────────────────────────────────────────────────────┐ TOKEN COUNT
│  BLOCK A — SYSTEM CONTEXT (cached; ~10 variants)      │
│  Role instructions, simulation rules, response schema  │  ~300 tokens
│  Keyed by: occupation_type                             │
├────────────────────────────────────────────────────────┤
│  BLOCK B — AGENT PROFILE (cached; 1 per agent)        │
│  Static demographic + household + health attributes    │  ~200 tokens
│  Keyed by: agent_id                                    │
├────────────────────────────────────────────────────────┤  ← prefix cache
│  BLOCK C — DYNAMIC STATE (computed each tick)         │  boundary
│  Disease state, behavioral vector, memory, policy,    │  ~150 tokens
│  co-location context, inbox messages                   │
├────────────────────────────────────────────────────────┤
│  BLOCK D — TASK (cached; 1 variant)                   │
│  The decision questions + JSON output schema           │  ~100 tokens
└────────────────────────────────────────────────────────┘
                                           TOTAL: ~750 tokens input
                                           RESPONSE: ~150 tokens
```

Blocks A+B = the prefix (500 tokens). With vLLM APC enabled, this is computed once per agent per model restart and served from KV cache on every subsequent tick.

Block C is the only per-tick network payload: ~150 tokens × 4 chars/token ≈ 600 bytes/agent/tick. This is roughly on par with the ChiSim MPI-serialized agent state.

### 5.2 Block A — System Context Template

One template per `occupation_type`. Example for `essential_worker`:

```
SYSTEM — CityCOVID LLM Agent (Essential Worker)

You are simulating a single Chicago resident who works an essential in-person job
during the COVID-19 pandemic. You reason and respond as this specific person.
You are NOT an AI assistant — you ARE this person. Do not break character.

Simulation rules:
- Your decisions cover one hour of simulated time.
- You only know what your character would plausibly know.
- You cannot choose locations that are physically impossible for your character.
- Respond ONLY with the JSON schema specified at the end. No commentary.

Disease state definitions you may be told:
  SUSCEPTIBLE: healthy, not immune
  EXPOSED: infected but not yet contagious (you don't know this)
  PRE-SYMPTOMATIC: contagious, no symptoms yet (you don't know this)
  ASYMPTOMATIC: contagious, no symptoms (you don't know this)
  SYMPTOMATIC-MILD: noticeably ill; you are aware
  SYMPTOMATIC-SEVERE: seriously ill; you are aware
  HOSPITALIZED | ICU | RECOVERED | DECEASED
```

### 5.3 Block B — Agent Profile Template

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR PROFILE (fixed for this simulation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name:              {synthetic_name}
Age:               {age}
Sex:               {sex}
Race/ethnicity:    {race_ethnicity}
Neighborhood:      {neighborhood}, Chicago (zip {zip_code})
Household:         {household_size}-person household
                   Members: {household_members_description}
Occupation:        {occupation_type}
Workplace:         {workplace_label}
Can work from home:{can_wfh}
Uses transit:      {uses_transit}
Health conditions: {comorbidities_description}
Healthcare access: {healthcare_access_description}
```

Render this once at simulation start and store as a string in the agent record (`profile_text`). Never re-render during the run.

### 5.4 Block C — Dynamic State Template

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT STATE  (Sim day {sim_day}, {day_of_week} {hour}:00)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Disease state:     {disease_state}  (day {days_in_state} in this state)
{symptom_line}
How you feel:      fear={fear_pct}%  fatigue={fatigue_pct}%
                   financial pressure={pressure_pct}%
                   risk perception={risk_pct}%

Memory: {episodic_memory}

Policy environment:
{policy_lines}

Local situation (what you know from news):
{local_news_summary}

Your normal schedule this hour: {baseline_place_label}

Where you are now: {current_place_label}

{colocation_context}

{inbox_messages}
```

**Rendering notes:**

- `{symptom_line}`: blank if SUSCEPTIBLE/EXPOSED/PRE-SYMPTOMATIC/ASYMPTOMATIC. For SYMPTOMATIC-MILD: `"Symptoms: mild fatigue, low fever since day {symptom_onset_day}."` For SYMPTOMATIC-SEVERE: `"Symptoms: significant shortness of breath, high fever."`
- `{fear_pct}` etc: `int(agent.fear_level * 100)`
- `{colocation_context}`: see Section 7.1
- `{inbox_messages}`: see Section 7.2; blank string if inbox is empty

### 5.5 Block D — Task and Response Schema

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR DECISION THIS HOUR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Decide where you go (or stay) and how you behave.

Location options: HOME | WORKPLACE | SCHOOL | ESSENTIAL_RETAIL |
  HOSPITAL_OR_CLINIC | SOCIAL_VISIT | RESTAURANT_BAR | PARK_OR_OUTDOOR |
  TRANSIT | OTHER

Respond with ONLY valid JSON:

{
  "agent_id": {agent_id},
  "sim_day": {sim_day},
  "hour": {hour},
  "location": "<LOCATION_TYPE>",
  "reasoning": "<1–2 sentences as this person>",
  "mask_wearing": <true|false|null>,
  "distancing": <true|false|null>,
  "health_seeking": <true|false>,
  "behavioral_deviation": "<description or null>"
}
```

---

## 6. Tick Execution Loop

### 6.1 One Full Tick (Worker Perspective)

```python
async def run_tick(worker: Worker, tick: int, pool: VLLMPool):
    sim_day  = tick // 24
    hour     = tick % 24
    hour_of_week = (sim_day % 7) * 24 + hour

    # ── Step 1: Pull agent shard from DB ──────────────────────────────
    agents = db.query(
        "SELECT * FROM agents WHERE worker_shard = ?", worker.shard_id
    )

    # ── Step 2: Apply policy filter to schedule ──────────────────────
    policy = db.get_policy_state()
    for agent in agents:
        scheduled_place = db.get_schedule(agent.agent_id, hour_of_week)
        agent.next_place_id = apply_policy_filter(agent, scheduled_place, policy)
        # apply_policy_filter: redirect to HOME if place_type is closed,
        # or if agent is HOSPITALIZED/DECEASED

    # ── Step 3: Build prompts ────────────────────────────────────────
    # Identify which agents need LLM decisions this tick
    # (skip HOSPITALIZED, DECEASED, fully-compliant archetypes if desired)
    llm_agents = [a for a in agents if needs_llm_decision(a)]

    prompts = []
    for agent in llm_agents:
        colocation_ctx = build_colocation_context(agent, tick, db)
        inbox_msgs     = build_inbox_messages(agent, db)
        dynamic_suffix = render_block_c(agent, tick, policy, colocation_ctx,
                                        inbox_msgs, db)
        # Block A and B are stored as agent.system_text and agent.profile_text
        # vLLM sees: system_text + profile_text + dynamic_suffix + task_text
        # The first two are prefix-cached after tick 0
        full_prompt = (agent.system_text + "\n" + agent.profile_text +
                       "\n" + dynamic_suffix + "\n" + TASK_TEXT)
        prompts.append(full_prompt)

    # ── Step 4: Scatter-gather LLM inference ─────────────────────────
    responses = await scatter_gather(pool, prompts)

    # ── Step 5: Parse and validate responses ─────────────────────────
    decisions = {}
    for agent, response in zip(llm_agents, responses):
        if response.success:
            try:
                d = json.loads(response.text)
                decisions[agent.agent_id] = d
            except json.JSONDecodeError:
                decisions[agent.agent_id] = fallback_decision(agent)
        else:
            decisions[agent.agent_id] = fallback_decision(agent)

    # For non-LLM agents: use schedule directly
    for agent in agents:
        if agent.agent_id not in decisions:
            decisions[agent.agent_id] = schedule_decision(agent)

    # ── Step 6: Write location decisions to DB ────────────────────────
    db.batch_update_locations(decisions)

    # ── Step 7: Update behavioral state vectors ───────────────────────
    events = compute_events(agents, decisions, db)
    db.batch_update_behavioral_state(agents, events, db.get_sim_state())

    # ── Step 8: Trigger episodic memory compression if due ────────────
    for agent in agents:
        if sim_day - agent.last_compression_day >= 7 and sim_day > 0:
            asyncio.create_task(compress_episodic_memory(agent, pool, db))
            # Fire-and-forget; result written async, used next tick

    # ── Step 9: Mark worker tick complete (barrier) ───────────────────
    db.mark_worker_done(worker.shard_id, tick)
```

### 6.2 Disease Engine (Central, Runs After All Workers Done)

```python
def run_disease_engine(tick: int, db: Database):
    """
    Runs after barrier: all workers have written tick N location decisions.
    """
    # ── Step 1: Build occupancy table ────────────────────────────────
    occupancy = defaultdict(list)   # place_id → [agent_id, ...]
    for agent_id, place_id in db.get_all_locations(tick):
        occupancy[place_id].append(agent_id)

    # ── Step 2: Compute exposure events ──────────────────────────────
    for place_id, occupant_ids in occupancy.items():
        place = db.get_place(place_id)
        occupants = [db.get_agent(aid) for aid in occupant_ids]
        infectious = [a for a in occupants if is_infectious(a)]
        susceptible = [a for a in occupants if a.disease_state == 'SUSCEPTIBLE']

        if not infectious or not susceptible:
            continue

        n_total = len(occupants)
        n_inf   = len(infectious)
        density_scale = density_scaling(n_inf, n_total)
        vent_factor   = VENTILATION_FACTOR[place.ventilation]

        for s in susceptible:
            for i in infectious:
                exposure = (BASE_TRANSMISSION_PROB
                           * density_scale
                           * mask_factor(s, i)
                           * vent_factor
                           * ASYMPTOMATIC_MULT if is_asymptomatic(i) else 1.0)
                db.add_exposure(s.agent_id, exposure)

    # ── Step 3: Advance disease state machines ───────────────────────
    for agent in db.get_all_agents():
        new_state, new_days = advance_state_machine(agent)
        db.update_disease_state(agent.agent_id, new_state, new_days)

    # ── Step 4: Write co-location events to inboxes ──────────────────
    write_colocation_inbox_events(occupancy, db)

    # ── Step 5: Update place event logs ──────────────────────────────
    update_place_event_logs(occupancy, db)
```

### 6.3 Tick Barrier and Coordination

```python
# Coordinator (single process, lightweight)
async def simulation_loop(n_workers: int, n_ticks: int):
    for tick in range(n_ticks):
        # Signal all workers to start tick
        db.set_tick(tick)

        # Wait for all workers to complete
        while True:
            done = db.count_workers_done(tick)
            if done == n_workers:
                break
            await asyncio.sleep(0.1)

        # Run disease engine (synchronous, ~milliseconds at 1M agents)
        run_disease_engine(tick, db)

        # Optionally: checkpoint, record metrics, update policy state
        record_metrics(tick, db)
```

For the 1K pilot: single process, no barrier needed — just call `run_tick` then `run_disease_engine` sequentially in a loop.

---

## 7. Agent Communication

### 7.1 Co-location Context (Block C injection)

Built by the orchestrator before prompt assembly. Uses the **previous tick's** occupancy table (current tick occupancy is not yet finalized when prompts are built).

```python
def build_colocation_context(agent, tick, db) -> str:
    if tick == 0:
        return ""
    prev_place_id = agent.current_place_id
    prev_occupants = db.get_occupancy(tick - 1, prev_place_id)

    if not prev_occupants:
        return ""

    n_total     = len(prev_occupants)
    n_symptomatic = sum(1 for a in prev_occupants
                        if a.disease_state in ('SYMPTOMATIC-MILD','SYMPTOMATIC-SEVERE'))
    n_masking     = sum(1 for a in prev_occupants
                        if a.agent_id in db.get_mask_wearers(tick - 1))
    place = db.get_place(prev_place_id)

    parts = [f"At your location last hour ({place.label}):"]
    parts.append(f"  - Approximately {n_total} people present")
    if n_symptomatic:
        parts.append(f"  - {n_symptomatic} person(s) appeared visibly ill")
    mask_pct = int(n_masking / n_total * 100) if n_total else 0
    parts.append(f"  - Mask wearing: ~{mask_pct}% of people")
    if place.event_log:
        log = json.loads(place.event_log)
        for event in log[-2:]:          # show at most last 2 events
            parts.append(f"  - {event}")

    return "\n".join(parts)
```

Target: ~40–60 tokens per agent per tick.

### 7.2 Social Network Inbox

Agents have a `social_contacts` relationship (stored as `agent_id → [contact_agent_id, ...]`). Written by the disease engine after each tick:

```python
def write_colocation_inbox_events(occupancy, db):
    """Write notable observable events to social contact inboxes."""
    # Disease state changes visible to social network
    for agent in db.get_agents_with_state_change_today():
        new_state = agent.disease_state
        if new_state in ('SYMPTOMATIC-MILD', 'SYMPTOMATIC-SEVERE',
                         'HOSPITALIZED', 'RECOVERED', 'DECEASED'):
            msg = format_social_message(agent, new_state)
            for contact_id in db.get_social_contacts(agent.agent_id):
                db.add_inbox_message(contact_id, msg, tick=current_tick)

def format_social_message(agent, state) -> str:
    occupation_desc = OCCUPATION_SHORT[agent.occupation_type]
    age_group = age_group_label(agent.age)
    msgs = {
        'SYMPTOMATIC-MILD':   f"A {age_group} {occupation_desc} you know reported feeling ill.",
        'SYMPTOMATIC-SEVERE': f"A {age_group} {occupation_desc} you know is seriously ill.",
        'HOSPITALIZED':       f"Someone you know ({age_group}, {occupation_desc}) was hospitalized.",
        'RECOVERED':          f"Someone you know recovered from COVID.",
        'DECEASED':           f"Someone in your social network has died.",
    }
    return msgs[state]
```

Inbox messages injected into Block C:

```python
def build_inbox_messages(agent, db) -> str:
    msgs = db.get_unread_inbox(agent.agent_id, limit=5)
    if not msgs:
        return ""
    db.mark_inbox_read(agent.agent_id)
    lines = ["Messages from your social network (since last hour):"]
    for m in msgs:
        lines.append(f"  - {m.message}")
    return "\n".join(lines)
```

Target: 0–75 tokens depending on social network activity.

### 7.3 Place Event Log

Updated by the disease engine. Rolling window of 3 entries per place.

```python
def update_place_event_logs(occupancy, db):
    for place_id, occupant_ids in occupancy.items():
        place = db.get_place(place_id)
        log = json.loads(place.event_log or '[]')
        occupants = [db.get_agent(aid) for aid in occupant_ids]

        # Notable thresholds that generate log entries
        n_symptomatic = sum(1 for a in occupants
                            if 'SYMPTOMATIC' in a.disease_state)
        n_total = len(occupants)

        if n_symptomatic >= 3 or (n_total > 5 and n_symptomatic / n_total > 0.2):
            log.append(f"Several people were visibly ill here recently.")
        if any(a.disease_state == 'HOSPITALIZED' for a in occupants):
            log.append(f"A person here was taken to hospital.")

        # Keep last 3 entries
        place.event_log = json.dumps(log[-3:])
        db.update_place(place)
```

---

## 8. Disease Engine — State Machine

### 8.1 States and Infectiousness

| State | Infectious | Aware | Schedule modified |
|---|---|---|---|
| SUSCEPTIBLE | No | Yes | No |
| EXPOSED | No | No | No |
| PRE-SYMPTOMATIC | Yes | No | No |
| ASYMPTOMATIC | Yes | No | No |
| SYMPTOMATIC-MILD | Yes | Yes | Partial isolation |
| SYMPTOMATIC-SEVERE | Yes | Yes | Strong isolation |
| HOSPITALIZED | Managed | Yes | Override to HOSPITAL |
| ICU | Managed | Yes | Override to HOSPITAL |
| RECOVERED | No | Yes | No |
| DECEASED | No | N/A | Removed from simulation |

### 8.2 Transition Logic

```python
def advance_state_machine(agent) -> tuple[str, int]:
    s = agent.disease_state
    d = agent.days_in_state
    r = agent.age_risk_mult

    if s == 'SUSCEPTIBLE':
        if agent.exposure_count > 0:
            if random() < exposure_to_prob(agent.exposure_count):
                return 'EXPOSED', 0
        agent.exposure_count = 0   # reset each tick
        return s, d + 1

    elif s == 'EXPOSED':
        if d >= sample_incubation_period():  # ~5 days, lognormal
            if random() < ASYMPTOMATIC_FRACTION:
                return 'ASYMPTOMATIC', 0
            return 'PRE-SYMPTOMATIC', 0
        return s, d + 1

    elif s == 'PRE-SYMPTOMATIC':
        if d >= PRE_SYMPTOMATIC_DAYS:        # ~2 days
            return 'SYMPTOMATIC-MILD', 0
        return s, d + 1

    elif s == 'ASYMPTOMATIC':
        if d >= ASYMPTOMATIC_DURATION:       # ~7 days
            return 'RECOVERED', 0
        return s, d + 1

    elif s == 'SYMPTOMATIC-MILD':
        if d >= MILD_DURATION:               # ~4 days
            if random() < SEVERE_PROB * r:
                return 'SYMPTOMATIC-SEVERE', 0
            return 'RECOVERED', 0
        return s, d + 1

    elif s == 'SYMPTOMATIC-SEVERE':
        if d >= SEVERE_DURATION:             # ~3 days
            if random() < HOSP_PROB * r:
                return 'HOSPITALIZED', 0
            return 'RECOVERED', 0
        return s, d + 1

    elif s == 'HOSPITALIZED':
        if d >= HOSP_DURATION:               # ~8 days
            if random() < ICU_PROB * r:
                return 'ICU', 0
            return 'RECOVERED', 0
        return s, d + 1

    elif s == 'ICU':
        if d >= ICU_DURATION:                # ~10 days
            if random() < DEATH_PROB * r:
                return 'DECEASED', 0
            return 'RECOVERED', 0
        return s, d + 1

    return s, d + 1   # RECOVERED, DECEASED: terminal states
```

### 8.3 Transmission Parameters (from Ozik et al. 2021, Table 1)

```python
BASE_TRANSMISSION_PROB = 0.06          # per infectious-susceptible co-location per hour
ASYMPTOMATIC_MULT      = 0.5           # relative infectivity vs symptomatic
DENSITY_SCALE_K        = 0.75          # scaling factor for n_inf/n_total
VENTILATION_FACTOR     = {
    'outdoor':  0.1,
    'high':     0.5,
    'medium':   1.0,
    'low':      1.5,
}
ASYMPTOMATIC_FRACTION  = 0.35          # fraction of exposed → asymptomatic

def density_scaling(n_inf, n_total):
    return DENSITY_SCALE_K * (n_inf / n_total)

def mask_factor(susceptible, infectious):
    # Rough multiplicative reduction for mask wearing
    s_mask = 0.5 if susceptible.mask_wearing else 1.0
    i_mask = 0.5 if infectious.mask_wearing  else 1.0
    return s_mask * i_mask

def exposure_to_prob(exposure_count):
    # Sigmoid; exposure_count accumulates across co-locations this tick
    return 1 - exp(-exposure_count)
```

---

## 9. Aurora Swarm Integration

### 9.1 VLLMPool Configuration

```python
from aurora_swarm import VLLMPool, parse_hostfile
from aurora_swarm.patterns.scatter_gather import scatter_gather

async def create_pool(hostfile: str) -> VLLMPool:
    endpoints = parse_hostfile(hostfile)
    pool = VLLMPool(
        endpoints,
        model="openai/gpt-oss-120b",   # or whatever is deployed
        max_tokens=200,                 # response budget (JSON ≈ 150 tokens)
        max_tokens_aggregation=1024,    # for episodic compression calls
        use_batch=True,                 # batch mode for throughput
        # model_max_context set from env or queried from vLLM at startup
    )
    return pool
```

### 9.2 Worker to Pool Assignment

For the pilot (1 worker, 1 pool): trivial.

For multi-worker deployments: all workers share a single `VLLMPool` (the pool's semaphore handles concurrency). The pool is created once by the coordinator and passed to workers.

Alternatively, partition endpoints across workers for complete isolation:

```python
# Worker 0 gets endpoints 0..K/W-1, Worker 1 gets K/W..2K/W-1, etc.
shard_endpoints = endpoints[shard_id * shard_size : (shard_id+1) * shard_size]
pool = VLLMPool(shard_endpoints, ...)
```

### 9.3 Fallback and Error Handling

```python
def fallback_decision(agent) -> dict:
    """Used when LLM call fails or returns unparseable JSON."""
    # Conservative: stay home if symptomatic, follow schedule otherwise
    if 'SYMPTOMATIC' in agent.disease_state or agent.disease_state == 'HOSPITALIZED':
        location = 'HOME' if 'SYMPTOMATIC' in agent.disease_state else 'HOSPITAL_OR_CLINIC'
    else:
        scheduled = db.get_schedule(agent.agent_id, current_hour_of_week)
        location = place_type_to_location(db.get_place(scheduled).place_type)
    return {
        "agent_id": agent.agent_id,
        "location": location,
        "reasoning": None,
        "mask_wearing": agent.fear_level > 0.5,
        "distancing": agent.fear_level > 0.5,
        "health_seeking": 'SEVERE' in agent.disease_state,
        "behavioral_deviation": None,
        "_fallback": True   # flag for diagnostics
    }
```

---

## 10. Activity Schedules

### 10.1 Archetype Approach (Pilot)

For 1K–10K agents, generate schedules from 6 archetypes rather than importing ATUS microdata:

```python
ARCHETYPES = {
    'essential_worker': {
        # (hour_of_day, place_type, day_types)
        'weekday': [(0,5,'HOME'), (5,6,'TRANSIT'), (6,14,'WORKPLACE'),
                    (14,15,'ESSENTIAL_RETAIL'), (15,16,'TRANSIT'),
                    (16,22,'HOME'), (22,24,'HOME')],
        'weekend': [(0,9,'HOME'), (9,11,'ESSENTIAL_RETAIL'), (11,22,'HOME'),
                    (22,24,'HOME')],
    },
    'remote_capable': {
        'weekday': [(0,8,'HOME'), (8,17,'HOME'), (17,22,'HOME'), (22,24,'HOME')],
        'weekend': [(0,9,'HOME'), (9,12,'PARK_OR_OUTDOOR'), (12,22,'HOME'),
                    (22,24,'HOME')],
    },
    'school_age': {
        'weekday': [(0,7,'HOME'), (7,8,'TRANSIT'), (8,15,'SCHOOL'),
                    (15,16,'TRANSIT'), (16,21,'HOME'), (21,24,'HOME')],
        'weekend': [(0,9,'HOME'), (9,12,'SOCIAL_VISIT'), (12,22,'HOME'),
                    (22,24,'HOME')],
    },
    'retired': {
        'weekday': [(0,8,'HOME'), (8,10,'ESSENTIAL_RETAIL'), (10,12,'PARK_OR_OUTDOOR'),
                    (12,22,'HOME'), (22,24,'HOME')],
        'weekend': [(0,9,'HOME'), (9,11,'SOCIAL_VISIT'), (11,22,'HOME'),
                    (22,24,'HOME')],
    },
    'student': {
        'weekday': [(0,8,'HOME'), (8,16,'SCHOOL'), (16,20,'HOME'), (20,24,'HOME')],
        'weekend': [(0,10,'HOME'), (10,14,'SOCIAL_VISIT'), (14,22,'HOME'),
                    (22,24,'HOME')],
    },
    'unemployed': {
        'all':     [(0,9,'HOME'), (9,12,'ESSENTIAL_RETAIL'), (12,22,'HOME'),
                    (22,24,'HOME')],
    },
}
```

Assign archetype by `occupation_type`. Assign specific `place_id` to each archetype slot from the place inventory (e.g., agent's `workplace_id` fills WORKPLACE slots).

### 10.2 Policy Filter

Applied at tick-time, not stored in the schedule:

```python
def apply_policy_filter(agent, scheduled_place_id, policy) -> int:
    place = db.get_place(scheduled_place_id)
    ptype = place.place_type

    # Hard overrides: hospitalized/deceased can't move
    if agent.disease_state in ('HOSPITALIZED', 'ICU'):
        return db.get_hospital_place_id()
    if agent.disease_state == 'DECEASED':
        return None

    # Policy closures
    if ptype == 'school' and not policy['schools_open']:
        return agent.household_id
    if ptype == 'restaurant' and not policy['restaurants_open']:
        return agent.household_id
    if policy['stay_at_home'] and ptype not in ('household','hospital','essential_retail'):
        # Essential workers exempt from stay-at-home
        if not (ptype == 'workplace' and agent.occupation_type == 'essential_worker'):
            return agent.household_id

    # Symptomatic self-isolation (schedule complies with LLM decision override)
    # LLM response supersedes this; this is only for non-LLM fallback agents
    if 'SYMPTOMATIC' in agent.disease_state:
        if random() < (0.4 + agent.fear_level * 0.4):
            return agent.household_id

    return scheduled_place_id
```

---

## 11. Synthetic Population Generation (Pilot)

For a 1K-agent pilot neighborhood, generate synthetically:

```python
import random

CHICAGO_NEIGHBORHOOD_DEMOGRAPHICS = {
    'age_distribution': [(0,17,0.22), (18,34,0.25), (35,54,0.28),
                         (55,74,0.18), (75,99,0.07)],
    'occupation_distribution': {
        'essential_worker': 0.28,
        'remote_capable':   0.22,
        'school_age':       0.18,
        'student':          0.08,
        'retired':          0.14,
        'unemployed':       0.10,
    },
    'avg_household_size': 2.8,
    'comorbidity_rate': 0.35,        # fraction with ≥1 comorbidity
    'can_wfh_rate': 0.45,
    'transit_use_rate': 0.38,
}

PLACE_RATIOS = {                     # places per 1,000 agents
    'household':       357,          # 1000 agents / 2.8 avg household size
    'workplace':        25,          # ~6 workers per workplace avg
    'school':            3,
    'essential_retail': 8,
    'park':              4,
    'hospital':          1,
    'restaurant':        6,
    'community_venue':   2,
}
```

---

## 12. Pilot (1K Agents) Implementation Notes

The pilot uses the full architecture above with these simplifications:

| Component | Full architecture | 1K Pilot |
|---|---|---|
| Database | Redis + Postgres | Python dicts |
| Workers | N sharded processes | Single async loop |
| Tick barrier | DB counter + coordinator | Sequential: tick → disease → repeat |
| Occupancy | DB table | `defaultdict(list)` rebuilt each tick |
| Schedules | DB table | Pre-built dict `agent_id → [place_id×168]` |
| Inboxes | DB table | Per-agent `deque(maxlen=5)` |
| Social graph | DB relationship | Dict `agent_id → [contact_ids]` |
| LLM pool | Shared VLLMPool | Single VLLMPool, any hostfile |
| Prefix caching | vLLM APC (automatic) | Same — no configuration needed |

Estimated wall-clock per simulated hour at 1K agents, 4 endpoints:

```
LLM inference: 1,000 prompts / (4 endpoints × 2 prompts/sec) = 125 sec
Orchestrator:  <1 sec (dict operations on 1K agents)
Disease engine:<1 sec
Total per tick: ~2 minutes
90-day run:    ~2,160 ticks × 2 min = ~72 hours   ← too slow for calibration

With 100 endpoints:
LLM inference: 1,000 / (100 × 2) = 5 sec
Total per tick: ~6 sec
90-day run:    ~3.6 hours   ← acceptable
```

For plumbing validation, run 1–3 simulated days (72 ticks). On 4 endpoints this completes in ~2.5 hours — fine for verifying the tick loop, prompt rendering, disease engine, and JSON parsing before scaling up.

---

## 13. Configuration Reference

```python
# sim_config.py

# ── Scale ────────────────────────────────────────────────────────────
N_AGENTS          = 1_000           # total agents
N_WORKERS         = 1               # orchestrator shards
HOSTFILE          = "agents.hostfile"

# ── Simulation time ──────────────────────────────────────────────────
SIM_START_DATE    = "2020-03-15"    # calendar anchor
N_SIM_DAYS        = 90
HOURS_PER_TICK    = 1

# ── LLM ─────────────────────────────────────────────────────────────
MODEL             = "openai/gpt-oss-120b"
MAX_TOKENS        = 200             # response cap
MAX_TOKENS_AGG    = 1024            # episodic compression calls
USE_BATCH         = True
PREFIX_CACHE      = True            # requires vLLM APC enabled on server

# ── Memory ───────────────────────────────────────────────────────────
COMPRESSION_INTERVAL_DAYS = 7      # episodic memory compression frequency
MAX_INBOX_MESSAGES        = 5      # unread messages retained per agent
MAX_PLACE_LOG_ENTRIES     = 3

# ── Disease engine ───────────────────────────────────────────────────
BASE_TRANSMISSION_PROB    = 0.06
ASYMPTOMATIC_FRACTION     = 0.35
ASYMPTOMATIC_MULT         = 0.50
INITIAL_INFECTED          = 5      # seed infections at sim start
RANDOM_SEED               = 42     # for disease engine stochastic draws only

# ── Behavioral state vector ──────────────────────────────────────────
FEAR_DECAY_PER_TICK               = 0.005
FEAR_RISE_HOUSEHOLD_SYMPTOMATIC   = 0.15
FEAR_RISE_CONTACT_SICK            = 0.08
FEAR_RISE_CONTACT_DIED            = 0.20
FATIGUE_RISE_PER_ISOLATED_DAY     = 0.03
FATIGUE_DECAY_PER_ACTIVE_DAY      = 0.05
PRESSURE_RISE_PER_MISSED_WORK_DAY = 0.04
TRUST_DECAY_ON_POLICY_CHANGE      = 0.05

# ── Initial policy state ─────────────────────────────────────────────
INITIAL_POLICY = {
    "schools_open":              True,
    "restaurants_open":          True,
    "stay_at_home":              False,
    "mask_mandate":              False,
    "essential_retail_capacity": 1.0,
}
```

---

## 14. File Layout (Proposed)

```
swarmsim/
├── sim_config.py            # Section 13 constants
├── db.py                    # Database abstraction (dict for pilot, SQL for scale)
├── population.py            # Synthetic population + place generation (Section 11)
├── schedules.py             # Archetype schedule generation (Section 10)
├── prompt.py                # Blocks A–D rendering (Section 5)
├── worker.py                # Tick execution loop (Section 6.1)
├── disease_engine.py        # State machine + transmission (Section 8)
├── communication.py         # Co-location context, inbox, place log (Section 7)
├── behavioral_state.py      # State vector update rules (Section 4.1)
├── memory.py                # Episodic compression (Section 4.2)
├── coordinator.py           # Barrier + disease engine orchestration (Section 6.2)
└── run_pilot.py             # Entry point: wire everything together, 1K pilot
```

---

*References: SwarmSim-Design.md (agent schema, prompt template, disease states), SwarmSim-Scaling.md (throughput analysis, operating points), Aurora Swarm README (VLLMPool, scatter_gather, batch mode).*
