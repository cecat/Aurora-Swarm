"""
population.py — Synthetic population and place inventory generator.

Default target: Logan Square, Chicago (Community Area 22).
Demographic distributions derived from 2020 ACS 5-year estimates for
Census tracts within Logan Square (tracts 2101–2108, Cook County, IL).

Public data sources used / available:
  Agents:  US Census ACS API  https://api.census.gov/data/2020/acs/acs5
           IPUMS PUMS          https://usa.ipums.org/usa/
           Cook County Health Atlas  https://cookcountyhealthatlas.org/
  Places:  OpenStreetMap via osmnx  https://github.com/gboeing/osmnx
           Chicago Data Portal       https://data.cityofchicago.org/
           TIGER/Line housing units  https://www.census.gov/geographies/mapping-files/
  Schedules: ATUS microdata         https://www.bls.gov/tus/data.htm

For the pilot, agent demographics are sampled from hardcoded ACS-derived
distributions so the script runs without any API keys or downloads.
Places are pulled from OpenStreetMap if osmnx is installed; otherwise a
fully synthetic place inventory is generated from the same ratios.

Usage:
    from swarmsim.population import generate_population
    agents, places = generate_population(n_agents=1000, seed=42)
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from typing import Optional

from .models import Agent, Place
from . import sim_config as cfg

# ── Optional OSM import ───────────────────────────────────────────────────────
try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False


# ═══════════════════════════════════════════════════════════════════════════════
# Logan Square demographic distributions
# Source: ACS 2020 5-year estimates, Census tracts 2101–2108, Cook County IL
# ═══════════════════════════════════════════════════════════════════════════════

# (age_min, age_max_inclusive, weight)
AGE_DISTRIBUTION = [
    (0,   4,  0.06),
    (5,  12,  0.10),
    (13, 17,  0.05),
    (18, 24,  0.10),
    (25, 34,  0.22),
    (35, 44,  0.17),
    (45, 54,  0.12),
    (55, 64,  0.09),
    (65, 74,  0.06),
    (75, 99,  0.03),
]

SEX_DISTRIBUTION = [("M", 0.49), ("F", 0.51)]

# ACS Hispanic origin + race categories
RACE_ETHNICITY_DISTRIBUTION = [
    ("White non-Hispanic",    0.49),
    ("Hispanic or Latino",    0.34),
    ("Black or African American", 0.08),
    ("Asian",                 0.06),
    ("Multiracial or other",  0.03),
]

# Population-level occupation type distribution
# Derived from ACS occupation + age data for Logan Square
OCCUPATION_DISTRIBUTION = [
    ("school_age",           0.21),   # ages 0–17 (school-age children)
    ("student",              0.06),   # ages 18–24 in full-time education
    ("essential_worker",     0.25),   # service, food, construction, transport
    ("remote_capable",       0.26),   # management, professional, office
    ("healthcare_worker",    0.04),   # clinical + support staff
    ("retired",              0.09),   # ages 65+ not working
    ("unemployed",           0.08),   # working-age, not employed
    ("nursing_home_resident",0.01),   # long-term care residents
]

# Comorbidity prevalence (independent, Cook County / Logan Square estimates)
# Source: Cook County Health Atlas, CDC PLACES 2022
COMORBIDITY_PREVALENCE = {
    "diabetes":         0.08,
    "hypertension":     0.22,
    "obesity":          0.28,
    "heart_disease":    0.04,
    "copd":             0.04,
    "immunocompromised":0.02,
}

# can_wfh by occupation
CAN_WFH_RATE = {
    "essential_worker":     0.00,
    "remote_capable":       0.95,
    "student":              0.50,   # hybrid classes
    "school_age":           0.00,
    "retired":              0.00,
    "unemployed":           0.00,
    "healthcare_worker":    0.05,
    "nursing_home_resident":0.00,
}

# uses_transit (Logan Square: ~38% overall)
TRANSIT_USE_RATE = {
    "essential_worker":     0.45,
    "remote_capable":       0.35,
    "student":              0.55,
    "school_age":           0.30,
    "retired":              0.25,
    "unemployed":           0.30,
    "healthcare_worker":    0.40,
    "nursing_home_resident":0.00,
}

# healthcare_access: 0–1 (probability of seeking care when symptomatic)
# Varies by income proxy (occupation)
HEALTHCARE_ACCESS_PARAMS = {
    "essential_worker":     (0.55, 0.15),   # (mean, std)
    "remote_capable":       (0.80, 0.10),
    "student":              (0.65, 0.15),
    "school_age":           (0.75, 0.10),
    "retired":              (0.70, 0.10),
    "unemployed":           (0.40, 0.15),
    "healthcare_worker":    (0.90, 0.05),
    "nursing_home_resident":(0.85, 0.05),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic name lists (reflects Logan Square demographic mix)
# Polish, Spanish, English, and mixed-heritage names
# ═══════════════════════════════════════════════════════════════════════════════

FIRST_NAMES_M = [
    "James", "Miguel", "Carlos", "David", "Kevin", "Marcus", "Daniel", "Luis",
    "Robert", "Anthony", "José", "Eric", "Brandon", "Jordan", "Alejandro",
    "Michael", "Christopher", "Jason", "Ryan", "Patrick", "Tomasz", "Andrzej",
    "William", "Matthew", "Andrew", "Joshua", "Nathan", "Brian", "Kevin", "Sean",
    "Diego", "Ricardo", "Fernando", "Eduardo", "Manuel", "Rafael", "Gabriel",
    "Isaiah", "Derrick", "Malik", "Tyrone", "Jerome", "DeAndre", "Jamal",
    "Kevin", "Piotr", "Krzysztof", "Marek", "Adam", "Łukasz", "Michał",
]

FIRST_NAMES_F = [
    "Maria", "Jennifer", "Ashley", "Jessica", "Sarah", "Lisa", "Angela",
    "Michelle", "Patricia", "Sandra", "Gabriela", "Valentina", "Alejandra",
    "Isabella", "Sofia", "Camila", "Daniela", "Mariana", "Lucia", "Rosa",
    "Aisha", "Keisha", "Tamara", "Latoya", "Shanice", "Destiny", "Brianna",
    "Emily", "Megan", "Lauren", "Rebecca", "Katherine", "Amanda", "Rachel",
    "Anna", "Magdalena", "Katarzyna", "Agnieszka", "Monika", "Joanna",
    "Christine", "Stephanie", "Nicole", "Heather", "Tiffany", "Crystal",
    "Yolanda", "Esperanza", "Marisol", "Xiomara", "Yesenia",
]

LAST_NAMES = [
    "Garcia", "Martinez", "Rodriguez", "Lopez", "Hernandez", "Gonzalez",
    "Rivera", "Ramirez", "Torres", "Flores", "Cruz", "Reyes", "Morales",
    "Smith", "Johnson", "Williams", "Brown", "Davis", "Miller", "Wilson",
    "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Thompson", "Robinson", "Clark", "Lewis", "Lee", "Walker",
    "Kowalski", "Nowak", "Wiśniewski", "Wójcik", "Kowalczyk", "Kaminski",
    "Lewandowski", "Zielinski", "Szymanski", "Woźniak", "Dąbrowski",
    "Washington", "Jefferson", "Coleman", "Jenkins", "Brooks", "Richardson",
    "Chen", "Kim", "Nguyen", "Patel", "Singh", "Ali", "Khan",
    "O'Brien", "Murphy", "Sullivan", "McCarthy", "Kelly", "Walsh",
]

NEIGHBORHOODS_LOGAN_SQUARE = [
    "Logan Square", "Palmer Square", "Bucktown (east edge)",
    "Avondale (south edge)", "Humboldt Park (west edge)",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Place inventory
# ═══════════════════════════════════════════════════════════════════════════════

# Places per 1,000 agents (rounded; households computed separately from HH size)
PLACE_RATIOS_PER_1000 = {
    "workplace":        30,    # ~6 workers/workplace average
    "school":            4,    # elementary, middle, high, community college
    "essential_retail": 10,    # grocery, pharmacy, hardware
    "hospital":          2,    # hospital + urgent care clinic
    "restaurant":       18,    # indoor dining + fast food
    "park":              5,    # neighborhood parks + playlots
    "transit":           6,    # CTA stops / bus stops
    "community_venue":   4,    # churches, community centers, libraries
    "nursing_home":      1,
}

# Capacity ranges (min, max) by place type
CAPACITY_RANGES = {
    "household":        (1,   6),
    "workplace":        (5,  40),
    "school":           (200, 700),
    "essential_retail": (8,  35),
    "hospital":         (80, 400),
    "restaurant":       (20,  80),
    "park":             (50, 500),
    "transit":          (20, 100),
    "community_venue":  (30, 200),
    "nursing_home":     (30, 120),
}

# Ventilation by place type
VENTILATION = {
    "household":        "medium",
    "workplace":        "medium",
    "school":           "medium",
    "essential_retail": "medium",
    "hospital":         "high",
    "restaurant":       "low",
    "park":             "outdoor",
    "transit":          "low",
    "community_venue":  "medium",
    "nursing_home":     "medium",
}

# Natural language label templates
LABEL_TEMPLATES = {
    "household":        "household ({size}-person, {zip})",
    "workplace":        "{adj} workplace (~{cap} workers/visitors, {vent} ventilation)",
    "school":           "{level} school (~{cap} students, indoor)",
    "essential_retail": "{retail_type} (~{cap} staff + customers, {vent} ventilation)",
    "hospital":         "{hosp_type} (high infection-control, PPE available)",
    "restaurant":       "restaurant/bar (indoor dining, ~{cap} occupancy, low ventilation)",
    "park":             "outdoor park (~{cap} typical visitors, outdoor)",
    "transit":          "CTA {transit_type} stop (enclosed, low ventilation)",
    "community_venue":  "{venue_type} (~{cap} capacity, {vent} ventilation)",
    "nursing_home":     "long-term care facility (~{cap} residents, restricted access)",
}

WORKPLACE_ADJ    = ["office", "warehouse", "retail", "food-service", "industrial",
                    "small-business", "commercial"]
SCHOOL_LEVELS    = ["elementary", "middle", "high", "K-8", "charter"]
RETAIL_TYPES     = ["grocery store", "pharmacy", "convenience store",
                    "hardware store", "dollar store"]
HOSP_TYPES       = ["hospital emergency department", "urgent care clinic",
                    "community health clinic"]
TRANSIT_TYPES    = ["Blue Line", "bus"]
VENUE_TYPES      = ["church/place of worship", "community center", "public library",
                    "neighborhood association hall"]


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _weighted_choice(choices: list[tuple]) -> object:
    """choices = [(value, weight), ...]. Weights need not sum to 1."""
    values, weights = zip(*choices)
    return random.choices(values, weights=weights, k=1)[0]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _sample_age(distribution: list[tuple]) -> int:
    band = _weighted_choice([(b, w) for b, w in
                             [((lo, hi), w) for lo, hi, w in distribution]])
    return random.randint(band[0], band[1])


def _age_risk_mult(age: int) -> float:
    for (lo, hi), mult in cfg.AGE_RISK_MULT.items():
        if lo <= age <= hi:
            return mult
    return 1.0


def _sample_comorbidities(age: int) -> list[str]:
    result = []
    for condition, base_prev in COMORBIDITY_PREVALENCE.items():
        # Comorbidity risk scales with age; children rarely have these conditions
        if age < 18:
            age_scale = 0.05
        elif age < 40:
            age_scale = 0.40
        elif age < 60:
            age_scale = 1.00
        else:
            age_scale = 2.20
        if random.random() < min(1.0, base_prev * age_scale):
            result.append(condition)
    return result


def _sample_healthcare_access(occupation: str) -> float:
    mean, std = HEALTHCARE_ACCESS_PARAMS[occupation]
    return _clamp(random.gauss(mean, std))


def _synthetic_name(sex: str) -> str:
    first = random.choice(FIRST_NAMES_M if sex == "M" else FIRST_NAMES_F)
    last  = random.choice(LAST_NAMES)
    return f"{first} {last}"


# ═══════════════════════════════════════════════════════════════════════════════
# Place generation
# ═══════════════════════════════════════════════════════════════════════════════

def _make_label(place_type: str, capacity: int, zip_code: str,
                household_size: int = 0) -> str:
    """Render a natural-language place label for LLM injection."""
    vent = VENTILATION[place_type]
    if place_type == "household":
        return LABEL_TEMPLATES["household"].format(
            size=household_size, zip=zip_code)
    if place_type == "workplace":
        return LABEL_TEMPLATES["workplace"].format(
            adj=random.choice(WORKPLACE_ADJ), cap=capacity, vent=vent)
    if place_type == "school":
        return LABEL_TEMPLATES["school"].format(
            level=random.choice(SCHOOL_LEVELS), cap=capacity)
    if place_type == "essential_retail":
        return LABEL_TEMPLATES["essential_retail"].format(
            retail_type=random.choice(RETAIL_TYPES), cap=capacity, vent=vent)
    if place_type == "hospital":
        return LABEL_TEMPLATES["hospital"].format(
            hosp_type=random.choice(HOSP_TYPES))
    if place_type == "restaurant":
        return LABEL_TEMPLATES["restaurant"].format(cap=capacity)
    if place_type == "park":
        return LABEL_TEMPLATES["park"].format(cap=capacity)
    if place_type == "transit":
        return LABEL_TEMPLATES["transit"].format(
            transit_type=random.choice(TRANSIT_TYPES))
    if place_type == "community_venue":
        return LABEL_TEMPLATES["community_venue"].format(
            venue_type=random.choice(VENUE_TYPES), cap=capacity, vent=vent)
    if place_type == "nursing_home":
        return LABEL_TEMPLATES["nursing_home"].format(cap=capacity)
    return f"{place_type} (capacity {capacity})"


def _generate_synthetic_places(n_agents: int, avg_hh_size: float,
                                zip_code: str) -> list[Place]:
    """
    Generate a synthetic place inventory scaled to n_agents.
    Used when osmnx is unavailable or the caller requests it explicitly.
    """
    places: list[Place] = []
    pid = 0

    # Households
    n_households = math.ceil(n_agents / avg_hh_size)
    for _ in range(n_households):
        hh_size = max(1, round(random.gauss(avg_hh_size, 1.0)))
        cap = max(1, min(hh_size, 6))
        places.append(Place(
            place_id=pid,
            place_type="household",
            label=_make_label("household", cap, zip_code, household_size=cap),
            zip_code=zip_code,
            capacity=cap,
            ventilation="medium",
        ))
        pid += 1

    # All other place types from ratios
    for ptype, ratio in PLACE_RATIOS_PER_1000.items():
        n_places = max(1, round(ratio * n_agents / 1000))
        lo, hi   = CAPACITY_RANGES[ptype]
        for _ in range(n_places):
            cap = random.randint(lo, hi)
            places.append(Place(
                place_id=pid,
                place_type=ptype,
                label=_make_label(ptype, cap, zip_code),
                zip_code=zip_code,
                capacity=cap,
                ventilation=VENTILATION[ptype],
            ))
            pid += 1

    return places


def _generate_osmnx_places(neighborhood: str, zip_code: str,
                            n_agents: int, avg_hh_size: float) -> list[Place]:
    """
    Pull real POI locations from OpenStreetMap for the target neighborhood,
    supplement with synthetic households. Falls back to fully synthetic on error.
    """
    if not HAS_OSMNX:
        print("[population] osmnx not available — using synthetic places.")
        return _generate_synthetic_places(n_agents, avg_hh_size, zip_code)

    query = f"{neighborhood}, Chicago, Illinois, USA"
    osm_tag_map = {
        "essential_retail": [
            {"shop": "supermarket"}, {"shop": "convenience"},
            {"shop": "pharmacy"}, {"amenity": "pharmacy"},
        ],
        "school":           [{"amenity": "school"}],
        "hospital":         [{"amenity": "hospital"}, {"amenity": "clinic"},
                             {"amenity": "doctors"}],
        "restaurant":       [{"amenity": "restaurant"}, {"amenity": "fast_food"},
                             {"amenity": "bar"}],
        "park":             [{"leisure": "park"}, {"leisure": "playground"}],
        "transit":          [{"public_transport": "stop_position"},
                             {"highway": "bus_stop"}],
        "community_venue":  [{"amenity": "place_of_worship"},
                             {"amenity": "community_centre"},
                             {"amenity": "library"}],
    }

    places: list[Place] = []
    pid = 0

    print(f"[population] Querying OpenStreetMap for {query} ...")
    for ptype, tag_list in osm_tag_map.items():
        for tags in tag_list:
            try:
                gdf = ox.features_from_place(query, tags=tags)
                for _, row in gdf.iterrows():
                    lo, hi = CAPACITY_RANGES[ptype]
                    cap    = random.randint(lo, hi)
                    name   = row.get("name", "")
                    label  = f"{name} — " if name else ""
                    label += _make_label(ptype, cap, zip_code)
                    places.append(Place(
                        place_id=pid,
                        place_type=ptype,
                        label=label,
                        zip_code=zip_code,
                        capacity=cap,
                        ventilation=VENTILATION[ptype],
                        lat=row.geometry.centroid.y if row.geometry else None,
                        lon=row.geometry.centroid.x if row.geometry else None,
                    ))
                    pid += 1
            except Exception as e:
                print(f"[population] OSM query failed for {ptype}/{tags}: {e}")

    # Workplaces and nursing homes are not well-represented in OSM — generate synthetically
    for ptype in ("workplace", "nursing_home"):
        ratio  = PLACE_RATIOS_PER_1000[ptype]
        n      = max(1, round(ratio * n_agents / 1000))
        lo, hi = CAPACITY_RANGES[ptype]
        for _ in range(n):
            cap = random.randint(lo, hi)
            places.append(Place(
                place_id=pid,
                place_type=ptype,
                label=_make_label(ptype, cap, zip_code),
                zip_code=zip_code,
                capacity=cap,
                ventilation=VENTILATION[ptype],
            ))
            pid += 1

    # Households — always synthetic
    n_households = math.ceil(n_agents / avg_hh_size)
    for _ in range(n_households):
        hh_size = max(1, round(random.gauss(avg_hh_size, 1.0)))
        cap     = max(1, min(hh_size, 6))
        places.append(Place(
            place_id=pid,
            place_type="household",
            label=_make_label("household", cap, zip_code, household_size=cap),
            zip_code=zip_code,
            capacity=cap,
            ventilation="medium",
        ))
        pid += 1

    print(f"[population] Generated {len(places)} places "
          f"({sum(1 for p in places if p.place_type == 'household')} households "
          f"from OSM + synthetic).")
    return places


# ═══════════════════════════════════════════════════════════════════════════════
# Agent generation
# ═══════════════════════════════════════════════════════════════════════════════

def _assign_occupation(age: int) -> str:
    """
    Occupation is partly age-determined, partly drawn from the distribution.
    Hard rules take priority; remaining population is distributed by OCCUPATION_DISTRIBUTION.
    """
    if age < 5:
        return "school_age"     # pre-school; tracked but behaviorally like school_age
    if 5 <= age <= 17:
        return "school_age"
    if age >= 75:
        # Most are retired; small fraction nursing home
        return "nursing_home_resident" if random.random() < 0.10 else "retired"
    if 65 <= age < 75:
        return "retired" if random.random() < 0.85 else "essential_worker"

    # Working-age adults: sample from conditional distribution
    # (school_age and retired already handled above)
    working_age_dist = [
        (occ, w) for occ, w in OCCUPATION_DISTRIBUTION
        if occ not in ("school_age", "retired", "nursing_home_resident")
    ]
    # Normalise
    total = sum(w for _, w in working_age_dist)
    working_age_dist = [(occ, w / total) for occ, w in working_age_dist]
    return _weighted_choice(working_age_dist)


def _generate_agents(n_agents: int, places: list[Place]) -> list[Agent]:
    """Generate n_agents synthetic Chicago residents."""

    # Index places by type for assignment
    by_type: dict[str, list[Place]] = defaultdict(list)
    for p in places:
        by_type[p.place_type].append(p)

    households     = by_type["household"]
    workplaces     = by_type["workplace"]
    schools        = by_type["school"]
    nursing_homes  = by_type["nursing_home"]

    if not households:
        raise ValueError("No household places generated — cannot assign agents.")

    agents: list[Agent] = []

    # Build household roster first so household_size is accurate
    # Assign agents to households in order until each household reaches capacity
    hh_rosters: dict[int, list[int]] = defaultdict(list)  # hh_id → [agent_ids]
    hh_cycle    = 0
    hh_sizes    = {p.place_id: p.capacity for p in households}

    for agent_id in range(n_agents):
        # Find a household with room; cycle if all full (rare at exact capacity)
        while True:
            hh = households[hh_cycle % len(households)]
            if len(hh_rosters[hh.place_id]) < hh_sizes[hh.place_id]:
                hh_rosters[hh.place_id].append(agent_id)
                assigned_hh = hh
                break
            hh_cycle += 1
            if hh_cycle > n_agents * 2:
                # Safety: just use next household
                assigned_hh = households[agent_id % len(households)]
                hh_rosters[assigned_hh.place_id].append(agent_id)
                break
        hh_cycle += 1

        # Demographics
        age  = _sample_age(AGE_DISTRIBUTION)
        sex  = _weighted_choice(SEX_DISTRIBUTION)
        race = _weighted_choice(RACE_ETHNICITY_DISTRIBUTION)
        occ  = _assign_occupation(age)

        # Place assignments
        workplace_id: Optional[int] = None
        school_id:    Optional[int] = None

        if occ == "school_age":
            if schools:
                school_id = random.choice(schools).place_id
        elif occ == "student":
            if schools:
                school_id = random.choice(schools).place_id
            if workplaces and random.random() < 0.35:   # part-time workers
                workplace_id = random.choice(workplaces).place_id
        elif occ == "nursing_home_resident":
            if nursing_homes:
                # nursing home IS their home for scheduling purposes
                workplace_id = random.choice(nursing_homes).place_id
        elif occ in ("essential_worker", "remote_capable",
                     "healthcare_worker", "unemployed"):
            if workplaces and occ != "unemployed":
                workplace_id = random.choice(workplaces).place_id
        # retired: no workplace or school

        comorbidities    = _sample_comorbidities(age)
        healthcare_access = _sample_healthcare_access(occ)
        can_wfh          = random.random() < CAN_WFH_RATE.get(occ, 0.0)
        uses_transit     = random.random() < TRANSIT_USE_RATE.get(occ, 0.30)
        fin_pressure     = cfg.FINANCIAL_PRESSURE_BASELINE.get(occ, 0.30)

        agent = Agent(
            agent_id          = agent_id,
            synthetic_name    = _synthetic_name(sex),
            age               = age,
            sex               = sex,
            race_ethnicity    = race,
            zip_code          = cfg.NEIGHBORHOOD_ZIP,
            neighborhood      = random.choice(NEIGHBORHOODS_LOGAN_SQUARE),
            household_id      = assigned_hh.place_id,
            household_size    = hh_sizes[assigned_hh.place_id],
            occupation_type   = occ,
            workplace_id      = workplace_id,
            school_id         = school_id,
            can_wfh           = can_wfh,
            uses_transit      = uses_transit,
            comorbidities     = comorbidities,
            healthcare_access = healthcare_access,
            age_risk_mult     = _age_risk_mult(age),
            financial_pressure = _clamp(random.gauss(fin_pressure, 0.10)),
            current_place_id  = assigned_hh.place_id,
        )
        agents.append(agent)

    return agents


# ═══════════════════════════════════════════════════════════════════════════════
# Social contact network
# ═══════════════════════════════════════════════════════════════════════════════

def _build_social_network(agents: list[Agent]) -> None:
    """
    Assign social contacts to each agent in-place.
    Three tiers:
      1. Household members (automatic, bidirectional)
      2. Workplace / school peers (2–4 per agent)
      3. Neighborhood acquaintances (1–2 per agent)
    """
    # Index by household and workplace/school
    hh_index: dict[int, list[int]]  = defaultdict(list)
    wp_index: dict[int, list[int]]  = defaultdict(list)

    for a in agents:
        hh_index[a.household_id].append(a.agent_id)
        if a.workplace_id is not None:
            wp_index[a.workplace_id].append(a.agent_id)
        if a.school_id is not None:
            wp_index[a.school_id].append(a.agent_id)

    all_ids = [a.agent_id for a in agents]

    for agent in agents:
        contacts: set[int] = set()

        # Tier 1: household members
        for mid in hh_index[agent.household_id]:
            if mid != agent.agent_id:
                contacts.add(mid)

        # Tier 2: workplace / school peers (sample up to 4)
        peer_pool_id = agent.workplace_id or agent.school_id
        if peer_pool_id is not None:
            peers = [aid for aid in wp_index[peer_pool_id]
                     if aid != agent.agent_id]
            contacts.update(random.sample(peers, min(4, len(peers))))

        # Tier 3: random neighborhood acquaintances (1–2)
        n_acquaint = random.randint(1, 2)
        candidates = [aid for aid in all_ids
                      if aid != agent.agent_id and aid not in contacts]
        if candidates:
            contacts.update(random.sample(candidates,
                                          min(n_acquaint, len(candidates))))

        agent.social_contacts = list(contacts)[: cfg.SOCIAL_CONTACTS_MAX]


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate_population(
    n_agents:    int  = cfg.N_AGENTS,
    seed:        int  = cfg.RANDOM_SEED,
    neighborhood: str = cfg.NEIGHBORHOOD,
    zip_code:    str  = cfg.NEIGHBORHOOD_ZIP,
    avg_hh_size: float = 2.6,
    use_osmnx:   bool  = True,
) -> tuple[list[Agent], list[Place]]:
    """
    Generate a synthetic population and place inventory.

    Returns:
        agents: list[Agent]   — all synthetic residents
        places: list[Place]   — all venues (households, workplaces, etc.)

    Place IDs are stable integers. Agent household_id and workplace_id
    reference place_id values in the places list.
    """
    random.seed(seed)

    print(f"[population] Generating {n_agents:,} agents for {neighborhood}, Chicago ...")

    # 1. Places
    if use_osmnx and HAS_OSMNX:
        places = _generate_osmnx_places(neighborhood, zip_code, n_agents, avg_hh_size)
    else:
        places = _generate_synthetic_places(n_agents, avg_hh_size, zip_code)

    print(f"[population] Place inventory: {len(places):,} places "
          f"({len([p for p in places if p.place_type == 'household'])} households)")

    # 2. Agents
    agents = _generate_agents(n_agents, places)

    # 3. Social contact network
    _build_social_network(agents)

    # 4. Summary stats
    _print_summary(agents, places)

    return agents, places


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def _print_summary(agents: list[Agent], places: list[Place]) -> None:
    n = len(agents)
    occ_counts: dict[str, int] = defaultdict(int)
    for a in agents:
        occ_counts[a.occupation_type] += 1

    place_counts: dict[str, int] = defaultdict(int)
    for p in places:
        place_counts[p.place_type] += 1

    print("\n── Population summary ────────────────────────────────")
    print(f"  Agents : {n:,}")
    print(f"  Places : {len(places):,}")
    print(f"  Age    : mean={sum(a.age for a in agents)/n:.1f}  "
          f"range={min(a.age for a in agents)}–{max(a.age for a in agents)}")
    print(f"  Comorbidity (≥1): "
          f"{sum(1 for a in agents if a.comorbidities)/n*100:.1f}%")
    print(f"\n  Occupation breakdown:")
    for occ, count in sorted(occ_counts.items(), key=lambda x: -x[1]):
        print(f"    {occ:<28} {count:>5}  ({count/n*100:.1f}%)")
    print(f"\n  Place breakdown:")
    for ptype, count in sorted(place_counts.items(), key=lambda x: -x[1]):
        print(f"    {ptype:<20} {count:>5}")

    contact_lens = [len(a.social_contacts) for a in agents]
    print(f"\n  Social contacts: mean={sum(contact_lens)/n:.1f}  "
          f"min={min(contact_lens)}  max={max(contact_lens)}")
    print("──────────────────────────────────────────────────────\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    agents, places = generate_population(
        n_agents=1_000,
        seed=42,
        use_osmnx=False,    # set True if osmnx is installed
    )
    # Spot-check a few agents
    for a in agents[:3]:
        print(f"  {a.synthetic_name}, age {a.age}, {a.occupation_type}, "
              f"HH={a.household_id}, WP={a.workplace_id}, "
              f"contacts={len(a.social_contacts)}, "
              f"comorbidities={a.comorbidities}")
    print(f"\nFirst place: {places[0].label}")
