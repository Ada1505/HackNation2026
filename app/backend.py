"""
MediAlert — Databricks Backend
================================
Runs entirely from local Python. Connects to Databricks cloud services
using only HTTP (requests + openai). No Spark, no dbutils, no notebooks.

Provides three capabilities:
  1. db_sql()        — run SQL on Databricks SQL Warehouse
  2. vector_search() — semantic search via Mosaic AI Vector Search REST API
  3. run_agent()     — multi-step reasoning via Databricks Foundation Models

Import in Streamlit:  from backend import vector_search, run_agent, db_sql
"""

import os, re, json, time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# MODULE-LEVEL SQL CACHE  (avoids repeated warehouse hits)
# Key = sql string, Value = (rows, timestamp)
# ─────────────────────────────────────────────────────────────
_SQL_CACHE: dict = {}
_SQL_CACHE_TTL   = 300  # seconds before a cached result expires

def _cache_get(sql: str):
    entry = _SQL_CACHE.get(sql)
    if entry and (time.time() - entry[1]) < _SQL_CACHE_TTL:
        return entry[0]
    return None

def _cache_set(sql: str, rows: list):
    _SQL_CACHE[sql] = (rows, time.time())

# ── MLflow 3 tracing setup ────────────────────────────────────
try:
    import mlflow
    from mlflow.entities import SpanType
    # Use Databricks-hosted MLflow when credentials available
    _db_host  = os.environ.get("DATABRICKS_HOST", "")
    _db_token = os.environ.get("DATABRICKS_TOKEN", "")
    if _db_host and _db_token:
        mlflow.set_tracking_uri("databricks")
        os.environ.setdefault("DATABRICKS_HOST",  _db_host)
        os.environ.setdefault("DATABRICKS_TOKEN", _db_token)
    else:
        mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("MediAlert-Agent")
    _MLFLOW = True
except Exception:
    _MLFLOW = False

# ─────────────────────────────────────────────────────────────
# CONFIG  (all come from .env)
# ─────────────────────────────────────────────────────────────
DATABRICKS_HOST   = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN  = os.environ.get("DATABRICKS_TOKEN", "")
DATABRICKS_SQL_WH = os.environ.get("DATABRICKS_SQL_WH", "")

# Vector Search
VS_ENDPOINT  = os.environ.get("VS_ENDPOINT",  "vs-readyalert-endpoint")
VS_INDEX     = os.environ.get("VS_INDEX",     "workspace.default.facilities_for_embedding_index")

# LLM (Databricks Foundation Model, OpenAI-compatible)
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")

# Table names (must match your Databricks Unity Catalog)
SQL_TABLE   = "workspace.default.facilities_sql"
EMBED_TABLE = "workspace.default.facilities_for_embedding"

# ─────────────────────────────────────────────────────────────
# VS INDEX AUTO-DISCOVERY
# Queries the endpoint to find the real index name at startup.
# This fixes "index does not exist" errors caused by wrong VS_INDEX.
# ─────────────────────────────────────────────────────────────
def _discover_vs_index() -> str:
    """
    Ask Databricks which indexes exist on VS_ENDPOINT.
    Returns the first ready index name, or VS_INDEX as-is if discovery fails.
    """
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN or not VS_ENDPOINT:
        return VS_INDEX
    try:
        r = requests.get(
            f"{DATABRICKS_HOST}/api/2.0/vector-search/endpoints/{VS_ENDPOINT}/indexes",
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            timeout=12,
        )
        if not r.ok:
            print(f"[VS-discover] {r.status_code}: {r.text[:200]}")
            return VS_INDEX
        data    = r.json()
        indexes = data.get("vector_indexes", data.get("indexes", []))
        if not indexes:
            print(f"[VS-discover] No indexes found on endpoint '{VS_ENDPOINT}'")
            return VS_INDEX
        # Prefer indexes with 'facilities' in the name and that are ready
        for idx in indexes:
            name   = idx.get("name", "")
            status = str(idx.get("status", {}).get("ready_for_queries", "")).lower()
            print(f"[VS-discover] found index: {name!r}  ready={status}")
            if "facilit" in name.lower() and status == "true":
                print(f"[VS-discover] ✅ using: {name!r}")
                return name
        # Fall back to first index found
        first = indexes[0].get("name", VS_INDEX)
        print(f"[VS-discover] using first index: {first!r}")
        return first
    except Exception as e:
        print(f"[VS-discover] exception: {e}")
        return VS_INDEX

# Run discovery once at import time (result cached in module-level var)
_RESOLVED_VS_INDEX: str = VS_INDEX  # will be overwritten below after helpers defined


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _headers() -> dict:
    return {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type":  "application/json",
    }

def is_configured() -> bool:
    """True when the minimum Databricks credentials are present."""
    return bool(DATABRICKS_HOST and DATABRICKS_TOKEN)

def is_vs_configured() -> bool:
    """True when Vector Search vars are present."""
    return is_configured() and bool(VS_ENDPOINT)

def is_llm_configured() -> bool:
    """True when LLM endpoint is reachable (same token as DB)."""
    return is_configured() and bool(LLM_ENDPOINT)


# ─────────────────────────────────────────────────────────────
# 1. SQL QUERIES  (Databricks SQL Warehouse REST API)
# ─────────────────────────────────────────────────────────────
def db_sql(sql: str, timeout: int = 30) -> list[dict]:
    """
    Execute SQL on Databricks SQL Warehouse.
    Returns list of row dicts, or empty list on error.
    Requires valid credentials — no fallback data.
    """
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN or not DATABRICKS_SQL_WH:
        print("[db_sql] Missing credentials — DATABRICKS_HOST / TOKEN / SQL_WH not set")
        return []

    h = _headers()
    # Auto-start warehouse if stopped
    try:
        state = requests.get(
            f"{DATABRICKS_HOST}/api/2.0/sql/warehouses/{DATABRICKS_SQL_WH}",
            headers=h, timeout=10,
        ).json().get("state", "")
        if state in ("STOPPED", "STOPPING"):
            requests.post(f"{DATABRICKS_HOST}/api/2.0/sql/warehouses/{DATABRICKS_SQL_WH}/start",
                         headers=h, timeout=10)
            for _ in range(24):
                time.sleep(5)
                state = requests.get(
                    f"{DATABRICKS_HOST}/api/2.0/sql/warehouses/{DATABRICKS_SQL_WH}",
                    headers=h, timeout=10,
                ).json().get("state", "")
                if state == "RUNNING":
                    break
    except Exception:
        pass

    # Check cache first
    cached = _cache_get(sql)
    if cached is not None:
        print(f"[db_sql] cache hit ({len(cached)} rows)")
        return cached

    try:
        r = requests.post(
            f"{DATABRICKS_HOST}/api/2.0/sql/statements",
            headers=h,
            json={"warehouse_id": DATABRICKS_SQL_WH, "statement": sql,
                  "wait_timeout": f"{timeout}s", "on_wait_timeout": "CANCEL",
                  "format": "JSON_ARRAY"},
            timeout=timeout + 5,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status", {}).get("state") != "SUCCEEDED":
            err = data.get("status",{}).get("error",{})
            print(f"[db_sql] FAILED: {err}")
            return []
        cols = [c["name"] for c in data.get("manifest",{}).get("schema",{}).get("columns",[])]
        rows = [dict(zip(cols, row)) for row in data.get("result",{}).get("data_array",[])]
        print(f"[db_sql] OK — {len(rows)} rows, cols={cols[:5]}")
        _cache_set(sql, rows)
        return rows
    except Exception as e:
        print(f"[db_sql] exception: {e}")
        return []


# Run VS index discovery now that _headers() is defined.
# We update _RESOLVED_VS_INDEX in the module's global namespace.
try:
    _RESOLVED_VS_INDEX = _discover_vs_index()
    print(f"[VS-discover] active index → {_RESOLVED_VS_INDEX!r}")
except Exception:
    _RESOLVED_VS_INDEX = VS_INDEX  # keep default if discovery itself crashes


# ─────────────────────────────────────────────────────────────
# 2. VECTOR SEARCH  (Mosaic AI Vector Search REST API)
# ─────────────────────────────────────────────────────────────
def _vs_query(url: str, payload: dict, timeout: int = 25) -> list[dict]:
    """
    Fire one Databricks Vector Search REST call.
    Handles manifest→col_names mapping and returns parsed row dicts.
    Returns [] on any error (caller decides what to do next).
    """
    try:
        r = requests.post(url, headers=_headers(), json=payload, timeout=timeout)
        if not r.ok:
            print(f"[VS] HTTP {r.status_code}: {r.text[:300]}")
            return []
        data = r.json()
        hits = data.get("result", {}).get("data_array", [])
        # Columns live in manifest, NOT in result
        col_names = [c["name"] for c in data.get("manifest", {}).get("columns", [])]
        print(f"[VS] {len(hits)} hits  cols={col_names}")
        if not col_names and hits:
            # Positional fallback — matches the columns we requested
            col_names = ["name", "notes_blob"]
            print("[VS] WARNING: manifest empty, using positional fallback")
        return [dict(zip(col_names, row)) for row in hits]
    except Exception as e:
        print(f"[VS] exception: {e}")
        return []


def _sql_keyword_search(
    query:        str,
    top_k:        int = 20,
    state_filter: Optional[str] = None,
    city_filter:  Optional[str] = None,
) -> list[dict]:
    """
    SQL LIKE keyword fallback — used when VS index is unavailable or returns 0 hits.
    Searches notes_blob for meaningful terms from the query.
    Always returns real Databricks data.
    """
    # Extract content keywords (drop short stop-words)
    stop = {"find","the","in","a","an","and","or","for","with","near","of","to","is","that","can","at"}
    raw_terms = re.findall(r"[a-zA-Z]{4,}", query)
    terms = [t for t in raw_terms if t.lower() not in stop][:5]
    if not terms:
        terms = raw_terms[:3]
    if not terms:
        terms = ["hospital"]

    # Build LIKE clauses — OR across terms so we cast a wide net
    note_clauses = " OR ".join(
        f"lower(e.notes_blob) LIKE lower('%{re.sub(chr(39), chr(39)*2, t)[:40]}%')"
        for t in terms
    )

    geo_clauses = []
    if state_filter:
        sf = re.sub(r"['\";\\]", "", state_filter)[:40]
        geo_clauses.append(f"lower(s.state) LIKE lower('%{sf}%')")
    if city_filter:
        cf = re.sub(r"['\";\\]", "", city_filter)[:40]
        geo_clauses.append(f"lower(s.city) LIKE lower('%{cf}%')")

    geo_where = ("AND (" + " OR ".join(geo_clauses) + ")") if geo_clauses else ""

    rows = db_sql(f"""
        SELECT s.name, s.city, s.state, s.pin_code, s.address_line1,
               s.facilityTypeId AS facility_type, s.num_doctors, s.capacity,
               s.officialPhone  AS phone,
               s.composite_trust AS trust_score,
               s.flag_icu_contradiction, s.flag_surgery_no_anaesthesia,
               s.gap_no_doctor_count, s.gap_no_equipment_data,
               e.notes_blob
        FROM   {SQL_TABLE}   s
        JOIN   {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
        WHERE  ({note_clauses})
        {geo_where}
        ORDER  BY s.composite_trust DESC
        LIMIT  {int(top_k)}
    """)

    print(f"[SQL-fallback] {len(rows)} rows for query={query!r} state={state_filter!r} city={city_filter!r}")

    # Normalise field names to match the VS-enriched schema
    results = []
    for r in rows:
        results.append({
            "name":                r.get("name", ""),
            "notes_blob":          r.get("notes_blob", ""),
            "trust_score":         float(r.get("trust_score") or 0),
            "num_doctors":         r.get("num_doctors"),
            "capacity":            r.get("capacity"),
            "facility_type":       r.get("facility_type"),
            "phone":               r.get("phone"),
            "address":             r.get("address_line1"),
            "city":                r.get("city", ""),
            "state":               r.get("state", ""),
            "pin_code":            r.get("pin_code", ""),
            "flag_icu_no_beds":    r.get("flag_icu_contradiction")      in (True, "true", "1", 1),
            "flag_surgery_no_anaes": r.get("flag_surgery_no_anaesthesia") in (True, "true", "1", 1),
            "gap_no_doctor_count": r.get("gap_no_doctor_count")         in (True, "true", "1", 1),
            "gap_no_equipment_data": r.get("gap_no_equipment_data")     in (True, "true", "1", 1),
        })
    return results


def vector_search(
    query:        str,
    top_k:        int = 10,
    state_filter: Optional[str] = None,
    city_filter:  Optional[str] = None,
    min_trust:    float = 0.0,
) -> list[dict]:
    """
    Semantic search over the 10k facility vector index.

    Strategy:
      1. Try Databricks Vector Search (semantic embeddings).
      2. If VS returns 0 hits for ANY reason (wrong index name, 404, timeout),
         automatically fall back to SQL LIKE keyword search on the same data.
      3. Post-filter by state/city using SQL-verified fields.

    This means the agent ALWAYS gets real data even without a working VS index.
    """
    results = []

    # ── Attempt VS (skip if not configured) ───────────────────
    if is_vs_configured():
        _active_index = _RESOLVED_VS_INDEX
        url = f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{_active_index}/query"
        fetch_k = min(50, top_k * 3)

        base_payload: dict = {
            "num_results": fetch_k,
            "columns":     ["name", "notes_blob"],
            "query_text":  query,
        }

        results = _vs_query(url, base_payload)

        if results:
            print(f"[VS] raw pool: {len(results)} candidates")
            # Enrich with SQL (adds city, state, trust, flags, phone)
            results = _enrich_with_sql(results)

            # Post-filter by state/city using verified SQL data
            if state_filter:
                geo = [r for r in results if state_filter.lower() in r.get("state", "").lower()]
                results = geo if geo else results   # graceful: keep all if no geo match
                print(f"[VS] state-filter={state_filter!r}: {len(results)} after filter")
            elif city_filter:
                geo = [r for r in results if city_filter.lower() in r.get("city", "").lower()]
                results = geo if geo else results
                print(f"[VS] city-filter={city_filter!r}: {len(results)} after filter")
        else:
            print("[VS] 0 hits — switching to SQL keyword fallback")
    else:
        print("[VS] not configured — using SQL keyword fallback directly")

    # ── SQL fallback (always runs when VS returns nothing) ─────
    if not results:
        results = _sql_keyword_search(query, top_k=top_k * 2,
                                      state_filter=state_filter,
                                      city_filter=city_filter)
        # If state-filtered SQL also returns nothing, retry without geo filter
        if not results and (state_filter or city_filter):
            print("[SQL-fallback] geo-filtered returned 0 — retrying without geo filter")
            results = _sql_keyword_search(query, top_k=top_k * 2)

    if not results:
        print("[search] FINAL: 0 results from both VS and SQL")
        return []

    if min_trust > 0:
        results = [r for r in results if float(r.get("trust_score", 0)) >= min_trust]

    return results[:top_k]


def _enrich_with_sql(vs_results: list[dict]) -> list[dict]:
    """
    Join VS hits with SQL table to add trust score, flags, phone, address.
    Uses case-insensitive LIKE matching so minor name differences don't block enrichment.
    """
    if not vs_results:
        return vs_results

    vs_names = [r.get("name", "") for r in vs_results if r.get("name")]
    if not vs_names:
        return vs_results

    # Build a LIKE OR clause — tolerates minor case/spacing differences
    like_clauses = " OR ".join(
        f"lower(name) LIKE lower('%{n.replace(chr(39), chr(39)*2)[:60]}%')"
        for n in vs_names
    )

    sql_rows = db_sql(f"""
        SELECT name, city, state, pin_code, address_line1, facilityTypeId,
               composite_trust   AS trust_score_raw,
               num_doctors, capacity, officialPhone,
               flag_icu_contradiction, flag_surgery_no_anaesthesia,
               gap_no_doctor_count, gap_no_equipment_data
        FROM   {SQL_TABLE}
        WHERE  {like_clauses}
        LIMIT  {len(vs_names) * 2}
    """)
    print(f"[enrich] SQL returned {len(sql_rows)} rows for {len(vs_names)} VS hits")

    # Build lookup: lowercase name → row  (handles case differences)
    lookup = {row["name"].lower(): row for row in sql_rows if row.get("name")}

    for r in vs_results:
        vs_name_lower = r.get("name", "").lower()
        # Try exact lower-case match first, then substring scan
        extra = lookup.get(vs_name_lower) or next(
            (v for k, v in lookup.items() if vs_name_lower in k or k in vs_name_lower), {}
        )
        if not extra:
            print(f"[enrich] WARNING: no SQL match for '{r.get('name')}'")

        r["trust_score"]           = float(extra.get("trust_score_raw") or 0)
        r["num_doctors"]           = extra.get("num_doctors")
        r["capacity"]              = extra.get("capacity")
        r["facility_type"]         = extra.get("facilityTypeId")
        r["phone"]                 = extra.get("officialPhone")
        r["address"]               = extra.get("address_line1")
        r["city"]                  = extra.get("city")   or r.get("city",  "")
        r["state"]                 = extra.get("state")  or r.get("state", "")
        r["pin_code"]              = extra.get("pin_code") or r.get("pin_code", "")
        r["flag_icu_no_beds"]      = extra.get("flag_icu_contradiction")      in (True, "true", "1", 1)
        r["flag_surgery_no_anaes"] = extra.get("flag_surgery_no_anaesthesia") in (True, "true", "1", 1)
        r["gap_no_doctor_count"]   = extra.get("gap_no_doctor_count")         in (True, "true", "1", 1)
        r["gap_no_equipment_data"] = extra.get("gap_no_equipment_data")       in (True, "true", "1", 1)

    return vs_results




# ─────────────────────────────────────────────────────────────
# 3. AGENT  (Full ReAct loop — Databricks Foundation Model)
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MediAlert, an Agentic Healthcare Intelligence System for India.
Your mission: reduce Discovery-to-Care time for 1.4 billion people.

You have access to 10,000+ verified medical facilities. Use your tools in a
Reason → Act → Observe loop until you can answer confidently.

═══════════════════════════════════════════════════════
MANDATORY SEARCH STRATEGY (follow this exactly):
═══════════════════════════════════════════════════════

STEP 1 — LOCATION SEARCH:
  • If a state is mentioned (e.g. "Bihar", "Gujarat"), call search_facilities
    with state="<StateName>" AND a capability query.
  • "rural Bihar" → state="Bihar"
  • If city mentioned → city="<CityName>"

STEP 2 — BROAD FALLBACK (REQUIRED if step 1 returns < 3 facilities):
  • Immediately call search_facilities AGAIN with a BROADER query and NO state/city filter.
  • Broaden the query: remove specific procedure names, use general category
    (e.g. "appendectomy" → "general surgery emergency care")
  • You MUST make at least 2 search calls before giving up.

STEP 3 — VERIFY:
  • Call get_facility_detail on the TOP 3-5 results to confirm capabilities.
  • Look for EXPLICIT mentions of the procedure in notes_blob.

STEP 4 — CLASSIFY:
  • TIER 1: notes_blob explicitly mentions the exact procedure or very close synonym.
  • TIER 2: facility has related general capability (e.g. surgical ward, OT, emergency dept)
    but procedure not explicitly confirmed. ALWAYS populate tier2 with at least 3 facilities.

STEP 5 — NEVER RETURN EMPTY:
  • If tier1 is empty, put the best matching facilities in tier2.
  • tier2 should have 3-6 facilities whenever the region has ANY surgical/emergency facility.
  • Only set both tiers empty if absolutely no relevant facility was found.

═══════════════════════════════════════════════════════
TRUST SCORING:
═══════════════════════════════════════════════════════
  trust_score 7.0-10.0 → VERIFIED
  trust_score 4.0-6.9  → LIKELY_RELIABLE
  trust_score < 4.0    → UNVERIFIED

FLAGS (check these, include in output):
  flag_icu_contradiction      → claims ICU but beds=0 (PENALIZE in tier placement)
  flag_surgery_no_anaesthesia → claims surgery, no anaesthesia (PENALIZE)
  gap_no_doctor_count         → no doctor count recorded (WARN only)
  gap_no_equipment_data       → no equipment data (WARN only)

═══════════════════════════════════════════════════════
OUTPUT FORMAT — respond with ONLY valid JSON, no markdown:
═══════════════════════════════════════════════════════
{
  "search_metadata": {
    "query_type": "procedure_specific" | "general_search" | "medical_desert",
    "requested_procedure": "<string or null>",
    "location": "<state/city extracted from query>",
    "location_type": "exact_city" | "state_wide" | "radius" | "all_india",
    "search_conducted": true,
    "verification_performed": true,
    "search_attempts": 1
  },
  "tier1_confirmed": [
    {
      "name": "<facility name>",
      "city": "<city>",
      "state": "<state>",
      "trust_score": 0.0,
      "trust_label": "VERIFIED" | "LIKELY_RELIABLE" | "UNVERIFIED",
      "distance_km": null,
      "evidence": "<exact quote from notes confirming the specific procedure/capability>",
      "contradiction_flags": [],
      "warnings": [],
      "contact": {"phone": "<phone or null>"}
    }
  ],
  "tier2_possible": [
    {
      "name": "<facility name>",
      "city": "<city>",
      "state": "<state>",
      "trust_score": 0.0,
      "trust_label": "VERIFIED" | "LIKELY_RELIABLE" | "UNVERIFIED",
      "related_capability": "<general category that makes this relevant>",
      "verification_note": "<specific reason why not tier1 — e.g. 'general surgery listed but appendectomy not explicitly mentioned'>",
      "recommendation": "Call facility to verify before visiting",
      "contact": {"phone": "<phone or null>"}
    }
  ],
  "confidence": {
    "level": "High" | "Medium" | "Low",
    "reasoning": "<1-2 line justification based on evidence quality>"
  },
  "meta": {
    "total_scanned": 10000,
    "matches_found": 0,
    "avg_trust_score": "Medium"
  },
  "no_results_message": "<null if results found; detailed explanation + alternatives if truly empty>"
}"""

# ── Tool schemas (sent to LLM for function calling) ──────────
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_facilities",
            "description": (
                "Semantic search over 10,000+ Indian medical facilities. "
                "Call this first to find candidates. Supports state and city filters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",  "description": "Capability description, e.g. 'ICU with ventilators in Bihar'"},
                    "top_k": {"type": "integer", "description": "Results to return (default 8)", "default": 8},
                    "state": {"type": "string",  "description": "Filter by Indian state name"},
                    "city":  {"type": "string",  "description": "Filter by exact city name"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_facility_detail",
            "description": (
                "Retrieve full notes and metadata for a specific facility by name. "
                "Use this to verify whether a procedure/capability is EXPLICITLY mentioned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "facility_name": {"type": "string", "description": "Full or partial facility name"},
                },
                "required": ["facility_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_medical_deserts",
            "description": "Find Indian states critically underserved for a medical specialty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "specialty_keyword":       {"type": "string",  "description": "Specialty e.g. Dialysis, Oncology"},
                    "min_facilities_threshold":{"type": "integer", "description": "Desert threshold (default 3)", "default": 3},
                },
                "required": ["specialty_keyword"],
            },
        },
    },
]


# ── Tool implementations (callable by the agent) ─────────────

def get_facility_detail(facility_name: str) -> dict:
    """Full record for a facility — used by agent to verify explicit capability."""
    safe = re.sub(r"['\";\\]", "", facility_name)[:80]
    rows = db_sql(f"""
        SELECT s.name, s.city, s.state, s.pin_code, s.address_line1,
               s.facilityTypeId AS facility_type, s.num_doctors, s.capacity,
               s.officialPhone AS phone,
               s.composite_trust AS trust_score,
               s.flag_icu_contradiction, s.flag_surgery_no_anaesthesia,
               s.gap_no_doctor_count, s.gap_no_equipment_data,
               e.notes_blob, e.trust_score_raw
        FROM   {SQL_TABLE}   s
        LEFT JOIN {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
        WHERE  lower(s.name) LIKE lower('%{safe}%')
        ORDER  BY s.composite_trust DESC
        LIMIT  1
    """)
    if not rows:
        return {"error": f"Facility '{facility_name}' not found"}
    r = rows[0]
    r["trust_score"] = float(r.get("trust_score") or r.get("trust_score_raw") or 0)
    return r


def find_medical_deserts(specialty_keyword: str,
                          min_facilities_threshold: int = 3) -> list[dict]:
    """States with critically low facility count for a specialty."""
    kw = re.sub(r"['\";\\]", "", specialty_keyword)[:50]
    rows = db_sql(f"""
        SELECT s.state,
               COUNT(*)                          AS facility_count,
               ROUND(AVG(s.composite_trust), 2) AS avg_trust
        FROM   {SQL_TABLE}   s
        JOIN   {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
        WHERE  lower(e.notes_blob) LIKE lower('%{kw}%')
        GROUP  BY s.state
        HAVING COUNT(*) < {int(min_facilities_threshold)}
        ORDER  BY facility_count ASC
        LIMIT  20
    """)
    return rows or [{"message": f"No desert found for '{specialty_keyword}' — coverage seems adequate."}]


def _parse_agent_json(raw: str) -> dict:
    """Extract JSON from agent's final answer and ensure required keys exist."""
    text = re.sub(r"```(?:json)?\s*", "", raw or "").strip().rstrip("```").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            parsed.setdefault("tier1_confirmed",  [])
            parsed.setdefault("tier2_possible",   [])
            parsed.setdefault("confidence",       {"level": "Medium", "reasoning": ""})
            parsed.setdefault("search_metadata",  {})
            parsed.setdefault("meta",             {"total_scanned": 10000,
                                                   "matches_found": 0,
                                                   "avg_trust_score": "Medium"})
            parsed.setdefault("no_results_message", None)
            return parsed
        except Exception:
            pass
    return {
        "tier1_confirmed":  [],
        "tier2_possible":   [],
        "confidence":       {"level": "Low", "reasoning": "Could not parse structured response"},
        "search_metadata":  {},
        "meta":             {"total_scanned": 10000, "matches_found": 0, "avg_trust_score": "N/A"},
        "no_results_message": "Agent response could not be parsed as JSON.",
        "_parse_error":     True,
        "_raw":             raw,
    }


def run_agent(user_query: str, context_facilities: list[dict] = None) -> dict:
    """
    Full ReAct agent with tool calling + MLflow 3 tracing.

    The agent:
      1. Calls search_facilities to find candidates
      2. Calls get_facility_detail to verify explicit capability
      3. Returns structured JSON with tier1_confirmed / tier2_possible
      4. Logs every span to MLflow 3

    Returns a dict with all agent output + trace data for Streamlit.
    """
    if not is_llm_configured():
        return {
            "tier1_confirmed": [], "tier2_possible": [],
            "confidence": {"level": "N/A", "reasoning": "No credentials"},
            "search_metadata": {}, "meta": {},
            "no_results_message": "LLM not configured. Add credentials to .env.",
            "error": "no_credentials", "trace": None,
        }

    t_total = time.time()
    trace_spans:    list[dict] = []
    tool_calls_log: list[dict] = []
    total_tokens = "N/A"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]

    # Inject any pre-fetched facilities as context (they still get verified via tool calls)
    if context_facilities:
        preview = "\n".join(
            f"[{i+1}] {f.get('name','?')} | {f.get('city','?')}, {f.get('state','?')} "
            f"| Trust: {float(f.get('trust_score',0)):.1f}"
            for i, f in enumerate(context_facilities[:6])
        )
        messages.append({
            "role": "user",
            "content": (
                f"Initial search returned these candidates — "
                f"call get_facility_detail to verify before placing them in tier1:\n\n{preview}"
            ),
        })

    TOOLS_MAP = {
        "search_facilities":  lambda **kw: vector_search(
            query        = kw.get("query", user_query),
            top_k        = kw.get("top_k", 10),
            state_filter = kw.get("state"),
            city_filter  = kw.get("city"),
        ),
        "get_facility_detail":  lambda **kw: get_facility_detail(kw["facility_name"]),
        "find_medical_deserts": lambda **kw: find_medical_deserts(
            kw["specialty_keyword"],
            kw.get("min_facilities_threshold", 3),
        ),
    }

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key  = DATABRICKS_TOKEN,
            base_url = f"{DATABRICKS_HOST}/serving-endpoints",
        )

        # ── MLflow parent span ────────────────────────────────
        _mlf_span = None
        if _MLFLOW:
            try:
                _mlf_span = mlflow.start_span(
                    name="MediAlert-ReAct-Agent", span_type=SpanType.AGENT)
                _mlf_span.__enter__()
                _mlf_span.set_inputs({"query": user_query})
            except Exception:
                _mlf_span = None

        # ── ReAct loop ────────────────────────────────────────
        max_iterations = 6
        iterations     = 0
        final_answer   = None

        while iterations < max_iterations:
            iterations += 1
            t_llm = time.time()

            if _MLFLOW:
                try:
                    with mlflow.start_span(name=f"llm_call_{iterations}",
                                           span_type=SpanType.LLM) as ls:
                        ls.set_inputs({"messages_count": len(messages)})
                        response = client.chat.completions.create(
                            model=LLM_ENDPOINT, messages=messages,
                            tools=TOOL_SCHEMAS, tool_choice="auto",
                            temperature=0.1, max_tokens=2048,
                        )
                        ls.set_outputs({"finish_reason": response.choices[0].finish_reason})
                except Exception:
                    response = client.chat.completions.create(
                        model=LLM_ENDPOINT, messages=messages,
                        tools=TOOL_SCHEMAS, tool_choice="auto",
                        temperature=0.1, max_tokens=2048,
                    )
            else:
                response = client.chat.completions.create(
                    model=LLM_ENDPOINT, messages=messages,
                    tools=TOOL_SCHEMAS, tool_choice="auto",
                    temperature=0.1, max_tokens=2048,
                )

            llm_ms       = int((time.time() - t_llm) * 1000)
            total_tokens = getattr(getattr(response, "usage", None), "total_tokens", "N/A")
            choice       = response.choices[0]
            message      = choice.message
            tc_count     = len(message.tool_calls or [])

            trace_spans.append({
                "name":    f"🧠 LLM Call #{iterations}",
                "type":    "LLM",
                "inputs":  {"messages": len(messages), "model": LLM_ENDPOINT},
                "outputs": {"tokens": total_tokens, "tool_calls": tc_count,
                            "final": not bool(message.tool_calls)},
                "latency_ms": llm_ms,
            })

            # No tool calls → final answer
            if not message.tool_calls:
                final_answer = message.content
                break

            # Append assistant turn (with tool_calls)
            asst_dict = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                asst_dict["tool_calls"] = [tc.model_dump() for tc in message.tool_calls]
            messages.append(asst_dict)

            # Execute each tool
            for tc in message.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except Exception:
                    fn_args = {}

                t_tool = time.time()
                if fn_name in TOOLS_MAP:
                    try:
                        with (mlflow.start_span(name=f"tool_{fn_name}",
                                                span_type=SpanType.TOOL) if _MLFLOW else _noop()) as ts:
                            if _MLFLOW and ts:
                                ts.set_inputs(fn_args)
                            tool_result = TOOLS_MAP[fn_name](**fn_args)
                            if _MLFLOW and ts:
                                ts.set_outputs({"count": len(tool_result)
                                                if isinstance(tool_result, list) else 1})
                    except Exception:
                        tool_result = TOOLS_MAP[fn_name](**fn_args)
                else:
                    tool_result = {"error": f"Unknown tool: {fn_name}"}

                tool_ms = int((time.time() - t_tool) * 1000)
                result_count = len(tool_result) if isinstance(tool_result, list) else 1

                tool_calls_log.append({
                    "iteration": iterations, "tool": fn_name,
                    "args": fn_args, "result_count": result_count,
                    "latency_ms": tool_ms,
                })
                trace_spans.append({
                    "name":    f"⚙️ {fn_name}",
                    "type":    "TOOL",
                    "inputs":  fn_args,
                    "outputs": {"count": result_count,
                                "preview": str(tool_result)[:200]},
                    "latency_ms": tool_ms,
                })

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(tool_result, default=str)[:6000],
                })

        # ── Parse structured JSON from agent's final answer ───
        parsed = _parse_agent_json(final_answer or "")

        # Ensure meta is populated
        t1 = parsed.get("tier1_confirmed", [])
        t2 = parsed.get("tier2_possible",  [])
        parsed.setdefault("meta", {}).update({
            "total_scanned": 10000,
            "matches_found": len(t1) + len(t2),
        })

        # ── Close MLflow parent span ──────────────────────────
        if _mlf_span:
            try:
                _mlf_span.set_outputs({
                    "tier1": len(t1), "tier2": len(t2),
                    "confidence": parsed.get("confidence", {}).get("level"),
                    "iterations": iterations,
                })
                _mlf_span.__exit__(None, None, None)
            except Exception:
                pass

        total_ms = int((time.time() - t_total) * 1000)
        trace_spans.append({
            "name":    "✅ Agent Complete",
            "type":    "AGENT",
            "inputs":  {"query": user_query},
            "outputs": {"tier1": len(t1), "tier2": len(t2),
                        "confidence": parsed.get("confidence", {}).get("level", "N/A"),
                        "iterations": iterations, "tool_calls": len(tool_calls_log)},
            "latency_ms": total_ms,
        })

        return {
            **parsed,
            "raw_answer": final_answer,
            "error":      None,
            "trace": {
                "spans":          trace_spans,
                "tool_calls_log": tool_calls_log,
                "total_ms":       total_ms,
                "model":          LLM_ENDPOINT,
                "total_tokens":   total_tokens,
                "iterations":     iterations,
                "mlflow_active":  _MLFLOW,
            },
        }

    except ImportError:
        return {
            "tier1_confirmed": [], "tier2_possible": [],
            "confidence": {"level": "N/A", "reasoning": "openai not installed"},
            "search_metadata": {}, "meta": {}, "no_results_message": None,
            "error": "import_error", "trace": None,
        }
    except Exception as e:
        return {
            "tier1_confirmed": [], "tier2_possible": [],
            "confidence": {"level": "N/A", "reasoning": str(e)},
            "search_metadata": {}, "meta": {}, "no_results_message": None,
            "error": str(e), "trace": None,
        }


class _noop:
    """No-op context manager — used when MLflow is off."""
    def __enter__(self): return None
    def __exit__(self, *_): pass


