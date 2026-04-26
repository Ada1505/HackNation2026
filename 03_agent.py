# Databricks notebook source
# DBTITLE 1,JSON Output API Documentation
# MAGIC %md
# MAGIC # MediAlert JSON Output API Documentation
# MAGIC
# MAGIC ## Overview
# MAGIC The agent now outputs structured JSON for easy frontend integration. Use `query_agent_json()` instead of `query_agent()` for JSON responses.
# MAGIC
# MAGIC ## JSON Response Schema
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "search_metadata": {
# MAGIC     "query_type": "procedure_specific" | "general_search" | "medical_desert",
# MAGIC     "requested_procedure": "<procedure name or null>",
# MAGIC     "location": "<state/city/coordinates>",
# MAGIC     "location_type": "exact_city" | "radius_Xkm" | "state_wide" | "all_india",
# MAGIC     "search_conducted": true,
# MAGIC     "verification_performed": true | false
# MAGIC   },
# MAGIC   "tier1_confirmed": [
# MAGIC     {
# MAGIC       "facility_id": "<id>",
# MAGIC       "name": "<facility name>",
# MAGIC       "city": "<city>",
# MAGIC       "state": "<state>",
# MAGIC       "pincode": "<6-digit or null>",
# MAGIC       "trust_score": 0.0-10.0,
# MAGIC       "trust_label": "VERIFIED" | "LIKELY_RELIABLE" | "UNVERIFIED",
# MAGIC       "distance_km": <number or null>,
# MAGIC       "evidence": "<exact quote confirming capability>",
# MAGIC       "contradiction_flags": [<active flags>],
# MAGIC       "warnings": [<user warnings>],
# MAGIC       "contact": {
# MAGIC         "phone": "<phone or null>",
# MAGIC         "email": "<email or null>",
# MAGIC         "website": "<url or null>"
# MAGIC       }
# MAGIC     }
# MAGIC   ],
# MAGIC   "tier2_possible": [
# MAGIC     {
# MAGIC       "facility_id": "<id>",
# MAGIC       "name": "<facility name>",
# MAGIC       "city": "<city>",
# MAGIC       "state": "<state>",
# MAGIC       "pincode": "<6-digit or null>",
# MAGIC       "trust_score": 0.0-10.0,
# MAGIC       "trust_label": "VERIFIED" | "LIKELY_RELIABLE" | "UNVERIFIED",
# MAGIC       "distance_km": <number or null>,
# MAGIC       "related_capability": "<general category>",
# MAGIC       "verification_note": "<why not tier 1>",
# MAGIC       "recommendation": "Call facility to verify...",
# MAGIC       "contact": {...}
# MAGIC     }
# MAGIC   ],
# MAGIC   "data_quality_notes": ["<notes about data gaps>"],
# MAGIC   "confidence": {
# MAGIC     "level": "high" | "medium" | "low" | "cannot_recommend",
# MAGIC     "reasoning": "<explanation>"
# MAGIC   },
# MAGIC   "summary": "<2-3 sentence summary>"
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ## Usage Examples
# MAGIC
# MAGIC ### Python (in this notebook)
# MAGIC ```python
# MAGIC result = query_agent_json("Find ICU facilities in Mumbai")
# MAGIC
# MAGIC # Access tier 1 facilities
# MAGIC for facility in result['tier1_confirmed']:
# MAGIC     print(f"{facility['name']} - Trust: {facility['trust_score']}")
# MAGIC     print(f"Evidence: {facility['evidence']}")
# MAGIC ```
# MAGIC
# MAGIC ### Streamlit Frontend
# MAGIC See cell below for complete Streamlit app example.
# MAGIC
# MAGIC ## Key Features
# MAGIC
# MAGIC 1. **Two-Tier System**:
# MAGIC    - **Tier 1**: Explicitly confirmed capabilities (verified in facility data)
# MAGIC    - **Tier 2**: Related capabilities (requires user verification)
# MAGIC
# MAGIC 2. **Trust Scoring**:
# MAGIC    - 7.0-10.0: VERIFIED (high confidence)
# MAGIC    - 4.0-6.9: LIKELY_RELIABLE (medium confidence)
# MAGIC    - 0.1-3.9: UNVERIFIED (low confidence)
# MAGIC
# MAGIC 3. **Contradiction Flags**:
# MAGIC    - `flag_surgery_no_anaesthesia`: Claims surgery but no anaesthetist
# MAGIC    - `flag_icu_contradiction`: Claims ICU but no beds
# MAGIC    - `flag_dialysis_no_support`: Claims dialysis but no equipment
# MAGIC    - Facilities with flags are demoted and warned
# MAGIC
# MAGIC 4. **Data Quality Transparency**:
# MAGIC    - Gap flags indicate missing data (not poor quality)
# MAGIC    - Rural facilities often have gaps due to data collection issues
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. Run the test cell below to see JSON output in action
# MAGIC 2. Use the Streamlit example to build your frontend
# MAGIC 3. Integrate with your backend using the JSON API
# MAGIC

# COMMAND ----------

# MediAlert — Notebook 03: Multi-Attribute Reasoning Agent
# ─────────────────────────────────────────────────────────
# Depends on:
#   workspace.default.facilities_sql          (from 01_cleaning)
#   workspace.default.facilities_for_embedding (from 01_cleaning)
#   main.default.facilities_vector_index       (from 02_embeddings)
#
# Paste each cell (between # COMMAND ----------) into a
# separate Databricks notebook cell. Run top-to-bottom.


# COMMAND ----------

# =============================================================
# CELL 0 — CONFIG  (matches your existing notebooks exactly)
# =============================================================

SQL_TABLE    = "workspace.default.facilities_sql"
EMBED_TABLE  = "workspace.default.facilities_for_embedding"
VS_INDEX     = "workspace.default.facilities_vector_index"
VS_ENDPOINT  = "vs-medialert-endpoint"          # your existing endpoint name

# Databricks Foundation Model to use for reasoning.
# Options (all free-tier): databricks-meta-llama-3-3-70b-instruct
#                           databricks-mixtral-8x7b-instruct
#                           databricks-dbrx-instruct
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

MLFLOW_EXPERIMENT = "/Shared/MediAlert_Agent"

print("Config ready.")
print(f"  SQL table    : {SQL_TABLE}")
print(f"  Vector index : {VS_INDEX}")
print(f"  LLM          : {LLM_ENDPOINT}")


# COMMAND ----------

# # =============================================================
# # CELL 1 — INSTALL DEPENDENCIES
# # =============================================================

# %pip install mlflow databricks-vectorsearch openai --quiet
# dbutils.library.restartPython()


# COMMAND ----------

# =============================================================
# CELL 2 — IMPORTS & MLFLOW SETUP
# =============================================================

import json
import re
import time
import mlflow
import mlflow.deployments
from mlflow.entities import SpanType
from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI
import pyspark.sql.functions as F

# Point OpenAI client at Databricks' Foundation Model endpoint
token  = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host   = spark.conf.get("spark.databricks.workspaceUrl")

openai_client = OpenAI(
    api_key    = token,
    base_url   = f"https://{host}/serving-endpoints",
)

# Vector Search
vsc          = VectorSearchClient()
vs_index     = vsc.get_index(VS_ENDPOINT, VS_INDEX)

# MLflow experiment
mlflow.set_experiment(MLFLOW_EXPERIMENT)
print("Imports complete.")


# COMMAND ----------

# =============================================================
# CELL 3 — TOOL DEFINITIONS
# Three tools the agent can call:
#   1. search_facilities   — semantic vector search
#   2. get_facility_detail — full SQL row for a facility
#   3. find_medical_deserts — state/district gap analysis
# =============================================================

# ── Tool 1: Semantic vector search ───────────────────────────
@mlflow.trace(span_type=SpanType.TOOL, name="search_facilities")
def search_facilities(
    query    : str,
    top_k    : int   = 8,
    state    : str   = None,
    city     : str   = None,
    lat      : float = None,
    lon      : float = None,
    radius_km: float = 50.0,
    near_city: str   = None,    # "near X" intent — search by radius, not exact city
) -> list[dict]:
    """
    Semantic search with location intelligence:
    1. Try exact city/state filter first
    2. If zero results, expand to radius_km around city coordinates
    3. Always sort by Haversine distance when lat/lon are known
    4. Attach distance_km to every result for the agent to cite
    """

    # Resolve coordinates from near_city if not directly provided
    if near_city and not lat:
        lat, lon = CITY_COORDS.get(near_city.lower(), (None, None))
    if city and not lat:
        lat, lon = CITY_COORDS.get(city.lower(), (None, None))

    # ── Pass 1: filtered search ───────────────────────────────
    filters = {}
    if state:
        filters["state"] = state.title()
    if city and not near_city:          # exact city only when NOT "near X"
        filters["city"] = city.title()

    # Define columns we're requesting
    requested_cols = ["facility_id", "name", "state", "city", "pin_code", "notes_blob", "lat", "lon"]
    
    response = vs_index.similarity_search(
        query_text  = query,
        columns     = requested_cols,
        num_results = top_k,
        filters     = filters if filters else None,
    )
    hits      = response.get("result", {}).get("data_array", [])
    
    # Vector search doesn't return column metadata, so use our requested order
    # Data format: [facility_id, name, state, city, pin_code, notes_blob, lat, lon, score]
    results = []
    for row in hits:
        if len(row) >= len(requested_cols):
            result = {}
            for i, col in enumerate(requested_cols):
                result[col] = row[i]
            # Last element is the similarity score
            if len(row) > len(requested_cols):
                result["similarity_score"] = row[-1]
            results.append(result)

    # ── Pass 2: radius fallback when Pass 1 yields nothing ────
    radius_fallback_used = False
    if not results and (lat is not None):
        print(f"  ↳ No results in {city or state} — expanding to {radius_km}km radius")
        radius_fallback_used = True
        # Search without location filter, filter by distance after
        response2 = vs_index.similarity_search(
            query_text  = query,
            columns     = requested_cols,
            num_results = 50,   # fetch more so radius filter has candidates
        )
        hits2 = response2.get("result", {}).get("data_array", [])
        
        all_results = []
        for row in hits2:
            if len(row) >= len(requested_cols):
                result = {}
                for i, col in enumerate(requested_cols):
                    result[col] = row[i]
                if len(row) > len(requested_cols):
                    result["similarity_score"] = row[-1]
                all_results.append(result)

        # Filter to radius
        results = []
        for r in all_results:
            r_lat = _safe_float(r.get("lat"))
            r_lon = _safe_float(r.get("lon"))
            if r_lat and r_lon:
                dist = haversine_km(lat, lon, r_lat, r_lon)
                if dist <= radius_km:
                    r["distance_km"] = round(dist, 1)
                    results.append(r)
        results = sorted(results, key=lambda x: x["distance_km"])[:top_k]

    # ── Attach distance to Pass 1 results when lat/lon known ──
    if lat and not radius_fallback_used:
        for r in results:
            r_lat = _safe_float(r.get("lat"))
            r_lon = _safe_float(r.get("lon"))
            if r_lat and r_lon:
                r["distance_km"] = round(haversine_km(lat, lon, r_lat, r_lon), 1)

    # ── Enrich with trust_score and flags from SQL table ──────
    if results:
        names_for_sql = [r.get("name", "") for r in results if r.get("name", "").strip()]

        if not names_for_sql:
            print("  WARNING: vector results had no name field, skipping SQL enrichment")
        else:
            names_escaped = ", ".join(
                f"'{n.replace(chr(39), chr(39)*2)}'" for n in names_for_sql
            )

            # Query all trust metrics, contradiction flags, and gap flags
            sql_rows = spark.sql(f"""
                SELECT name, city, state,
                    composite_trust, trust_ci_lower, trust_ci_upper, trust_label,
                    pillar_presence, pillar_capability, pillar_integrity, pillar_engagement,
                    flag_icu_contradiction, flag_surgery_no_anaesthesia, 
                    flag_dialysis_no_support, flag_availability_contradiction,
                    flag_overreach_claim, flag_nicu_no_support, contradiction_count,
                    gap_no_notes, gap_no_doctor_count, gap_no_equipment_data,
                    gap_no_capacity_data, gap_no_specialties, gap_no_procedures,
                    gap_no_capabilities, gap_skeleton_record, gap_count,
                    num_doctors, capacity, facilityTypeId,
                    officialPhone, address_line1
                FROM {SQL_TABLE}
                WHERE name IN ({names_escaped})
            """).toPandas()

            # Sort by composite_trust and deduplicate
            sql_rows = sql_rows.sort_values("composite_trust", ascending=False)
            sql_rows = sql_rows.drop_duplicates(subset=["name"], keep="first")
            name_map = sql_rows.set_index("name").to_dict("index")

            for r in results:
                extra = name_map.get(r.get("name"), {})
                
                # Trust metrics
                r["trust_score"]      = float(extra.get("composite_trust", 0) or 0)
                r["trust_ci_lower"]   = float(extra.get("trust_ci_lower", 0) or 0)
                r["trust_ci_upper"]   = float(extra.get("trust_ci_upper", 0) or 0)
                r["trust_label"]      = extra.get("trust_label", "UNVERIFIED")
                
                # Trust pillars
                r["pillar_presence"]   = float(extra.get("pillar_presence", 0) or 0)
                r["pillar_capability"] = float(extra.get("pillar_capability", 0) or 0)
                r["pillar_integrity"]  = float(extra.get("pillar_integrity", 0) or 0)
                r["pillar_engagement"] = float(extra.get("pillar_engagement", 0) or 0)
                
                # Contradiction flags (RED FLAGS)
                r["flag_icu_contradiction"]          = bool(extra.get("flag_icu_contradiction", False))
                r["flag_surgery_no_anaesthesia"]     = bool(extra.get("flag_surgery_no_anaesthesia", False))
                r["flag_dialysis_no_support"]        = bool(extra.get("flag_dialysis_no_support", False))
                r["flag_availability_contradiction"] = bool(extra.get("flag_availability_contradiction", False))
                r["flag_overreach_claim"]            = bool(extra.get("flag_overreach_claim", False))
                r["flag_nicu_no_support"]            = bool(extra.get("flag_nicu_no_support", False))
                r["contradiction_count"]             = int(extra.get("contradiction_count", 0) or 0)
                
                # Gap flags (MISSING DATA, NOT RED FLAGS)
                r["gap_no_notes"]         = bool(extra.get("gap_no_notes", False))
                r["gap_no_doctor_count"]  = bool(extra.get("gap_no_doctor_count", False))
                r["gap_no_equipment_data"]= bool(extra.get("gap_no_equipment_data", False))
                r["gap_no_capacity_data"] = bool(extra.get("gap_no_capacity_data", False))
                r["gap_no_specialties"]   = bool(extra.get("gap_no_specialties", False))
                r["gap_no_procedures"]    = bool(extra.get("gap_no_procedures", False))
                r["gap_no_capabilities"]  = bool(extra.get("gap_no_capabilities", False))
                r["gap_skeleton_record"]  = bool(extra.get("gap_skeleton_record", False))
                r["gap_count"]            = int(extra.get("gap_count", 0) or 0)
                
                # Facility details
                r["num_doctors"] = extra.get("num_doctors")
                r["capacity"]    = extra.get("capacity")
                r["phone"]       = extra.get("officialPhone")
                r["address"]     = extra.get("address_line1")
            
            # Sort by composite trust for ranking, but demote high-contradiction facilities
            results.sort(key=lambda x: (
                -x.get("contradiction_count", 0),  # Contradictions first (ascending = worse first)
                -x.get("trust_score", 0)           # Then by trust (descending = better first)
            ))

    # Tag so agent can tell the user
    for r in results:
        r["radius_fallback"] = radius_fallback_used

    return results


def _safe_float(val):
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None


# ── Tool 2: Full SQL detail for one facility ─────────────────
@mlflow.trace(span_type=SpanType.TOOL, name="get_facility_detail")
def get_facility_detail(facility_name: str) -> dict:
    """
    Pull the complete SQL record for a named facility.
    Used by the agent to cite specific fields that justify its recommendation.
    """
    rows = spark.sql(f"""
        SELECT *
        FROM {SQL_TABLE}
        WHERE lower(name) LIKE lower('%{facility_name.replace("'", "''")}%')
        LIMIT 1
    """).toPandas()

    if rows.empty:
        return {"error": f"No facility found matching '{facility_name}'"}

    row = rows.iloc[0].dropna().to_dict()
    # Also pull notes_blob from embedding table for the citation text
    notes_rows = spark.sql(f"""
        SELECT notes_blob
        FROM {EMBED_TABLE}
        WHERE lower(name) LIKE lower('%{facility_name.replace("'", "''")}%')
        LIMIT 1
    """).toPandas()
    if not notes_rows.empty:
        row["notes_blob"] = notes_rows.iloc[0]["notes_blob"]
    return row


# ── Tool 3: Medical desert finder ────────────────────────────
@mlflow.trace(span_type=SpanType.TOOL, name="find_medical_deserts")
def find_medical_deserts(specialty_keyword: str, min_facilities_threshold: int = 3) -> dict:
    """
    Find states/cities where a given specialty (e.g., 'Oncology', 'Dialysis')
    is critically underserved. Returns states with fewer than threshold facilities.
    """
    df = spark.sql(f"""
        SELECT e.state, e.city, e.pin_code, e.name, e.notes_blob
        FROM {EMBED_TABLE} e
        WHERE lower(e.notes_blob) LIKE lower('%{specialty_keyword}%')
    """).toPandas()

    total_by_state = (
        spark.sql(f"SELECT state, COUNT(*) AS total FROM {EMBED_TABLE} GROUP BY state")
        .toPandas().set_index("state")["total"].to_dict()
    )

    if df.empty:
        return {"specialty": specialty_keyword, "covered_states": [], "desert_states": list(total_by_state.keys())}

    covered = df.groupby("state").size().reset_index(name="facility_count")
    covered_dict = covered.set_index("state")["facility_count"].to_dict()

    # States with coverage below threshold are deserts
    deserts = {
        st: {"total_facilities": total_by_state.get(st, 0), "with_specialty": covered_dict.get(st, 0)}
        for st in total_by_state
        if covered_dict.get(st, 0) < min_facilities_threshold
    }

    return {
        "specialty"      : specialty_keyword,
        "total_with_specialty": int(df.shape[0]),
        "covered_states" : list(covered_dict.keys()),
        "desert_states"  : deserts,
        "sample_facilities": df[["name", "state", "city", "pin_code"]].head(10).to_dict("records"),
    }


# Register tools for agent
TOOLS = {
    "search_facilities"   : search_facilities,
    "get_facility_detail" : get_facility_detail,
    "find_medical_deserts": find_medical_deserts,
}

print("Tools defined.")

# COMMAND ----------

# =============================================================
# CELL 3b — LOCATION HELPERS  (INSERT BETWEEN CELL 3 AND CELL 4)
# =============================================================

import math
import re

# Coordinates for major Indian cities — extend this list from your dataset
CITY_COORDS = {
    "mumbai": (19.0760, 72.8777), "delhi": (28.6139, 77.2090),
    "bangalore": (12.9716, 77.5946), "bengaluru": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867), "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639), "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714), "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462), "noida": (28.5355, 77.3910),
    "gurgaon": (28.4595, 77.0266), "gurugram": (28.4595, 77.0266),
    "chandigarh": (30.7333, 76.7794), "indore": (22.7196, 75.8577),
    "bhopal": (23.2599, 77.4126), "patna": (25.5941, 85.1376),
    "nagpur": (21.1458, 79.0882), "surat": (21.1702, 72.8311),
    "agra": (27.1767, 78.0081), "varanasi": (25.3176, 82.9739),
    "coimbatore": (11.0168, 76.9558), "kochi": (9.9312, 76.2673),
    "visakhapatnam": (17.6868, 83.2185), "ranchi": (23.3441, 85.3096),
    "guwahati": (26.1445, 91.7362), "bhubaneswar": (20.2961, 85.8245),
    "amritsar": (31.6340, 74.8723), "dehradun": (30.3165, 78.0322),
}

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance between two lat/lon points in km."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))

def extract_location(text: str) -> dict:
    """
    Parse a user query for location intent.
    Returns: { city, state, pin_code, near_me, near_city, lat, lon }
    """
    lower = text.lower()

    # "near <city>" pattern — distinct from exact city match
    near_city = None
    near_match = re.search(
        r'\bnear\s+([a-z\s]{3,25}?)(?:\s+(?:city|district|area|region))?\b',
        lower
    )
    if near_match:
        candidate = near_match.group(1).strip().rstrip()
        if candidate in CITY_COORDS:
            near_city = candidate

    # Exact city mention
    city_found = None
    for city in CITY_COORDS:
        # Use word boundary check to avoid partial matches
        if re.search(r'\b' + re.escape(city) + r'\b', lower):
            city_found = city
            break

    # PIN code
    pin_match = re.search(r'\b[1-9][0-9]{5}\b', text)
    pin_code = pin_match.group() if pin_match else None

    # State names (add more as needed)
    STATE_NAMES = [
        "bihar", "uttar pradesh", "maharashtra", "rajasthan", "gujarat",
        "karnataka", "tamil nadu", "kerala", "telangana", "andhra pradesh",
        "west bengal", "madhya pradesh", "odisha", "punjab", "haryana",
        "assam", "jharkhand", "uttarakhand", "himachal pradesh", "goa",
    ]
    state_found = next((s for s in STATE_NAMES if s in lower), None)

    near_me = any(p in lower for p in ["near me", "nearby", "closest", "nearest"])

    # Resolve coordinates
    ref_city = near_city or city_found
    lat, lon = CITY_COORDS.get(ref_city, (None, None)) if ref_city else (None, None)

    return {
        "city"     : city_found,
        "near_city": near_city,           # user said "near X" — use radius not exact match
        "state"    : state_found,
        "pin_code" : pin_code,
        "near_me"  : near_me,
        "lat"      : lat,
        "lon"      : lon,
        "found"    : bool(city_found or near_city or pin_code or state_found or near_me),
    }

# Smoke test
print(extract_location("Find dialysis center near Pune"))
print(extract_location("ICU hospital in Bihar with emergency"))
print(extract_location("nearest trauma center near me"))

# COMMAND ----------

# =============================================================
# CELL 4 — TOOL SCHEMAS (sent to the LLM)
# =============================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_facilities",
            "description": (
                "Semantic search over 10,000 Indian medical facilities. "
                "Supports exact city/state filter and radius-based 'near X' search. "
                "If city search returns nothing, automatically expands to radius_km around that city. "
                "Always returns distance_km when coordinates are known."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query"    : {"type": "string",  "description": "Natural-language capability description"},
                    "top_k"    : {"type": "integer", "description": "Results to return (default 8)", "default": 8},
                    "state"    : {"type": "string",  "description": "Restrict to Indian state name"},
                    "city"     : {"type": "string",  "description": "Exact city name match"},
                    "lat"      : {"type": "number",  "description": "User or reference latitude"},
                    "lon"      : {"type": "number",  "description": "User or reference longitude"},
                    "radius_km": {"type": "number",  "description": "Search radius in km (default 50)", "default": 50},
                    "near_city": {"type": "string",  "description": "City name for 'near X' queries — uses radius not exact match"},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_facility_detail",
            "description": (
                "Retrieve the complete database record for a specific facility by name. "
                "Use this to verify claims, find phone/address, or extract the exact "
                "sentence from facility notes that justifies a recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "facility_name": {
                        "type": "string",
                        "description": "The name of the facility (partial match is fine)"
                    }
                },
                "required": ["facility_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_medical_deserts",
            "description": (
                "Find regions of India critically underserved for a given medical specialty. "
                "Returns states where fewer than a threshold number of facilities offer "
                "the specialty — these are 'medical deserts'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "specialty_keyword": {
                        "type": "string",
                        "description": "The specialty to search for (e.g., Oncology, Dialysis, Trauma)"
                    },
                    "min_facilities_threshold": {
                        "type": "integer",
                        "description": "States with fewer than this many facilities are flagged as deserts",
                        "default": 3
                    }
                },
                "required": ["specialty_keyword"]
            }
        }
    }
]

print("Tool schemas ready.")


# COMMAND ----------

# =============================================================
# CELL 5 — SYSTEM PROMPT (Enhanced: Evidence + MLflow metadata)
# =============================================================

SYSTEM_PROMPT = """You are MediAlert, an Agentic Healthcare Intelligence System for India.
Your mission is to reduce Discovery-to-Care time for 1.4 billion people.

You have access to a database of 10,000+ medical facilities across India,
each with structured metadata and free-form capability notes.

CRITICAL REASONING RULES:

1. PROCEDURE-SPECIFIC SEARCHES (appendectomy, dialysis, chemotherapy, etc.):
   
   Step 1: Use search_facilities with the procedure name
   Step 2: Call get_facility_detail on TOP 5 results to verify explicit mentions
   Step 3: Split results into TWO TIERS:
   
   TIER 1 — EXPLICITLY CONFIRMED:
   - Facility notes/procedures/capabilities contain the EXACT procedure name
   - OR contains clear synonyms (appendectomy = appendicectomy = laparoscopic appendectomy)
   - Show exact quote from notes as evidence
   - Rank by: (1) trust_score, (2) no contradiction flags
   
   TIER 2 — RELATED CAPABILITY (NOT CONFIRMED):
   - Has general capability (\"surgery\", \"emergency\") but procedure NOT mentioned
   - Present honestly: \"Has surgical capability but [procedure] not explicitly confirmed\"
   - User should verify by calling before visiting
   - Only show if TIER 1 is empty or user asks for \"possible\" alternatives
   
   NEVER present a TIER 2 facility as if it has confirmed capability.
   If TIER 1 is empty, state clearly: \"No facilities with explicit [procedure] capability found.\"

2. TRUST SCORE RANKING:
   - Primary: composite_trust (0-10 scale)
   - Higher = more verified, complete data
   - trust_score < 3.0 = UNVERIFIED (flag prominently)
   - trust_score < 2.0 = HIGH RISK (demote to bottom or exclude)

3. FLAG-BASED FILTERING (CONTRADICTION FLAGS):
   These indicate LOGICAL CONTRADICTIONS in facility claims — ALWAYS APPLY PENALTIES.
   Always include active flags in the response.

4. GAP FLAGS (MISSING DATA — DO NOT PENALIZE):
   These indicate ABSENT fields in source data — NOT evidence of poor quality.
   Include gap information for transparency but don't demote facilities.

5. RANKING LOGIC (apply in order):
   a) EXCLUDE: trust_score < 1.5 OR (procedure-specific AND not in TIER 1)
   b) DEMOTE TO BOTTOM: contradiction_count > 1
   c) PRIMARY SORT: composite_trust DESC (highest first)
   d) SECONDARY SORT: distance_km ASC (if coordinates provided)
   e) TERTIARY: gap_count ASC (prefer more complete data)

6. EVIDENCE REQUIREMENTS:
   - For EVERY recommendation, cite the specific field that justifies it
   - Quote exact sentence from notes_blob if claiming a specific capability
   - If inferring from general category, state explicitly: \"Inferred from [field]\"
   - Include the source field name (e.g., \"procedure_text\", \"capability_text\", \"description\")

LOCATION REASONING RULES:
1. \"near <city>\" → use near_city + radius_km (default 50km)
2. Exact city → use city= parameter
3. State-level → use state= parameter
4. If city returns 0 results → automatically expands to radius, note this
5. User coordinates → pass lat/lon for distance ranking

OUTPUT FORMAT - STRICT JSON ONLY:
You MUST respond with ONLY valid JSON (no markdown, no code blocks, no extra text).
Use this exact structure:

{
  \"mlflow_metadata\": {
    \"run_id\": \"<will be added by wrapper>\",
    \"trace_id\": \"<will be added by wrapper>\",
    \"model_endpoint\": \"<LLM endpoint used>\",
    \"total_tool_calls\": <number>,
    \"iterations\": <number>
  },
  \"search_metadata\": {
    \"query_type\": \"procedure_specific\" or \"general_search\" or \"medical_desert\",
    \"requested_procedure\": \"<procedure name or null>\",
    \"location\": \"<state/city/coordinates>\",
    \"location_type\": \"exact_city\" or \"radius_Xkm\" or \"state_wide\" or \"all_india\",
    \"search_conducted\": true,
    \"verification_performed\": true or false
  },
  \"tier1_confirmed\": [
    {
      \"facility_id\": \"<id>\",
      \"name\": \"<facility name>\",
      \"city\": \"<city>\",
      \"state\": \"<state>\",
      \"pincode\": \"<6-digit pincode or null>\",
      \"address\": \"<full address or null>\",
      \"coordinates\": {
        \"lat\": <latitude or null>,
        \"lon\": <longitude or null>
      },
      \"trust_score\": <0.0-10.0>,
      \"trust_label\": \"VERIFIED\" or \"LIKELY_RELIABLE\" or \"UNVERIFIED\",
      \"trust_ci_lower\": <lower confidence bound or null>,
      \"trust_ci_upper\": <upper confidence bound or null>,
      \"distance_km\": <number or null>,
      \"evidence\": {
        \"primary_quote\": \"<exact sentence from notes confirming capability>\",
        \"source_field\": \"description\" or \"procedure_text\" or \"capability_text\" or \"specialties_text\",
        \"additional_evidence\": [\"<other supporting quotes or null>\"]
      },
      \"facility_details\": {
        \"facility_type\": \"<hospital/clinic/etc or null>\",
        \"operator_type\": \"<government/private/etc or null>\",
        \"year_established\": <year or null>,
        \"num_doctors\": <count or null>,
        \"capacity\": <bed count or null>,
        \"specialties\": [\"<list of specialties or empty array>\"],
        \"procedures\": [\"<list of procedures or empty array>\"],
        \"equipment\": [\"<list of equipment or empty array>\"]
      },
      \"contradiction_flags\": [
        {
          \"flag_name\": \"flag_surgery_no_anaesthesia\",
          \"description\": \"Claims surgery but no anaesthesiologist on record\"
        }
      ],
      \"data_gaps\": [
        {
          \"gap_name\": \"gap_no_doctor_count\",
          \"description\": \"Doctor count not recorded in source data\"
        }
      ],
      \"warnings\": [\"<user-facing warnings based on flags or empty array>\"],
      \"contact\": {
        \"phone\": \"<phone or null>\",
        \"email\": \"<email or null>\",
        \"website\": \"<website or null>\"
      }
    }
  ],
  \"tier2_possible\": [
    {
      \"facility_id\": \"<id>\",
      \"name\": \"<facility name>\",
      \"city\": \"<city>\",
      \"state\": \"<state>\",
      \"pincode\": \"<6-digit pincode or null>\",
      \"trust_score\": <0.0-10.0>,
      \"trust_label\": \"VERIFIED\" or \"LIKELY_RELIABLE\" or \"UNVERIFIED\",
      \"distance_km\": <number or null>,
      \"related_capability\": \"<general category like 'surgery' or 'emergency'>\",
      \"evidence\": {
        \"general_quote\": \"<quote showing related capability>\",
        \"source_field\": \"<field name>\"
      },
      \"verification_note\": \"<explain why not in tier 1>\",
      \"recommendation\": \"Call facility to verify specific procedure before visiting\",
      \"contact\": {
        \"phone\": \"<phone or null>\",
        \"email\": \"<email or null>\",
        \"website\": \"<website or null>\"
      }
    }
  ],
  \"data_quality_notes\": [\"<notes about facilities with high gap_count or limited data>\"],
  \"confidence\": {
    \"level\": \"high\" or \"medium\" or \"low\" or \"cannot_recommend\",
    \"reasoning\": \"<explanation of confidence level>\",
    \"factors\": {
      \"explicit_confirmation_found\": true or false,
      \"contradiction_count\": <number>,
      \"avg_trust_score\": <number>,
      \"data_completeness\": \"high\" or \"medium\" or \"low\"
    }
  },
  \"summary\": \"<2-3 sentence natural language summary of findings>\"
}

IMPORTANT:
- If TIER 1 is empty for procedure-specific searches, set tier1_confirmed to empty array []
- Only populate tier2_possible if TIER 1 is empty
- Always include evidence with source field for transparency
- Include facility_details for tier1 to give frontend full context
- Always include all required fields (use null for missing values)
- Ensure valid JSON syntax (proper escaping of quotes, no trailing commas)
- Do NOT wrap the JSON in markdown code blocks
- Do NOT add any text before or after the JSON

Be brutally honest. Never over-promise. Patient safety depends on conservative recommendations.
If you cannot find explicit confirmation, reflect that in the tier structure and confidence."""


# COMMAND ----------

# =============================================================
# CELL 6 — AGENT LOOP (ReAct: Reason → Act → Observe → Repeat)
# =============================================================

@mlflow.trace(span_type=SpanType.AGENT, name="MediAlert_Agent")
def run_agent(user_query: str, max_iterations: int = 8) -> dict:
    """
    Run the MediAlert ReAct agent.

    Args:
        user_query    : Natural language healthcare query
        max_iterations: Safety cap on tool-call rounds

    Returns:
        dict with keys: answer, citations, tool_calls_log, iterations
    """
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_query},
    ]

    tool_calls_log = []
    iterations     = 0

    with mlflow.start_span(name="agent_loop", span_type=SpanType.CHAIN) as loop_span:
        loop_span.set_inputs({"query": user_query})

        while iterations < max_iterations:
            iterations += 1

            # ── LLM call ─────────────────────────────────────
            with mlflow.start_span(name=f"llm_call_{iterations}", span_type=SpanType.LLM) as llm_span:
                llm_span.set_inputs({"messages": messages})

                response = openai_client.chat.completions.create(
                    model       = LLM_ENDPOINT,
                    messages    = messages,
                    tools       = TOOL_SCHEMAS,
                    tool_choice = "auto",
                    temperature = 0.1,
                    max_tokens  = 2048,
                )
                llm_span.set_outputs({"response": response.model_dump()})

            choice  = response.choices[0]
            message = choice.message

            # ── No tool call → final answer ───────────────────
            if not message.tool_calls:
                loop_span.set_outputs({"answer": message.content, "iterations": iterations})
                return {
                    "answer"         : message.content,
                    "tool_calls_log" : tool_calls_log,
                    "iterations"     : iterations,
                }

            # ── Execute each tool call ────────────────────────
            messages.append(message)   # append assistant turn with tool_calls

            for tc in message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                print(f"  [iter {iterations}] Calling tool: {fn_name}({fn_args})")

                if fn_name not in TOOLS:
                    tool_result = {"error": f"Unknown tool: {fn_name}"}
                else:
                    try:
                        tool_result = TOOLS[fn_name](**fn_args)
                    except Exception as e:
                        tool_result = {"error": str(e)}

                tool_calls_log.append({
                    "iteration" : iterations,
                    "tool"      : fn_name,
                    "args"      : fn_args,
                    "result_len": len(str(tool_result)),
                })

                # Append tool result to conversation
                messages.append({
                    "role"         : "tool",
                    "tool_call_id" : tc.id,
                    "content"      : json.dumps(tool_result, default=str)[:6000],  # truncate large results
                })

        # Safety exit
        loop_span.set_outputs({"answer": "Max iterations reached", "iterations": iterations})
        return {
            "answer"        : "Agent reached maximum iterations without a final answer.",
            "tool_calls_log": tool_calls_log,
            "iterations"    : iterations,
        }


print("Agent loop defined.")


# COMMAND ----------

# =============================================================
# CELL 7 — RUN SINGLE QUERY (with MLflow run) - TEXT VERSION
# Use query_agent_json() instead for structured JSON output
# =============================================================

def query_agent(user_query: str, run_name: str = None) -> str:
    """Wrapper that logs the full agent run to MLflow. Returns raw text."""
    run_name = run_name or f"agent_query_{int(time.time())}"
    print(f"\nQuery: {user_query}\n{'─'*60}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("query", user_query)
        mlflow.log_param("llm_endpoint", LLM_ENDPOINT)

        result = run_agent(user_query)

        mlflow.log_metric("iterations", result["iterations"])
        mlflow.log_metric("tool_calls", len(result["tool_calls_log"]))
        mlflow.log_text(result["answer"], "answer.txt")
        mlflow.log_dict({"tool_calls": result["tool_calls_log"]}, "tool_calls.json")

    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nTool calls made: {len(result['tool_calls_log'])} over {result['iterations']} iterations")
    return result["answer"]


print("Text-based query function defined. Use query_agent_json() for JSON output.")


# COMMAND ----------

# DBTITLE 1,JSON Response Parser for Frontend
# =============================================================
# CELL 7.5 — JSON PARSER FOR STREAMLIT FRONTEND (with MLflow)
# =============================================================

import json
import re

def query_agent_json(user_query: str, run_name: str = None) -> dict:
    """
    Wrapper around query_agent that ensures JSON output and parses it.
    Returns a Python dict ready for Streamlit consumption with MLflow metadata.
    
    Returns:
        dict: Parsed JSON response with facility data and MLflow tracking info
        
    Raises:
        ValueError: If response is not valid JSON
    """
    run_name = run_name or f"agent_query_{int(time.time())}"
    print(f"\nQuery: {user_query}\n{'─'*60}")
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        
        mlflow.log_param("query", user_query)
        mlflow.log_param("llm_endpoint", LLM_ENDPOINT)
        mlflow.log_param("output_format", "json")
        
        result = run_agent(user_query)
        
        mlflow.log_metric("iterations", result["iterations"])
        mlflow.log_metric("tool_calls", len(result["tool_calls_log"]))
        
        raw_answer = result["answer"]
        
        # Try to extract JSON if wrapped in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', raw_answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = raw_answer
        
        # Parse JSON
        try:
            parsed_json = json.loads(json_str)
            
            # Inject MLflow metadata into the response
            if "mlflow_metadata" not in parsed_json:
                parsed_json["mlflow_metadata"] = {}
            
            parsed_json["mlflow_metadata"].update({
                "run_id": run_id,
                "run_name": run_name,
                "model_endpoint": LLM_ENDPOINT,
                "total_tool_calls": len(result["tool_calls_log"]),
                "iterations": result["iterations"],
                "mlflow_ui_url": f"#mlflow/experiments/runs/{run_id}"
            })
            
            mlflow.log_dict(parsed_json, "response.json")
            mlflow.log_text(json.dumps(parsed_json, indent=2), "response_formatted.json")
            
            print("\n✓ Successfully parsed JSON response")
            print(f"  - MLflow Run ID: {run_id}")
            print(f"  - Tier 1 (Confirmed): {len(parsed_json.get('tier1_confirmed', []))} facilities")
            print(f"  - Tier 2 (Possible): {len(parsed_json.get('tier2_possible', []))} facilities")
            print(f"  - Confidence: {parsed_json.get('confidence', {}).get('level', 'unknown')}")
            
            return parsed_json
            
        except json.JSONDecodeError as e:
            print(f"\n✗ JSON parsing failed: {e}")
            print(f"\nRaw response:\n{raw_answer[:500]}...")
            
            mlflow.log_text(raw_answer, "raw_response_failed.txt")
            
            # Return error structure with MLflow metadata
            return {
                "mlflow_metadata": {
                    "run_id": run_id,
                    "run_name": run_name,
                    "model_endpoint": LLM_ENDPOINT,
                    "total_tool_calls": len(result["tool_calls_log"]),
                    "iterations": result["iterations"],
                    "mlflow_ui_url": f"#mlflow/experiments/runs/{run_id}"
                },
                "error": "Failed to parse JSON response",
                "raw_response": raw_answer,
                "parse_error": str(e),
                "tier1_confirmed": [],
                "tier2_possible": [],
                "confidence": {"level": "error", "reasoning": "Response parsing failed"}
            }

print("JSON parser function defined. Use query_agent_json() for structured output with MLflow tracking.")


# COMMAND ----------

# DBTITLE 1,Streamlit Integration - Query with Validation
# # =============================================================
# # CELL 7.6 — STREAMLIT ENTRY POINT (Agent + Validator)
# # =============================================================
# # Single function that Streamlit calls:
# #   1. Generates agent JSON response
# #   2. Validates all facilities through 04_validator
# #   3. Returns enhanced JSON with validation results
# # =============================================================

# # Load validator functions from 04_validator notebook
# %run ./04_validator

# @mlflow.trace(span_type=SpanType.CHAIN, name="Query_With_Validation")
# def query_with_validation(user_query: str, run_name: str = None, validate: bool = True) -> dict:
#     """
#     Complete pipeline for Streamlit integration:
#     1. Run agent to get facility recommendations
#     2. Validate each facility through rule-based + LLM validator
#     3. Return enhanced JSON with validation results
    
#     Args:
#         user_query: Natural language query from user
#         run_name: Optional MLflow run name
#         validate: If False, skip validation (faster, for testing)
    
#     Returns:
#         Enhanced JSON with structure:
#         {
#           "mlflow_metadata": {...},
#           "search_metadata": {...},
#           "tier1_confirmed": [
#             {
#               ...facility fields...,
#               "validation": {
#                 "final_verdict": "PASS" | "FAIL" | "PASS_WITH_WARNINGS",
#                 "confidence": "High" | "Medium" | "Low",
#                 "violations": [...],
#                 "warnings": [...],
#                 "supporting_cite": "...",
#                 "contradiction": "...",
#                 "corrective_action": "..."
#               },
#               "validated": true | false
#             }
#           ],
#           "tier2_possible": [...],
#           "validation_summary": {
#             "total_facilities": 5,
#             "passed": 3,
#             "failed": 2,
#             "tier1_passed": 2,
#             "tier1_failed": 1,
#             "claimed_specialty": "Surgery"
#           },
#           "confidence": {...},
#           "summary": "..."
#         }
#     """
#     print(f"\n{'='*70}")
#     print(f"QUERY WITH VALIDATION: {user_query[:80]}...")
#     print(f"{'='*70}")
    
#     # Step 1: Get agent recommendations
#     print("\n[1/2] Running agent to find facilities...")
#     agent_json = query_agent_json(user_query, run_name=run_name or "query_with_validation")
    
#     tier1_count = len(agent_json.get('tier1_confirmed', []))
#     tier2_count = len(agent_json.get('tier2_possible', []))
#     print(f"      Agent found: {tier1_count} tier1, {tier2_count} tier2 facilities")
    
#     # Step 2: Validate facilities (if enabled)
#     if validate and (tier1_count > 0 or tier2_count > 0):
#         print("\n[2/2] Validating facilities against medical standards...")
#         try:
#             validated_json = validate_agent_results(agent_json)
            
#             val_summary = validated_json.get('validation_summary', {})
#             print(f"      Validation complete:")
#             print(f"        - Passed: {val_summary.get('passed', 0)}/{val_summary.get('total_facilities', 0)}")
#             print(f"        - Failed: {val_summary.get('failed', 0)}/{val_summary.get('total_facilities', 0)}")
#             print(f"        - Specialty checked: {val_summary.get('claimed_specialty', 'N/A')}")
            
#             return validated_json
            
#         except Exception as e:
#             print(f"      WARNING: Validation failed with error: {str(e)}")
#             print(f"      Returning agent results without validation")
#             agent_json['validation_error'] = str(e)
#             return agent_json
#     else:
#         print("\n[2/2] Skipping validation (no facilities or validate=False)")
#         return agent_json


# print("\n" + "="*70)
# print("STREAMLIT INTEGRATION READY")
# print("="*70)
# print("""Use this function in Streamlit:

#   result = query_with_validation(
#       user_query="Find ICU in Mumbai",
#       validate=True
#   )
  
#   # Access results
#   for facility in result['tier1_confirmed']:
#       print(facility['name'])
#       print(facility['validation']['final_verdict'])
#       print(facility['validation']['supporting_cite'])
# """)
# print("="*70)

# COMMAND ----------

# DBTITLE 1,Test - Query with Validation
# # =============================================================
# # CELL 7.7 — TEST QUERY WITH VALIDATION
# # =============================================================
# # Test the complete pipeline
# # =============================================================

# print("\n" + "="*70)
# print("TESTING COMPLETE PIPELINE: Agent + Validator")
# print("="*70)

# test_query = "Find facilities in Bihar that can perform appendectomy"

# with mlflow.start_run(run_name="test_complete_pipeline"):
#     result = query_with_validation(
#         user_query=test_query,
#         validate=True
#     )
    
#     # Log complete result
#     mlflow.log_dict(result, "complete_validated_result.json")

# print("\n" + "="*70)
# print("RESULTS SUMMARY")
# print("="*70)

# # Display validation summary
# val_summary = result.get('validation_summary', {})
# print(f"\nValidation Summary:")
# print(f"  Total facilities: {val_summary.get('total_facilities', 0)}")
# print(f"  Passed: {val_summary.get('passed', 0)}")
# print(f"  Failed: {val_summary.get('failed', 0)}")
# print(f"  Specialty: {val_summary.get('claimed_specialty', 'N/A')}")

# # Display tier1 with validation
# print(f"\nTier 1 Confirmed ({len(result.get('tier1_confirmed', []))}):") 
# for i, fac in enumerate(result.get('tier1_confirmed', [])[:3], 1):
#     val = fac.get('validation', {})
#     verdict = val.get('final_verdict', 'UNKNOWN')
#     confidence = val.get('confidence', 'Low')
    
#     verdict_icon = "✓" if verdict == "PASS" else ("⚠" if "WARNING" in verdict else "✗")
    
#     print(f"\n  {verdict_icon} {i}. {fac.get('name')}")
#     print(f"     Location: {fac.get('city')}, {fac.get('state')}")
#     print(f"     Trust Score: {fac.get('trust_score')}/10")
#     print(f"     Validation: {verdict} ({confidence} confidence)")
    
#     if val.get('supporting_cite'):
#         cite = val['supporting_cite'][:100]
#         print(f"     Evidence: \"{cite}...\"")
    
#     if val.get('violations'):
#         print(f"     Violations: {len(val['violations'])}")
#         for v in val['violations'][:2]:
#             print(f"       - {v}")
    
#     if val.get('warnings'):
#         print(f"     Warnings: {len(val['warnings'])}")

# print("\n" + "="*70)
# print("Test complete! Result logged to MLflow.")
# print("="*70)

# COMMAND ----------

# DBTITLE 1,Streamlit Integration Guide
# MAGIC %md
# MAGIC # Streamlit Integration Guide
# MAGIC
# MAGIC ## Overview
# MAGIC The MediAlert system now has a complete **Agent → Validator** pipeline ready for Streamlit integration.
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC Streamlit App
# MAGIC      ↓
# MAGIC   query_with_validation(user_query)
# MAGIC      ↓
# MAGIC   ┌─────────────────┐
# MAGIC   │  03_agent       │  → Finds facilities using vector search + reasoning
# MAGIC   │  query_agent_json() │  → Returns structured JSON (tier1, tier2)
# MAGIC   └─────────────────┘
# MAGIC      ↓
# MAGIC   ┌─────────────────┐
# MAGIC   │  04_validator   │  → Rule-based checks (equipment, staff, flags)
# MAGIC   │  validate_agent_results() │  → LLM validation (evidence extraction)
# MAGIC   └─────────────────┘  → Returns enhanced JSON with validation
# MAGIC      ↓
# MAGIC   Enhanced JSON to Streamlit
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Single Entry Point for Streamlit
# MAGIC
# MAGIC **Function**: `query_with_validation(user_query, validate=True)`
# MAGIC
# MAGIC **Location**: Notebook `03_agent`, Cell 7.6
# MAGIC
# MAGIC **Usage**:
# MAGIC ```python
# MAGIC result = query_with_validation(
# MAGIC     user_query="Find ICU facilities in Mumbai",
# MAGIC     validate=True  # Set False to skip validation for speed
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Response Structure
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "mlflow_metadata": {
# MAGIC     "run_id": "...",
# MAGIC     "model_endpoint": "...",
# MAGIC     "total_tool_calls": 2,
# MAGIC     "iterations": 3,
# MAGIC     "mlflow_ui_url": "#mlflow/experiments/runs/..."
# MAGIC   },
# MAGIC   
# MAGIC   "search_metadata": {
# MAGIC     "query_type": "procedure_specific",
# MAGIC     "requested_procedure": "appendectomy",
# MAGIC     "location": "Bihar",
# MAGIC     "search_conducted": true,
# MAGIC     "verification_performed": true
# MAGIC   },
# MAGIC   
# MAGIC   "tier1_confirmed": [
# MAGIC     {
# MAGIC       "facility_id": "123",
# MAGIC       "name": "Apollo Hospital",
# MAGIC       "city": "Mumbai",
# MAGIC       "state": "Maharashtra",
# MAGIC       "trust_score": 8.5,
# MAGIC       "evidence": {
# MAGIC         "primary_quote": "24/7 ICU with ventilators",
# MAGIC         "source_field": "procedures"
# MAGIC       },
# MAGIC       
# MAGIC       "validation": {
# MAGIC         "final_verdict": "PASS",
# MAGIC         "confidence": "High",
# MAGIC         "rule_verdict": "PASS",
# MAGIC         "supporting_cite": "ICU with 20 beds, 24/7 intensivist coverage",
# MAGIC         "contradiction": null,
# MAGIC         "violations": [],
# MAGIC         "warnings": ["No equipment list available"],
# MAGIC         "corrective_action": null,
# MAGIC         "reasoning_chain": ["Found ICU keyword", "Staff verified", "No contradictions"]
# MAGIC       },
# MAGIC       
# MAGIC       "validated": true,
# MAGIC       "validation_confidence": "High"
# MAGIC     }
# MAGIC   ],
# MAGIC   
# MAGIC   "tier2_possible": [...],
# MAGIC   
# MAGIC   "validation_summary": {
# MAGIC     "total_facilities": 5,
# MAGIC     "passed": 3,
# MAGIC     "failed": 2,
# MAGIC     "tier1_passed": 2,
# MAGIC     "tier1_failed": 1,
# MAGIC     "tier2_passed": 1,
# MAGIC     "tier2_failed": 1,
# MAGIC     "claimed_specialty": "ICU"
# MAGIC   },
# MAGIC   
# MAGIC   "confidence": {
# MAGIC     "level": "high",
# MAGIC     "reasoning": "..."
# MAGIC   },
# MAGIC   
# MAGIC   "summary": "Found 5 facilities..."
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Streamlit Display Logic
# MAGIC
# MAGIC ### 1. Show Validation Summary
# MAGIC ```python
# MAGIC val_summary = result['validation_summary']
# MAGIC st.metric("Total Facilities", val_summary['total_facilities'])
# MAGIC st.metric("✓ Passed", val_summary['passed'])
# MAGIC st.metric("✗ Failed", val_summary['failed'])
# MAGIC ```
# MAGIC
# MAGIC ### 2. Display Tier 1 Facilities (with validation)
# MAGIC ```python
# MAGIC for facility in result['tier1_confirmed']:
# MAGIC     validation = facility['validation']
# MAGIC     verdict = validation['final_verdict']
# MAGIC     
# MAGIC     # Color code by verdict
# MAGIC     if verdict == 'PASS':
# MAGIC         st.success(f"✓ {facility['name']} - VERIFIED")
# MAGIC     elif 'WARNING' in verdict:
# MAGIC         st.warning(f"⚠ {facility['name']} - PASS WITH WARNINGS")
# MAGIC     else:
# MAGIC         st.error(f"✗ {facility['name']} - FAILED VALIDATION")
# MAGIC     
# MAGIC     # Show evidence
# MAGIC     if validation['supporting_cite']:
# MAGIC         st.info(f"Evidence: {validation['supporting_cite']}")
# MAGIC     
# MAGIC     # Show violations
# MAGIC     if validation['violations']:
# MAGIC         with st.expander("⚠ Violations"):
# MAGIC             for v in validation['violations']:
# MAGIC                 st.write(f"- {v}")
# MAGIC     
# MAGIC     # Show warnings
# MAGIC     if validation['warnings']:
# MAGIC         with st.expander("📋 Warnings"):
# MAGIC             for w in validation['warnings']:
# MAGIC                 st.write(f"- {w}")
# MAGIC ```
# MAGIC
# MAGIC ### 3. Handle Tier 2 (Unconfirmed)
# MAGIC ```python
# MAGIC if result['tier2_possible']:
# MAGIC     st.subheader("⚠ Tier 2: Possible Matches (Unconfirmed)")
# MAGIC     st.caption("These facilities have related capabilities but lack explicit confirmation")
# MAGIC     
# MAGIC     for facility in result['tier2_possible']:
# MAGIC         validation = facility.get('validation', {})
# MAGIC         with st.expander(f"{facility['name']} - {facility.get('city')}"):
# MAGIC             st.write(f"**Related Capability**: {facility.get('related_capability')}")
# MAGIC             st.write(f"**Trust Score**: {facility.get('trust_score')}/10")
# MAGIC             st.write(f"**Validation**: {validation.get('final_verdict', 'N/A')}")
# MAGIC             
# MAGIC             if validation.get('corrective_action'):
# MAGIC                 st.info(f"💡 {validation['corrective_action']}")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Optional: Show MLflow Debug Info
# MAGIC ```python
# MAGIC with st.expander("🔍 Debug Info (MLflow)"):
# MAGIC     mlflow_meta = result['mlflow_metadata']
# MAGIC     st.json(mlflow_meta)
# MAGIC     st.markdown(f"[View in MLflow]({mlflow_meta['mlflow_ui_url']})")
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Benefits of This Integration
# MAGIC
# MAGIC 1. **Single Function Call**: Streamlit just calls `query_with_validation()`
# MAGIC 2. **Automatic Validation**: All facilities validated against medical standards
# MAGIC 3. **Evidence-Based**: Every claim backed by exact quotes from facility notes
# MAGIC 4. **Contradiction Detection**: Flags facilities claiming capabilities they lack
# MAGIC 5. **Two-Layer Validation**: Rule-based + LLM reasoning
# MAGIC 6. **Structured Output**: Easy to parse and display in Streamlit
# MAGIC 7. **MLflow Tracking**: Full observability for debugging
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Testing
# MAGIC
# MAGIC Run Cell 7.7 to test the complete pipeline:
# MAGIC ```python
# MAGIC result = query_with_validation(
# MAGIC     "Find facilities in Bihar that can perform appendectomy"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC Check the output for:
# MAGIC - ✓ Validation summary statistics
# MAGIC - ✓ Tier 1 facilities with verdicts
# MAGIC - ✓ Evidence citations
# MAGIC - ✓ Violations and warnings
# MAGIC - ✓ MLflow tracking

# COMMAND ----------

# =============================================================
# CELL 7b — CONFIDENCE SCORING  (NEW — Areas of Research Q1)
# Statistical prediction intervals on trust scores
# =============================================================

import numpy as np
from scipy import stats

@mlflow.trace(span_type=SpanType.TOOL, name="compute_confidence_interval")
def compute_confidence_interval(facility_name: str) -> dict:
    """
    Computes a statistical confidence interval around a facility's
    trust score using a Wilson score interval approach.

    This answers the hackathon's Areas of Research Q1:
    "Can we use statistics-based methods to create prediction intervals?"

    The Wilson interval treats each trust signal (social presence,
    staff record, logo, facts, followers) as a Bernoulli trial.
    The interval width tells you how much to trust the trust score.
    """
    rows = spark.sql(f"""
        SELECT composite_trust, trust_social_count, trust_has_staff,
               trust_has_logo, trust_fact_count, trust_followers,
               gap_no_notes, gap_no_doctor_count, gap_no_equipment_data,
               flag_icu_contradiction, flag_surgery_no_anaesthesia,
               num_doctors, capacity
        FROM {SQL_TABLE}
        WHERE lower(name) LIKE lower('%{facility_name.replace("'","''")}%')
        LIMIT 1
    """).toPandas()

    if rows.empty:
        return {"error": f"Facility '{facility_name}' not found"}

    r = rows.iloc[0]

    # Each binary signal is a "trial" — did it pass?
    signals = {
        "has_social_presence" : min(float(r.get("trust_social_count", 0) or 0) / 3, 1.0),
        "has_staff_record"    : float(r.get("trust_has_staff", 0) or 0),
        "has_logo"            : float(r.get("trust_has_logo", 0) or 0),
        "has_facts"           : min(float(r.get("trust_fact_count", 0) or 0) / 5, 1.0),
        "has_followers"       : 1.0 if float(r.get("trust_followers", 0) or 0) > 100 else 0.0,
        "has_doctors"         : 0.0 if bool(r.get("gap_no_doctor_count", False)) else 1.0,
        "has_notes"           : 0.0 if bool(r.get("gap_no_notes", False)) else 1.0,
        "no_icu_contradiction": 0.0 if bool(r.get("flag_icu_contradiction", False)) else 1.0,
        "no_surg_contradiction": 0.0 if bool(r.get("flag_surgery_no_anaesthesia", False)) else 1.0,
    }

    n       = len(signals)          # number of signals (trials)
    k       = sum(signals.values()) # number passing
    p_hat   = k / n                 # observed proportion

    # Wilson score interval (better than normal approx for small n)
    z       = 1.96                  # 95% confidence
    denom   = 1 + z**2 / n
    centre  = (p_hat + z**2 / (2*n)) / denom
    margin  = (z * math.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / denom

    lower_ci = max(0.0, round(centre - margin, 3))
    upper_ci = min(1.0, round(centre + margin, 3))

    # Convert to 0–10 trust scale
    trust_lower = round(lower_ci * 10, 2)
    trust_upper = round(upper_ci * 10, 2)
    trust_point = round(p_hat   * 10, 2)

    # Interval width tells us data completeness
    width = upper_ci - lower_ci
    data_quality = "High"   if width < 0.15 else \
                   "Medium" if width < 0.30 else "Low"

    return {
        "facility_name"      : facility_name,
        "trust_point_est"    : trust_point,
        "trust_ci_lower"     : trust_lower,
        "trust_ci_upper"     : trust_upper,
        "trust_interval_str" : f"{trust_lower}–{trust_upper} / 10",
        "data_quality"       : data_quality,
        "signals_checked"    : n,
        "signals_passed"     : k,
        "signals_detail"     : signals,
        "interpretation"     : (
            f"With 95% confidence, the true trust score lies between "
            f"{trust_lower} and {trust_upper} out of 10. "
            f"{'Narrow interval = reliable data.' if width < 0.2 else 'Wide interval = incomplete data, interpret with caution.'}"
        )
    }

# Test it
print(compute_confidence_interval("Apollo"))

# COMMAND ----------

# DBTITLE 1,Test: Appendectomy search with enhanced logic
# =============================================================
# TEST: Verify improved flag/gap logic and procedure verification
# =============================================================

print("\n" + "="*70)
print("TEST 1: Bihar appendectomy query (previously hallucinated)")
print("="*70)

answer_bihar = query_agent(
    "Find the nearest facility in rural Bihar that can perform an emergency "
    "appendectomy and typically leverages part-time doctors.",
    run_name="test_bihar_appendectomy_fixed"
)

print("\n" + "="*70)
print("Expected: Should now return TIER 2 (unconfirmed) or state no explicit capability found")
print("="*70)

# COMMAND ----------

# DBTITLE 1,TEST: JSON Output for Streamlit
# =============================================================
# TEST: JSON OUTPUT FOR STREAMLIT FRONTEND
# =============================================================

print("\n" + "="*70)
print("TEST: Bihar appendectomy query with JSON output")
print("="*70)

# Run query with JSON parsing
result_json = query_agent_json(
    "Find the nearest facility in rural Bihar that can perform an emergency "
    "appendectomy and typically leverages part-time doctors.",
    run_name="test_json_output_bihar"
)

# Display structured results for frontend consumption
print("\n" + "="*70)
print("STRUCTURED JSON RESULT (ready for Streamlit)")
print("="*70)
print(json.dumps(result_json, indent=2))

print("\n" + "="*70)
print("FRONTEND USAGE EXAMPLE:")
print("="*70)
print(f"Query Type: {result_json.get('search_metadata', {}).get('query_type')}")
print(f"Location: {result_json.get('search_metadata', {}).get('location')}")
print(f"Confidence: {result_json.get('confidence', {}).get('level')}")
print(f"\nTier 1 Facilities: {len(result_json.get('tier1_confirmed', []))}")
for i, fac in enumerate(result_json.get('tier1_confirmed', [])[:3], 1):
    print(f"  {i}. {fac.get('name')} - Trust: {fac.get('trust_score')}/10")
    print(f"     {fac.get('city')}, {fac.get('state')} - {fac.get('pincode')}")
    evidence = fac.get('evidence', {})
    if isinstance(evidence, dict):
        primary_quote = evidence.get('primary_quote', 'N/A')
        print(f"     Evidence: {primary_quote[:100]}..." if len(primary_quote) > 100 else f"     Evidence: {primary_quote}")
    else:
        print(f"     Evidence: {str(evidence)[:100]}...")

if result_json.get('tier2_possible'):
    print(f"\nTier 2 Facilities (Unconfirmed): {len(result_json.get('tier2_possible', []))}")
    for i, fac in enumerate(result_json.get('tier2_possible', [])[:2], 1):
        print(f"  {i}. {fac.get('name')} - Trust: {fac.get('trust_score')}/10")
        verification_note = fac.get('verification_note', 'N/A')
        print(f"     Note: {verification_note[:80]}..." if len(verification_note) > 80 else f"     Note: {verification_note}")


# COMMAND ----------

# =============================================================
# CELL 8 — BATCH EVALUATION QUERIES
# Run all benchmark queries and log to a single MLflow run.
# =============================================================

EVAL_QUERIES = [
    "Find a facility with a functional ICU and neonatal beds available 24/7 in Uttar Pradesh.",
    "Which states have the fewest dialysis centres? Identify the top medical deserts.",
    "Find a cancer treatment centre with chemotherapy capability in a rural district.",
    "List facilities in Rajasthan that have emergency trauma care and a blood bank.",
    "Find a hospital in Maharashtra claiming advanced surgery but no anaesthesiologist — flag it.",
    "Where is oncology most critically absent across India?",
]

all_results = []
with mlflow.start_run(run_name="MediAlert_Batch_Eval"):
    for i, q in enumerate(EVAL_QUERIES):
        print(f"\n{'='*60}\nQuery {i+1}/{len(EVAL_QUERIES)}: {q[:80]}")
        result = run_agent(q)
        all_results.append({"query": q, "answer": result["answer"], "iters": result["iterations"]})
        mlflow.log_text(result["answer"], f"query_{i+1}_answer.txt")
        mlflow.log_metric(f"query_{i+1}_iterations", result["iterations"])

    mlflow.log_metric("total_queries", len(EVAL_QUERIES))
    mlflow.log_dict(all_results, "batch_eval_results.json")

print(f"\nAll {len(EVAL_QUERIES)} queries complete. Check MLflow Experiments for traces.")


# COMMAND ----------

# =============================================================
# CELL 9 — INTERACTIVE MODE (run one custom query)
# Change the string below and re-run this cell anytime.
# =============================================================

MY_QUERY = """
Find the 3 best facilities in Tamil Nadu with ICU beds, dialysis,
and at least 1 doctor on record. Flag any that have contradictions
in their data and show me the exact text from their notes.
"""

final_answer = query_agent(MY_QUERY, run_name="custom_query")


# COMMAND ----------

# =============================================================
# CELL 10 — VERIFY: Check MLflow traces are recorded
# =============================================================

runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT], max_results=5)
print(f"MLflow runs recorded: {len(runs)}")
if not runs.empty:
    print(runs[["run_id", "tags.mlflow.runName", "metrics.iterations", "metrics.tool_calls"]].to_string())
    print("\nOpen the Experiments tab in Databricks to view full agent traces.")
else:
    print("No runs found — check MLFLOW_EXPERIMENT path.")

# =============================================================
# END OF NOTEBOOK 03
# Next: 04_validator.py  (Self-Correction / Validator Agent)
# =============================================================

# COMMAND ----------

