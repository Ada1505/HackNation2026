"""
MediAlert — Streamlit App
=========================
Full conversion of the Flask/WhatsApp backend into a Streamlit UI.

Tabs:
  1. 🔍 Find Facilities   — keyword search across Databricks
  2. 📋 Facility Status   — lookup by ID with flag warnings
  3. 🗺  Desert Map        — medical desert analysis by specialty
  4. 📍 Near Me           — Distance comparison (OpenStreetMap / Nominatim)
  5. 💬 WhatsApp          — Twilio WhatsApp sandbox simulator + sender

Run:  streamlit run streamlit_app.py
"""

import os, re, time, math, json
from collections import defaultdict
from typing import Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Backend (Databricks vector search + agent) ─────────────────
try:
    from backend import (
        vector_search   as _vs_search,
        run_agent       as _run_agent,
        is_vs_configured  as _vs_ok,
        is_llm_configured as _llm_ok,
    )
    _BACKEND_LOADED = True
except ImportError:
    _BACKEND_LOADED = False
    def _vs_search(*a, **kw): return []
    def _run_agent(*a, **kw): return {
        "tier1_confirmed": [], "tier2_possible": [],
        "confidence": {"level": "N/A", "reasoning": "backend.py not found"},
        "search_metadata": {}, "meta": {"total_scanned": 0, "matches_found": 0, "avg_trust_score": "N/A"},
        "no_results_message": "backend.py could not be imported.",
        "error": "import_error", "trace": None,
    }
    def _vs_ok(): return False
    def _llm_ok(): return False

# ─────────────────────────────────────────────────────────────
# CONFIG  (reads from .env or Streamlit secrets)
# ─────────────────────────────────────────────────────────────
def _cfg(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

DATABRICKS_HOST   = _cfg("DATABRICKS_HOST")
DATABRICKS_TOKEN  = _cfg("DATABRICKS_TOKEN")
DATABRICKS_SQL_WH = _cfg("DATABRICKS_SQL_WH")
TWILIO_SID        = _cfg("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN      = _cfg("TWILIO_AUTH_TOKEN")
TWILIO_FROM       = _cfg("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

SQL_TABLE    = "workspace.default.facilities_sql"
EMBED_TABLE  = "workspace.default.facilities_for_embedding"
AUDIT_TABLE  = "workspace.default.facilities_audit"
NOMINATIM_UA = "MediAlert/1.0"

# ─────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediAlert",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1923 0%, #1a2a3a 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c8dff0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label { color: #7aaecf !important; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }

/* Main header */
.medi-header {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 60%, #0f2d4a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px 20px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.medi-header::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 320px; height: 100%;
    background: radial-gradient(ellipse at 80% 50%, rgba(30,140,255,0.08) 0%, transparent 70%);
}
.medi-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem; margin: 0;
    background: linear-gradient(90deg, #e8f4ff, #7ec8ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.medi-header p { color: #7aaecf; font-size: 0.92rem; margin: 6px 0 0; }

/* Cards */
.facility-card {
    background: #0d1f33;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
}
.facility-card:hover { border-color: #2d6aad; }
.facility-card h4 { color: #e8f4ff; font-size: 1.05rem; margin: 0 0 6px; }
.facility-card .meta { color: #7aaecf; font-size: 0.83rem; }
.facility-card .trust-high { color: #4ade80; font-weight: 600; }
.facility-card .trust-mid  { color: #facc15; font-weight: 600; }
.facility-card .trust-low  { color: #f87171; font-weight: 600; }

/* Flag badges */
.flag-badge {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    color: #fca5a5;
    border-radius: 20px;
    font-size: 0.72rem;
    padding: 2px 10px;
    margin: 3px 3px 0 0;
}
.flag-ok {
    background: rgba(74,222,128,0.12);
    border-color: rgba(74,222,128,0.35);
    color: #86efac;
}

/* Desert bars */
.desert-row {
    display: flex; align-items: center; gap: 12px;
    margin: 8px 0; font-size: 0.88rem;
}
.desert-bar-wrap { flex: 1; background: #0d1f33; border-radius: 4px; height: 14px; overflow: hidden; }
.desert-bar { height: 100%; border-radius: 4px; transition: width 0.6s ease; }

/* Metric tiles */
.metric-tile {
    background: #0d1f33;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-tile .val { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #7ec8ff; }
.metric-tile .lbl { color: #7aaecf; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }

/* Chat bubbles */
.chat-wrap { max-height: 440px; overflow-y: auto; padding: 10px 0; }
.bubble { border-radius: 12px; padding: 10px 16px; margin: 6px 0; max-width: 80%; font-size: 0.88rem; line-height: 1.5; white-space: pre-wrap; }
.bubble-user { background: #1a4a7a; color: #e8f4ff; margin-left: auto; border-bottom-right-radius: 2px; }
.bubble-bot  { background: #0d2137; border: 1px solid #1e3a5f; color: #c8dff0; border-bottom-left-radius: 2px; }
.bubble-wrap-user { display: flex; justify-content: flex-end; }
.bubble-wrap-bot  { display: flex; justify-content: flex-start; }

/* Distance compare table */
.dist-table { width: 100%; border-collapse: collapse; font-size: 0.87rem; }
.dist-table th { background: #0d1f33; color: #7aaecf; padding: 8px 12px; text-align: left; border-bottom: 1px solid #1e3a5f; font-weight: 500; font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; }
.dist-table td { padding: 10px 12px; border-bottom: 1px solid #0d1f33; color: #c8dff0; }
.dist-table tr:hover td { background: #0d1f33; }
.dist-winner { color: #4ade80; font-weight: 600; }

/* Location badge */
.loc-badge {
    background: rgba(30,140,255,0.12);
    border: 1px solid rgba(30,140,255,0.35);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.85rem;
    color: #7ec8ff;
    margin-bottom: 12px;
}

/* Input overrides */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] select {
    background: #0d1f33 !important;
    border: 1px solid #1e3a5f !important;
    color: #e8f4ff !important;
    border-radius: 8px !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1a5fa8, #0f4080) !important;
    color: #e8f4ff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
}
div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #2270c0, #1a5fa8) !important;
}

/* Tab styling */
[data-testid="stTab"] { color: #7aaecf !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# INDIA REFERENCE DATA  (cities + conditions for quick-select)
# ─────────────────────────────────────────────────────────────
INDIA_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad",
    "Chennai", "Kolkata", "Pune", "Jaipur", "Surat",
    "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
    "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad",
    "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut",
    "Rajkot", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad",
    "Amritsar", "Allahabad", "Ranchi", "Coimbatore", "Jodhpur",
    "Madurai", "Raipur", "Kochi", "Chandigarh", "Guwahati",
    "Thiruvananthapuram", "Bhubaneswar", "Dehradun", "Mysore",
]

MAJOR_CONDITIONS = [
    "ICU", "Dialysis", "Oncology", "Surgery", "Neonatal",
    "Emergency", "Blood Bank", "Cardiac", "Orthopaedic", "Neurology",
]

MAJOR_CITIES_BUTTONS = [
    "Mumbai", "Delhi", "Bangalore", "Chennai",
    "Hyderabad", "Pune", "Kolkata", "Jaipur",
]


# ─────────────────────────────────────────────────────────────
# DATABRICKS SQL HELPER
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def db_sql(sql: str) -> list[dict]:
    """Execute SQL via Databricks SQL Warehouse REST API. Returns [] on any error."""
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN or not DATABRICKS_SQL_WH:
        st.warning("⚠️ Databricks not configured — check .env for HOST / TOKEN / SQL_WH.")
        return []
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
    # Ensure warehouse is running
    try:
        state = requests.get(f"{DATABRICKS_HOST}/api/2.0/sql/warehouses/{DATABRICKS_SQL_WH}",
                             headers=headers, timeout=10).json().get("state", "")
        if state in ("STOPPED", "STOPPING"):
            requests.post(f"{DATABRICKS_HOST}/api/2.0/sql/warehouses/{DATABRICKS_SQL_WH}/start",
                          headers=headers, timeout=10)
            for _ in range(24):
                time.sleep(5)
                state = requests.get(f"{DATABRICKS_HOST}/api/2.0/sql/warehouses/{DATABRICKS_SQL_WH}",
                                     headers=headers, timeout=10).json().get("state", "")
                if state == "RUNNING":
                    break
    except Exception:
        pass
    try:
        r = requests.post(
            f"{DATABRICKS_HOST}/api/2.0/sql/statements",
            headers=headers,
            json={"warehouse_id": DATABRICKS_SQL_WH, "statement": sql,
                  "wait_timeout": "30s", "on_wait_timeout": "CANCEL", "format": "JSON_ARRAY"},
            timeout=35,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status", {}).get("state") != "SUCCEEDED":
            st.error(f"SQL error: {data.get('status', {}).get('error', {})}")
            return []
        cols = [c["name"] for c in data.get("manifest", {}).get("schema", {}).get("columns", [])]
        return [dict(zip(cols, row)) for row in data.get("result", {}).get("data_array", [])]
    except Exception as e:
        st.error(f"Databricks error: {e}")
        return []


def sanitize(text: str, max_len: int = 120) -> str:
    return re.sub(r"['\";\\]", "", text).strip()[:max_len]


# ─────────────────────────────────────────────────────────────
# GEO HELPERS  (Nominatim / OpenStreetMap only — no API key)
# ─────────────────────────────────────────────────────────────
def geocode(place: str) -> Optional[Tuple[float, float]]:
    """
    Geocode a place name using OpenStreetMap Nominatim.
    No API key required. Returns (lat, lng) or None.
    """
    place = sanitize(place)
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params  = {"q": place, "format": "json", "limit": 1},
            headers = {"User-Agent": NOMINATIM_UA},
            timeout = 10,
        )
        results = r.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None


def reverse_geocode(lat: float, lng: float) -> str:
    """
    Reverse geocode coordinates to a human-readable address using Nominatim.
    Returns a formatted address string or coordinate fallback.
    """
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params  = {"lat": lat, "lon": lng, "format": "json"},
            headers = {"User-Agent": NOMINATIM_UA},
            timeout = 10,
        )
        data = r.json()
        return data.get("display_name", f"{lat:.4f}, {lng:.4f}")
    except Exception:
        return f"{lat:.4f}, {lng:.4f}"


def _extract_coords_from_url(url: str) -> Optional[str]:
    """
    Extract lat,lng string from a map URL.
    Handles Google Maps, Apple Maps, WhatsApp shares, and goo.gl short links.
    Returns "lat,lng" string or None.
    """
    # Follow redirects for shortened URLs (goo.gl/maps, maps.app.goo.gl, etc.)
    if any(s in url for s in ["goo.gl", "maps.app", "bit.ly", "tinyurl"]):
        try:
            r = requests.head(url, allow_redirects=True, timeout=8,
                              headers={"User-Agent": NOMINATIM_UA})
            url = r.url
        except Exception:
            pass  # Use original URL if redirect fails

    patterns = [
        # Google Maps /@lat,lng or @lat,lng,zoom
        r"@(-?\d+\.?\d+),(-?\d+\.?\d+)",
        # ?q=lat,lng or ?q=lat%2Clng
        r"[?&]q=(-?\d+\.?\d+)[,%2C]+(-?\d+\.?\d+)",
        # ?ll=lat,lng  (Apple Maps, some Google variants)
        r"[?&]ll=(-?\d+\.?\d+),(-?\d+\.?\d+)",
        # ?center=lat,lng
        r"[?&]center=(-?\d+\.?\d+),(-?\d+\.?\d+)",
        # /place/...!3dlat!4dlng
        r"!3d(-?\d+\.?\d+)!4d(-?\d+\.?\d+)",
        # Bare lat,lng anywhere in URL (last resort)
        r"(-?\d{1,2}\.\d{4,}),(-?\d{2,3}\.\d{4,})",
    ]

    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            lat, lng = float(m.group(1)), float(m.group(2))
            # Sanity check — India bounding box (roughly)
            if 6 <= lat <= 38 and 68 <= lng <= 98:
                return f"{lat:.6f},{lng:.6f}"
            # Accept anyway if outside India box (user might be testing)
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                return f"{lat:.6f},{lng:.6f}"

    return None


def haversine(lat1, lng1, lat2, lng2) -> float:
    R  = 6371
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lng2 - lng1)
    a  = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 2)


# ─────────────────────────────────────────────────────────────
# TWILIO WHATSAPP HELPER
# ─────────────────────────────────────────────────────────────
def twilio_send(to: str, body: str) -> bool:
    """Send via Twilio WhatsApp sandbox. Returns True on success."""
    if not TWILIO_SID or not TWILIO_TOKEN:
        return False
    try:
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
            auth = (TWILIO_SID, TWILIO_TOKEN),
            data = {"From": TWILIO_FROM, "To": f"whatsapp:{to}", "Body": body[:1600]},
            timeout = 15,
        )
        return r.status_code == 201
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# TRUST SCORE COLOUR
# ─────────────────────────────────────────────────────────────
def trust_class(score) -> str:
    try:
        s = float(score)
        if s >= 7:  return "trust-high"
        if s >= 4:  return "trust-mid"
        return "trust-low"
    except Exception:
        return "trust-mid"


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 MediAlert")
    st.markdown("---")
    st.markdown("**Connections**")
    db_ok     = bool(DATABRICKS_HOST and DATABRICKS_TOKEN)
    twilio_ok = bool(TWILIO_SID and TWILIO_TOKEN)
    vs_ok  = _vs_ok()
    llm_ok = _llm_ok()
    st.markdown(f"{'🟢' if db_ok     else '🔴'} Databricks SQL {'connected' if db_ok else 'not configured'}")
    st.markdown(f"{'🟢' if vs_ok     else '🟡'} Vector Search {'ready' if vs_ok else 'needs VS_INDEX'}")
    st.markdown(f"{'🟢' if llm_ok    else '🟡'} LLM Agent {'ready' if llm_ok else 'needs LLM_ENDPOINT'}")
    st.markdown(f"{'🟢' if twilio_ok else '🟡'} Twilio WhatsApp {'ready' if twilio_ok else 'not configured'}")
    st.markdown("🟢 OpenStreetMap (Nominatim) — always on")

    st.markdown("---")
    st.markdown("**Quick filters**")
    sidebar_state = st.text_input("Filter by state", placeholder="e.g. Maharashtra")
    sidebar_city  = st.text_input("Filter by city",  placeholder="e.g. Pune")

    # ── Agent System Insights (populated after agent runs) ────
    st.markdown("---")
    st.markdown("**📊 System Insights**")
    if "agent_insights" in st.session_state and st.session_state.agent_insights:
        ins = st.session_state.agent_insights
        st.metric("Facilities Scanned", f"{ins.get('total_scanned', 10000):,}")
        st.metric("Matches Found",      ins.get("matches_found", 0))
        st.metric("Avg Trust Score",    ins.get("avg_trust_score", "—"))
        if ins.get("iterations"):
            st.markdown(f"🔄 **{ins['iterations']}** reasoning iterations")
        if ins.get("tool_calls"):
            st.markdown(f"⚙️ **{ins['tool_calls']}** tool calls made")
        if ins.get("total_ms"):
            st.markdown(f"⏱ **{ins['total_ms']} ms** total latency")
    else:
        st.markdown("<span style='color:#4a7a9b;font-size:0.82rem'>Run the AI Agent to see live insights</span>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.72rem;color:#4a7a9b'>"
        "🔎 This system uses AI to analyze real hospital data and detect inconsistencies.<br><br>"
        "MediAlert v3.0 · 10k+ verified facilities<br>"
        "Powered by Databricks + Llama 3.3-70B<br>"
        "Geocoding by OpenStreetMap · Messaging by Twilio"
        "</span>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="medi-header">
  <h1>MediAlert</h1>
  <p>India's healthcare facility intelligence platform · 10,000+ verified facilities · Real-time Databricks data</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_agent, tab_find, tab_deserts, tab_near, tab_wa = st.tabs([
    "🤖 AI Agent",
    "🔍 Find Facilities",
    "🗺️ Desert Map",
    "📍 Near Me",
    "💬 WhatsApp",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — FIND FACILITIES
# ══════════════════════════════════════════════════════════════
with tab_find:
    st.markdown("#### Search verified healthcare facilities")

    # ── Session state — initialise KEYS that inputs will use ──
    if "find_kw_input"   not in st.session_state: st.session_state.find_kw_input   = ""
    if "find_city_input" not in st.session_state: st.session_state.find_city_input = ""

    # ── Condition quick buttons ────────────────────────────────
    st.markdown("<span style='color:#4a7a9b;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em'>Quick conditions</span>", unsafe_allow_html=True)
    cond_cols = st.columns(len(MAJOR_CONDITIONS))
    for i, (col, cond) in enumerate(zip(cond_cols, MAJOR_CONDITIONS)):
        with col:
            if st.button(cond, key=f"cond_{i}", use_container_width=True):
                st.session_state.find_kw_input = cond   # write directly to input key
                st.rerun()

    # ── City quick buttons ─────────────────────────────────────
    st.markdown("<span style='color:#4a7a9b;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em'>Quick cities</span>", unsafe_allow_html=True)
    city_btn_cols = st.columns(len(MAJOR_CITIES_BUTTONS))
    for i, (col, city) in enumerate(zip(city_btn_cols, MAJOR_CITIES_BUTTONS)):
        with col:
            if st.button(city, key=f"fcity_{i}", use_container_width=True):
                st.session_state.find_city_input = city  # write directly to input key
                st.rerun()

    st.markdown("")
    col_q, col_city, col_sp, col_btn = st.columns([3, 2, 2, 1])
    with col_q:
        # key= is the single source of truth — no value= needed
        keyword = st.text_input(
            "Keyword",
            placeholder="dialysis, ICU, oncology …",
            label_visibility="collapsed",
            key="find_kw_input",
        )
    with col_city:
        city_typed = st.text_input(
            "City",
            placeholder="City (or pick above)",
            label_visibility="collapsed",
            key="find_city_input",
        )
    with col_sp:
        specialty_filter = st.selectbox(
            "Specialty",
            ["Any", "ICU", "Surgery", "Dialysis", "Oncology", "Neonatal", "Emergency", "Cardiac", "Neurology"],
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("Search", use_container_width=True)

    if search_clicked or keyword:
        q_parts = []
        if keyword:
            kw = sanitize(keyword)
            q_parts.append(f"lower(e.notes_blob) LIKE lower('%{kw}%')")
        if specialty_filter != "Any":
            sp = sanitize(specialty_filter)
            q_parts.append(f"lower(e.notes_blob) LIKE lower('%{sp.lower()}%')")
        if city_typed:
            q_parts.append(f"lower(s.city) LIKE lower('%{sanitize(city_typed)}%')")
        if sidebar_state:
            q_parts.append(f"lower(s.state) LIKE lower('%{sanitize(sidebar_state)}%')")
        if sidebar_city:
            q_parts.append(f"lower(s.city) LIKE lower('%{sanitize(sidebar_city)}%')")

        where = ("WHERE " + " AND ".join(q_parts)) if q_parts else ""

        with st.spinner("Querying Databricks …"):
            rows = db_sql(f"""
                SELECT s.name, s.state, s.city, s.pin_code,
                       e.trust_score_raw, s.num_doctors, s.capacity,
                       s.facilityTypeId, s.officialPhone, s.address_line1,
                       s.flag_icu_contradiction, s.flag_surgery_no_anaesthesia,
                       s.gap_no_doctor_count, s.gap_no_equipment_data
                FROM   {SQL_TABLE}   s
                JOIN   {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
                {where}
                ORDER  BY e.trust_score_raw DESC
                LIMIT  20
            """)

        if not rows:
            st.info("No facilities found. Try a broader keyword.")
        else:
            mc1, mc2, mc3, mc4 = st.columns(4)
            avg_trust  = sum(float(r.get("trust_score_raw") or 0) for r in rows) / len(rows)
            total_docs = sum(int(float(r.get("num_doctors") or 0)) for r in rows)
            flagged    = sum(1 for r in rows if any([
                r.get("flag_icu_contradiction"), r.get("flag_surgery_no_anaesthesia"),
                r.get("gap_no_doctor_count"),
            ]))
            for col, val, lbl in [
                (mc1, len(rows),          "Results"),
                (mc2, f"{avg_trust:.1f}/10", "Avg Trust"),
                (mc3, total_docs,         "Total Doctors"),
                (mc4, flagged,            "Flagged"),
            ]:
                with col:
                    st.markdown(f'<div class="metric-tile"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            for r in rows:
                flags_html = ""
                if r.get("flag_icu_contradiction")  in ("true", True, "1", 1):
                    flags_html += '<span class="flag-badge">⚠ ICU/no beds</span>'
                if r.get("flag_surgery_no_anaesthesia") in ("true", True, "1", 1):
                    flags_html += '<span class="flag-badge">⚠ Surgery/no anaes.</span>'
                if r.get("gap_no_doctor_count")          in ("true", True, "1", 1):
                    flags_html += '<span class="flag-badge">⚠ Zero doctors</span>'
                if r.get("gap_no_equipment_data")          in ("true", True, "1", 1):
                    flags_html += '<span class="flag-badge">⚠ No equipment</span>'
                if not flags_html:
                    flags_html = '<span class="flag-badge flag-ok">✓ No flags</span>'

                tc = trust_class(r.get("trust_score_raw"))
                st.markdown(f"""
<div class="facility-card">
  <h4>{r.get('name','N/A')}</h4>
  <div class="meta">
    📍 {r.get('city','')}, {r.get('state','')} &nbsp;·&nbsp; PIN {r.get('pin_code','')}
    &nbsp;·&nbsp; {r.get('facilityTypeId','')}
  </div>
  <div class="meta" style="margin-top:6px">
    ⭐ Trust: <span class="{tc}">{r.get('trust_score_raw','N/A')}/10</span>
    &nbsp;·&nbsp; 📞 {r.get('officialPhone','N/A')}
  </div>
  <div style="margin-top:8px">{flags_html}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — DESERT MAP
# ══════════════════════════════════════════════════════════════
with tab_deserts:
    st.markdown("#### Medical desert analysis — states with fewest facilities")

    col_sp2, col_th, col_run = st.columns([3, 2, 1])
    with col_sp2:
        desert_specialty = st.selectbox(
            "Specialty",
            ["oncology","dialysis","ICU","surgery","neonatal","emergency","blood bank"],
            label_visibility="collapsed",
        )
    with col_th:
        threshold = st.slider("Critical threshold (facilities)", 1, 10, 3, label_visibility="collapsed")
    with col_run:
        desert_clicked = st.button("Analyse", use_container_width=True)

    if desert_clicked:
        sp = sanitize(desert_specialty)
        with st.spinner("Running desert analysis …"):
            # state lives in SQL_TABLE — embed table has no state column
            coverage = db_sql(f"""
                SELECT s.state, COUNT(*) AS facility_count
                FROM   {SQL_TABLE}   s
                JOIN   {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
                WHERE  lower(e.notes_blob) LIKE lower('%{sp}%')
                  AND  s.state IS NOT NULL
                GROUP  BY s.state
                ORDER  BY facility_count ASC
            """)
            totals = db_sql(f"""
                SELECT state, COUNT(*) AS total
                FROM   {SQL_TABLE}
                WHERE  state IS NOT NULL
                GROUP  BY state
                ORDER  BY state
            """)

        if not coverage and not totals:
            st.warning("⚠️ No data returned. Check Databricks connection in sidebar.")
        elif not coverage:
            st.info(f"No facilities found mentioning '{sp}'. Try a different specialty.")
        elif not totals:
            st.warning("⚠️ Could not fetch state totals from Databricks.")
        else:
            total_map = {r["state"]: int(float(r["total"])) for r in totals if r.get("state") and r.get("total") is not None}
            cov_map   = {r["state"]: int(float(r["facility_count"])) for r in coverage if r.get("state") and r.get("facility_count") is not None}

            all_states = []
            for state, total in total_map.items():
                with_sp = cov_map.get(state, 0)
                pct = round(with_sp / total * 100, 1) if total else 0
                all_states.append({"state": state, "with_sp": with_sp, "total": total, "pct": pct})
            all_states.sort(key=lambda x: x["pct"])

            deserts  = [s for s in all_states if s["with_sp"] < threshold]
            adequate = [s for s in all_states if s["with_sp"] >= threshold]

            mc1, mc2, mc3 = st.columns(3)
            for col, val, lbl in [
                (mc1, len(deserts),  f"Critical (<{threshold})"),
                (mc2, len(adequate), "Adequate states"),
                (mc3, f"{sum(s['with_sp'] for s in all_states):,}", f"Total {sp} facilities"),
            ]:
                with col:
                    st.markdown(f'<div class="metric-tile"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Coverage breakdown for *{sp}***")

            for s in all_states[:25]:
                pct   = s["pct"]
                color = "#ef4444" if s["with_sp"] < threshold else ("#facc15" if pct < 15 else "#4ade80")
                icon  = "🔴" if s["with_sp"] < threshold else ("🟡" if pct < 15 else "🟢")
                bar_w = int(pct * 3)
                st.markdown(f"""
<div class="desert-row">
  <span style="width:170px;color:#c8dff0">{icon} {s['state']}</span>
  <div class="desert-bar-wrap">
    <div class="desert-bar" style="width:{bar_w}px;max-width:100%;background:{color}"></div>
  </div>
  <span style="width:110px;color:#7aaecf;text-align:right">{s['with_sp']}/{s['total']} ({pct}%)</span>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"🔴 **Critical deserts** (< {threshold} facilities): " +
                        ", ".join(s["state"] for s in deserts[:10]) or "None")


# ══════════════════════════════════════════════════════════════
# TAB 4 — NEAR ME  (OpenStreetMap / Nominatim only)
# ══════════════════════════════════════════════════════════════
with tab_near:
    st.markdown("#### Nearby facilities — distance comparison")
    st.caption("📍 Enter your location or paste lat,lng coordinates. Uses OpenStreetMap — no API key required.")

    # ── Session state — initialise KEYS that inputs will use ──
    if "near_loc_input"  not in st.session_state: st.session_state.near_loc_input  = ""
    if "near_kw_input"   not in st.session_state: st.session_state.near_kw_input   = "hospital"
    if "url_parse_err"   not in st.session_state: st.session_state.url_parse_err   = ""

    # ── Map URL paste extractor ────────────────────────────────
    with st.expander("🔗 Paste a map link to auto-fill location"):
        st.markdown(
            "<span style='color:#7aaecf;font-size:0.83rem'>"
            "Works with Google Maps, Apple Maps, WhatsApp location shares, and shortened goo.gl links."
            "</span>",
            unsafe_allow_html=True,
        )
        url_col, btn_col = st.columns([5, 1])
        with url_col:
            pasted_url = st.text_input(
                "Map URL",
                placeholder="Paste Google Maps / Apple Maps / WhatsApp location URL here …",
                label_visibility="collapsed",
                key="map_url_input",
            )
        with btn_col:
            parse_clicked = st.button("📍 Extract", use_container_width=True, key="parse_url_btn")

        st.markdown(
            "<span style='color:#4a7a9b;font-size:0.75rem'>"
            "How to copy: Open Google Maps → long-press your location → tap the coordinates at the top → Share → Copy link"
            "</span>",
            unsafe_allow_html=True,
        )

        if parse_clicked and pasted_url.strip():
            extracted = _extract_coords_from_url(pasted_url.strip())
            if extracted:
                st.session_state.near_loc_input = extracted   # write directly to input key
                st.session_state.url_parse_err  = ""
                st.success(f"✅ Coordinates extracted: **{extracted}** — location field auto-filled!")
                st.rerun()
            else:
                st.session_state.url_parse_err = (
                    "❌ Couldn't find coordinates in that URL. "
                    "Try opening the link first in your browser, then copy the full expanded URL."
                )

        if st.session_state.url_parse_err:
            st.error(st.session_state.url_parse_err)

    # ── City quick buttons ─────────────────────────────────────
    st.markdown("<span style='color:#4a7a9b;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em'>Quick cities</span>", unsafe_allow_html=True)
    near_city_cols = st.columns(len(MAJOR_CITIES_BUTTONS))
    for i, (col, city) in enumerate(zip(near_city_cols, MAJOR_CITIES_BUTTONS)):
        with col:
            if st.button(city, key=f"ncity_{i}", use_container_width=True):
                st.session_state.near_loc_input = city   # write directly to input key
                st.rerun()

    # ── Condition quick buttons ────────────────────────────────
    st.markdown("<span style='color:#4a7a9b;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em'>Quick conditions</span>", unsafe_allow_html=True)
    near_cond_cols = st.columns(len(MAJOR_CONDITIONS))
    for i, (col, cond) in enumerate(zip(near_cond_cols, MAJOR_CONDITIONS)):
        with col:
            if st.button(cond, key=f"ncond_{i}", use_container_width=True):
                st.session_state.near_kw_input = cond    # write directly to input key
                st.rerun()

    st.markdown("")
    col_loc, col_kw2, col_near_btn = st.columns([3, 2, 1])
    with col_loc:
        user_location = st.text_input(
            "Your location",
            placeholder="City name or lat,lng (e.g. 19.0760,72.8777)",
            label_visibility="collapsed",
            key="near_loc_input",
        )
    with col_kw2:
        near_keyword = st.text_input(
            "Keyword",
            placeholder="hospital, clinic, ICU …",
            label_visibility="collapsed",
            key="near_kw_input",
        )
    with col_near_btn:
        near_clicked = st.button("Find & Compare", use_container_width=True)

    # ── Bonus: pre-fill from last saved location ───────────────
    if "last_location" in st.session_state and st.session_state.last_location:
        lat_p, lng_p = st.session_state.last_location
        st.markdown(f"""
<div class="loc-badge">
  📍 <b>Last saved location:</b> {lat_p:.5f}, {lng_p:.5f}
  &nbsp;·&nbsp; <span style="opacity:0.7">{st.session_state.get('last_address','')[:60]}</span>
</div>""", unsafe_allow_html=True)
        if st.button("Use last saved location", key="use_tg_loc"):
            user_location = f"{lat_p},{lng_p}"

    if near_clicked and user_location:
        # Parse lat,lng or geocode
        coords = None
        latlon_match = re.match(r"^(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)$", user_location.strip())
        if latlon_match:
            coords = float(latlon_match.group(1)), float(latlon_match.group(2))
        else:
            with st.spinner("Geocoding via OpenStreetMap …"):
                coords = geocode(user_location)

        if not coords:
            st.error("Could not geocode that location. Try a city name or lat,lng coordinates.")
        else:
            user_lat, user_lng = coords

            # Reverse geocode for a readable label
            with st.spinner("Resolving address …"):
                address_label = reverse_geocode(user_lat, user_lng)

            st.success(f"📍 Location resolved: **{user_lat:.5f}, {user_lng:.5f}**")
            st.caption(address_label)

            # Store in session (Bonus)
            st.session_state["last_location"] = (user_lat, user_lng)
            st.session_state["last_address"]  = address_label[:80]

            # ── Databricks: same-city / nearby verified facilities ──
            city_guess = sanitize(user_location.split(",")[0].strip(), 60)
            with st.spinner("Querying Databricks for nearby verified facilities …"):
                db_rows = db_sql(f"""
                    SELECT s.name, s.state, s.city, s.pin_code,
                           e.trust_score_raw, s.num_doctors, s.officialPhone,
                           s.gap_no_doctor_count, s.flag_icu_contradiction,
                           s.lat, s.lon
                    FROM   {SQL_TABLE}   s
                    JOIN   {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
                    WHERE  lower(s.city) LIKE lower('%{city_guess}%')
                       AND lower(e.notes_blob) LIKE lower('%{sanitize(near_keyword)}%')
                    ORDER  BY e.trust_score_raw DESC
                    LIMIT  15
                """)

            # Compute haversine distance where coords available
            for row in db_rows:
                try:
                    rlat = float(row.get("lat") or 0)
                    rlng = float(row.get("lon") or 0)
                    if rlat and rlng:
                        row["dist_km"] = haversine(user_lat, user_lng, rlat, rlng)
                    else:
                        row["dist_km"] = None
                except Exception:
                    row["dist_km"] = None

            # Sort by distance if available
            db_rows_sorted = sorted(
                db_rows,
                key=lambda x: x["dist_km"] if x["dist_km"] is not None else 9999
            )

            st.markdown("---")
            st.markdown("**🏥 Verified facilities (Databricks + OpenStreetMap distance)**")

            if not db_rows_sorted:
                st.info(f"No verified facilities found in '{city_guess}' for '{near_keyword}'.\nTry a broader city name or keyword.")
            else:
                table_html = """
<table class="dist-table">
<thead><tr><th>#</th><th>Name</th><th>Trust</th><th>Distance</th><th>Doctors</th><th>Phone</th></tr></thead><tbody>
"""
                for i, r in enumerate(db_rows_sorted, 1):
                    tc      = trust_class(r.get("trust_score_raw"))
                    warn    = "⚠️ " if r.get("gap_no_doctor_count") in ("true", True, "1", 1) else ""
                    dist    = f"{r['dist_km']} km" if r.get("dist_km") is not None else "—"
                    winner  = ' class="dist-winner"' if i == 1 and r.get("dist_km") else ""
                    table_html += f"""<tr>
<td{winner}>{i}</td>
<td{winner}>{warn}{r['name']}</td>
<td{winner} class="{tc}">{r.get('trust_score_raw','—')}</td>
<td{winner}>{dist}</td>
<td{winner}>{r.get('num_doctors','—')}</td>
<td{winner} style="font-size:0.78rem">{r.get('officialPhone','—')}</td>
</tr>"""
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)

                # Distance bar chart
                has_dist = [r for r in db_rows_sorted if r.get("dist_km") is not None]
                if len(has_dist) >= 2:
                    st.markdown("<br>**Distance comparison**", unsafe_allow_html=True)
                    max_dist = max(r["dist_km"] for r in has_dist)
                    for r in has_dist[:6]:
                        bar_pct = int((r["dist_km"] / max_dist) * 100) if max_dist else 0
                        color   = "#4ade80" if r == has_dist[0] else "#3b82f6"
                        st.markdown(f"""
<div class="desert-row">
  <span style="width:180px;color:#c8dff0;font-size:0.82rem">{r['name'][:24]}</span>
  <div class="desert-bar-wrap"><div class="desert-bar" style="width:{bar_pct}%;background:{color}"></div></div>
  <span style="width:70px;color:#7aaecf;text-align:right;font-size:0.82rem">{r['dist_km']} km</span>
</div>""", unsafe_allow_html=True)

            # ── Folium OSM map ─────────────────────────────────
            if db_rows:
                st.markdown("---")
                st.markdown("**OpenStreetMap view**")
                try:
                    import folium
                    from streamlit_folium import st_folium

                    m = folium.Map(
                        location=[user_lat, user_lng],
                        zoom_start=13,
                        tiles="OpenStreetMap",
                    )

                    # User pin
                    folium.Marker(
                        [user_lat, user_lng],
                        popup="📍 Your Location",
                        icon=folium.Icon(color="blue", icon="home"),
                    ).add_to(m)

                    # Facility pins
                    for r in db_rows_sorted[:10]:
                        try:
                            rlat = float(r.get("lat") or 0)
                            rlng = float(r.get("lon") or 0)
                            if rlat and rlng:
                                dist_str = f" · {r['dist_km']} km" if r.get("dist_km") else ""
                                folium.Marker(
                                    [rlat, rlng],
                                    popup=f"{r['name']}{dist_str}",
                                    icon=folium.Icon(color="red", icon="plus-sign"),
                                ).add_to(m)
                        except Exception:
                            pass

                    st_folium(m, width=None, height=380)
                except ImportError:
                    st.info("Install `streamlit-folium` for map view: `pip install streamlit-folium`")


# ══════════════════════════════════════════════════════════════
# TAB 5 — WHATSAPP  (Twilio sandbox)
# ══════════════════════════════════════════════════════════════
with tab_wa:
    st.markdown("#### WhatsApp — Simulate or send real Twilio messages")

    if not twilio_ok:
        st.info(
            "⚠️ Twilio not yet configured. Add these to your `.env` and restart:\n"
            "```\nTWILIO_ACCOUNT_SID=ACxxxxxxxx\n"
            "TWILIO_AUTH_TOKEN=your_auth_token\n"
            "TWILIO_WHATSAPP_FROM=whatsapp:+14155238886\n```\n\n"
            "The sandbox number above is the Twilio WhatsApp sandbox default — "
            "keep it unless you have an approved sender number."
        )

    # ── Session state ──────────────────────────────────────────
    if "wa_chat"             not in st.session_state: st.session_state.wa_chat             = []
    if "wa_awaiting_location" not in st.session_state: st.session_state.wa_awaiting_location = False
    if "wa_pending_cmd"      not in st.session_state: st.session_state.wa_pending_cmd      = None
    if "last_location"       not in st.session_state: st.session_state.last_location       = None
    if "last_address"        not in st.session_state: st.session_state.last_address        = ""

    col_mode, col_phone = st.columns([2, 3])
    with col_mode:
        mode = st.selectbox("Mode", ["Simulate (local)", "Send real WhatsApp"], label_visibility="collapsed")
    with col_phone:
        phone_to = st.text_input(
            "Phone (with country code)",
            placeholder="+923001234567",
            label_visibility="collapsed",
            disabled=(mode == "Simulate (local)"),
        )

    st.markdown("---")

    # ─────────────────────────────────────────────────────────
    # CHATBOT LOGIC  (unchanged from WhatsApp tab — same commands)
    # ─────────────────────────────────────────────────────────
    def bot_process(user_text: str) -> str:
        """
        Rule-based dispatcher — same logic as the original WhatsApp bot.
        """
        text  = user_text.strip()
        lower = text.lower()

        # ── Awaiting location reply ───────────────────────────
        if st.session_state.wa_awaiting_location:
            st.session_state.wa_awaiting_location = False
            pending = st.session_state.wa_pending_cmd or ""
            st.session_state.wa_pending_cmd = None
            if pending == "near":
                return _handle_near(text)
            elif pending in ("find_with_q",):
                return _handle_find_with_city(text, "")
            else:
                return _handle_near(text)

        # ── Help ──────────────────────────────────────────────
        if re.match(r"^(help|hi|hello|start|menu|commands?)$", lower):
            return (
                "🏥 *MediAlert Commands*\n\n"
                "*find <keyword>* — Search facilities by condition\n"
                "*near <place>* — Hospitals near a location\n"
                "*deserts <specialty>* — Medical desert analysis\n\n"
                "📍 Examples:\n  find ICU Mumbai\n  near Patna\n  deserts dialysis\n\n"
                "💡 No location? Say *find dialysis* and I'll ask you!"
            )

        # ── status <id> ───────────────────────────────────────
        m = re.match(r"^status\s+(\S+)$", lower)
        if m:
            return _handle_status(m.group(1))

        # ── deserts <specialty> ───────────────────────────────
        m = re.match(r"^deserts?\s+(.+)$", lower, re.IGNORECASE)
        if m:
            return _handle_deserts(m.group(1).strip())

        # ── near <place> ──────────────────────────────────────
        m = re.match(r"^near\s+(.+)$", lower, re.IGNORECASE)
        if m:
            place = m.group(1).strip()
            if len(place) < 2:
                return _prompt_location("near")
            return _handle_near(place)

        # ── find <keyword> ────────────────────────────────────
        m = re.match(r"^find\s+(.+)$", lower, re.IGNORECASE)
        if m:
            query     = m.group(1).strip()
            city_hint = _extract_city(query)
            if not city_hint:
                st.session_state.wa_awaiting_location = True
                st.session_state.wa_pending_cmd       = "find_with_q"
                st.session_state["_pending_find_q"]   = query
                return (
                    f"📍 You asked to find *{query}*.\n\n"
                    "Which city or area should I search in?\n"
                    "_Reply with a city name, e.g._ *Lahore* _or_ *Patna*"
                )
            return _handle_find_with_city(city_hint, query)

        # ── Lone city word after location prompt ──────────────
        if st.session_state.wa_awaiting_location:
            return _handle_near(text)

        return (
            "⚠️ Command not recognised.\n\n"
            "Try:\n  status 1234\n  near Lahore\n  find ICU\n  deserts oncology\n\n"
            "Type *help* for the full list."
        )

    def _prompt_location(cmd: str) -> str:
        st.session_state.wa_awaiting_location = True
        st.session_state.wa_pending_cmd       = cmd
        return (
            "📍 I need your location to search nearby.\n\n"
            "Please reply with:\n"
            "• Your *city or area* name _(e.g. Andheri, Mumbai)_\n"
            "• Or share your *WhatsApp location pin* 📎"
        )

    def _extract_city(query: str) -> str:
        known = {c.lower() for c in INDIA_CITIES}
        for word in query.lower().split():
            if word in known:
                return word.title()
        return ""

    def _handle_status(fid_raw: str) -> str:
        fid = re.sub(r"[^\d]", "", fid_raw)
        if not fid:
            return "❌ ID must be numeric. e.g. *status 4521*"
        rows = db_sql(f"""
            SELECT name, state, city, composite_trust AS trust_score_raw, num_doctors, capacity,
                   facilityTypeId, officialPhone,
                   flag_icu_contradiction, flag_surgery_no_anaesthesia, gap_no_doctor_count AS flag_zero_doctors
            FROM   {SQL_TABLE}
            WHERE  CAST(ROW_NUMBER() OVER (ORDER BY name) - 1 AS STRING) = '{fid}'
            LIMIT  1
        """)
        if not rows:
            return f"❌ No facility found with ID {fid}."
        r     = rows[0]
        flags = []
        if r.get("flag_icu_contradiction")  in ("true", True, "1", 1): flags.append("⚠️ ICU claimed / no beds")
        if r.get("flag_surgery_no_anaesthesia") in ("true", True, "1", 1): flags.append("⚠️ Surgery / no anaes.")
        if r.get("gap_no_doctor_count")          in ("true", True, "1", 1): flags.append("⚠️ Zero doctors on record")
        flag_str = ("\n" + "\n".join(flags)) if flags else "\n✅ No flags"
        return (
            f"🏥 *{r.get('name','N/A')}* (ID: {fid})\n"
            f"📍 {r.get('city','')}, {r.get('state','')}\n"
            f"🔖 {r.get('facilityTypeId','N/A')}\n"
            f"👨‍⚕️ {r.get('num_doctors','N/A')} doctors  |  🛏 {r.get('capacity','N/A')} beds\n"
            f"⭐ Trust: {r.get('trust_score_raw','N/A')}/10\n"
            f"📞 {r.get('officialPhone','N/A')}"
            + flag_str
        )

    def _handle_near(place: str) -> str:
        place  = sanitize(place, 80)
        coords = geocode(place)
        if not coords:
            return f"❌ Couldn't find *{place}* on the map.\nTry a city name like *Mumbai* or *Patna*."
        lat, lng = coords
        # Store location in session (Bonus)
        st.session_state.last_location = (lat, lng)
        rows = db_sql(f"""
            SELECT s.name, s.city, e.trust_score_raw, s.officialPhone, s.num_doctors
            FROM   {SQL_TABLE}   s
            JOIN   {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
            WHERE  lower(s.city) LIKE lower('%{place}%')
            ORDER  BY e.trust_score_raw DESC
            LIMIT  5
        """)
        lines = [f"📍 Facilities near *{place.title()}*:\n"]
        if rows:
            for i, r in enumerate(rows, 1):
                lines.append(f"{i}. *{r['name']}*\n   ⭐{r.get('trust_score_raw','?')}  👨‍⚕️{r.get('num_doctors','?')}  📞{r.get('officialPhone','N/A')}")
        else:
            lines.append("No verified facilities found in our database for this city.")
        lines.append("\nUse *find <keyword> <city>* to filter by specialty.")
        return "\n".join(lines)

    def _handle_find_with_city(city: str, query: str) -> str:
        if not query:
            query = st.session_state.pop("_pending_find_q", "hospital")
        city  = sanitize(city, 60)
        query = sanitize(query, 100)
        rows  = db_sql(f"""
            SELECT s.name, s.city, s.state, e.trust_score_raw, s.officialPhone
            FROM   {SQL_TABLE}   s
            JOIN   {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
            WHERE  lower(s.city) LIKE lower('%{city}%')
              AND  lower(e.notes_blob) LIKE lower('%{query}%')
            ORDER  BY e.trust_score_raw DESC
            LIMIT  5
        """)
        if not rows:
            return f"😔 No *{query}* facilities found in *{city.title()}*.\nTry a different city or keyword."
        lines = [f"🔍 *{query.title()}* in *{city.title()}*:\n"]
        for i, r in enumerate(rows, 1):
            lines.append(f"{i}. *{r['name']}*  ⭐{r.get('trust_score_raw','?')}/10\n   📞{r.get('officialPhone','N/A')}")
        lines.append("\nUse *find <keyword> <city>* to refine your search.")
        return "\n".join(lines)

    def _handle_deserts(specialty: str) -> str:
        sp   = sanitize(specialty, 50)
        rows = db_sql(f"""
            SELECT s.state, COUNT(*) AS facility_count
            FROM   {SQL_TABLE}   s
            JOIN   {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
            WHERE  lower(e.notes_blob) LIKE lower('%{sp}%')
              AND  s.state IS NOT NULL
            GROUP  BY s.state
            ORDER  BY facility_count ASC
            LIMIT  8
        """)
        totals = db_sql(f"""
            SELECT state, COUNT(*) AS total
            FROM   {SQL_TABLE}
            WHERE  state IS NOT NULL
            GROUP  BY state
        """)
        if not rows:
            return f"⚠️ No data for *{sp}*.\nTry: ICU, dialysis, oncology, surgery."
        if not totals or "total" not in totals[0]:
            return "⚠️ Databricks not connected — desert analysis needs live data."
        total_map = {r["state"]: int(float(r["total"])) for r in totals if r.get("state") and r.get("total") is not None}
        lines     = [f"🗺 *Medical Deserts — {sp.title()}*\n"]
        for r in rows:
            state = r.get("state", "Unknown")
            count = int(float(r.get("facility_count") or 0))
            total = total_map.get(state, 1)
            pct   = round(count / total * 100, 1)
            icon  = "🔴" if count < 3 else ("🟡" if pct < 15 else "🟢")
            lines.append(f"{icon} {state}: {count}/{total} ({pct}%)")
        lines.append("\n🔴=Critical  🟡=Underserved  🟢=Adequate")
        return "\n".join(lines)

    # ── Chat UI ───────────────────────────────────────────────
    chat_html = '<div class="chat-wrap">'
    for msg in st.session_state.wa_chat:
        cls_wrap = "bubble-wrap-user" if msg["role"] == "user" else "bubble-wrap-bot"
        cls_bub  = "bubble-user"      if msg["role"] == "user" else "bubble-bot"
        chat_html += f'<div class="{cls_wrap}"><div class="bubble {cls_bub}">{msg["text"]}</div></div>'
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    if st.session_state.wa_awaiting_location:
        st.info("💬 Bot is waiting for your location. Type a city name below.")

    # ── Input row ─────────────────────────────────────────────
    col_inp, col_send, col_clear = st.columns([5, 1, 1])
    with col_inp:
        user_msg = st.text_input(
            "Message",
            placeholder="Type a command … e.g. find dialysis",
            label_visibility="collapsed",
            key="wa_input",
        )
    with col_send:
        send_clicked = st.button("Send ▶", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.wa_chat             = []
            st.session_state.wa_awaiting_location = False
            st.session_state.wa_pending_cmd       = None
            st.rerun()

    if send_clicked and user_msg.strip():
        st.session_state.wa_chat.append({"role": "user", "text": user_msg.strip()})

        with st.spinner("Processing …"):
            reply = bot_process(user_msg.strip())

        st.session_state.wa_chat.append({"role": "bot", "text": reply})

        # Send real WhatsApp message via Twilio
        if mode == "Send real WhatsApp" and phone_to.strip():
            ok = twilio_send(phone_to.strip(), reply)
            if ok:
                st.success(f"✅ Sent to {phone_to} via Twilio")
            else:
                st.error("Twilio send failed — check credentials / phone format.")

        st.rerun()

    # ── Quick-command buttons ──────────────────────────────────
    st.markdown("**Quick commands:**")
    qcols = st.columns(5)
    quick = ["help", "find ICU Mumbai", "near Delhi", "deserts oncology", "find dialysis Patna"]
    for i, (col, cmd) in enumerate(zip(qcols, quick)):
        with col:
            if st.button(cmd, key=f"qc_{i}", use_container_width=True):
                st.session_state.wa_chat.append({"role": "user", "text": cmd})
                with st.spinner("…"):
                    rep = bot_process(cmd)
                st.session_state.wa_chat.append({"role": "bot", "text": rep})
                if mode == "Send real WhatsApp" and phone_to.strip():
                    twilio_send(phone_to.strip(), rep)
                st.rerun()



# ══════════════════════════════════════════════════════════════
# TAB 6 — AI AGENT  (Full ReAct · Vector Search · MLflow 3)
# ══════════════════════════════════════════════════════════════
with tab_agent:

    # ── Init session state ─────────────────────────────────────
    if "agent_history"  not in st.session_state: st.session_state.agent_history  = []
    if "agent_insights" not in st.session_state: st.session_state.agent_insights = {}
    if "agent_mode"     not in st.session_state: st.session_state.agent_mode     = "structured"

    # ── Header ─────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#071a2e,#0a2040);border:1px solid #1e3a5f;
     border-radius:12px;padding:20px 28px 14px;margin-bottom:20px">
  <h2 style="margin:0;color:#e8f4ff;font-size:1.4rem">&#127973; Agentic Healthcare Finder</h2>
  <p style="margin:6px 0 0;color:#7aaecf;font-size:0.88rem">
    Find verified healthcare facilities based on real capabilities, not just listings.<br>
    Powered by Databricks Vector Search &middot; Llama 3.3-70B &middot; MLflow 3 Tracing
  </p>
</div>""", unsafe_allow_html=True)

    # ── Connection banner ──────────────────────────────────────
    _vs_ready  = _vs_ok()
    _llm_ready = _llm_ok()
    if not _vs_ready or not _llm_ready:
        st.warning(
            "⚠️ **Agent not fully configured** — results will use SQL keyword fallback. "
            "Add `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `VS_INDEX`, `LLM_ENDPOINT` to `.env`."
        )
    else:
        st.success("✅ Databricks Vector Search + Llama 3.3-70B Agent ready")

    # ── Input mode toggle ──────────────────────────────────────
    mode_col1, mode_col2, _ = st.columns([2, 2, 6])
    with mode_col1:
        if st.button("📋 Structured Input",
                     type="primary" if st.session_state.agent_mode == "structured" else "secondary",
                     key="mode_struct"):
            st.session_state.agent_mode = "structured"
            st.rerun()
    with mode_col2:
        if st.button("💬 Natural Language",
                     type="primary" if st.session_state.agent_mode == "natural" else "secondary",
                     key="mode_natural"):
            st.session_state.agent_mode = "natural"
            st.rerun()

    st.markdown("")

    # ── INPUT SECTION ──────────────────────────────────────────
    agent_query   = ""
    agent_clicked = False

    if st.session_state.agent_mode == "structured":
        c1, c2, c3, c4 = st.columns([2, 2, 3, 1])
        with c1:
            find_type = st.selectbox(
                "FIND",
                ["Hospital", "ICU", "Doctor", "Dialysis", "Oncology",
                 "Trauma / Emergency", "Maternity / Neonatal", "Surgery", "Pharmacy", "Any"],
                key="agent_find_type",
            )
        with c2:
            find_location = st.text_input("LOCATION", placeholder="City or State",
                                          key="agent_location")
        with c3:
            find_condition = st.text_input(
                "CONDITION (optional)",
                placeholder="e.g. breathing issue, fracture, pregnancy emergency",
                key="agent_condition",
            )
        with c4:
            st.markdown("<br>", unsafe_allow_html=True)
            agent_clicked = st.button("🔍 Search", use_container_width=True, key="agent_btn_struct")

        if agent_clicked:
            _loc = find_location.strip()
            parts = [f"Find {find_type} in {_loc}" if _loc else f"Find {find_type}"]
            if find_condition.strip():
                parts.append(f"for {find_condition.strip()}")
            agent_query = " ".join(parts)

    else:
        q_col, btn_col = st.columns([6, 1])
        with q_col:
            agent_nl = st.text_input(
                "Ask in plain English",
                placeholder="e.g. Best ICU with dialysis near Mumbai, or hospitals with neonatal care in Bihar",
                label_visibility="collapsed",
                key="agent_nl_query",
            )
        with btn_col:
            agent_clicked = st.button("🔍 Ask", use_container_width=True, key="agent_btn_nl")
        if agent_clicked:
            agent_query = agent_nl.strip()

    # ── Quick example prompts ──────────────────────────────────
    st.markdown(
        "<div style='margin:8px 0 4px;color:#4a7a9b;font-size:0.78rem;font-weight:500'>"
        "Quick examples — click to run:</div>",
        unsafe_allow_html=True,
    )
    _examples = [
        "Dialysis centres in Gujarat",
        "ICU + neonatal in Rajasthan",
        "Oncology near Mumbai",
        "States with fewest dialysis centres",
        "Emergency trauma in rural Bihar",
    ]
    ex_cols = st.columns(len(_examples))
    for _ei, (_ex, _ec) in enumerate(zip(_examples, ex_cols)):
        with _ec:
            if st.button(_ex, key=f"ex_{_ei}", use_container_width=True):
                agent_query   = _ex
                agent_clicked = True

    st.markdown("---")

    # ── Run agent ──────────────────────────────────────────────
    if agent_clicked and agent_query.strip():
        with st.spinner("🤖 Agent reasoning … (searching → verifying → synthesising)"):
            result = _run_agent(agent_query.strip())

        _meta  = result.get("meta",  {})
        _trace = result.get("trace") or {}
        st.session_state.agent_insights = {
            "total_scanned":   _meta.get("total_scanned", 10000),
            "matches_found":   _meta.get("matches_found", 0),
            "avg_trust_score": _meta.get("avg_trust_score", "N/A"),
            "iterations":  _trace.get("iterations", "—"),
            "tool_calls":  len(_trace.get("tool_calls_log", [])),
            "total_ms":    _trace.get("total_ms", "—"),
        }

        st.session_state.agent_history.append({
            "query":  agent_query.strip(),
            "result": result,
        })

    # ── Render results (newest first) ─────────────────────────
    for _entry in reversed(st.session_state.agent_history):
        _q      = _entry["query"]
        _result = _entry["result"]

        st.markdown(f"""
<div style="background:#071320;border:1px solid #1e3a5f;border-radius:10px;
     padding:14px 20px;margin-bottom:16px">
  <div style="color:#7aaecf;font-size:0.72rem;text-transform:uppercase;
       letter-spacing:.1em;margin-bottom:4px">Query</div>
  <div style="color:#e8f4ff;font-size:1rem;font-weight:500">{_q}</div>
</div>""", unsafe_allow_html=True)

        _err = _result.get("error")
        _t1  = _result.get("tier1_confirmed", [])
        _t2  = _result.get("tier2_possible",  [])

        if _err and not _t1 and not _t2:
            st.error(f"Agent error: {_err}")
            continue

        # Search metadata + confidence
        _smeta       = _result.get("search_metadata", {})
        _conf        = _result.get("confidence", {})
        _conf_level  = _conf.get("level",     "N/A") if isinstance(_conf, dict) else str(_conf)
        _conf_reason = _conf.get("reasoning", "")    if isinstance(_conf, dict) else ""
        _conf_color  = {"High": "#4ade80", "Medium": "#facc15", "Low": "#f87171"}.get(_conf_level, "#7aaecf")

        _mp = []
        if _smeta.get("query_type"):             _mp.append(f"Type: <b>{_smeta['query_type']}</b>")
        if _smeta.get("location"):               _mp.append(f"📍 {_smeta['location']}")
        if _smeta.get("location_type"):          _mp.append(_smeta["location_type"])
        if _smeta.get("verification_performed"): _mp.append("✓ Verified")

        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;'
            'background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;'
            'padding:10px 16px;margin-bottom:12px;font-size:0.8rem;color:#c8dff0">'
            + (" &nbsp;·&nbsp; ".join(_mp) if _mp else "")
            + f'&nbsp;&nbsp;<span style="background:rgba(74,222,128,.12);border:1px solid '
            f'rgba(74,222,128,.3);border-radius:20px;padding:2px 12px;color:{_conf_color};'
            f'font-size:0.75rem">Confidence: {_conf_level}</span>'
            + (f'<span style="color:#7aaecf;font-size:0.75rem;font-style:italic"> — {_conf_reason}</span>'
               if _conf_reason else '')
            + '</div>',
            unsafe_allow_html=True,
        )

        # ── DECISION TRAIL ─────────────────────────────────────
        _trace = _result.get("trace")
        if _trace:
            _spans    = _trace.get("spans", [])
            _total_ms = _trace.get("total_ms", 0)
            _tokens   = _trace.get("total_tokens", "N/A")
            _mlf      = _trace.get("mlflow_active", False)
            _iters    = _trace.get("iterations", "—")
            _tclogs   = _trace.get("tool_calls_log", [])

            with st.expander(
                f"🔬 Decision Trail  —  {_iters} iterations  ·  "
                f"{len(_tclogs)} tool calls  ·  {_total_ms} ms  ·  {_tokens} tokens",
                expanded=True,
            ):
                if _tclogs:
                    st.markdown(
                        "<div style='color:#7aaecf;font-size:0.74rem;font-weight:600;"
                        "text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>"
                        "Tool Calls</div>", unsafe_allow_html=True)
                    for _tc in _tclogs:
                        _tn  = _tc.get("tool", "?")
                        _ta  = _tc.get("args", {})
                        _trc = _tc.get("result_count", 0)
                        _tms = _tc.get("latency_ms", 0)
                        _ti  = _tc.get("iteration", "?")
                        _as  = ", ".join(f"{k}={repr(v)[:35]}" for k, v in _ta.items())
                        st.markdown(
                            f'<div style="background:#071320;border-left:3px solid #1e6ea8;'
                            f'border-radius:0 6px 6px 0;padding:5px 12px;margin:3px 0;'
                            f'font-family:monospace;font-size:0.77rem">'
                            f'<span style="color:#4a7a9b">iter {_ti}</span>&nbsp; '
                            f'<span style="color:#a07af5">&#9881;&#65039; {_tn}</span>'
                            f'<span style="color:#7aaecf">({_as})</span>'
                            f'&nbsp;→&nbsp;<span style="color:#4ade80">{_trc} results</span>'
                            f'&nbsp;<span style="color:#4a9f7a;font-size:0.7rem">⏱ {_tms} ms</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    "<div style='color:#7aaecf;font-size:0.74rem;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:.08em;margin:10px 0 6px'>"
                    "Reasoning Spans</div>", unsafe_allow_html=True)

                _scfg = {
                    "LLM":   ("#1a1040", "#6c3fd4", "&#129504;"),
                    "TOOL":  ("#0d3349", "#1e6ea8", "&#9881;"),
                    "AGENT": ("#0d2a1a", "#2a8c55",  "&#9989;"),
                }
                for _sp in _spans:
                    _stype            = _sp.get("type", "AGENT")
                    _sbg, _sbr, _sic  = _scfg.get(_stype, ("#0a1628", "#1e3a5f", "&#9881;"))
                    _sname            = _sp.get("name", _stype)
                    _sms              = _sp.get("latency_ms", 0)
                    _sin              = _sp.get("inputs",  {})
                    _sout             = _sp.get("outputs", {})

                    def _fkv(d, mv=50):
                        _ps = []
                        for _k, _v in (d or {}).items():
                            if isinstance(_v, list):
                                _vs = "[" + ", ".join(str(_x)[:22] for _x in _v[:3]) + ("…" if len(_v)>3 else "") + "]"
                            else:
                                _vs = str(_v)[:mv]
                            _ps.append(f"<b style='color:#7aaecf'>{_k}</b>=<span style='color:#c8dff0'>{_vs}</span>")
                        return " &nbsp;·&nbsp; ".join(_ps)

                    _bar = min(100, max(3, int(_sms / max(_total_ms, 1) * 100)))
                    st.markdown(f"""
<div style="background:{_sbg};border-left:3px solid {_sbr};border-radius:0 8px 8px 0;
     padding:9px 14px;margin:4px 0;font-size:0.79rem">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px">
    <span style="color:{_sbr};font-weight:600">{_sic} {_sname}</span>
    <span style="color:#4a9f7a;font-size:0.72rem">{'⚡ instant' if _sms==0 else f'⏱ {_sms} ms'}</span>
    <div style="flex:1;height:3px;background:#0f1923;border-radius:2px">
      <div style="width:{_bar}%;height:100%;background:{_sbr};border-radius:2px"></div>
    </div>
  </div>
  <div style="font-size:0.73rem;line-height:1.8">
    <span style="color:#4a7a9b">IN &nbsp;</span>{_fkv(_sin)}<br>
    <span style="color:#4a7a9b">OUT</span>&nbsp;{_fkv(_sout)}
  </div>
</div>""", unsafe_allow_html=True)

                if _mlf and DATABRICKS_HOST:
                    st.markdown(
                        f'<div style="margin-top:10px;font-size:0.74rem;color:#4a7a9b">'
                        f'📊 Full trace logged in '
                        f'<a href="{DATABRICKS_HOST}/#mlflow/experiments" target="_blank" '
                        f'style="color:#4a9f7a">Databricks MLflow — MediAlert-Agent experiment</a>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── NO RESULTS ─────────────────────────────────────────
        _no_res = _result.get("no_results_message")
        if not _t1 and not _t2:
            st.markdown(f"""
<div style="background:#1a0f00;border:1px solid #7a4a00;border-radius:10px;
     padding:16px 20px;margin:12px 0">
  <div style="color:#facc15;font-size:0.95rem;font-weight:600;margin-bottom:6px">
    ⚠️ No exact match found</div>
  <div style="color:#c8dff0;font-size:0.85rem">
    {_no_res or 'Try a nearby city, broader condition, or different specialty.'}
  </div>
</div>""", unsafe_allow_html=True)
            if _result.get("_parse_error") and _result.get("_raw"):
                with st.expander("Raw agent response (debug)"):
                    st.text(_result["_raw"])

        # ── TIER 1 — EXPLICITLY CONFIRMED ─────────────────────
        if _t1:
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin:16px 0 10px">
  <div style="background:#0d2a1a;border:2px solid #2a8c55;border-radius:20px;
       padding:4px 16px;color:#4ade80;font-size:0.8rem;font-weight:700">
    🟢 TIER 1 — EXPLICITLY CONFIRMED &nbsp;({len(_t1)} facilit{'y' if len(_t1)==1 else 'ies'})
  </div>
  <div style="color:#4a7a9b;font-size:0.75rem">Capability verified in facility records</div>
</div>""", unsafe_allow_html=True)

            _cols1 = st.columns(min(len(_t1), 2))
            for _fi, _fac in enumerate(_t1):
                _ts     = float(_fac.get("trust_score", 0) or 0)
                _tlabel = _fac.get("trust_label", "LIKELY_RELIABLE")
                _ticon  = "🟢" if _tlabel == "VERIFIED" else ("🟡" if _tlabel == "LIKELY_RELIABLE" else "🔴")
                _tcol   = "#4ade80" if _tlabel == "VERIFIED" else ("#facc15" if _tlabel == "LIKELY_RELIABLE" else "#f87171")
                _tcc    = trust_class(_ts)
                _fh     = ""
                for _fl in _fac.get("contradiction_flags", []):
                    _fh += f'<span class="flag-badge">⚠ {_fl}</span>'
                for _wn in _fac.get("warnings", []):
                    _fh += (f'<span style="background:rgba(250,204,21,.08);border:1px solid rgba(250,204,21,.25);'
                            f'color:#facc15;font-size:0.7rem;padding:2px 8px;border-radius:10px;margin:2px">{_wn}</span>')
                if not _fh:
                    _fh = '<span class="flag-badge flag-ok">✓ No flags</span>'
                _phone = (_fac.get("contact") or {}).get("phone") or _fac.get("phone", "")
                _dist  = _fac.get("distance_km")
                _dh    = f'<span style="color:#4a9f7a">📏 {_dist} km</span>' if _dist else ""
                _evid  = _fac.get("evidence", "") or ""

                with _cols1[_fi % 2]:
                    st.markdown(f"""
<div class="facility-card" style="border-left:4px solid #2a8c55">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
    <h4 style="font-size:1rem;margin:0;color:#e8f4ff">{_fac.get('name','N/A')}</h4>
    <span style="font-size:0.72rem;background:#0d2a1a;border:1px solid #2a8c55;
          border-radius:12px;padding:2px 10px;color:#4ade80">Tier 1</span>
  </div>
  <div class="meta">📍 {_fac.get('city','')}, {_fac.get('state','')}&nbsp;&nbsp;{_dh}</div>
  <div class="meta" style="margin-top:6px">
    {_ticon} <span style="color:{_tcol};font-weight:600">{_tlabel}</span>
    &nbsp;·&nbsp; ⭐ Trust: <span class="{_tcc}">{_ts:.1f}/10</span>
  </div>
  <div style="background:#071a1a;border-left:3px solid #2a8c55;border-radius:0 6px 6px 0;
       padding:8px 12px;margin:10px 0;font-size:0.8rem;color:#a8d8c0;font-style:italic">
    📄 <b>Evidence:</b> {(_evid[:280]+'…') if len(_evid)>280 else (_evid or 'See facility notes')}
  </div>
  {(f'<div class="meta" style="margin-top:4px">📞 {_phone}</div>') if _phone else ''}
  <div style="margin-top:8px">{_fh}</div>
</div>""", unsafe_allow_html=True)

        # ── TIER 2 — POSSIBLE ──────────────────────────────────
        if _t2:
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin:18px 0 10px">
  <div style="background:#1a1400;border:2px solid #a07a00;border-radius:20px;
       padding:4px 16px;color:#facc15;font-size:0.8rem;font-weight:700">
    🟡 TIER 2 — RELATED CAPABILITY &nbsp;({len(_t2)} facilit{'y' if len(_t2)==1 else 'ies'})
  </div>
  <div style="color:#4a7a9b;font-size:0.75rem">📞 Call to verify before visiting</div>
</div>""", unsafe_allow_html=True)

            _cols2 = st.columns(min(len(_t2), 2))
            for _fi, _fac in enumerate(_t2):
                _ts     = float(_fac.get("trust_score", 0) or 0)
                _tlabel = _fac.get("trust_label", "LIKELY_RELIABLE")
                _ticon  = "🟢" if _tlabel == "VERIFIED" else ("🟡" if _tlabel == "LIKELY_RELIABLE" else "🔴")
                _tcol   = "#4ade80" if _tlabel == "VERIFIED" else ("#facc15" if _tlabel == "LIKELY_RELIABLE" else "#f87171")
                _tcc    = trust_class(_ts)
                _phone  = (_fac.get("contact") or {}).get("phone") or _fac.get("phone", "")
                _vnote  = _fac.get("verification_note", "Not explicitly confirmed")
                _relcap = _fac.get("related_capability", "General healthcare")
                _rec    = _fac.get("recommendation", "Call before visiting")

                with _cols2[_fi % 2]:
                    st.markdown(f"""
<div class="facility-card" style="border-left:4px solid #a07a00;opacity:0.92">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
    <h4 style="font-size:1rem;margin:0;color:#e8f4ff">{_fac.get('name','N/A')}</h4>
    <span style="font-size:0.72rem;background:#1a1400;border:1px solid #a07a00;
          border-radius:12px;padding:2px 10px;color:#facc15">Tier 2</span>
  </div>
  <div class="meta">📍 {_fac.get('city','')}, {_fac.get('state','')}</div>
  <div class="meta" style="margin-top:6px">
    {_ticon} <span style="color:{_tcol};font-weight:600">{_tlabel}</span>
    &nbsp;·&nbsp; ⭐ Trust: <span class="{_tcc}">{_ts:.1f}/10</span>
  </div>
  <div style="margin-top:6px;color:#c8a84a;font-size:0.8rem">🩺 Related: {_relcap}</div>
  <div style="background:#1a1000;border-left:3px solid #a07a00;border-radius:0 6px 6px 0;
       padding:8px 12px;margin:8px 0;font-size:0.78rem;color:#d4b87a;font-style:italic">
    ⚠ {_vnote}
  </div>
  <div style="color:#facc15;font-size:0.75rem;margin-top:4px">📋 {_rec}</div>
  {(f'<div class="meta" style="margin-top:4px">📞 {_phone}</div>') if _phone else ''}
</div>""", unsafe_allow_html=True)

        st.markdown("---")

    # ── Clear button ───────────────────────────────────────────
    if st.session_state.agent_history:
        if st.button("🗑 Clear history", key="clear_agent_hist"):
            st.session_state.agent_history  = []
            st.session_state.agent_insights = {}
            st.rerun()
