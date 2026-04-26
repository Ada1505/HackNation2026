"""
MediAlert — Streamlit App
=========================
Full conversion of the Flask/WhatsApp backend into a Streamlit UI.

Tabs:
  1. 🔍 Find Facilities   — keyword search across Databricks
  2. 📋 Facility Status   — lookup by ID with flag warnings
  3. 🗺  Desert Map        — medical desert analysis by specialty
  4. 📍 Near Me           — Maps distance comparison (user-provided location)
  5. 💬 WhatsApp Tester   — simulate / send real Twilio messages with auto location-prompt

Run:  streamlit run streamlit_app.py
"""

import os, re, time, math, json
from collections import defaultdict

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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
MAPS_API_KEY      = _cfg("GOOGLE_MAPS_API_KEY")
TWILIO_SID        = _cfg("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN      = _cfg("TWILIO_AUTH_TOKEN")
TWILIO_FROM       = _cfg("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

SQL_TABLE   = "workspace.default.facilities_sql"
EMBED_TABLE = "workspace.default.facilities_for_embedding"
AUDIT_TABLE = "workspace.default.facilities_audit"
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

/* WhatsApp chat bubble */
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
# DATABRICKS SQL HELPER
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def db_sql(sql: str) -> list[dict]:
    """Execute SQL on Databricks SQL Warehouse, return list of dicts."""
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        return []
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = {
        "warehouse_id":   DATABRICKS_SQL_WH,
        "statement":      sql,
        "wait_timeout":   "30s",
        "on_wait_timeout":"CANCEL",
        "format":         "JSON_ARRAY",
    }
    try:
        r = requests.post(
            f"{DATABRICKS_HOST}/api/2.0/sql/statements",
            headers=headers, json=payload, timeout=35
        )
        r.raise_for_status()
        data      = r.json()
        state     = data.get("status", {}).get("state")
        if state != "SUCCEEDED":
            err = data.get("status", {}).get("error", {})
            st.error(f"SQL error [{state}]: {err.get('message', str(data))}")
            return []
        schema    = data.get("manifest", {}).get("schema", {}).get("columns", [])
        col_names = [c["name"] for c in schema]
        rows      = data.get("result", {}).get("data_array", [])
        return [dict(zip(col_names, row)) for row in rows]
    except Exception as e:
        st.error(f"Databricks connection error: {e}")
        return []


def sanitize(text: str, max_len: int = 120) -> str:
    return re.sub(r"['\";\\]", "", text).strip()[:max_len]


# ─────────────────────────────────────────────────────────────
# MAPS / GEO HELPERS
# ─────────────────────────────────────────────────────────────
def geocode(place: str) -> tuple[float, float] | None:
    place = sanitize(place)
    if MAPS_API_KEY:
        r = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                         params={"address": place, "key": MAPS_API_KEY}, timeout=10)
        results = r.json().get("results", [])
        if results:
            loc = results[0]["geometry"]["location"]
            return float(loc["lat"]), float(loc["lng"])
    else:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": place, "format": "json", "limit": 1},
                         headers={"User-Agent": NOMINATIM_UA}, timeout=10)
        results = r.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    return None


def haversine(lat1, lng1, lat2, lng2) -> float:
    R = 6371
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lng2 - lng1)
    a  = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 2)


def nearby_hospitals_maps(lat: float, lng: float, radius_m: int = 15000) -> list[dict]:
    """Google Places nearby — only called when user provides location."""
    if not MAPS_API_KEY:
        return []
    r = requests.get(
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
        params={"location": f"{lat},{lng}", "radius": radius_m,
                "keyword": "hospital clinic", "key": MAPS_API_KEY},
        timeout=10,
    )
    results = r.json().get("results", [])[:8]
    places  = []
    for p in results:
        plat = p["geometry"]["location"]["lat"]
        plng = p["geometry"]["location"]["lng"]
        places.append({
            "name":    p.get("name", ""),
            "address": p.get("vicinity", ""),
            "dist_km": haversine(lat, lng, plat, plng),
            "rating":  p.get("rating", "N/A"),
            "lat": plat, "lng": plng,
        })
    return sorted(places, key=lambda x: x["dist_km"])


# ─────────────────────────────────────────────────────────────
# TWILIO HELPER
# ─────────────────────────────────────────────────────────────
def twilio_send(to: str, body: str) -> bool:
    """Send via Twilio REST. Returns True on success."""
    if not TWILIO_SID or not TWILIO_TOKEN:
        return False
    try:
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
            auth=(TWILIO_SID, TWILIO_TOKEN),
            data={"From": TWILIO_FROM, "To": f"whatsapp:{to}", "Body": body[:1600]},
            timeout=15,
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
    st.markdown("**Data source**")
    db_ok = bool(DATABRICKS_HOST and DATABRICKS_TOKEN)
    st.markdown(f"{'🟢' if db_ok else '🔴'} Databricks {'connected' if db_ok else 'not configured'}")
    maps_ok = bool(MAPS_API_KEY)
    st.markdown(f"{'🟢' if maps_ok else '🟡'} Maps {'API key set' if maps_ok else '(Nominatim fallback)'}")
    twilio_ok = bool(TWILIO_SID and TWILIO_TOKEN)
    st.markdown(f"{'🟢' if twilio_ok else '🔴'} Twilio {'ready' if twilio_ok else 'not configured'}")

    st.markdown("---")
    st.markdown("**Quick filters**")
    sidebar_state = st.text_input("Filter by state", placeholder="e.g. Maharashtra")
    sidebar_city  = st.text_input("Filter by city",  placeholder="e.g. Pune")

    st.markdown("---")
    st.markdown("<span style='font-size:0.75rem;color:#4a7a9b'>MediAlert v2.0 · 10k+ verified facilities</span>",
                unsafe_allow_html=True)


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
tab_find, tab_status, tab_deserts, tab_near, tab_wa = st.tabs([
    "🔍 Find Facilities",
    "📋 Facility Status",
    "🗺️ Desert Map",
    "📍 Near Me",
    "💬 WhatsApp",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — FIND FACILITIES
# ══════════════════════════════════════════════════════════════
with tab_find:
    st.markdown("#### Search verified healthcare facilities")

    col_q, col_sp, col_btn = st.columns([3, 2, 1])
    with col_q:
        keyword = st.text_input("Keyword", placeholder="dialysis, ICU, oncology, surgery …", label_visibility="collapsed")
    with col_sp:
        specialty_filter = st.selectbox("Specialty", ["Any", "ICU", "Surgery", "Dialysis", "Oncology", "Neonatal", "Emergency"], label_visibility="collapsed")
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
        if sidebar_state:
            q_parts.append(f"lower(s.state) LIKE lower('%{sanitize(sidebar_state)}%')")
        if sidebar_city:
            q_parts.append(f"lower(s.city) LIKE lower('%{sanitize(sidebar_city)}%')")

        where = ("WHERE " + " AND ".join(q_parts)) if q_parts else ""

        with st.spinner("Querying Databricks …"):
            rows = db_sql(f"""
                SELECT s.name, s.state, s.city, s.pin_code,
                       s.trust_score_raw, s.num_doctors, s.capacity,
                       s.facilityTypeId, s.officialPhone, s.address_line1,
                       s.flag_icu_claimed_no_beds, s.flag_surgery_no_anesthesia,
                       s.flag_zero_doctors, s.flag_no_equipment,
                       ROW_NUMBER() OVER (ORDER BY s.trust_score_raw DESC) - 1 AS facility_id
                FROM   {SQL_TABLE}   s
                JOIN   {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
                {where}
                ORDER  BY s.trust_score_raw DESC
                LIMIT  20
            """)

        if not rows:
            st.info("No facilities found. Try a broader keyword.")
        else:
            # Summary metrics
            mc1, mc2, mc3, mc4 = st.columns(4)
            avg_trust = sum(float(r.get("trust_score_raw") or 0) for r in rows) / len(rows)
            total_docs = sum(int(r.get("num_doctors") or 0) for r in rows)
            flagged    = sum(1 for r in rows if any([
                r.get("flag_icu_claimed_no_beds"), r.get("flag_surgery_no_anesthesia"),
                r.get("flag_zero_doctors"),
            ]))
            for col, val, lbl in [
                (mc1, len(rows), "Results"),
                (mc2, f"{avg_trust:.1f}/10", "Avg Trust"),
                (mc3, total_docs, "Total Doctors"),
                (mc4, flagged, "Flagged"),
            ]:
                with col:
                    st.markdown(f'<div class="metric-tile"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            for r in rows:
                flags_html = ""
                if r.get("flag_icu_claimed_no_beds")   in ("true", True, "1", 1):
                    flags_html += '<span class="flag-badge">⚠ ICU/no beds</span>'
                if r.get("flag_surgery_no_anesthesia")  in ("true", True, "1", 1):
                    flags_html += '<span class="flag-badge">⚠ Surgery/no anaes.</span>'
                if r.get("flag_zero_doctors")           in ("true", True, "1", 1):
                    flags_html += '<span class="flag-badge">⚠ Zero doctors</span>'
                if r.get("flag_no_equipment")           in ("true", True, "1", 1):
                    flags_html += '<span class="flag-badge">⚠ No equipment</span>'
                if not flags_html:
                    flags_html = '<span class="flag-badge flag-ok">✓ No flags</span>'

                tc = trust_class(r.get("trust_score_raw"))
                st.markdown(f"""
<div class="facility-card">
  <h4>{r.get('name','N/A')} <span style="font-size:0.78rem;color:#4a7a9b">ID: {r.get('facility_id','?')}</span></h4>
  <div class="meta">
    📍 {r.get('city','')}, {r.get('state','')} &nbsp;·&nbsp; PIN {r.get('pin_code','')}
    &nbsp;·&nbsp; {r.get('facilityTypeId','')}
  </div>
  <div class="meta" style="margin-top:6px">
    ⭐ Trust: <span class="{tc}">{r.get('trust_score_raw','N/A')}/10</span>
    &nbsp;·&nbsp; 👨‍⚕️ {r.get('num_doctors','N/A')} doctors
    &nbsp;·&nbsp; 🛏 {r.get('capacity','N/A')} beds
    &nbsp;·&nbsp; 📞 {r.get('officialPhone','N/A')}
  </div>
  <div style="margin-top:8px">{flags_html}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — FACILITY STATUS
# ══════════════════════════════════════════════════════════════
with tab_status:
    st.markdown("#### Facility detail by ID")
    col_id, col_go = st.columns([3, 1])
    with col_id:
        fid_input = st.text_input("Facility ID", placeholder="e.g. 4521", label_visibility="collapsed")
    with col_go:
        status_clicked = st.button("Lookup", use_container_width=True)

    if status_clicked and fid_input:
        fid = re.sub(r"[^\d]", "", fid_input)
        if not fid:
            st.error("ID must be numeric.")
        else:
            with st.spinner("Fetching …"):
                rows = db_sql(f"""
                    SELECT name, state, city, pin_code, address_line1,
                           trust_score_raw, num_doctors, capacity,
                           facilityTypeId, officialPhone, email,
                           flag_icu_claimed_no_beds, flag_surgery_no_anesthesia,
                           flag_zero_doctors, flag_no_equipment,
                           flag_no_specialties, flag_no_procedures
                    FROM   {SQL_TABLE}
                    WHERE  CAST(ROW_NUMBER() OVER (ORDER BY name) - 1 AS STRING) = '{fid}'
                    LIMIT  1
                """)

            if not rows:
                st.error(f"No facility found with ID {fid}.")
            else:
                r = rows[0]
                col_l, col_r = st.columns([3, 2])
                with col_l:
                    tc = trust_class(r.get("trust_score_raw"))
                    st.markdown(f"""
<div class="facility-card" style="padding:24px 28px">
  <h4 style="font-size:1.3rem">{r.get('name','N/A')}</h4>
  <div class="meta">🏷 {r.get('facilityTypeId','N/A')}</div>
  <div class="meta" style="margin-top:10px">
    📍 {r.get('address_line1','')}<br>
    &nbsp;&nbsp;&nbsp;{r.get('city','')}, {r.get('state','')} — {r.get('pin_code','')}
  </div>
  <div class="meta" style="margin-top:10px">
    📞 {r.get('officialPhone','N/A')}<br>
    ✉️ {r.get('email','N/A')}
  </div>
  <div style="margin-top:14px;display:flex;gap:20px">
    <div><div class="lbl" style="font-size:0.7rem;color:#4a7a9b;text-transform:uppercase;letter-spacing:0.06em">Trust Score</div>
         <div class="{tc}" style="font-size:1.5rem;font-family:'DM Serif Display',serif">{r.get('trust_score_raw','N/A')}/10</div></div>
    <div><div class="lbl" style="font-size:0.7rem;color:#4a7a9b;text-transform:uppercase;letter-spacing:0.06em">Doctors</div>
         <div style="font-size:1.5rem;color:#7ec8ff;font-family:'DM Serif Display',serif">{r.get('num_doctors','N/A')}</div></div>
    <div><div class="lbl" style="font-size:0.7rem;color:#4a7a9b;text-transform:uppercase;letter-spacing:0.06em">Beds</div>
         <div style="font-size:1.5rem;color:#7ec8ff;font-family:'DM Serif Display',serif">{r.get('capacity','N/A')}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

                with col_r:
                    st.markdown("**Quality flags**")
                    flag_map = {
                        "flag_icu_claimed_no_beds":   "ICU claimed but no beds",
                        "flag_surgery_no_anesthesia": "Surgery / no anaesthesiologist",
                        "flag_zero_doctors":          "Zero doctors on record",
                        "flag_no_equipment":          "No equipment data",
                        "flag_no_specialties":        "No specialties listed",
                        "flag_no_procedures":         "No procedures listed",
                    }
                    any_flag = False
                    for col_key, label in flag_map.items():
                        val = r.get(col_key)
                        is_flagged = val in ("true", True, "1", 1)
                        icon = "🔴" if is_flagged else "🟢"
                        st.markdown(f"{icon} {label}")
                        if is_flagged: any_flag = True
                    if not any_flag:
                        st.success("No contradictions detected")


# ══════════════════════════════════════════════════════════════
# TAB 3 — DESERT MAP
# ══════════════════════════════════════════════════════════════
with tab_deserts:
    st.markdown("#### Medical desert analysis — states with fewest facilities")

    col_sp2, col_th, col_run = st.columns([3, 2, 1])
    with col_sp2:
        desert_specialty = st.selectbox("Specialty", ["oncology","dialysis","ICU","surgery","neonatal","emergency","blood bank"], label_visibility="collapsed")
    with col_th:
        threshold = st.slider("Critical threshold (facilities)", 1, 10, 3, label_visibility="collapsed")
    with col_run:
        desert_clicked = st.button("Analyse", use_container_width=True)

    # ── NEW: Heatmap filters ───────────────────────────────────
    with st.expander("🗺️ Facility Heatmap Filters", expanded=False):
        hm_col1, hm_col2, hm_col3, hm_col4 = st.columns(4)

        with hm_col1:
            hm_specialty = st.selectbox("Specialty", 
                ["All","ICU","Surgery","Dialysis","Oncology",
                 "Neonatal","Emergency","Blood Bank","Radiology"],
                key="hm_specialty"
            )
        with hm_col2:
            hm_state = st.selectbox("State",
                ["All"] + sorted(s["state"] for s in db_sql(
                    f"SELECT DISTINCT state FROM {SQL_TABLE} ORDER BY state"
                ) if s.get("state")),
                key="hm_state"
            )
        with hm_col3:
            hm_trust = st.slider("Min Trust Score", 
                0.0, 7.0, 0.0, 0.5, key="hm_trust"
            )
        with hm_col4:
            hm_high_risk = st.checkbox("High Risk Only (3+ flags)", 
                key="hm_high_risk"
            )

        hm_clicked = st.button("Generate Heatmap", use_container_width=True)

    # ── NEW: Build and render heatmap ─────────────────────────
    if hm_clicked:
        SPECIALTY_KEYWORDS = {
            "ICU"       : "icu",
            "Surgery"   : "surgery",
            "Dialysis"  : "dialysis",
            "Oncology"  : "oncolog",
            "Neonatal"  : "neonatal",
            "Emergency" : "emergency",
            "Blood Bank": "blood bank",
            "Radiology" : "radiolog",
        }

        # Build WHERE clause from filters
        conditions = [
            "s.lat IS NOT NULL",
            "s.lon IS NOT NULL",
            f"s.trust_score_raw >= {hm_trust}",
        ]

        if hm_specialty != "All":
            kw = SPECIALTY_KEYWORDS[hm_specialty]
            conditions.append(
                f"lower(e.notes_blob) LIKE '%{kw}%'"
            )

        if hm_state != "All":
            conditions.append(
                f"lower(s.state) = lower('{hm_state}')"
            )

        if hm_high_risk:
            conditions.append("""(
                CAST(s.flag_icu_claimed_no_beds    AS INT) +
                CAST(s.flag_surgery_no_anesthesia  AS INT) +
                CAST(s.flag_no_equipment           AS INT) +
                CAST(s.flag_zero_doctors           AS INT) +
                CAST(s.flag_no_specialties         AS INT)
            ) >= 3""")

        where = "WHERE " + " AND ".join(conditions)

        with st.spinner("Loading facilities for heatmap..."):
            hm_rows = db_sql(f"""
                SELECT s.name, s.state, s.city,
                       s.lat, s.lon,
                       s.trust_score_raw,
                       s.flag_icu_claimed_no_beds,
                       s.flag_surgery_no_anesthesia,
                       s.flag_zero_doctors,
                       s.flag_no_equipment,
                       s.flag_no_specialties,
                       (CAST(s.flag_icu_claimed_no_beds   AS INT) +
                        CAST(s.flag_surgery_no_anesthesia AS INT) +
                        CAST(s.flag_no_equipment          AS INT) +
                        CAST(s.flag_zero_doctors          AS INT) +
                        CAST(s.flag_no_specialties        AS INT)
                       ) AS flag_count
                FROM {SQL_TABLE} s
                LEFT JOIN {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
                {where}
                LIMIT 2000
            """)

        if not hm_rows:
            st.warning("No facilities match the selected filters.")
        else:
            # ── Summary metrics ────────────────────────────────
            hm_c1, hm_c2, hm_c3, hm_c4 = st.columns(4)
            avg_t    = sum(float(r.get("trust_score_raw") or 0) for r in hm_rows) / len(hm_rows)
            high_r   = sum(1 for r in hm_rows if int(r.get("flag_count") or 0) >= 3)
            states_n = len(set(r.get("state") for r in hm_rows))

            for col, val, lbl in [
                (hm_c1, f"{len(hm_rows):,}",  "Facilities"),
                (hm_c2, states_n,              "States"),
                (hm_c3, f"{avg_t:.1f}/7",      "Avg Trust"),
                (hm_c4, high_r,                "High Risk"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-tile">'
                        f'<div class="val">{val}</div>'
                        f'<div class="lbl">{lbl}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Build Folium map ───────────────────────────────
            try:
                import folium
                from folium.plugins import HeatMap, MarkerCluster
                from streamlit_folium import st_folium

                m = folium.Map(
                    location   = [20.5937, 78.9629],
                    zoom_start = 5,
                    tiles      = "CartoDB dark_matter"
                )

                # Layer 1 — Heatmap
                heat_data = [
                    [float(r["lat"]), float(r["lon"])]
                    for r in hm_rows
                    if r.get("lat") and r.get("lon")
                ]
                HeatMap(
                    heat_data,
                    radius      = 8,
                    blur        = 10,
                    min_opacity = 0.3,
                    name        = "Facility Density"
                ).add_to(m)

                # Layer 2 — Individual markers if small result set
                if len(hm_rows) <= 150:
                    cluster = MarkerCluster(name="Facilities").add_to(m)
                    for r in hm_rows:
                        if not r.get("lat") or not r.get("lon"):
                            continue
                        flag_count = int(r.get("flag_count") or 0)
                        color = (
                            "red"    if flag_count >= 3 else
                            "orange" if float(r.get("trust_score_raw") or 0) < 2 else
                            "green"
                        )
                        folium.CircleMarker(
                            location     = [float(r["lat"]), float(r["lon"])],
                            radius       = 5,
                            color        = color,
                            fill         = True,
                            fill_opacity = 0.8,
                            tooltip      = r.get("name", ""),
                            popup        = folium.Popup(f"""
                                <b>{r.get('name','')}</b><br>
                                {r.get('city','')}, {r.get('state','')}<br>
                                Trust: {r.get('trust_score_raw','N/A')}/7<br>
                                Flags: {flag_count}
                            """, max_width=200),
                        ).add_to(cluster)
                    st.caption("🟢 Trusted  🟠 Low trust  🔴 High risk  — individual markers shown (≤150 results)")

                folium.LayerControl(collapsed=False).add_to(m)
                st_folium(m, width=None, height=550, returned_objects=[])

            except ImportError:
                st.error("Run: `pip install folium streamlit-folium` to enable map view.")

            # ── Download filtered results ──────────────────────
            import pandas as pd
            hm_df = pd.DataFrame(hm_rows)
            st.download_button(
                label     = "⬇️ Download Filtered Facilities CSV",
                data      = hm_df.to_csv(index=False),
                file_name = f"medialert_{hm_specialty}_{hm_state}.csv".replace(" ","_"),
                mime      = "text/csv"
            )

    # ── EXISTING desert bar chart code below (unchanged) ──────
    if desert_clicked:
        sp = sanitize(desert_specialty)
        with st.spinner("Running desert analysis …"):
            coverage = db_sql(f"""
                SELECT e.state, COUNT(*) AS facility_count
                FROM   {EMBED_TABLE} e
                WHERE  lower(e.notes_blob) LIKE lower('%{sp}%')
                GROUP  BY e.state
                ORDER  BY facility_count ASC
            """)
            totals = db_sql(f"""
                SELECT state, COUNT(*) AS total
                FROM   {EMBED_TABLE}
                GROUP  BY state
            """)

        if not coverage:
            st.info(f"No facilities found with '{sp}' in their profile.")
        else:
            total_map = {r["state"]: int(r["total"]) for r in totals}
            cov_map   = {r["state"]: int(r["facility_count"]) for r in coverage}
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

            max_total = max((s["total"] for s in all_states), default=1)
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
# TAB 4 — NEAR ME  (Maps API only when user provides location)
# ══════════════════════════════════════════════════════════════
with tab_near:
    st.markdown("#### Nearby facilities — distance comparison within same city")
    st.caption("📍 Enter your location to fetch nearby clinics via Maps API and compare distances to verified DB facilities.")

    if not MAPS_API_KEY:
        st.warning("⚠️ Google Maps API key not set — using Nominatim geocoding (no nearby search). Add `GOOGLE_MAPS_API_KEY` to enable full distance comparison.")

    col_loc, col_kw2, col_near_btn = st.columns([3, 2, 1])
    with col_loc:
        user_location = st.text_input("Your location", placeholder="e.g. Defence, Karachi  or  lat,lng", label_visibility="collapsed")
    with col_kw2:
        near_keyword = st.text_input("Keyword", value="hospital", placeholder="hospital, clinic …", label_visibility="collapsed")
    with col_near_btn:
        near_clicked = st.button("Find & Compare", use_container_width=True)

    if near_clicked and user_location:
        # Parse lat,lng OR geocode text
        coords = None
        latlon_match = re.match(r"^(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)$", user_location.strip())
        if latlon_match:
            coords = float(latlon_match.group(1)), float(latlon_match.group(2))
        else:
            with st.spinner("Geocoding your location …"):
                coords = geocode(user_location)

        if not coords:
            st.error("Could not geocode that location. Try a city name or lat,lng.")
        else:
            user_lat, user_lng = coords
            st.success(f"📍 Location resolved: {user_lat:.4f}, {user_lng:.4f}")

            # ── Maps nearby (Google Places) ───────────────────
            maps_places = []
            if MAPS_API_KEY:
                with st.spinner("Fetching nearby places via Maps API …"):
                    maps_places = nearby_hospitals_maps(user_lat, user_lng)

            # ── Databricks: same-city facilities ─────────────
            city_guess = sanitize(user_location.split(",")[0].strip(), 60)
            with st.spinner("Querying Databricks for city facilities …"):
                db_rows = db_sql(f"""
                    SELECT s.name, s.state, s.city, s.pin_code,
                           s.trust_score_raw, s.num_doctors, s.officialPhone,
                           s.flag_zero_doctors, s.flag_icu_claimed_no_beds
                    FROM   {SQL_TABLE}   s
                    JOIN   {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
                    WHERE  lower(s.city) LIKE lower('%{city_guess}%')
                       AND lower(e.notes_blob) LIKE lower('%{sanitize(near_keyword)}%')
                    ORDER  BY s.trust_score_raw DESC
                    LIMIT  10
                """)

            col_maps, col_db = st.columns(2)

            with col_maps:
                st.markdown("**🗺️ Maps API — nearby places**")
                if not MAPS_API_KEY:
                    st.info("Google Maps key required for this section.")
                elif not maps_places:
                    st.info("No places returned by Maps API.")
                else:
                    table_html = """
<table class="dist-table">
<thead><tr><th>#</th><th>Name</th><th>Distance</th><th>Rating</th></tr></thead><tbody>
"""
                    min_dist = maps_places[0]["dist_km"] if maps_places else 0
                    for i, p in enumerate(maps_places, 1):
                        winner = ' class="dist-winner"' if p["dist_km"] == min_dist else ""
                        table_html += f"""<tr>
<td{winner}>{i}</td>
<td{winner}>{p['name']}</td>
<td{winner}>{p['dist_km']} km</td>
<td{winner}>{'⭐' + str(p['rating']) if p['rating'] != 'N/A' else '—'}</td>
</tr>"""
                    table_html += "</tbody></table>"
                    st.markdown(table_html, unsafe_allow_html=True)

                    # Distance comparison chart data
                    if len(maps_places) >= 2:
                        st.markdown("<br>**Distance comparison**", unsafe_allow_html=True)
                        for p in maps_places[:5]:
                            bar_pct = int((p["dist_km"] / maps_places[-1]["dist_km"]) * 100)
                            color = "#4ade80" if p["dist_km"] == min_dist else "#3b82f6"
                            st.markdown(f"""
<div class="desert-row">
  <span style="width:160px;color:#c8dff0;font-size:0.82rem">{p['name'][:22]}</span>
  <div class="desert-bar-wrap"><div class="desert-bar" style="width:{bar_pct}%;background:{color}"></div></div>
  <span style="width:70px;color:#7aaecf;text-align:right;font-size:0.82rem">{p['dist_km']} km</span>
</div>""", unsafe_allow_html=True)

            with col_db:
                st.markdown("**🏥 Databricks — verified DB facilities in city**")
                if not db_rows:
                    st.info(f"No verified facilities found in '{city_guess}' for '{near_keyword}'.\nTry a broader city name or keyword.")
                else:
                    table_html = """
<table class="dist-table">
<thead><tr><th>Name</th><th>Trust</th><th>Doctors</th><th>Phone</th></tr></thead><tbody>
"""
                    for r in db_rows:
                        tc = trust_class(r.get("trust_score_raw"))
                        warn = "⚠️ " if r.get("flag_zero_doctors") in ("true", True, "1", 1) else ""
                        table_html += f"""<tr>
<td>{warn}{r['name']}</td>
<td class="{tc}">{r.get('trust_score_raw','—')}</td>
<td>{r.get('num_doctors','—')}</td>
<td style="font-size:0.78rem">{r.get('officialPhone','—')}</td>
</tr>"""
                    table_html += "</tbody></table>"
                    st.markdown(table_html, unsafe_allow_html=True)

            # ── Folium map (if Maps key available for geocoding facilities) ──
            if maps_places or db_rows:
                st.markdown("---")
                st.markdown("**Combined map view**")
                try:
                    import folium
                    from streamlit_folium import st_folium

                    m = folium.Map(location=[user_lat, user_lng], zoom_start=13,
                                   tiles="CartoDB dark_matter")

                    # User marker
                    folium.Marker(
                        [user_lat, user_lng],
                        popup="📍 Your Location",
                        icon=folium.Icon(color="blue", icon="home"),
                    ).add_to(m)

                    # Maps places
                    for p in maps_places:
                        folium.Marker(
                            [p["lat"], p["lng"]],
                            popup=f"{p['name']} ({p['dist_km']} km)",
                            icon=folium.Icon(color="red", icon="plus-sign"),
                        ).add_to(m)

                    st_folium(m, width=None, height=380)
                except ImportError:
                    st.info("Install `streamlit-folium` for map view: `pip install streamlit-folium`")


# ══════════════════════════════════════════════════════════════
# TAB 5 — WHATSAPP (Twilio simulator + sender)
# ══════════════════════════════════════════════════════════════
with tab_wa:
    st.markdown("#### WhatsApp — Simulate or send real Twilio messages")

    # ── Session state for chat ────────────────────────────────
    if "wa_chat" not in st.session_state:
        st.session_state.wa_chat = []  # list of {"role": "user"/"bot", "text": str}
    if "wa_awaiting_location" not in st.session_state:
        st.session_state.wa_awaiting_location = False
    if "wa_pending_cmd" not in st.session_state:
        st.session_state.wa_pending_cmd = None   # stores partial command awaiting location

    col_mode, col_phone = st.columns([2, 3])
    with col_mode:
        mode = st.selectbox("Mode", ["Simulate (local)", "Send real WhatsApp"], label_visibility="collapsed")
    with col_phone:
        phone_to = st.text_input("Phone (with country code)", placeholder="+923001234567",
                                  label_visibility="collapsed",
                                  disabled=(mode == "Simulate (local)"))

    # ── Location-prompt logic ─────────────────────────────────
    def bot_process(user_text: str) -> str:
        """
        Rule-based dispatcher — same logic as Flask backend.
        If the user doesn't mention a city/location for 'near' or 'find' commands,
        the bot PROMPTS for it and stores the pending command.
        """
        text   = user_text.strip()
        lower  = text.lower()

        # ── If we are awaiting a location reply ───────────────
        if st.session_state.wa_awaiting_location:
            st.session_state.wa_awaiting_location = False
            pending = st.session_state.wa_pending_cmd or ""
            st.session_state.wa_pending_cmd = None
            # treat this message as the location for the pending command
            if pending == "near":
                return _handle_near(text)
            elif pending == "find":
                return _handle_find_with_city(text, "")
            else:
                return _handle_near(text)

        # ── Help ──────────────────────────────────────────────
        if re.match(r"^(help|hi|hello|start|menu|commands?)$", lower):
            return (
                "🏥 *MediAlert Commands*\n\n"
                "*status <ID>* — Full facility details\n"
                "*find <keyword>* — Search facilities\n"
                "*near <place>* — Hospitals near a location\n"
                "*deserts <specialty>* — Medical desert analysis\n\n"
                "📍 No location? Just say *find dialysis* and I'll ask you where!"
            )

        # ── status <id> ───────────────────────────────────────
        m = re.match(r"^status\s+(\S+)$", lower)
        if m:
            return _handle_status(m.group(1))

        # ── deserts <specialty> ───────────────────────────────
        m = re.match(r"^deserts?\s+(.+)$", lower, re.IGNORECASE)
        if m:
            return _handle_deserts(m.group(1).strip())

        # ── near <place> — check if place given ───────────────
        m = re.match(r"^near\s+(.+)$", lower, re.IGNORECASE)
        if m:
            place = m.group(1).strip()
            if len(place) < 2:
                # no location given
                return _prompt_location("near")
            return _handle_near(place)

        # ── find <keyword> — check if city mentioned ──────────
        m = re.match(r"^find\s+(.+)$", lower, re.IGNORECASE)
        if m:
            query = m.group(1).strip()
            # Heuristic: does the query contain a known city/state word?
            city_hint = _extract_city(query)
            if not city_hint:
                # Store pending and ask for location
                st.session_state.wa_awaiting_location = True
                st.session_state.wa_pending_cmd = "find_with_q"
                # Store the query part separately
                st.session_state["_pending_find_q"] = query
                return (
                    f"📍 You asked to find *{query}*.\n\n"
                    "Which city or area should I search in?\n"
                    "_Reply with a city name, e.g._ *Lahore* _or_ *Patna*"
                )
            return _handle_find_with_city(city_hint, query)

        # ── lone location word (no command prefix) ────────────
        # e.g. user replies "Karachi" after being prompted
        if st.session_state.wa_awaiting_location:
            return _handle_near(text)

        return (
            "⚠️ Command not recognised.\n\n"
            "Try:\n  status 1234\n  near Lahore\n  find ICU\n  deserts oncology\n\n"
            "Type *help* for the full list."
        )

    def _prompt_location(cmd: str) -> str:
        st.session_state.wa_awaiting_location = True
        st.session_state.wa_pending_cmd = cmd
        return (
            "📍 I need your location to search nearby.\n\n"
            "Please reply with:\n"
            "• Your *city or area* name _(e.g. Defence, Karachi)_\n"
            "• Or share your *WhatsApp location pin* 📎"
        )

    def _extract_city(query: str) -> str:
        """Very light heuristic: last capitalised word or known city list."""
        known = {"karachi","lahore","islamabad","rawalpindi","mumbai","delhi","chennai",
                 "hyderabad","bangalore","pune","kolkata","patna","lucknow","jaipur",
                 "ahmedabad","surat","agra","bhopal","indore","nagpur","visakhapatnam",
                 "chandigarh","ludhiana","amritsar","coimbatore","kochi","thiruvananthapuram"}
        for word in query.lower().split():
            if word in known:
                return word.title()
        return ""

    def _handle_status(fid_raw: str) -> str:
        fid = re.sub(r"[^\d]", "", fid_raw)
        if not fid:
            return "❌ ID must be numeric. e.g. *status 4521*"
        rows = db_sql(f"""
            SELECT name, state, city, trust_score_raw, num_doctors, capacity,
                   facilityTypeId, officialPhone,
                   flag_icu_claimed_no_beds, flag_surgery_no_anesthesia, flag_zero_doctors
            FROM   {SQL_TABLE}
            WHERE  CAST(ROW_NUMBER() OVER (ORDER BY name) - 1 AS STRING) = '{fid}'
            LIMIT  1
        """)
        if not rows:
            return f"❌ No facility found with ID {fid}."
        r = rows[0]
        flags = []
        if r.get("flag_icu_claimed_no_beds")   in ("true", True, "1", 1): flags.append("⚠️ ICU claimed / no beds")
        if r.get("flag_surgery_no_anesthesia")  in ("true", True, "1", 1): flags.append("⚠️ Surgery / no anaes.")
        if r.get("flag_zero_doctors")           in ("true", True, "1", 1): flags.append("⚠️ Zero doctors on record")
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
        place = sanitize(place, 80)
        coords = geocode(place)
        if not coords:
            return f"❌ Couldn't find *{place}* on the map.\nTry a city name like *Karachi* or *Patna*."
        lat, lng = coords
        rows = db_sql(f"""
            SELECT s.name, s.city, s.trust_score_raw, s.officialPhone, s.num_doctors
            FROM   {SQL_TABLE} s
            WHERE  lower(s.city) LIKE lower('%{place}%')
            ORDER  BY s.trust_score_raw DESC
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
        # If city came from awaiting state, query is in session
        if not query:
            query = st.session_state.pop("_pending_find_q", "hospital")
        city  = sanitize(city, 60)
        query = sanitize(query, 100)
        rows = db_sql(f"""
            SELECT s.name, s.city, s.state, s.trust_score_raw, s.officialPhone,
                   ROW_NUMBER() OVER (ORDER BY s.trust_score_raw DESC) - 1 AS facility_id
            FROM   {SQL_TABLE}   s
            JOIN   {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
            WHERE  lower(s.city) LIKE lower('%{city}%')
              AND  lower(e.notes_blob) LIKE lower('%{query}%')
            ORDER  BY s.trust_score_raw DESC
            LIMIT  5
        """)
        if not rows:
            return f"😔 No *{query}* facilities found in *{city.title()}*.\nTry a different city or keyword."
        lines = [f"🔍 *{query.title()}* in *{city.title()}*:\n"]
        for i, r in enumerate(rows, 1):
            lines.append(f"{i}. *{r['name']}*  ⭐{r.get('trust_score_raw','?')}/10\n   📞{r.get('officialPhone','N/A')}  🆔{r.get('facility_id','?')}")
        lines.append("\nUse *status <ID>* for full details.")
        return "\n".join(lines)

    def _handle_deserts(specialty: str) -> str:
        sp = sanitize(specialty, 50)
        rows = db_sql(f"""
            SELECT e.state, COUNT(*) AS facility_count
            FROM   {EMBED_TABLE} e
            WHERE  lower(e.notes_blob) LIKE lower('%{sp}%')
            GROUP  BY e.state
            ORDER  BY facility_count ASC
            LIMIT  8
        """)
        totals = db_sql(f"SELECT state, COUNT(*) AS total FROM {EMBED_TABLE} GROUP BY state")
        if not rows:
            return f"⚠️ No data for *{sp}*.\nTry: ICU, dialysis, oncology, surgery."
        total_map = {r["state"]: int(r["total"]) for r in totals}
        lines = [f"🗺 *Medical Deserts — {sp.title()}*\n"]
        for r in rows:
            state = r["state"]
            count = int(r["facility_count"])
            total = total_map.get(state, 1)
            pct   = round(count / total * 100, 1)
            icon  = "🔴" if count < 3 else ("🟡" if pct < 15 else "🟢")
            lines.append(f"{icon} {state}: {count}/{total} ({pct}%)")
        lines.append("\n🔴=Critical  🟡=Underserved  🟢=Adequate")
        return "\n".join(lines)

    # ── Chat UI ───────────────────────────────────────────────
    st.markdown("---")

    # Render chat history
    chat_html = '<div class="chat-wrap">'
    for msg in st.session_state.wa_chat:
        cls_wrap = "bubble-wrap-user" if msg["role"] == "user" else "bubble-wrap-bot"
        cls_bub  = "bubble-user"      if msg["role"] == "user" else "bubble-bot"
        chat_html += f'<div class="{cls_wrap}"><div class="bubble {cls_bub}">{msg["text"]}</div></div>'
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Prompt hint if awaiting location
    if st.session_state.wa_awaiting_location:
        st.info("💬 Bot is waiting for your location. Type a city name below.")

    # Input row
    col_inp, col_send, col_clear = st.columns([5, 1, 1])
    with col_inp:
        user_msg = st.text_input("Message", placeholder="Type a command … e.g. find dialysis",
                                  label_visibility="collapsed", key="wa_input")
    with col_send:
        send_clicked = st.button("Send ▶", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.wa_chat = []
            st.session_state.wa_awaiting_location = False
            st.session_state.wa_pending_cmd = None
            st.rerun()

    if send_clicked and user_msg.strip():
        # Add user bubble
        st.session_state.wa_chat.append({"role": "user", "text": user_msg.strip()})

        # Generate bot reply
        with st.spinner("Processing …"):
            reply = bot_process(user_msg.strip())

        st.session_state.wa_chat.append({"role": "bot", "text": reply})

        # Real Twilio send if enabled
        if mode == "Send real WhatsApp" and phone_to.strip():
            ok = twilio_send(phone_to.strip(), reply)
            if ok:
                st.success(f"✅ Sent to {phone_to} via Twilio")
            else:
                st.error("Twilio send failed — check credentials / phone format.")

        st.rerun()

    # Quick-command buttons
    st.markdown("**Quick commands:**")
    qcols = st.columns(5)
    quick = ["help", "find ICU", "near Karachi", "deserts oncology", "status 42"]
    for i, (col, cmd) in enumerate(zip(qcols, quick)):
        with col:
            if st.button(cmd, key=f"qc_{i}", use_container_width=True):
                st.session_state.wa_chat.append({"role": "user", "text": cmd})
                with st.spinner("…"):
                    rep = bot_process(cmd)
                st.session_state.wa_chat.append({"role": "bot", "text": rep})
                st.rerun()
