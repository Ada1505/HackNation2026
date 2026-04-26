# Databricks notebook source
# MediAlert — Notebook 05: Medical Desert Analysis + Map Dashboard
# ────────────────────────────────────────────────────────────────
# Stretch goal: "Create a visual dashboard that overlays your agent's
# findings onto a map of India. Highlight highest-risk medical
# deserts by PIN code."
#
# Depends on:
#   workspace.default.facilities_sql     (from 01_cleaning)
#   workspace.default.facilities_audit   (from 04_validator)
#   workspace.default.facilities_for_embedding (from 01_cleaning)


# COMMAND ----------

# =============================================================
# CELL 0 — CONFIG
# =============================================================

SQL_TABLE   = "workspace.default.facilities_sql"
AUDIT_TABLE = "workspace.default.facilities_audit"
EMBED_TABLE = "workspace.default.facilities_for_embedding"

# Output HTML map saved here (download from DBFS after running)
MAP_OUTPUT_PATH  = "/dbfs/tmp/medi_alert_map.html"
DESERT_CSV_PATH  = "/dbfs/tmp/medi_alert_deserts.csv"

MLFLOW_EXPERIMENT = "/Shared/MediAlert_Deserts"

print("Config ready.")


# COMMAND ----------

# =============================================================
# CELL 1 — INSTALL DEPENDENCIES
# =============================================================

%pip install folium mlflow plotly --quiet
dbutils.library.restartPython()


# COMMAND ----------

# =============================================================
# CELL 2 — IMPORTS
# =============================================================

import mlflow
import pandas as pd
import numpy as np
import json
import folium
from folium.plugins import MarkerCluster, HeatMap
import plotly.graph_objects as go
import plotly.express as px
import pyspark.sql.functions as F

mlflow.set_experiment(MLFLOW_EXPERIMENT)
print("Imports complete.")


# COMMAND ----------

# =============================================================
# CELL 3 — LOAD & VALIDATE SOURCE DATA
# =============================================================

sql_df   = spark.table(SQL_TABLE).toPandas()
audit_df = spark.table(AUDIT_TABLE).toPandas()
embed_df = spark.table(EMBED_TABLE).toPandas()

print(f"SQL table    : {len(sql_df):,} rows")
print(f"Audit table  : {len(audit_df):,} rows")
print(f"Embed table  : {len(embed_df):,} rows")

# Basic sanity checks
assert len(sql_df)   > 9000, "SQL table too small — re-run 01_cleaning.ipynb"
assert len(audit_df) > 9000, "Audit table missing — run 04_validator.py first"
assert "lat" in sql_df.columns and "lon" in sql_df.columns, \
    "lat/lon columns missing from SQL table"

# Merge: sql (has lat/lon) + audit (has flags) on name
merged = sql_df.merge(
    audit_df[["name", "total_flag_count",
              "flag_icu_claimed_no_beds", "flag_surgery_no_anesthesia",
              "flag_dialysis_no_nephrologist", "flag_nicu_no_equipment",
              "flag_oncology_no_treatment"]],
    on="name", how="left", suffixes=("", "_audit")
)

# Merge in notes_blob
merged = merged.merge(embed_df[["name", "notes_blob"]], on="name", how="left")

# Keep only rows with valid coordinates
geo_df = merged.dropna(subset=["lat", "lon"]).copy()
geo_df["lat"] = pd.to_numeric(geo_df["lat"], errors="coerce")
geo_df["lon"] = pd.to_numeric(geo_df["lon"], errors="coerce")
geo_df = geo_df.dropna(subset=["lat", "lon"])
geo_df = geo_df[(geo_df["lat"].between(6, 38)) & (geo_df["lon"].between(66, 98))]  # India bounding box

print(f"\nGeo-located facilities: {len(geo_df):,} / {len(sql_df):,}")
print(f"States covered: {geo_df['state'].nunique()}")
print(f"Cities covered: {geo_df['city'].nunique()}")


# COMMAND ----------

# =============================================================
# CELL 4 — SPECIALTY COVERAGE ANALYSIS
# ─────────────────────────────────────────────────────────────
# For each specialty, count facilities per state and flag
# states below the desert threshold.
# =============================================================

SPECIALTIES = {
    "ICU"       : ["icu", "intensive care", "critical care", "ventilator"],
    "Surgery"   : ["surgery", "surgical", "operation theatre", "ot "],
    "Dialysis"  : ["dialysis", "hemodialysis", "renal replacement"],
    "Oncology"  : ["oncolog", "cancer", "chemotherapy", "radiotherapy"],
    "Neonatal"  : ["neonatal", "nicu", "newborn icu", "premature"],
    "Emergency" : ["emergency", "24/7", "24 hour", "casualty", "trauma"],
    "Blood Bank": ["blood bank", "blood transfusion"],
    "Radiology" : ["mri", "ct scan", "x-ray", "radiolog", "imaging"],
}

DESERT_THRESHOLD = 5   # fewer than this → medical desert for that specialty

def detect_specialty(notes: str, keywords: list) -> bool:
    if pd.isna(notes):
        return False
    n = str(notes).lower()
    return any(kw in n for kw in keywords)

# Build specialty presence columns
notes_col = geo_df["notes_blob"].fillna("")
for specialty, keywords in SPECIALTIES.items():
    geo_df[f"has_{specialty.lower().replace(' ','_')}"] = notes_col.apply(
        lambda n: detect_specialty(n, keywords)
    )

# State-level aggregation
state_counts = geo_df.groupby("state").agg(
    total_facilities      = ("name", "count"),
    avg_trust_score       = ("trust_score_raw", "mean"),
    avg_flag_count        = ("total_flag_count", "mean"),
    high_risk_facilities  = ("total_flag_count", lambda x: (x >= 3).sum()),
    **{f"count_{sp.lower().replace(' ','_')}": (f"has_{sp.lower().replace(' ','_')}", "sum")
       for sp in SPECIALTIES}
).reset_index()

# Desert score: percentage of specialties underserved
specialty_cols = [f"count_{sp.lower().replace(' ','_')}" for sp in SPECIALTIES]
for col in specialty_cols:
    state_counts[f"desert_{col}"] = state_counts[col] < DESERT_THRESHOLD

state_counts["desert_specialties"] = state_counts[[f"desert_{c}" for c in specialty_cols]].sum(axis=1)
state_counts["desert_score"]       = (state_counts["desert_specialties"] / len(SPECIALTIES) * 100).round(1)

# Rank states: higher = more underserved
state_counts = state_counts.sort_values("desert_score", ascending=False)

print("── Top 15 Most Underserved States ───────────────────────")
print(state_counts[["state", "total_facilities", "desert_specialties", "desert_score", "avg_trust_score"]]
      .head(15).to_string(index=False))


# COMMAND ----------

# =============================================================
# CELL 5 — SAVE DESERT ANALYSIS CSV
# =============================================================

state_counts.to_csv(DESERT_CSV_PATH, index=False)
print(f"Desert analysis saved to: {DESERT_CSV_PATH}")
print("Download from DBFS: Files tab → tmp → medi_alert_deserts.csv")


# COMMAND ----------

# =============================================================
# CELL 6 — BUILD FOLIUM MAP
# ─────────────────────────────────────────────────────────────
# Layers:
#   1. Heatmap          — facility density
#   2. Clustered markers — individual facilities (colour by trust)
#   3. High-risk markers — facilities with >= 3 flags (red)
#   4. Desert overlay   — state-level desert score as circle
# =============================================================

def trust_color(trust_score):
    """Green = trusted, Orange = medium, Red = suspicious."""
    if pd.isna(trust_score) or trust_score < 2:
        return "red"
    elif trust_score < 4:
        return "orange"
    return "green"


# ── Base map centred on India ──────────────────────────────
m = folium.Map(
    location   = [20.5937, 78.9629],
    zoom_start = 5,
    tiles      = "CartoDB positron",
)

# ── Layer 1: Heatmap (density) ────────────────────────────
heat_data = [[row["lat"], row["lon"]] for _, row in geo_df.iterrows()]
HeatMap(heat_data, radius=8, blur=10, min_opacity=0.3,
        name="Facility Density Heatmap").add_to(m)

# ── Layer 2: Clustered markers (all facilities) ───────────
cluster = MarkerCluster(name="All Facilities").add_to(m)

for _, row in geo_df.iterrows():
    flags_text = ""
    if row.get("flag_icu_claimed_no_beds"):
        flags_text += "<br><b style='color:red'>⚠ ICU claimed but no beds</b>"
    if row.get("flag_surgery_no_anesthesia"):
        flags_text += "<br><b style='color:red'>⚠ Surgery but no anaesthesiologist</b>"
    if row.get("flag_zero_doctors"):
        flags_text += "<br><b style='color:orange'>⚠ Zero doctors on record</b>"

    popup_html = f"""
    <div style='font-family:Arial;min-width:220px'>
      <b>{row['name']}</b><br>
      {row.get('city','')}, {row.get('state','')} — {row.get('pin_code','')}<br>
      <hr style='margin:4px 0'>
      Trust Score: <b>{row.get('trust_score_raw', 'N/A')}</b> / 7<br>
      Doctors: {row.get('num_doctors', 'N/A')} &nbsp;|&nbsp;
      Capacity: {row.get('capacity', 'N/A')}<br>
      Type: {row.get('facilityTypeId', 'N/A')}<br>
      {flags_text}
      <hr style='margin:4px 0'>
      <small>{str(row.get('notes_blob',''))[:200]}...</small>
    </div>"""

    folium.CircleMarker(
        location = [row["lat"], row["lon"]],
        radius   = 5,
        color    = trust_color(row.get("trust_score_raw")),
        fill     = True,
        fill_opacity = 0.7,
        popup    = folium.Popup(popup_html, max_width=300),
        tooltip  = row["name"],
    ).add_to(cluster)

# ── Layer 3: High-risk facilities (standalone red markers) ─
high_risk_layer = folium.FeatureGroup(name="High-Risk Facilities (>=3 flags)")
for _, row in geo_df[geo_df.get("total_flag_count", pd.Series(0)) >= 3].iterrows():
    folium.Marker(
        location = [row["lat"], row["lon"]],
        icon     = folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon"),
        tooltip  = f"HIGH RISK: {row['name']}",
        popup    = f"<b>{row['name']}</b><br>Flags: {row.get('total_flag_count','?')}",
    ).add_to(high_risk_layer)
high_risk_layer.add_to(m)

# ── Layer 4: State-level desert score circles ─────────────
# Compute approximate state centroids from facility coordinates
state_centroids = geo_df.groupby("state").agg(
    lat=("lat", "mean"), lon=("lon", "mean")
).reset_index()
desert_layer = folium.FeatureGroup(name="Medical Desert Scores by State")

state_info = state_counts.set_index("state").to_dict("index")
for _, row in state_centroids.iterrows():
    info  = state_info.get(row["state"], {})
    score = info.get("desert_score", 0)
    total = info.get("total_facilities", 0)

    # Colour by desert severity
    if score >= 70:
        colour = "#d73027"   # deep red
    elif score >= 50:
        colour = "#fc8d59"   # orange
    elif score >= 30:
        colour = "#fee090"   # yellow
    else:
        colour = "#91bfdb"   # blue = well-served

    desert_popup = f"""
    <div style='font-family:Arial;min-width:180px'>
      <b>{row['state']}</b><br>
      <hr style='margin:4px 0'>
      Desert Score: <b>{score:.0f}%</b><br>
      Total Facilities: {total}<br>
      Underserved Specialties: {info.get('desert_specialties', 'N/A')} / {len(SPECIALTIES)}<br>
      Avg Trust Score: {info.get('avg_trust_score', 0):.1f} / 7
    </div>"""

    folium.Circle(
        location     = [row["lat"], row["lon"]],
        radius       = max(score * 2000, 20000),   # bigger = worse
        color        = colour,
        fill         = True,
        fill_opacity = 0.35,
        popup        = folium.Popup(desert_popup, max_width=250),
        tooltip      = f"{row['state']}: {score:.0f}% underserved",
    ).add_to(desert_layer)

desert_layer.add_to(m)

# ── Layer control ──────────────────────────────────────────
folium.LayerControl(collapsed=False).add_to(m)

# ── Legend ─────────────────────────────────────────────────
legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
            background:white;padding:12px;border-radius:8px;
            border:1px solid #ccc;font-family:Arial;font-size:12px;">
  <b>MediAlert — Trust Score</b><br>
  <span style="color:green">●</span> Trusted (score ≥ 4)<br>
  <span style="color:orange">●</span> Moderate (2–4)<br>
  <span style="color:red">●</span> Low / Suspicious (< 2)<br>
  <br>
  <b>Desert Score (circles)</b><br>
  <span style="color:#d73027">●</span> Critical desert ≥ 70%<br>
  <span style="color:#fc8d59">●</span> Severe 50–70%<br>
  <span style="color:#fee090">●</span> Moderate 30–50%<br>
  <span style="color:#91bfdb">●</span> Adequate < 30%
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))

# ── Save map ───────────────────────────────────────────────
m.save(MAP_OUTPUT_PATH)
print(f"Map saved: {MAP_OUTPUT_PATH}")
print("To download: Databricks Files tab → /tmp → medi_alert_map.html")


# COMMAND ----------

# =============================================================
# CELL 7 — PLOTLY BAR CHART: TOP MEDICAL DESERTS
# (displays inline in Databricks)
# =============================================================

top_deserts = state_counts.head(20)

fig = px.bar(
    top_deserts,
    x           = "desert_score",
    y           = "state",
    orientation = "h",
    color       = "desert_score",
    color_continuous_scale = "RdYlGn_r",
    range_color = [0, 100],
    text        = "desert_score",
    labels      = {"desert_score": "Desert Score (%)", "state": "State"},
    title       = "MediAlert — Top 20 Medical Deserts in India",
    hover_data  = ["total_facilities", "desert_specialties", "avg_trust_score"],
)
fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
fig.update_layout(
    height       = 600,
    yaxis        = dict(autorange="reversed"),
    coloraxis_showscale = True,
    plot_bgcolor = "white",
    paper_bgcolor= "white",
)
fig.show()


# COMMAND ----------

# =============================================================
# CELL 8 — PLOTLY SPECIALTY HEATMAP (State × Specialty)
# =============================================================

specialty_matrix = state_counts.set_index("state")[
    [f"count_{sp.lower().replace(' ','_')}" for sp in SPECIALTIES]
].head(20)
specialty_matrix.columns = list(SPECIALTIES.keys())

fig2 = px.imshow(
    specialty_matrix.values,
    x            = list(SPECIALTIES.keys()),
    y            = specialty_matrix.index.tolist(),
    color_continuous_scale = "RdYlGn",
    labels       = dict(color="Facility Count"),
    title        = "Specialty Coverage by State (top 20 most underserved)",
    text_auto    = True,
)
fig2.update_layout(height=700, coloraxis_showscale=True)
fig2.show()


# COMMAND ----------

# =============================================================
# CELL 9 — PINCODE-LEVEL DESERT REPORT (printable table)
# ─────────────────────────────────────────────────────────────
# For each specialty, show the top 10 PIN codes with zero
# coverage — directly actionable for NGO planners.
# =============================================================

print("── PIN-code Level Gaps (Zero Coverage) ─────────────────\n")

all_pincodes = geo_df["pin_code"].dropna().unique()
pincode_df   = geo_df.copy()

desert_rows = []
for specialty, keywords in SPECIALTIES.items():
    col = f"has_{specialty.lower().replace(' ','_')}"
    zero_coverage = pincode_df[pincode_df[col] == False]
    top_gaps = (
        zero_coverage.groupby(["state", "pin_code", "city"])
                     .size().reset_index(name="facilities_in_pincode")
                     .sort_values("facilities_in_pincode", ascending=False)
                     .head(10)
    )
    top_gaps["missing_specialty"] = specialty
    desert_rows.append(top_gaps)
    print(f"{specialty:12s}: {len(zero_coverage):,} facilities in PIN codes with zero {specialty} coverage")

desert_pincodes_df = pd.concat(desert_rows, ignore_index=True)
display(spark.createDataFrame(
    desert_pincodes_df[["missing_specialty","state","city","pin_code","facilities_in_pincode"]]
))


# COMMAND ----------

# =============================================================
# CELL 10 — LOG EVERYTHING TO MLFLOW
# =============================================================

with mlflow.start_run(run_name="MedicalDesertAnalysis"):

    # Metrics
    mlflow.log_metric("total_geo_facilities", len(geo_df))
    mlflow.log_metric("states_analyzed",      state_counts.shape[0])
    mlflow.log_metric("avg_desert_score",     float(state_counts["desert_score"].mean()))
    mlflow.log_metric("critical_deserts",     int((state_counts["desert_score"] >= 70).sum()))

    for sp in SPECIALTIES:
        col = f"count_{sp.lower().replace(' ','_')}"
        if col in state_counts.columns:
            total_sp = int(state_counts[col].sum())
            mlflow.log_metric(f"facilities_with_{sp.lower().replace(' ','_')}", total_sp)

    # Artifacts
    state_counts.to_csv("/tmp/state_desert_report.csv", index=False)
    mlflow.log_artifact("/tmp/state_desert_report.csv")
    mlflow.log_artifact(MAP_OUTPUT_PATH.replace("/dbfs",""))
    mlflow.log_text(
        "\n".join([f"{r['state']:30s} {r['desert_score']:5.1f}%"
                   for _, r in state_counts.iterrows()]),
        "desert_ranking.txt"
    )

    print("All desert analysis metrics and artifacts logged to MLflow.")


# COMMAND ----------

# =============================================================
# CELL 11 — FINAL VERIFICATION CHECKS
# =============================================================

print("Running final checks...\n")

# 1. Map file exists and is non-empty
import os
map_size = os.path.getsize(MAP_OUTPUT_PATH)
print(f"  Map file size: {map_size:,} bytes  {'OK' if map_size > 50000 else 'WARNING — file too small'}")

# 2. Desert analysis has all states
n_states = state_counts.shape[0]
print(f"  States in analysis: {n_states}  {'OK' if n_states >= 25 else 'WARNING — expected 28+ states'}")

# 3. All specialty columns were computed
spec_cols = [f"has_{sp.lower().replace(' ','_')}" for sp in SPECIALTIES]
missing_spec_cols = [c for c in spec_cols if c not in geo_df.columns]
print(f"  Specialty columns: {len(spec_cols) - len(missing_spec_cols)}/{len(spec_cols)} present  "
      f"{'OK' if not missing_spec_cols else 'MISSING: ' + str(missing_spec_cols)}")

# 4. MLflow experiment has at least one run
runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT], max_results=1)
print(f"  MLflow runs: {len(runs)}  {'OK' if len(runs) > 0 else 'WARNING — no runs found'}")

# 5. Geo coverage is reasonable
pct_geo = len(geo_df) / len(sql_df) * 100
print(f"  Geo coverage: {pct_geo:.1f}%  {'OK' if pct_geo > 80 else 'WARNING — many facilities missing coordinates'}")

print("\nAll checks complete.")
print(f"\nMap download path : {MAP_OUTPUT_PATH}")
print(f"CSV download path : {DESERT_CSV_PATH}")
print("Open Files tab in Databricks, navigate to /tmp/, download both files.")

# =============================================================
# END OF NOTEBOOK 05
# Project pipeline complete:
#   01_cleaning.ipynb  → cleaned Delta tables
#   02_embeddings.ipynb → vector index
#   03_agent.py         → reasoning agent + MLflow traces
#   04_validator.py     → self-correction + audit table
#   05_medical_deserts.py → map + desert analysis
# =============================================================