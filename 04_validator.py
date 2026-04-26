# Databricks notebook source
# MediAlert — Notebook 04: Validator / Self-Correction Agent
# ──────────────────────────────────────────────────────────
# Stretch goal: "Implement a Validator Agent that cross-references
# extracted data against known medical standards."
#
# Depends on:
#   workspace.default.facilities_sql          (from 01_cleaning)
#   workspace.default.facilities_for_embedding (from 01_cleaning)
#   main.default.facilities_vector_index       (from 02_embeddings)
#   Run 03_agent.py first so the agent function is familiar.


# COMMAND ----------

# =============================================================
# CELL 0 — CONFIG
# =============================================================

SQL_TABLE    = "workspace.default.facilities_sql"
EMBED_TABLE  = "workspace.default.facilities_for_embedding"
VS_INDEX     = "workspace.default.facilities_vector_index"
VS_ENDPOINT  = "medi-alert-vs"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

MLFLOW_EXPERIMENT = "/Shared/MediAlert_Validator"

print("Validator config ready.")


# COMMAND ----------

# =============================================================
# CELL 1 — INSTALL & RESTART
# =============================================================

%pip install mlflow databricks-vectorsearch openai --quiet
dbutils.library.restartPython()


# COMMAND ----------

# =============================================================
# CELL 2 — IMPORTS
# =============================================================

import json
import mlflow
import pandas as pd
from mlflow.entities import SpanType
from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI
import pyspark.sql.functions as F

token  = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host   = spark.conf.get("spark.databricks.workspaceUrl")

openai_client = OpenAI(
    api_key  = token,
    base_url = f"https://{host}/serving-endpoints",
)

vsc      = VectorSearchClient()
vs_index = vsc.get_index(VS_ENDPOINT, VS_INDEX)

mlflow.set_experiment(MLFLOW_EXPERIMENT)
print("Imports complete.")


# COMMAND ----------

# =============================================================
# CELL 3 — MEDICAL STANDARDS RULEBOOK
# ─────────────────────────────────────────────────────────────
# These are evidence-based minimum requirements derived from
# Indian Medical Council / NABH standards.
# The Validator Agent checks each agent recommendation against
# these rules and flags violations.
# =============================================================

MEDICAL_STANDARDS = {
    "ICU": {
        "required_keywords" : ["icu", "intensive care", "ventilator", "critical care"],
        "required_staff"    : ["intensivist", "critical care", "icu nurse", "anesthes"],
        "required_equipment": ["ventilator", "monitor", "defibrillator"],
        "min_bed_count"     : 1,
        "description"       : "ICU requires ventilators, monitoring equipment, and trained intensivists.",
    },
    "Surgery": {
        "required_keywords" : ["surgery", "surgical", "operation theatre", "ot "],
        "required_staff"    : ["surgeon", "anesthes", "anaesth", "scrub nurse"],
        "required_equipment": ["operation theatre", "ot ", "anesthesia machine", "surgical"],
        "min_bed_count"     : 5,
        "description"       : "Surgical facilities require an OT, anaesthesiologist, and minimum beds.",
    },
    "Emergency": {
        "required_keywords" : ["emergency", "casualty", "trauma", "24/7", "24 hour"],
        "required_staff"    : ["emergency", "casualty", "er doctor", "trauma"],
        "required_equipment": ["oxygen", "defibrillator", "emergency"],
        "min_bed_count"     : 1,
        "description"       : "Emergency services require 24/7 staffing and resuscitation equipment.",
    },
    "Dialysis": {
        "required_keywords" : ["dialysis", "hemodialysis", "renal", "kidney"],
        "required_staff"    : ["nephrolog", "dialysis technician", "renal"],
        "required_equipment": ["dialysis machine", "hemodialysis", "RO water"],
        "min_bed_count"     : 0,
        "description"       : "Dialysis centres require machines, RO water supply, and nephrology staff.",
    },
    "Oncology": {
        "required_keywords" : ["oncolog", "cancer", "chemotherapy", "radiotherapy", "tumor"],
        "required_staff"    : ["oncolog", "radiolog", "chemo"],
        "required_equipment": ["chemotherapy", "radiation", "biopsy", "oncolog"],
        "min_bed_count"     : 5,
        "description"       : "Oncology requires oncologists, chemo/radio infrastructure, and inpatient capacity.",
    },
    "Neonatal": {
        "required_keywords" : ["neonatal", "nicu", "newborn", "infant", "premature"],
        "required_staff"    : ["neonatolog", "pediatric", "nicu nurse"],
        "required_equipment": ["incubator", "phototherapy", "nicu", "neonatal warmer"],
        "min_bed_count"     : 1,
        "description"       : "NICU requires neonatologists, incubators, and phototherapy units.",
    },
}

print(f"Medical standards loaded for {len(MEDICAL_STANDARDS)} specialties:")
for k, v in MEDICAL_STANDARDS.items():
    print(f"  - {k}: {v['description'][:60]}...")


# COMMAND ----------

# =============================================================
# CELL 4 — RULE-BASED VALIDATOR FUNCTION
# ─────────────────────────────────────────────────────────────
# This is deterministic (no LLM) — it checks the raw notes_blob
# against the rulebook above and returns structured violations.
# =============================================================

@mlflow.trace(span_type=SpanType.TOOL, name="rule_based_validator")
def validate_facility_rules(facility_name: str, claimed_specialty: str) -> dict:
    """
    Check whether a facility's notes_blob meets minimum standards
    for a claimed specialty.

    Returns a structured validation report.
    """
    # Fetch facility record
    rows = spark.sql(f"""
        SELECT s.name, s.state, s.city, s.trust_score_raw, s.num_doctors,
               s.capacity, s.facilityTypeId,
               s.flag_icu_claimed_no_beds, s.flag_surgery_no_anesthesia,
               s.flag_no_equipment, s.flag_zero_doctors,
               e.notes_blob
        FROM {SQL_TABLE} s
        LEFT JOIN {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
        WHERE lower(s.name) LIKE lower('%{facility_name.replace("'", "''")}%')
        LIMIT 1
    """).toPandas()

    if rows.empty:
        return {"error": f"Facility '{facility_name}' not found in database."}

    row       = rows.iloc[0]
    notes     = str(row.get("notes_blob", "")).lower()
    specialty = claimed_specialty.strip().title()
    standards = MEDICAL_STANDARDS.get(specialty)

    report = {
        "facility_name"     : row["name"],
        "state"             : row.get("state"),
        "city"              : row.get("city"),
        "claimed_specialty" : specialty,
        "trust_score"       : float(row.get("trust_score_raw", 0)),
        "num_doctors"       : row.get("num_doctors"),
        "capacity"          : row.get("capacity"),
        "violations"        : [],
        "warnings"          : [],
        "passed_checks"     : [],
        "verdict"           : "UNKNOWN",
        "confidence"        : "Low",
    }

    # Check existing flag columns
    if bool(row.get("flag_icu_claimed_no_beds", False)):
        report["violations"].append("Claims ICU capability but reported bed count is 0 or missing.")
    if bool(row.get("flag_surgery_no_anesthesia", False)):
        report["violations"].append("Claims surgical capability but no anaesthesiologist found in staff.")
    if bool(row.get("flag_no_equipment", False)):
        report["warnings"].append("No equipment information available — cannot verify physical readiness.")
    if bool(row.get("flag_zero_doctors", False)):
        report["violations"].append("Zero doctors on record — facility may be unstaffed.")

    # Trust score check
    if report["trust_score"] < 2:
        report["warnings"].append(f"Low trust score ({report['trust_score']:.1f}/7) — data may be incomplete.")
    else:
        report["passed_checks"].append(f"Trust score {report['trust_score']:.1f}/7 is acceptable.")

    if standards:
        # Check required keywords in notes
        kw_found = [kw for kw in standards["required_keywords"] if kw.lower() in notes]
        kw_miss  = [kw for kw in standards["required_keywords"] if kw.lower() not in notes]
        if kw_found:
            report["passed_checks"].append(f"Notes mention specialty keywords: {kw_found}")
        if kw_miss:
            report["warnings"].append(f"Expected keywords not found in notes: {kw_miss}")

        # Check required staff
        staff_found = [s for s in standards["required_staff"] if s.lower() in notes]
        staff_miss  = [s for s in standards["required_staff"] if s.lower() not in notes]
        if staff_found:
            report["passed_checks"].append(f"Required staff found: {staff_found}")
        if len(staff_miss) == len(standards["required_staff"]):
            report["violations"].append(f"No required staff found for {specialty}: expected {standards['required_staff']}")

        # Check required equipment
        equip_found = [e for e in standards["required_equipment"] if e.lower() in notes]
        if equip_found:
            report["passed_checks"].append(f"Required equipment mentioned: {equip_found}")
        else:
            report["warnings"].append(f"None of the expected equipment mentioned: {standards['required_equipment']}")

        # Check bed count
        cap = row.get("capacity")
        if pd.notna(cap) and standards["min_bed_count"] > 0:
            if float(cap) < standards["min_bed_count"]:
                report["violations"].append(
                    f"Capacity {cap} is below minimum {standards['min_bed_count']} for {specialty}."
                )
            else:
                report["passed_checks"].append(f"Bed count {cap} meets minimum for {specialty}.")

    # Final verdict
    if len(report["violations"]) == 0 and len(report["warnings"]) <= 1:
        report["verdict"]    = "PASS"
        report["confidence"] = "High"
    elif len(report["violations"]) == 0:
        report["verdict"]    = "PASS_WITH_WARNINGS"
        report["confidence"] = "Medium"
    elif len(report["violations"]) <= 2:
        report["verdict"]    = "FAIL_SOFT"
        report["confidence"] = "Low"
    else:
        report["verdict"]    = "FAIL"
        report["confidence"] = "Very Low"

    return report


# Quick smoke test
test_report = validate_facility_rules("Apollo", "Surgery")
print(json.dumps(test_report, indent=2, default=str))


# COMMAND ----------

# =============================================================
# CELL 5 — LLM VALIDATOR AGENT
# ─────────────────────────────────────────────────────────────
# The LLM layer sits ON TOP of the rule-based validator.
# It reads the rule report and the raw notes, then:
#   (a) Confirms or overrides the verdict with reasoning
#   (b) Extracts the specific sentence that justifies / contradicts
#   (c) Suggests corrective action if the facility fails
# =============================================================

VALIDATOR_SYSTEM_PROMPT = """You are MediAlert-Validator, a medical standards compliance agent.

You will receive:
1. A rule-based validation report for a facility claiming a specialty
2. The facility's raw notes_blob text

Your job is to:
A. CONFIRM or OVERRIDE the rule-based verdict with explicit reasoning.
   - Cite the EXACT sentence from notes_blob that supports or contradicts each finding.
   - If the rules flagged a violation but the notes_blob clearly mentions the required
     item in a different phrasing, override to PASS and explain why.
   - If the rules passed but notes_blob contains contradictions, override to FAIL.

B. OUTPUT a JSON object with this exact structure:
{
  "final_verdict": "PASS" | "FAIL" | "PASS_WITH_WARNINGS",
  "confidence": "High" | "Medium" | "Low",
  "override_reason": "string or null",
  "supporting_citation": "exact sentence from notes that SUPPORTS the claim, or null",
  "contradiction_citation": "exact sentence from notes that CONTRADICTS the claim, or null",
  "corrective_action": "what the facility must fix to pass, or null",
  "reasoning_chain": ["step 1", "step 2", ...]
}

Return ONLY the JSON. No preamble or explanation outside the JSON."""


@mlflow.trace(span_type=SpanType.AGENT, name="LLM_Validator")
def llm_validate(rule_report: dict, notes_blob: str) -> dict:
    """
    LLM layer that reads a rule-based report + raw notes and
    produces a final reasoned verdict with citations.
    """
    user_msg = f"""RULE-BASED REPORT:
{json.dumps(rule_report, indent=2, default=str)}

RAW FACILITY NOTES:
{notes_blob[:3000]}

Produce the final validation JSON."""

    response = openai_client.chat.completions.create(
        model       = LLM_ENDPOINT,
        messages    = [
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature = 0.0,
        max_tokens  = 1024,
    )
    raw = response.choices[0].message.content.strip()

    # Parse JSON robustly
    try:
        # Find JSON block even if wrapped in markdown fences
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(match.group(0)) if match else {"error": "No JSON in response", "raw": raw}
    except json.JSONDecodeError:
        return {"error": "JSON parse error", "raw": raw}


import re   # ensure re is imported after restart

# Test
sample_report = validate_facility_rules("Apollo", "ICU")
sample_notes  = spark.sql(f"""
    SELECT notes_blob FROM {EMBED_TABLE}
    WHERE lower(name) LIKE '%apollo%' LIMIT 1
""").collect()
sample_notes_text = sample_notes[0][0] if sample_notes else "No notes available."

llm_result = llm_validate(sample_report, sample_notes_text)
print(json.dumps(llm_result, indent=2, default=str))


# COMMAND ----------

# =============================================================
# CELL 6 — FULL SELF-CORRECTION PIPELINE
# ─────────────────────────────────────────────────────────────
# Takes a PRIMARY AGENT RECOMMENDATION (list of facilities
# with claimed specialties) and runs every one through the
# two-layer validator. Returns a corrected, de-hallucinatied
# final answer.
# =============================================================

@mlflow.trace(span_type=SpanType.CHAIN, name="SelfCorrectionPipeline")
def self_correct(primary_recommendations: list[dict]) -> list[dict]:
    """
    Run each primary agent recommendation through the two-layer
    validator (rules + LLM). Returns a corrected list with
    final verdicts and citations attached.

    primary_recommendations: list of dicts, each with keys:
        facility_name, claimed_specialty, [any other agent fields]
    """
    corrected = []

    for i, rec in enumerate(primary_recommendations):
        fname     = rec.get("facility_name") or rec.get("name", "")
        specialty = rec.get("claimed_specialty") or rec.get("specialty", "")

        if not fname or not specialty:
            rec["validator_result"] = {"error": "Missing facility_name or claimed_specialty"}
            corrected.append(rec)
            continue

        print(f"  [{i+1}/{len(primary_recommendations)}] Validating: {fname} for {specialty}")

        # Layer 1: rule-based
        rule_report = validate_facility_rules(fname, specialty)

        # Fetch notes for LLM layer
        notes_rows = spark.sql(f"""
            SELECT notes_blob FROM {EMBED_TABLE}
            WHERE lower(name) LIKE lower('%{fname.replace("'","''")}%') LIMIT 1
        """).collect()
        notes = notes_rows[0][0] if notes_rows else ""

        # Layer 2: LLM
        llm_result = llm_validate(rule_report, notes or "")

        # Combine
        rec["rule_verdict"]     = rule_report.get("verdict")
        rec["final_verdict"]    = llm_result.get("final_verdict", rule_report.get("verdict"))
        rec["confidence"]       = llm_result.get("confidence", rule_report.get("confidence"))
        rec["supporting_cite"]  = llm_result.get("supporting_citation")
        rec["contradiction"]    = llm_result.get("contradiction_citation")
        rec["corrective_action"]= llm_result.get("corrective_action")
        rec["reasoning_chain"]  = llm_result.get("reasoning_chain", [])
        rec["violations"]       = rule_report.get("violations", [])
        rec["warnings"]         = rule_report.get("warnings", [])

        corrected.append(rec)

    return corrected


# ── Example: validate a set of hardcoded recommendations ──────
# In practice, feed the output of run_agent() from 03_agent.py here.
SAMPLE_RECS = [
    {"facility_name": "Apollo",        "claimed_specialty": "ICU"},
    {"facility_name": "Fortis",        "claimed_specialty": "Surgery"},
    {"facility_name": "Narayana",      "claimed_specialty": "Oncology"},
    {"facility_name": "Care Hospital", "claimed_specialty": "Dialysis"},
]

with mlflow.start_run(run_name="SelfCorrection_Sample"):
    corrected_recs = self_correct(SAMPLE_RECS)
    mlflow.log_dict(corrected_recs, "corrected_recommendations.json")
    mlflow.log_metric("total_validated", len(corrected_recs))
    mlflow.log_metric("pass_count",  sum(1 for r in corrected_recs if r.get("final_verdict") == "PASS"))
    mlflow.log_metric("fail_count",  sum(1 for r in corrected_recs if "FAIL" in str(r.get("final_verdict", ""))))

for r in corrected_recs:
    verdict_icon = "✓" if r.get("final_verdict") == "PASS" else "✗"
    print(f"\n{verdict_icon} {r['facility_name']} [{r.get('claimed_specialty')}]")
    print(f"   Verdict    : {r.get('final_verdict')}  ({r.get('confidence')} confidence)")
    if r.get("supporting_cite"):
        print(f"   Evidence   : \"{r['supporting_cite'][:100]}...\"")
    if r.get("contradiction"):
        print(f"   Contradiction: \"{r['contradiction'][:100]}...\"")
    if r.get("corrective_action"):
        print(f"   Fix needed : {r['corrective_action'][:100]}")


# COMMAND ----------

# =============================================================
# CELL 7 — BULK CONTRADICTION AUDIT (all 10k rows)
# ─────────────────────────────────────────────────────────────
# Rule-based only (no LLM) for speed. Produces a full audit
# table you can use for the medical desert map.
# =============================================================

@mlflow.trace(span_type=SpanType.TOOL, name="bulk_contradiction_audit")
def bulk_contradiction_audit() -> "pyspark.sql.DataFrame":
    """
    Apply rule-based contradiction checks to all facilities at once
    using Spark SQL. Returns a Spark DataFrame ready for analysis.
    """
    df = spark.sql(f"""
        SELECT
            s.name,
            s.state,
            s.city,
            s.pin_code,
            s.trust_score_raw,
            s.num_doctors,
            s.capacity,
            s.facilityTypeId,

            -- Existing flags from cleaning notebook
            s.flag_icu_claimed_no_beds,
            s.flag_surgery_no_anesthesia,
            s.flag_no_equipment,
            s.flag_zero_doctors,
            s.flag_no_specialties,
            s.flag_no_procedures,
            s.flag_no_capabilities,

            -- New derived flags from notes text
            CASE WHEN lower(e.notes_blob) LIKE '%dialysis%'
                  AND lower(e.notes_blob) NOT LIKE '%nephrolog%'
                  AND lower(e.notes_blob) NOT LIKE '%renal%'
                 THEN true ELSE false END AS flag_dialysis_no_nephrologist,

            CASE WHEN lower(e.notes_blob) LIKE '%nicu%'
                  AND lower(e.notes_blob) NOT LIKE '%neonatolog%'
                  AND lower(e.notes_blob) NOT LIKE '%incubator%'
                 THEN true ELSE false END AS flag_nicu_no_equipment,

            CASE WHEN lower(e.notes_blob) LIKE '%oncolog%'
                  AND lower(e.notes_blob) NOT LIKE '%chemotherapy%'
                  AND lower(e.notes_blob) NOT LIKE '%radiotherapy%'
                  AND lower(e.notes_blob) NOT LIKE '%cancer%'
                 THEN true ELSE false END AS flag_oncology_no_treatment,

            CASE WHEN lower(e.notes_blob) LIKE '%blood bank%'
                  AND lower(e.notes_blob) NOT LIKE '%transfusion%'
                  AND lower(e.notes_blob) NOT LIKE '%storage%'
                 THEN true ELSE false END AS flag_blood_bank_unverified,

            -- Total flag count per facility
            (CAST(s.flag_icu_claimed_no_beds    AS INT) +
             CAST(s.flag_surgery_no_anesthesia  AS INT) +
             CAST(s.flag_no_equipment           AS INT) +
             CAST(s.flag_zero_doctors           AS INT) +
             CAST(s.flag_no_specialties         AS INT)) AS total_flag_count

        FROM {SQL_TABLE} s
        LEFT JOIN {EMBED_TABLE} e ON lower(s.name) = lower(e.name)
    """)
    return df


with mlflow.start_run(run_name="BulkContradictionAudit"):
    audit_df = bulk_contradiction_audit()

    # Write audit table for use in notebook 05
    (audit_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable("workspace.default.facilities_audit"))

    total    = audit_df.count()
    flagged  = audit_df.filter(F.col("total_flag_count") > 0).count()
    high_risk= audit_df.filter(F.col("total_flag_count") >= 3).count()

    mlflow.log_metric("total_facilities",    total)
    mlflow.log_metric("flagged_facilities",  flagged)
    mlflow.log_metric("high_risk_facilities",high_risk)

    print(f"Audit complete: {total:,} facilities")
    print(f"  Flagged (>=1 issue): {flagged:,}  ({flagged/total*100:.1f}%)")
    print(f"  High risk (>=3 issues): {high_risk:,}  ({high_risk/total*100:.1f}%)")

display(
    audit_df.orderBy(F.desc("total_flag_count"))
            .select("name","state","city","trust_score_raw","total_flag_count",
                    "flag_icu_claimed_no_beds","flag_surgery_no_anesthesia",
                    "flag_zero_doctors","flag_dialysis_no_nephrologist")
            .limit(20)
)


# COMMAND ----------

# =============================================================
# CELL 8 — VERIFY OUTPUTS
# =============================================================

print("Verifying validator outputs...")

# 1. Audit table exists
count = spark.table("workspace.default.facilities_audit").count()
print(f"  facilities_audit rows: {count:,}  (expected ~10,000)")
assert count > 9000, "Audit table seems incomplete — check the join in bulk_contradiction_audit()"

# 2. Check flag columns exist
cols = spark.table("workspace.default.facilities_audit").columns
expected_flags = [
    "flag_icu_claimed_no_beds", "flag_surgery_no_anesthesia",
    "flag_dialysis_no_nephrologist", "flag_nicu_no_equipment",
    "flag_oncology_no_treatment", "total_flag_count"
]
missing = [c for c in expected_flags if c not in cols]
if missing:
    print(f"  WARNING — Missing columns: {missing}")
else:
    print(f"  All {len(expected_flags)} flag columns present")

# 3. MLflow experiment has runs
runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT], max_results=3)
print(f"  MLflow runs: {len(runs)}  (open Experiments tab to see traces)")

print("\nValidator notebook complete.")
print("Next: run 05_medical_deserts.py for the map dashboard.")

# =============================================================
# END OF NOTEBOOK 04
# Output table: workspace.default.facilities_audit
# Next: 05_medical_deserts.py
# =============================================================