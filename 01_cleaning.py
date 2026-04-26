# Databricks notebook source
import pandas as pd
import re

# COMMAND ----------

df = pd.read_excel("/Volumes/workspace/default/main/VF_Hackathon_Dataset_India_Large.xlsx") 

# COMMAND ----------

# DBTITLE 1,Create facility_id
df["facility_id"] = df.index.astype(str)
print(f"Created facility_id for {len(df)} records")

# COMMAND ----------

display(df.head())

# COMMAND ----------

print(df.isnull().sum().sort_values(ascending=False))

# COMMAND ----------

def safe_parse_list(val):
    """Turn '["a","b"]' or null or [] into a plain string or empty string."""
    if pd.isna(val) or val in ("[]", "", "null"):
        return ""
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return " | ".join(str(i) for i in parsed)
        return str(parsed)
    except Exception:
        return str(val)

df["specialties_text"]  = df["specialties"].apply(safe_parse_list)
df["procedure_text"]    = df["procedure"].apply(safe_parse_list)
df["capability_text"]   = df["capability"].apply(safe_parse_list)
df["equipment_text"]    = df["equipment"].apply(safe_parse_list)

# COMMAND ----------

def build_notes(row):
    parts = []
    if row["description"]:             parts.append(f"About: {row['description']}")
    if row["capability_text"]:         parts.append(f"Capabilities: {row['capability_text']}")
    if row["specialties_text"]:        parts.append(f"Specialties: {row['specialties_text']}")
    if row["procedure_text"]:          parts.append(f"Procedures: {row['procedure_text']}")
    if row["equipment_text"]:          parts.append(f"Equipment: {row['equipment_text']}")
    return " | ".join(parts) if parts else None

df["notes_blob"] = df.apply(build_notes, axis=1)
df["has_notes"]  = df["notes_blob"].notna() & (df["notes_blob"].str.len() > 40)

print(f"Rows with usable notes: {df['has_notes'].sum()} / {len(df)}")

# COMMAND ----------

# DBTITLE 1,Clean and validate basic fields
# Clean basic fields
df["name"]       = df["name"].str.strip().str.title()
df["state"]      = df["address_stateOrRegion"].str.strip().str.title()
df["city"]       = df["address_city"].str.strip().str.title()
df["lat"]        = pd.to_numeric(df["latitude"], errors="coerce")
df["lon"]        = pd.to_numeric(df["longitude"], errors="coerce")
df["num_doctors"]= pd.to_numeric(df["numberDoctors"], errors="coerce")
df["capacity"]   = pd.to_numeric(df["capacity"], errors="coerce")

# Pin code validation (Indian pincodes are 6 digits)
def clean_pincode(val):
    """Extract valid 6-digit pincode or return None."""
    if pd.isna(val):
        return None
    
    # Convert to string and strip
    pin_str = str(val).strip()
    
    # Check for null-like strings
    if pin_str.lower() in ('nan', 'null', '', 'none'):
        return None
    
    # Remove any non-digit characters and pad/truncate to 6 digits
    pin_digits = re.sub(r'\D', '', pin_str)
    
    # Valid Indian pincode is exactly 6 digits
    if len(pin_digits) == 6 and pin_digits.isdigit():
        return pin_digits
    
    # Try to parse if it's too short (pad with zeros)
    if len(pin_digits) > 0 and len(pin_digits) < 6 and pin_digits.isdigit():
        return pin_digits.zfill(6)
    
    # Invalid pincode
    return None

df["pin_code"] = df["address_zipOrPostcode"].apply(clean_pincode)

print(f"Valid pincodes: {df['pin_code'].notna().sum():,} / {len(df):,}")
print(f"Missing pincodes: {df['pin_code'].isna().sum():,}")

# COMMAND ----------

df["trust_social_count"]   = pd.to_numeric(df["distinct_social_media_presence_count"], errors="coerce").fillna(0)
df["trust_has_staff"]      = df["affiliated_staff_presence"].fillna(0).astype(int)
df["trust_has_logo"]       = df["custom_logo_presence"].fillna(0).astype(int)
df["trust_fact_count"]     = pd.to_numeric(df["number_of_facts_about_the_organization"], errors="coerce").fillna(0)
df["trust_followers"]      = pd.to_numeric(df["engagement_metrics_n_followers"], errors="coerce").fillna(0)


# COMMAND ----------

# DBTITLE 1,Optional: Fill missing pincodes using coordinates
# =============================================================
# OPTIONAL: Use reverse geocoding to fill missing pincodes
# This uses the geopy library with Nominatim (OpenStreetMap)
# Rate limited to 1 request/second to be respectful
# =============================================================

%pip install geopy

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

# Find facilities with missing pincodes but valid coordinates
missing_pins = df[
    df["pin_code"].isna() & 
    df["lat"].notna() & 
    df["lon"].notna()
].copy()

print(f"Facilities with missing pincodes (but have coordinates): {len(missing_pins)}")

if len(missing_pins) > 0:
    response = input(f"\nDo you want to reverse geocode {len(missing_pins)} locations? (yes/no): ")
    
    if response.lower() == 'yes':
        geolocator = Nominatim(user_agent="medialert_facility_search")
        filled_count = 0
        
        print("\nReverse geocoding... (this may take a few minutes)")
        
        for idx in missing_pins.index:
            lat = df.loc[idx, "lat"]
            lon = df.loc[idx, "lon"]
            
            try:
                location = geolocator.reverse(f"{lat}, {lon}", timeout=10)
                
                if location and location.raw.get('address'):
                    address = location.raw['address']
                    pincode = address.get('postcode')
                    
                    if pincode:
                        # Clean and validate the pincode
                        pin_digits = re.sub(r'\D', '', pincode)
                        if len(pin_digits) == 6 and pin_digits.isdigit():
                            df.loc[idx, "pin_code"] = pin_digits
                            filled_count += 1
                            
                            if filled_count % 10 == 0:
                                print(f"  Filled {filled_count} pincodes...")
                
                # Rate limiting: 1 request per second
                time.sleep(1)
                
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                print(f"  Warning: Geocoding failed for {df.loc[idx, 'name']}: {e}")
                continue
        
        print(f"\n✓ Filled {filled_count} pincodes using coordinates")
        print(f"Remaining missing pincodes: {df['pin_code'].isna().sum()}")
    else:
        print("Skipped reverse geocoding. Pincodes remain null.")
else:
    print("No missing pincodes to fill.")

# COMMAND ----------

# =============================================================
# EVERYTHING BELOW REPLACES from "df["flag_no_location"]" 
# onwards in your cleaning notebook
# Keep everything above (notes_blob, basic cleaning) unchanged
# =============================================================

import math
import ast
import numpy as np

# =============================================================
# MEDICAL STANDARDS — borrowed from validator notebook
# Used here to score capability QUALITY, not just quantity
# =============================================================

MEDICAL_STANDARDS = {
    "ICU":        ["icu", "intensive care", "ventilator", "critical care"],
    "Surgery":    ["surgery", "surgical", "operation theatre", "ot "],
    "Emergency":  ["emergency", "casualty", "trauma", "24/7", "24 hour"],
    "Dialysis":   ["dialysis", "hemodialysis", "renal", "kidney"],
    "Oncology":   ["oncolog", "cancer", "chemotherapy", "radiotherapy"],
    "Neonatal":   ["neonatal", "nicu", "newborn", "incubator"],
    "BloodBank":  ["blood bank", "blood storage", "transfusion"],
    "Trauma":     ["trauma", "accident", "emergency surgery"],
    "Maternity":  ["maternity", "obstetric", "delivery", "labour"],
    "Cardiology": ["cardiology", "cardiac", "ecg", "echo", "heart"],
    "Orthopedic": ["orthop", "fracture", "bone", "joint replacement"],
    "Pediatric":  ["pediatric", "paediatric", "child", "neonat"],
}

REQUIRED_SUPPORT = {
    "ICU":       ["ventilator", "monitor", "critical care"],
    "Surgery":   ["anaes", "anesth", "operation theatre", "ot"],
    "Dialysis":  ["nephrolog", "dialysis machine", "hemodialysis"],
    "Neonatal":  ["incubator", "phototherapy", "neonatolog"],
    "Oncology":  ["chemotherapy", "radiotherapy", "oncolog"],
}

# =============================================================
# HELPER FUNCTIONS
# =============================================================

def count_list_items(val) -> int:
    if pd.isna(val) or str(val).strip() in ("[]", "", "null", "nan"):
        return 0
    try:
        parsed = ast.literal_eval(str(val))
        return len(parsed) if isinstance(parsed, list) else 1
    except Exception:
        return 1

def notes_contains(notes: str, keywords: list) -> bool:
    n = str(notes or "").lower()
    return any(kw.lower() in n for kw in keywords)

def count_specialties_matched(notes: str) -> int:
    """How many medical standard categories does this facility cover."""
    n = str(notes or "").lower()
    return sum(
        1 for kws in MEDICAL_STANDARDS.values()
        if any(kw in n for kw in kws)
    )

# =============================================================
# STEP 1 — TRUE CONTRADICTIONS
# Both sides of the contradiction must be present and conflict.
# =============================================================

notes = df["notes_blob"].fillna("").str.lower()
cap   = df["capability_text"].fillna("").str.lower()
proc  = df["procedure_text"].fillna("").str.lower()
spec  = df["specialties_text"].fillna("").str.lower()
equip = df["equipment_text"].fillna("").str.lower()

# Claimed ICU in notes BUT capacity is 0 or null
df["flag_icu_contradiction"] = (
    (notes.str.contains("icu|intensive care", na=False))
    & (df["capacity"].fillna(0) == 0)
)

# Claimed surgery in notes BUT no anaesthesia keyword anywhere
df["flag_surgery_no_anaesthesia"] = (
    (notes.str.contains("surgery|surgical|operation", na=False))
    & (~notes.str.contains("anaes|anesth|anaesth", na=False))
    & (df["notes_blob"].str.len().fillna(0) > 60)  # only if notes are substantial
)

# Claimed dialysis BUT no nephrologist or machine
df["flag_dialysis_no_support"] = (
    (notes.str.contains("dialysis|hemodialysis", na=False))
    & (~notes.str.contains("nephrolog|dialysis machine|renal technician", na=False))
)

# Claimed 24/7 BUT notes say appointment/OPD only
df["flag_availability_contradiction"] = (
    (notes.str.contains("24/7|24 hour|round the clock", na=False))
    & (notes.str.contains("appointment only|opd only|by appointment", na=False))
)

# Facility type is "clinic" but claims hospital-level procedures
df["flag_overreach_claim"] = (
    df["facilityTypeId"].fillna("").str.lower().str.contains("clinic", na=False)
    & notes.str.contains("cardiac surgery|neurosurgery|organ transplant|open heart", na=False)
)

# Claimed neonatal/NICU but no incubator or neonatologist
df["flag_nicu_no_support"] = (
    (notes.str.contains("nicu|neonatal", na=False))
    & (~notes.str.contains("incubator|phototherapy|neonatolog", na=False))
)

# Count only true contradictions for severity
df["contradiction_count"] = (
    df["flag_icu_contradiction"].astype(int)
    + df["flag_surgery_no_anaesthesia"].astype(int)
    + df["flag_dialysis_no_support"].astype(int)
    + df["flag_availability_contradiction"].astype(int)
    + df["flag_overreach_claim"].astype(int)
    + df["flag_nicu_no_support"].astype(int)
)



# COMMAND ----------

# =============================================================
# STEP 2 — DATA GAPS (missing info — NOT red flags)
# =============================================================

df["gap_no_location"]      = df[["lat", "lon"]].isna().any(axis=1)
df["gap_no_notes"]         = ~df["has_notes"]
df["gap_no_doctor_count"]  = df["num_doctors"].isna()
df["gap_no_equipment_data"]= df["equipment_text"].eq("") | df["equipment_text"].isna()
df["gap_no_capacity_data"] = df["capacity"].isna()
df["gap_no_specialties"]   = df["specialties_text"].eq("") | df["specialties_text"].isna()
df["gap_no_procedures"]    = df["procedure_text"].eq("") | df["procedure_text"].isna()
df["gap_no_capabilities"]  = df["capability_text"].eq("") | df["capability_text"].isna()

# Skeleton = has almost nothing useful
df["gap_skeleton_record"] = (
    df["gap_no_notes"]
    & df["gap_no_doctor_count"]
    & df["gap_no_specialties"]
    & df["gap_no_procedures"]
)

df["gap_count"] = (
    df["gap_no_notes"].astype(int)
    + df["gap_no_doctor_count"].astype(int)
    + df["gap_no_equipment_data"].astype(int)
    + df["gap_no_capacity_data"].astype(int)
    + df["gap_no_specialties"].astype(int)
    + df["gap_no_procedures"].astype(int)
)



# COMMAND ----------

# =============================================================
# STEP 3 — TRUST SCORE v2 (4 pillars, replaces trust_score_raw)
# FIXED: Added NaN handling + duplicate column prevention
# =============================================================

def score_presence(row) -> float:
    """30% weight — how discoverable and verifiable is this entity."""
    s = 0.0
    s += min(float(row.get("trust_social_count") or 0) / 3.0, 1.0) * 2.5
    s += 2.0 if float(row.get("trust_has_staff") or 0) == 1 else 0.0
    s += 1.0 if float(row.get("trust_has_logo") or 0) == 1 else 0.0
    s += min(math.log1p(float(row.get("trust_fact_count") or 0))
             / math.log1p(10), 1.0) * 2.0
    has_web = pd.notna(row.get("officialWebsite")) and \
              str(row.get("officialWebsite", "")).strip() not in ("", "null", "nan")
    s += 1.5 if has_web else 0.0
    has_email = pd.notna(row.get("email")) and \
                "@" in str(row.get("email") or "")
    s += 1.0 if has_email else 0.0
    result = min(s / 10.0 * 10.0, 10.0)
    # Ensure no NaN
    return 0.0 if (math.isnan(result) or math.isinf(result)) else result


def score_capability(row) -> float:
    """35% weight — how rich and specific is the medical capability data."""
    s = 0.0
    notes = str(row.get("notes_blob") or "")
    note_len = len(notes.strip())

    # Notes richness
    if note_len == 0:      s -= 2.0
    elif note_len < 60:    s += 0.5
    elif note_len < 200:   s += 1.5
    elif note_len < 500:   s += 2.5
    else:                  s += 3.5

    # How many medical standard categories covered
    covered = count_specialties_matched(notes)
    if covered > 0:
        s += min(covered / 4.0, 1.0) * 2.0

    # Structured list richness (log scale) - with safe division
    n_spec = count_list_items(row.get("specialties"))
    n_proc = count_list_items(row.get("procedure"))
    n_cap  = count_list_items(row.get("capability"))
    n_equip= count_list_items(row.get("equipment"))

    if n_spec > 0:
        s += min(math.log1p(n_spec)  / math.log1p(10), 1.0) * 1.5
    if n_proc > 0:
        s += min(math.log1p(n_proc)  / math.log1p(15), 1.0) * 2.0
    if n_cap > 0:
        s += min(math.log1p(n_cap)   / math.log1p(10), 1.0) * 1.5
    if n_equip > 0:
        s += 0.5

    # Doctor count bonus - handle null/NaN
    try:
        docs = float(row.get("num_doctors") or 0)
        if not math.isnan(docs) and not math.isinf(docs) and docs > 0:
            s += min(docs / 5.0, 1.0) * 1.0
    except (ValueError, TypeError):
        pass  # Skip if conversion fails

    result = (s + 2.0) / 16.0 * 10.0
    result = min(max(result, 0.0), 10.0)
    
    # Ensure no NaN or Inf
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def score_integrity(row) -> float:
    """25% weight — starts at 10, deducts ONLY for true contradictions."""
    s = 10.0
    s -= float(row.get("flag_icu_contradiction") or False)        * 4.0
    s -= float(row.get("flag_surgery_no_anaesthesia") or False)   * 4.0
    s -= float(row.get("flag_dialysis_no_support") or False)      * 2.0
    s -= float(row.get("flag_availability_contradiction") or False)* 2.0
    s -= float(row.get("flag_overreach_claim") or False)          * 3.0
    s -= float(row.get("flag_nicu_no_support") or False)          * 2.0
    # Soft deduction for skeleton records
    s -= float(row.get("gap_skeleton_record") or False)           * 1.5
    result = min(max(s, 0.0), 10.0)
    return 0.0 if (math.isnan(result) or math.isinf(result)) else result


def score_engagement(row) -> float:
    """10% weight — social activity signals real patient interaction."""
    s = 0.0
    try:
        followers = float(row.get("trust_followers") or 0)
        if not math.isnan(followers):
            if followers > 1000:  s += 4.0
            elif followers > 100: s += 2.5
            elif followers > 10:  s += 1.0

        posts = float(row.get("post_metrics_post_count") or 0)
        if not math.isnan(posts):
            if posts > 50:   s += 3.0
            elif posts > 10: s += 2.0
            elif posts > 0:  s += 1.0

        engagements = float(row.get("engagement_metrics_n_engagements") or 0)
        if not math.isnan(engagements):
            if engagements > 500:  s += 3.0
            elif engagements > 50: s += 2.0
            elif engagements > 0:  s += 1.0
    except (ValueError, TypeError):
        pass  # Default s=0 if conversions fail

    result = s / 10.0 * 10.0
    result = min(result, 10.0)
    return 0.0 if (math.isnan(result) or math.isinf(result)) else result


WEIGHTS = {"presence": 0.30, "capability": 0.35,
           "integrity": 0.25, "engagement": 0.10}


def compute_trust(row) -> dict:
    p1 = score_presence(row)
    p2 = score_capability(row)
    p3 = score_integrity(row)
    p4 = score_engagement(row)

    composite = round(
        p1 * WEIGHTS["presence"]    +
        p2 * WEIGHTS["capability"]  +
        p3 * WEIGHTS["integrity"]   +
        p4 * WEIGHTS["engagement"], 2
    )
    composite = min(max(composite, 0.0), 10.0)
    
    # Ensure composite is not NaN
    if math.isnan(composite) or math.isinf(composite):
        composite = 0.0

    # Wilson confidence interval - with NaN protection
    signals = [p1/10, p2/10, p3/10, p4/10]
    # Filter out any NaN values
    signals = [s for s in signals if not (math.isnan(s) or math.isinf(s))]
    
    if len(signals) > 0:
        p_hat   = sum(signals) / len(signals)
        z, n    = 1.96, len(signals)
        denom   = 1 + z**2 / n
        ctr     = (p_hat + z**2 / (2*n)) / denom
        margin  = (z * math.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / denom
        ci_lo   = round(max(0, ctr - margin) * 10, 2)
        ci_hi   = round(min(1, ctr + margin) * 10, 2)
    else:
        # Fallback if all signals are NaN
        ci_lo, ci_hi = 0.0, 10.0

    if composite >= 8:   label = "VERIFIED"
    elif composite >= 5: label = "LIKELY_RELIABLE"
    elif composite >= 3: label = "UNVERIFIED"
    else:                label = "UNRELIABLE"

    return {
        "composite_trust"   : composite,
        "trust_ci_lower"    : ci_lo,
        "trust_ci_upper"    : ci_hi,
        "trust_label"       : label,
        "pillar_presence"   : round(p1, 2),
        "pillar_capability" : round(p2, 2),
        "pillar_integrity"  : round(p3, 2),
        "pillar_engagement" : round(p4, 2),
    }


# Drop existing trust columns if they exist (prevents duplicates)
trust_cols_to_drop = ["composite_trust", "trust_ci_lower", "trust_ci_upper", "trust_label",
                      "pillar_presence", "pillar_capability", "pillar_integrity", "pillar_engagement"]
df = df.drop(columns=[c for c in trust_cols_to_drop if c in df.columns])

print("Computing trust scores (this takes ~1 min for 10k rows)...")
trust_df = df.apply(compute_trust, axis=1, result_type="expand")
df       = pd.concat([df, trust_df], axis=1)

print("\n=== Trust score distribution ===" )
print(df["trust_label"].value_counts())
print("\n=== Composite trust stats ===" )
print(f"Mean: {df['composite_trust'].mean():.2f}")
print(f"Median: {df['composite_trust'].median():.2f}")
print(f"Null count: {df['composite_trust'].isna().sum()}")
print("\n=== Contradiction summary (genuine red flags) ===" )
print(f"ICU contradiction:         {df['flag_icu_contradiction'].sum():>6,}")
print(f"Surgery no anaesthesia:    {df['flag_surgery_no_anaesthesia'].sum():>6,}")
print(f"Dialysis no support:       {df['flag_dialysis_no_support'].sum():>6,}")
print(f"Availability contradiction:{df['flag_availability_contradiction'].sum():>6,}")
print(f"Overreach claims:          {df['flag_overreach_claim'].sum():>6,}")
print(f"NICU no support:           {df['flag_nicu_no_support'].sum():>6,}")
print(f"\nTotal with ≥1 contradiction: {(df['contradiction_count']>0).sum():,}")
print("\n=== Gap summary (missing data only) ===" )
print(f"No notes:                  {df['gap_no_notes'].sum():>6,}")
print(f"No doctor count:           {df['gap_no_doctor_count'].sum():>6,}")
print(f"No equipment data:         {df['gap_no_equipment_data'].sum():>6,}")
print(f"Skeleton records:          {df['gap_skeleton_record'].sum():>6,}")

# COMMAND ----------

# DBTITLE 1,Cell 11
# =============================================================
# STEP 4 — REBUILD SQL TABLE ONLY
# (facilities_for_embedding is untouched)
# =============================================================

SQL_COLS = [
    # Identity
    "facility_id", "name", "officialPhone", "email", "officialWebsite",
    # Location
    "address_line1", "city", "state", "pin_code", "lat", "lon",
    # Facility info
    "facilityTypeId", "operatorTypeId", "yearEstablished",
    "num_doctors", "capacity",
    # Raw signals (kept for backwards compat)
    "trust_social_count", "trust_has_staff", "trust_has_logo",
    "trust_fact_count", "trust_followers",
    # Trust score v2
    "composite_trust", "trust_ci_lower", "trust_ci_upper", "trust_label",
    "pillar_presence", "pillar_capability", "pillar_integrity", "pillar_engagement",
    # True contradictions
    "flag_icu_contradiction", "flag_surgery_no_anaesthesia",
    "flag_dialysis_no_support", "flag_availability_contradiction",
    "flag_overreach_claim", "flag_nicu_no_support",
    "contradiction_count",
    # Data gaps
    "gap_no_notes", "gap_no_doctor_count", "gap_no_equipment_data",
    "gap_no_capacity_data", "gap_no_specialties", "gap_no_procedures",
    "gap_no_capabilities", "gap_skeleton_record", "gap_count",
]

df_sql = df[SQL_COLS].copy()
for col in df_sql.select_dtypes(include=["object"]).columns:
    df_sql[col] = df_sql[col].astype(str)

spark.createDataFrame(df_sql) \
     .write.format("delta") \
     .mode("overwrite") \
     .option("overwriteSchema", "true") \
     .saveAsTable("workspace.default.facilities_sql")

print(f"\nSQL table saved: {len(df_sql):,} rows, {len(SQL_COLS)} columns")
print("facilities_for_embedding unchanged.")



# COMMAND ----------

# =============================================================
# STEP 5 — VERIFY
# =============================================================

verify = spark.table("workspace.default.facilities_sql")
print(f"\nVerification — row count: {verify.count():,}")
print("Columns:", verify.columns)

display(
    verify.orderBy("composite_trust", ascending=False)
          .select("name", "city", "state",
                  "composite_trust", "trust_label",
                  "trust_ci_lower", "trust_ci_upper",
                  "contradiction_count", "gap_count",
                  "flag_icu_contradiction", "flag_surgery_no_anaesthesia")
          .limit(20)
)

# COMMAND ----------

# DBTITLE 1,Cell 12
VECTOR_COLS = [
    "facility_id",  # Primary key for vector search
    "name", "pin_code", "state", "city", "lat", "lon",
    "facilityTypeId", "notes_blob", "trust_score_raw",
]

# COMMAND ----------

df_sql    = df[SQL_COLS].copy()
df_vector = df[VECTOR_COLS].dropna(subset=["notes_blob"]).copy()

print(f"SQL table rows:    {len(df_sql)}")
print(f"Vector table rows: {len(df_vector)}")

# COMMAND ----------


for col in df_sql.select_dtypes(include=['object']).columns:
    df_sql[col] = df_sql[col].astype(str)

for col in df_vector.select_dtypes(include=['object']).columns:
    df_vector[col] = df_vector[col].astype(str)

#spark.createDataFrame(df_sql).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.default.facilities_sql")
spark.createDataFrame(df_vector).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.default.facilities_for_embedding")

# COMMAND ----------



# COMMAND ----------

df_sql = spark.read.table("workspace.default.facilities_sql")
df_vec = spark.read.table("workspace.default.facilities_for_embedding")

df_sql_kz = df_sql.filter(df_sql["name"] == "K Z Hospital Baisi")
df_vec_kz = df_vec.filter(df_vec["name"] == "K Z Hospital Baisi")
df_kz = df[df["name"] == "K Z Hospital Baisi"]

display(df_sql_kz)
display(df_vec_kz)
display(df_kz)

# COMMAND ----------

display(df_sql)

# COMMAND ----------

