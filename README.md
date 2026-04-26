# MediAlert 

**India's AI-powered healthcare facility intelligence platform.**  
Built for Hackathon Challenge 03 "Serving A Nation" — 10,000+ verified Indian medical facilities on Databricks.

---

## What it does

MediAlert helps patients and healthcare workers find the right facility fast. It uses a full ReAct AI agent to reason over real facility data, classify results by confidence tier, and explain every decision it makes.

- **AI Agent** — Ask in plain English: *"Find an emergency appendectomy facility in rural Bihar"*. The agent searches, verifies, and returns Tier 1 (confirmed) vs Tier 2 (possible) results with evidence quotes.
- **Find Facilities** — SQL-powered keyword + specialty + city search across all 10k facilities with trust scores and quality flags.
- **Desert Map** — Identify which Indian states are critically underserved for a given specialty (dialysis, oncology, ICU, etc.).
- **Near Me** — Paste a location or map URL, get nearby verified facilities with distance comparison and OpenStreetMap view.
- **WhatsApp Bot** — Twilio-powered chatbot: `find ICU Mumbai`, `near Patna`, `deserts dialysis`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data platform | Databricks (Unity Catalog, SQL Warehouse) |
| Semantic search | Mosaic AI Vector Search |
| LLM / Agent | Llama 3.3-70B via Databricks Foundation Models |
| Agent framework | Custom ReAct loop (OpenAI-compatible API) |
| Tracing | MLflow 3 (spans per LLM call + tool call) |
| Frontend | Streamlit |
| Geocoding | OpenStreetMap / Nominatim (no API key) |
| Messaging | Twilio WhatsApp sandbox |

---

## Project Structure

```
.
├── backend.py          # Databricks connector: SQL, Vector Search, ReAct agent, MLflow tracing
├── streamlit_app.py    # Full Streamlit UI (5 tabs)
├── 01_cleaning.py # Notebook: data cleaning
├── 02_embeddings.py # Notebook: 
├── 03_agent.py         # Notebook: agent development + MLflow experiments
├── 04_validator.py     # Notebook: validator / self-correction layer
├── 05_medical_deserts.py # Notebook: desert analysis dashboard
├── filtered_heatmaps_text.py # Notebook: streamlit ui with filtered heatmaps
├── facilities_clean_sql # data after cleaning
├── facilities_for_embedding.csv # facility description data
├── requirements.txt # Python dependencies
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file (or set Streamlit secrets):

```env
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-personal-access-token
DATABRICKS_SQL_WH=your-warehouse-id

VS_ENDPOINT=vs-readyalert-endpoint
VS_INDEX=workspace.default.facilities_for_embedding_index

LLM_ENDPOINT=databricks-meta-llama-3-3-70b-instruct
```

Optional (WhatsApp):
```env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
```

### 3. Run

```bash
streamlit run streamlit_app.py
```

---

## Data Tables (Unity Catalog)

| Table | Description |
|---|---|
| `workspace.default.facilities_sql` | Structured metadata: state, city, trust score, flags, phone, capacity |
| `workspace.default.facilities_for_embedding` | Notes blobs used for semantic search and embeddings |

---

## Agent Architecture

```
User Query
    │
    ▼
ReAct Loop (max 6 iterations)
    │
    ├─ search_facilities()  →  Vector Search → SQL enrichment
    ├─ get_facility_detail() →  Full record + notes_blob verification
    └─ find_medical_deserts() → State-level coverage analysis
    │
    ▼
Structured JSON Response
    ├─ tier1_confirmed  (capability explicitly verified in notes)
    ├─ tier2_possible   (related capability, call to confirm)
    └─ confidence + evidence + contradiction flags
    │
    ▼
MLflow 3 Trace  (1 span per LLM call + 1 span per tool call)
```

**Fallback:** If Vector Search is unavailable, the agent automatically falls back to SQL keyword search on `notes_blob` — real data is always returned.

---

## Trust Scoring

| Score | Label | Meaning |
|---|---|---|
| 7.0 – 10.0 | VERIFIED | High confidence, complete data |
| 4.0 – 6.9 | LIKELY_RELIABLE | Adequate data, minor gaps |
| < 4.0 | UNVERIFIED | Incomplete or contradictory data |

*Computes a Wilson confidence interval around a facility's
    trust score using a Wilson score interval approach.*

**Quality Flags:**
- `flag_icu_contradiction` — claims ICU but 0 beds recorded
- `flag_surgery_no_anaesthesia` — claims surgery, no anaesthesiologist
- `gap_no_doctor_count` — no doctor count on record
- `gap_no_equipment_data` — no equipment data

---

## Team
- Ayesha Farooqui
- Abdul Ahad
- Hassan Shakil Pasha
