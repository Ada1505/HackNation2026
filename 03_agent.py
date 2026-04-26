# Databricks notebook source
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
VS_INDEX     = "main.default.facilities_vector_index"
VS_ENDPOINT  = "medi-alert-vs"          # your existing endpoint name

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

# =============================================================
# CELL 1 — INSTALL DEPENDENCIES
# =============================================================

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
def search_facilities(query: str, top_k: int = 8, state: str = None) -> list[dict]:
    """
    Search the 10k facility vector index using semantic similarity.
    Returns top_k facilities with their trust_score, name, state, pin_code.
    Optionally filters by state name.
    """
    filters = {}
    if state:
        filters["state"] = state

    response = vs_index.similarity_search(
        query_text  = query,
        columns     = ["facility_id", "name", "state", "pin_code", "notes_blob"],
        num_results = top_k,
        filters     = filters if filters else None,
    )
    hits      = response.get("result", {}).get("data_array", [])
    col_names = response.get("result", {}).get("columns", [])
    results   = [dict(zip(col_names, row)) for row in hits]

    # Enrich with trust_score and flags from SQL table
    if results:
        ids_list = ", ".join(f"'{r['facility_id']}'" for r in results)
        sql_rows = spark.sql(f"""
            SELECT CAST(ROW_NUMBER() OVER (ORDER BY name) - 1 AS STRING) AS facility_id,
                   name, state, city, pin_code, trust_score_raw,
                   flag_icu_claimed_no_beds, flag_surgery_no_anesthesia,
                   flag_no_equipment, flag_zero_doctors, num_doctors, capacity,
                   facilityTypeId, officialPhone, email, address_line1
            FROM {SQL_TABLE}
        """).toPandas()
        # Build a lookup by name for enrichment (facility_id was set as index in notebook 02)
        name_map = sql_rows.set_index("name").to_dict("index")
        for r in results:
            extra = name_map.get(r.get("name"), {})
            r["trust_score"]              = extra.get("trust_score_raw", 0)
            r["flag_icu_no_beds"]         = bool(extra.get("flag_icu_claimed_no_beds", False))
            r["flag_surgery_no_anaes"]    = bool(extra.get("flag_surgery_no_anesthesia", False))
            r["flag_no_equipment"]        = bool(extra.get("flag_no_equipment", False))
            r["flag_zero_doctors"]        = bool(extra.get("flag_zero_doctors", False))
            r["num_doctors"]              = extra.get("num_doctors")
            r["capacity"]                 = extra.get("capacity")
            r["facility_type"]            = extra.get("facilityTypeId")
            r["phone"]                    = extra.get("officialPhone")
            r["address"]                  = extra.get("address_line1")
            r["city"]                     = extra.get("city")

    return results


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
        "sample_facilities": df[["name", "state", "city", "pin_code"]].head(5).to_dict("records"),
    }


# Tool registry for the agent loop
TOOLS = {
    "search_facilities"    : search_facilities,
    "get_facility_detail"  : get_facility_detail,
    "find_medical_deserts" : find_medical_deserts,
}

print("Tools defined:", list(TOOLS.keys()))


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
                "Use this for queries about capabilities, specialties, equipment, "
                "or procedures. Returns ranked results with trust scores and flags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of the required capability"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 8, max 20)",
                        "default": 8
                    },
                    "state": {
                        "type": "string",
                        "description": "Optional: restrict search to a specific Indian state"
                    }
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
# CELL 5 — SYSTEM PROMPT
# =============================================================

SYSTEM_PROMPT = """You are MediAlert, an Agentic Healthcare Intelligence System for India.
Your mission is to reduce Discovery-to-Care time for 1.4 billion people.

You have access to a database of 10,000+ medical facilities across India,
each with structured metadata and free-form capability notes.

REASONING RULES:
1. ALWAYS use search_facilities first to retrieve candidates.
2. For any facility you recommend, call get_facility_detail to verify the claim
   and extract the EXACT sentence from notes_blob that supports your recommendation.
3. Check trust_score: facilities with trust_score < 2 should be flagged as unverified.
4. Check contradiction flags: flag_icu_no_beds, flag_surgery_no_anaes, flag_no_equipment.
   If a flag is True, note the contradiction in your response.
5. For desert/gap queries, use find_medical_deserts first.
6. Always cite the specific facility field or notes sentence that justifies each claim.
7. Structure your final answer as:
   - RECOMMENDATION: ranked list with name, city, state, pin_code, trust_score
   - EVIDENCE: exact sentence from notes_blob for each recommendation
   - FLAGS: any contradictions or low-trust warnings
   - CONFIDENCE: your overall confidence (High/Medium/Low) and why

Be specific. Never guess. If data is insufficient, say so."""


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
# CELL 7 — RUN SINGLE QUERY (with MLflow run)
# =============================================================

def query_agent(user_query: str, run_name: str = None) -> str:
    """Wrapper that logs the full agent run to MLflow."""
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


# ── Test query 1: Multi-attribute capability search ──────────
answer1 = query_agent(
    "Find the nearest facility in rural Bihar that can perform an emergency "
    "appendectomy and typically leverages part-time doctors.",
    run_name="test_appendectomy_Bihar"
)


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