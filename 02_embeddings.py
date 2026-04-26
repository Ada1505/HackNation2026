# Databricks notebook source
from databricks.sdk import WorkspaceClient
import pandas as pd

# COMMAND ----------

df = spark.table("workspace.default.facilities_for_embedding").toPandas()

# COMMAND ----------



# w = WorkspaceClient()

# def embed_batch(texts: list[str]) -> list[list[float]]:
#     embeddings = []
#     for t in texts:
#         response = w.serving_endpoints.query(
#             name="databricks-gte-large-en",
#             input=t
#         )
#         embeddings.append(response.data[0].embedding)
#     return embeddings

# # Embed in batches of 50 (rate limit safety)
# BATCH = 50
# all_embeddings = []
# for i in range(0, len(df), BATCH):
#     batch = df["notes_blob"].iloc[i:i+BATCH].fillna("").tolist()
#     all_embeddings.extend(embed_batch(batch))
#     print(f"Embedded {min(i+BATCH, len(df))}/{len(df)}")

# df["embedding"] = all_embeddings


# COMMAND ----------


spark.createDataFrame(
    df[["facility_id", "notes_blob", "name", "state", "pin_code"]]
).write.mode("append").saveAsTable("workspace.default.facilities_for_embedding")

print("Source table updated. Vector index will sync automatically.")

# COMMAND ----------

