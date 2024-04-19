# Databricks notebook source
import requests
import json

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
folder_name = dbutils.widgets.get("folder_name")

# COMMAND ----------

workspaceUrl = 'https://' + spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
current_user = spark.sql("select current_user()").first()[0]

# COMMAND ----------

url = f"{workspaceUrl}/api/2.1/jobs/create"

headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer {token}"
}

payload = json.dumps(
{
  "name": "job_gradio_app",
  "timeout_seconds": 0,
  "continuous": {
    "pause_status": "PAUSED"
  },
  "max_concurrent_runs": 1,
  "tasks": [
    {
      "task_key": "sample_chromadb_llms",
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": f"/Repos/{current_user}/{folder_name}/Sample Chroma Vector DB with LLMs",
        "base_parameters": {
          "catalog": catalog,
          "schema": schema
        },
        "source": "WORKSPACE"
      },
      "job_cluster_key": "rag_poc_cluster"
    }
  ],
  "job_clusters": [
    {
      "job_cluster_key": "rag_poc_cluster",
      "new_cluster": {
        "cluster_name": "",
        "spark_version": "13.3.x-scala2.12",
        "spark_conf": {
          "spark.master": "local[*, 4]",
          "spark.databricks.cluster.profile": "singleNode"
        },
        "aws_attributes": {
          "first_on_demand": 1,
          "availability": "SPOT_WITH_FALLBACK",
          "zone_id": "auto",
          "spot_bid_price_percent": 100,
          "ebs_volume_type": "GENERAL_PURPOSE_SSD",
          "ebs_volume_count": 1,
          "ebs_volume_size": 100
        },
        "node_type_id": "r5.2xlarge",
        "driver_node_type_id": "r5.2xlarge",
        "custom_tags": {
          "ResourceClass": "SingleNode",
          "Application": "pdf_rag_poc"
        },
        "enable_elastic_disk": True,
        "data_security_mode": "SINGLE_USER",
        "runtime_engine": "STANDARD",
        "num_workers": 0
      }
    }
  ],
  "run_as": {
    "user_name": current_user
  }
}
)

response = requests.request("POST", url, headers=headers, data=payload)

if response.status_code == 200:
  dbutils.notebook.exit("Job for gradio app created successfully! Job id: " + str(response.json()['job_id']))
else:
  dbutils.notebook.exit("Job creation failed")
