# Retreival Augmented Generation (RAG) Demo

Sample RAG implementation with open source ChromaDB Vector Search, bge-large-en embedding model & DBRX-Instruct LLM. Install this code repo in your workspace for a quick experiment on Databricks

## Usage:

Copy and paste the code below in your Databricks notebook cell, configure the set of to-be-configured parameters, and run the cell
This should clone the repo in your user folder as well as create sample Databricks jobs for pdf processing on top of your configured volume and also a gradio app job that you can run continuously

## To be Configured
### Set Catalog Configurations
```
catalog = "sarbani_dbrx_catalog"
schema = "india_fe_demo"
volume = "fe_demo_pdf"
```


### No need to configure
```
gitProvider = "gitHub"
gitUrl = "https://github.com/Vishesh8/india_fe_rag"

# Create Repo in the User Folder
import requests
import json

workspaceUrl = 'https://' + spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get() # Not recommended and can be replaced your your own user token
current_user = spark.sql("select current_user()").first()[0]
folder_name = gitUrl.split("/")[-1]

url = f"{workspaceUrl}/api/2.0/repos"

payload = json.dumps({
  "url": gitUrl,
  "provider": gitProvider,
  "path": f"/Repos/{current_user}/{folder_name}"
})

headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer {token}"
}

response = requests.request("POST", url, headers=headers, data=payload).json()

# Create Jobs for a scheduled or continuous run
status_pdf = dbutils.notebook.run(f"/Repos/{current_user}/{folder_name}/_resources/create-job-pdf-processing", 3600, {"catalog": catalog, "folder_name": folder_name, "schema": schema, "volume": volume})
gradio_app = dbutils.notebook.run(f"/Repos/{current_user}/{folder_name}/_resources/create-job-gradio-app", 3600, {"catalog": catalog, "folder_name": folder_name, "schema": schema})

print(response)
print(status_pdf)
print(gradio_app)
```
