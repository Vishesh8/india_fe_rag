# Databricks notebook source
# MAGIC %pip install chromadb textstat gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import pandas as pd
import mlflow
import chromadb
from langchain.chains import RetrievalQA
from langchain.llms import Databricks
from langchain.chat_models import ChatDatabricks
from langchain.embeddings.databricks import DatabricksEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

pdf_emb_table = spark.read.table(f"{catalog}.{schema}.pdf_demo_tbl").toPandas()
pdf_emb_dict = pdf_emb_table.to_dict("records")
len(pdf_emb_dict)

# COMMAND ----------

pdf_emb_dict

# COMMAND ----------

client = chromadb.Client()
collection = client.create_collection("pdf_collection")

# COMMAND ----------

for each_item in pdf_emb_dict:
  collection.add(
    embeddings=[each_item['embedding'].tolist()],
    uris=each_item['url'],
    documents=each_item['content'],
    ids=[str(each_item['id'])],
  )
collection.count()

# COMMAND ----------

llm = ChatDatabricks(
    endpoint="databricks-dbrx-instruct",
    max_tokens=256
)

# create the embedding function using Databricks Foundation Model APIs
embedding_function = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
docsearch = Chroma(
    client=client,
    collection_name="pdf_collection",
    embedding_function=embedding_function,
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(fetch_k=3),
    return_source_documents=True,
)

# COMMAND ----------

# qa("what is Crisil India mission?")

# COMMAND ----------

def respond(message, history):
  if len(message.strip()) == 0:
    return "ERROR the question should not be empty"
  # q = {"inputs": [message]}
  # try:
  response_data=qa(message)["result"]
  # except Exception as error:
  #   response_data = f"ERROR status_code: {type(error).__name__}"
  # # print(response.json())
  return response_data

# COMMAND ----------

examples = respond("can you write 3 questions specific to Crisil India", "the document is a business report")

# COMMAND ----------

examples.split("\n\n")

# COMMAND ----------



# COMMAND ----------

import gradio as gr
from gradio.themes.utils import sizes

theme = gr.themes.Soft(
    text_size=sizes.text_sm,radius_size=sizes.radius_sm, spacing_size=sizes.spacing_sm,
)
demo = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(show_label=False, container=False, show_copy_button=True, bubble_full_width=True),
    textbox=gr.Textbox(placeholder="Ask me a question",
                       container=False, scale=7),
    title="Databricks LLM RAG demo - Chat with DBRX Databricks model serving endpoint",
    description="This chatbot is a demo example for the dbdemos llm chatbot. <br>This content is provided as a LLM RAG educational example, without support. It is using DBRX, can hallucinate and should not be used as production content.<br>Please review our dbdemos license and terms for more details.",
    #examples=[["Summarize the business report?"],
              #["What is the recent GDP estimates?"]],
    examples= examples.split("\n\n"),
    cache_examples=False,
    theme=theme,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)

demo.launch(share=True)

# COMMAND ----------


