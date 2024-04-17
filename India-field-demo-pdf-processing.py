# Databricks notebook source
# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.1.5 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.10.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# %pip install transformers "unstructured[pdf,docx,image,md]" langchain llama-index  pydantic mlflow chromadb gradio langchain-openai
# dbutils.library.restartPython()

# COMMAND ----------

from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from llama_index import Document, set_global_tokenizer
from transformers import AutoTokenizer

# COMMAND ----------

#install poppler on the cluster (should be done by init scripts)
def install_ocr_on_nodes():
    """
    install poppler on the cluster (should be done by init scripts)
    """
    # from pyspark.sql import SparkSession
    import subprocess
    num_workers = max(1,int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers")))
    command = "sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get purge && sudo apt-get clean && sudo apt-get update && sudo apt-get install poppler-utils tesseract-ocr -y" 
    def run_subprocess(command):
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
            return output.decode()
        except subprocess.CalledProcessError as e:
            raise Exception("An error occurred installing OCR libs:"+ e.output.decode())
    #install on the driver
    run_subprocess(command)
    def run_command(iterator):
        for x in iterator:
            yield run_subprocess(command)
    # spark = SparkSession.builder.getOrCreate()
    data = spark.sparkContext.parallelize(range(num_workers), num_workers) 
    # Use mapPartitions to run command in each partition (worker)
    output = data.mapPartitions(run_command)
    try:
        output.collect();
        print("OCR libraries installed")
    except Exception as e:
        print(f"Couldn't install on all node: {e}")
        raise e

# COMMAND ----------

# For production use-case, install the libraries at your cluster level with an init script instead. 
install_ocr_on_nodes()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Let's extract text from our PDF. Then load it in delta table

# COMMAND ----------

#pdf_loc = '/Volumes/sarbani_dbrx_catalog/india_fe_demo/pdf_volume/India_outlook_2023.pdf'
pdf_loc = '/Volumes/sarbani_dbrx_catalog/india_fe_demo/fe_demo_pdf'

# COMMAND ----------

display(dbutils.fs.ls(pdf_loc))

# COMMAND ----------



# COMMAND ----------

df = (spark.readStream
        .format('cloudFiles')
        .option('cloudFiles.format', 'BINARYFILE')
        .option("pathGlobFilter", "*.pdf")
        .load(pdf_loc))

# Write the data as a Delta table
(df.writeStream
  .trigger(availableNow=True)
  .option("checkpointLocation", f'pdf_loc/checkpoints/raw_docs')
  .table("`sarbani_dbrx_catalog`.`india_fe_demo`.`pdf_raw`")
  .awaitTermination())

# COMMAND ----------

# MAGIC %sql SELECT * FROM `sarbani_dbrx_catalog`.`india_fe_demo`.`pdf_raw` 

# COMMAND ----------



# COMMAND ----------

from unstructured.partition.auto import partition
from unstructured.documents.elements import Element
import re
import io

def extract_doc_text(x : bytes) -> str:
  # Read files and extract the values with unstructured
  sections = partition(file=io.BytesIO(x))
  def clean_section(txt):
    txt = re.sub(r'\n', '', txt)
    return re.sub(r' ?\.', '.', txt)
  # Default split is by section of document, concatenate them all together because we want to split by sentence instead.
  return "\n".join([clean_section(s.text) for s in sections]) 

# COMMAND ----------



# COMMAND ----------

# import io
# import re
# with requests.get('dbfs:/Volumes/sarbani_dbrx_catalog/india_fe_demo/fe_demo_pdf/India_outlook_2023.pdf') as pdf:
#   doc = extract_doc_text(pdf.content)  
#   print(doc)

# COMMAND ----------

# with open(pdf_loc, 'rb') as f:
#     pdf_bytes = f.read()
# text = extract_doc_text(pdf_bytes)

# COMMAND ----------

from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from llama_index import Document, set_global_tokenizer
from transformers import AutoTokenizer
from pyspark.sql.functions import pandas_udf
from typing import Iterator
import pandas as pd

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #set llama2 as tokenizer to match our model size (will stay below BGE 1024 limit)
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    def extract_and_split(b):
      txt = extract_doc_text(b)
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

from mlflow.deployments import get_deploy_client

# bge-large-en Foundation models are available using the /serving-endpoints/databricks-bge-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
#print(embeddings)

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS `sarbani_dbrx_catalog`.`india_fe_demo`.`pdf_demo_tbl` (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

from pyspark.sql import functions as F

(spark.readStream.table("`sarbani_dbrx_catalog`.`india_fe_demo`.`pdf_raw`")
      .withColumn("content", F.explode(read_as_chunk("content")))
      .withColumn("embedding", get_embedding("content"))
      .selectExpr('path as url', 'content', 'embedding')
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'pdf_loc/checkpoints/pdf_chunk2')
    .table("`sarbani_dbrx_catalog`.`india_fe_demo`.`pdf_demo_tbl`").awaitTermination())


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM `sarbani_dbrx_catalog`.`india_fe_demo`.`pdf_demo_tbl`

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------



# COMMAND ----------

# from unstructured.partition.pdf import partition_pdf

# # Returns a List[Element] present in the pages of the parsed pdf document
# elements = partition_pdf(pdf_loc, strategy="hi_res")

# COMMAND ----------



# COMMAND ----------

# cnt=0
# for element in elements:
#   print("#########count######### :",cnt)
#   print(element.metadata.fields)
  
#   cnt = cnt+1
# # print("count :",cnt)
# # print(elements[100])
  

# COMMAND ----------

# elements[1060].metadata.fields

# COMMAND ----------

# for x in range(960, 978):
#   #print("#########count#########")
#   print(elements[x].text)
#   print(elements[x].metadata.page_number)

# COMMAND ----------

# elements[3000].metadata.fields

# COMMAND ----------

# elements[3000].text

# COMMAND ----------

# from collections import Counter
# display(Counter(type(element) for element in elements))
# print("")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Image

# COMMAND ----------

# image_loc = "/Volumes/sarbani_dbrx_catalog/india_fe_demo/pdf_volume/screen-4.png"

# COMMAND ----------

# from unstructured.partition.image import partition_image

# # Returns a List[Element] present in the pages of the parsed pdf document
# elements = partition_image("/Volumes/sarbani_dbrx_catalog/india_fe_demo/pdf_volume/screen-4.png")


# COMMAND ----------

# cnt=0
# for element in elements:
#   print("#########count######### :",cnt)
#   print(element.text)
#   cnt = cnt+1
# # print("count :",cnt)
# # print(elements[100])
  

# COMMAND ----------

# MAGIC %md
# MAGIC images look better in terms of extraction..lets try to convert pdf to images and then create embeddings using clip

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert the pdf to images

# COMMAND ----------

# from pdf2image import convert_from_path

# pdf_images = convert_from_path(pdf_loc)



# COMMAND ----------

# MAGIC %md
# MAGIC #### store the images in volume

# COMMAND ----------

# pdfname_pgno_imgloc = {}
# for idx in range(len(pdf_images)):
#     img_save_path = '/Volumes/sarbani_dbrx_catalog/india_fe_demo/pdf_volume/' + 'pdf_page_'+ str(idx+1) +'.png'
    
   
#     pdf_images[idx].save('/Volumes/sarbani_dbrx_catalog/india_fe_demo/pdf_volume/' + 'pdf_page_'+ str(idx+1) +'.png', 'PNG')
# print("Successfully converted PDF to images")

# COMMAND ----------

# pdfname_pgno_imgloc = {}
# for idx in range(len(pdf_images)):
#     img_save_path = '/Volumes/sarbani_dbrx_catalog/india_fe_demo/pdf_volume/' + 'pdf_page_'+ str(idx+1) +'.png'
#     pdfname_pgno_imgloc[idx+1] = img_save_path
#     pdf_images[idx].save(img_save_path, 'PNG')

# COMMAND ----------

# import pandas as pd

# # Convert the dictionary to a list of tuples
# pdfname_pgno_imgloc_list = list(pdfname_pgno_imgloc.items())

# # Create the DataFrame with the appropriate columns
# df_pdf_page_img = pd.DataFrame(pdfname_pgno_imgloc_list, columns=['pdf_page', 'page_image_loc'])
# df_pdf_page_img

# COMMAND ----------

# import pyspark.pandas as ps
# df_pdf_page_img_ps = ps.from_pandas(df_pdf_page_img)
# df_pdf_page_img_ps

# COMMAND ----------

# from unstructured.partition.image import partition_image

# filename = "/Volumes/sarbani_dbrx_catalog/india_fe_demo/pdf_volume/pdf_page_18.png"
# elements = partition_image(filename=filename)

# COMMAND ----------

# cnt=0
# for element in elements:
#   print("#########count######### :",cnt)
#   print(element)
#   cnt = cnt+1

# COMMAND ----------

# from unstructured.partition.image import partition_image

# # Returns a List[Element] present in the pages of the parsed pdf document
# elements = partition_image("/Volumes/sarbani_dbrx_catalog/india_fe_demo/pdf_volume/screen-4.png")

# COMMAND ----------

# # Replace the import statement with the following one
# import pyspark.pandas as ps

# # Convert column to list
# page_image_loc_list = df_pdf_page_img_ps['page_image_loc'].tolist()

# # Apply partition_image function and assign the result back to the DataFrame column
# df_pdf_page_img_ps['text_extract'] = [partition_image(loc) for loc in page_image_loc_list]

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract text from images now

# COMMAND ----------

# cnt=0
# for element in elements:
#   print("#########count######### :",cnt)
#   print(element.text)
#   cnt = cnt+1
# # print("count :",cnt)
# # print(elements[100])
  

# COMMAND ----------

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# from langchain_community.embeddings.sentence_transformer import (
#   SentenceTransformerEmbeddings, HuggingFaceEmbeddings
# )
# from langchain_community.vectorstores import Chroma

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Index Part

# COMMAND ----------

# loader = PyPDFLoader(pdf_loc)
# pages = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200))
# len(pages)

# COMMAND ----------

# vector_db_path='/Users/sriharsha.jana@databricks.com/demo_pdf_vecdb'
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(pages, embedding_function)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read part

# COMMAND ----------

# retriever = db.as_retriever(search_kwargs={"k": 2})
# docs = retriever.get_relevant_documents("what does zero optimizer provide advantage over other techniques?")
# len(docs)

# COMMAND ----------

# print(docs[0])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Generate Humanlike/Chat/any task using DBRX

# COMMAND ----------

# from langchain.chat_models import ChatDatabricks

# chat_model = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct", max_tokens=256)
# print(chat_model.invoke('What is Apache Spark'))

# COMMAND ----------

# Test Databricks Foundation LLM model
#from langchain_community.chat_models import ChatDatabricks

import langchain_community
from langchain_community.chat_models import ChatDatabricks

dbrx_chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200)
print(f"Test chat model: {dbrx_chat_model.predict('What is mixture oif expert in LLM')}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create a QA Chain from Langchains

# COMMAND ----------

# MAGIC %md
# MAGIC https://nakamasato.medium.com/enhancing-langchains-retrievalqa-for-real-source-links-53713c7d802a

# COMMAND ----------

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm=dbrx_chat_model, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)

# COMMAND ----------

query = """What is data lake?"""
llm_response = qa_chain(query)

# COMMAND ----------

llm_response['result']

# COMMAND ----------

doc_pages = 'Document Page Numbers ='+','.join(set([str(doc.metadata['page']) for doc in llm_response['source_documents']]))
doc_name = "Document ="+','.join(set([str(doc.metadata['source']) for doc in llm_response['source_documents']]))
doc_content = "Document Details ="+'.\n'.join(set([str(doc.page_content) for doc in llm_response['source_documents']]))

# COMMAND ----------

doc_pages

# COMMAND ----------

# import gradio as gr

# def question_answer(question, image):
#     output = qa_chain(question)
#     doc_pages = ','.join(set([str(doc.metadata['page']) for doc in output['source_documents']]))
#     doc_name = ','.join(set([str(doc.metadata['source']) for doc in output['source_documents']]))
#     doc_link = "<a href='https://docs.databricks.com/en/machine-learning/mlops/mlops-workflow.html' target='_blank'>Click here to visit Example.com</a>"
#     doc_content = '.\n'.join(set([str(doc.page_content) for doc in output['source_documents']]))
    
#     return output['result'], doc_pages, doc_name, doc_link, doc_content

# gr.Interface(fn=question_answer, inputs=["text", gr.Image(value='/Volumes/sarbani_catalog/sarbani_llm_rag/volume_databricks_documentation/rag-workflow.png')], outputs=[gr.Textbox(label='Chatbot Response'),gr.Textbox(label='Document Page#'),gr.Textbox(label='Document Location'),gr.Textbox(label=('doc_link') ) ,gr.Textbox(label='Document Context')]).launch(share=True)



# COMMAND ----------

# MAGIC %md
# MAGIC ### GRADIO - WIP Sarbani

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


