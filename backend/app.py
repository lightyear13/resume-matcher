import os
import time
import logging
import csv
import openai
from io import BytesIO, StringIO
import pandas as pd

from quart import Quart, request, jsonify, Blueprint, current_app
from azure.storage.blob import BlobServiceClient

bp = Blueprint("routes", __name__, static_folder="static")

@bp.route("/", defaults={"path": "index.html"})
@bp.route("/<path:path>")
async def static_file(path):
    return await bp.send_static_file(path)

import os
from openai import AzureOpenAI
    
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
conn_str = os.getenv("BLOB_CONN_STRING")
container_name = os.getenv("BLOB_CONTAINER")
blob_service_client = BlobServiceClient.from_connection_string(conn_str=conn_str)

blob_client = blob_service_client.get_blob_client(container=container_name, blob="job_dataset.csv")
blob_data = blob_client.download_blob()
blob_string = blob_data.content_as_text()
jobs = []

csv_reader = csv.DictReader(StringIO(blob_string))

for row in csv_reader:
    jobs.append(row)


def load_jobs_from_csv(file_path):
    jobs = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            jobs.append(row)
    return jobs

# Example usage
file_path = 'job_dataset.csv'
#jobs = load_jobs_from_csv(file_path)

@bp.route("/embedQuery", methods=["POST"])
async def embed_query():
    try:
        request_json = await request.get_json()
        query = request_json["query"]
        response = await openai.Embedding.acreate(
            input=query, engine="text-embedding-3-small"
        )
        return response["data"][0]["embedding"], 200
    except Exception as e:
        logging.exception("Exception in /embedQuery")
        return jsonify({"error": str(e)}), 500
    

def get_embedding(text, model="team333openaideploy"): 
    return client.embeddings.create(input = [text], model=model).data[0].embedding

@bp.route("/job-details", methods=["GET"])
async def getJobSearchResults():
    job_id = request.args.get('jobId')
    
    # Find the job with the given jobId
    job = next((job for job in jobs if str(job["job_id"]) == job_id), None)
    
    job["job_emd"]=get_embedding(job["job_description"])
    
    
    if job:
        return jsonify(job)
    else:
        return jsonify({"error": "Job not found"}), 404
    


def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    return app
