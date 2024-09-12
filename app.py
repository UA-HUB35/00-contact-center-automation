import os
import json
import boto3
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
from flask import Flask, request, Response
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

import warnings
warnings.filterwarnings("ignore")

application = Flask(__name__)
CORS(application)
load_dotenv()


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
api_key = os.getenv("api_key")
email = os.getenv("email")
cohere_api_key = os.getenv("cohere_api_key")

bedrock_client = boto3.client("bedrock-runtime", aws_access_key_id = AWS_ACCESS_KEY_ID, 
                              aws_secret_access_key = AWS_SECRET_ACCESS_KEY, region_name="us-east-1")
embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v2:0")
compressor = CohereRerank(model= "rerank-english-v3.0", top_n=1, cohere_api_key=cohere_api_key)



@application.route("/", methods=["GET"])
@cross_origin()
def home():
    return {"health":200}

@application.route("/history-db", methods=["POST"])
@cross_origin()
def history():
    try:
        data_string = request.get_data()
        data = json.loads(data_string)
        description = data.get("query")
        local_db = FAISS.load_local("History_DB", embeddings, allow_dangerous_deserialization=True)
        retriever = local_db.as_retriever(search_type="mmr", search_kwargs={"k":5})
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        compressed_docs = compression_retriever.invoke(description)[0]
        data = compressed_docs.metadata
        data["Description"] = compressed_docs.page_content
        return {"response" : data}

    except Exception as e:
        return Response(
            f"bad request! - {e} ",
            400,
        )
        
@application.route("/jira-db", methods=["POST"])
@cross_origin()
def jira():
    try:
        data_string = request.get_data()
        data = json.loads(data_string)
        description = data.get("query")
        local_db = FAISS.load_local("Jira_DB", embeddings, allow_dangerous_deserialization=True)
        retriever = local_db.as_retriever(search_type="mmr", search_kwargs={"k":5})
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        compressed_docs = compression_retriever.invoke(description)[0]
        data = compressed_docs.metadata
        data["Description"] = compressed_docs.page_content
        return {"response" : data}
       
    except Exception as e:
        return Response(
            f"bad request! - {e} ",
            400,
        )

if __name__ == "__main__":
    application.run(debug=True)
