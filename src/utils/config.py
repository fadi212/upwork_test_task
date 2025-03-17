import os
from pinecone import Pinecone

logs_folder = "logs"
os.makedirs(logs_folder, exist_ok=True)

LOCAL_INDEX_LOGS_PATH = os.path.join(logs_folder, "index_logs.json")
LOCAL_UPSERT_LOGS_PATH = os.path.join(logs_folder, "upsert_logs.json")
LOCAL_CHUNK_OUTPUT_DIR = "chunked_docs"

ARYN_API_KEY = os.environ["ARYN_API_KEY"]
INDEX_NAME = "new-index"

os.environ["PINECONE_ENVIRONMENT"] = "us-east-1-aws"

aryn_json_output_dir = 'tmp_aryn_json_dir'

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT")
)
