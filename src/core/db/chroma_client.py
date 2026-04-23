import os
import time
import chromadb
from dotenv import load_dotenv
from chromadb.api import ClientAPI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()

def get_chroma_client(retries: int = 10, delay: float = 2.0) -> ClientAPI:
    host = os.getenv("CHROMA_HOST", "chroma")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    last_error = None

    for _ in range(retries):
        try:
            client = chromadb.HttpClient(host=host, port=port)
            client.heartbeat()
            return client
        except Exception as e:
            last_error = e
            time.sleep(delay)

    raise RuntimeError(f"Failed to connect to Chroma: {last_error}")


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2"  # fallback
)   

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

def get_or_create_emv_collection():
    client = get_chroma_client()
    # client.delete_collection("emv_knowledge")
    collection_name = os.getenv("CHROMA_COLLECTION", "emv_knowledge")
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"description": "EMV knowledge base"}
    )
