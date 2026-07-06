import os
import chromadb

from langchain_chroma import Chroma

from src.core.db.chroma_client import (
    LocalSentenceTransformerEmbeddingFunction,
)

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "emv_knowledge_all_docs")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-large-en-v1.5",
)


def get_embedding_model():
    return LocalSentenceTransformerEmbeddingFunction(
        EMBEDDING_MODEL_NAME
    )


def get_langchain_chroma():
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
    )

    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
    )


def get_retriever(k: int = 5):
    return get_langchain_chroma().as_retriever(
        search_kwargs={"k": k}
    )