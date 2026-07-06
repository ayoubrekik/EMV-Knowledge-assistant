import os
import time
from pathlib import Path

HF_CACHE_DIR = "/root/.cache/huggingface"

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from dotenv import load_dotenv
from chromadb.api import ClientAPI
from chromadb.api.types import Documents, Embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-large-en-v1.5"
)


def resolve_hf_snapshot(model_name: str) -> str:
    cache_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = (
        Path(HF_CACHE_DIR)
        / "hub"
        / cache_name
        / "snapshots"
    )

    if not snapshots_dir.exists():
        raise RuntimeError(
            f"Model '{model_name}' is not downloaded.\n"
            f"Expected: {snapshots_dir}"
        )

    snapshots = sorted(
        p for p in snapshots_dir.iterdir() if p.is_dir()
    )

    if not snapshots:
        raise RuntimeError(
            f"No snapshot found for {model_name}"
        )

    return str(snapshots[-1])


class LocalSentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(
            resolve_hf_snapshot(model_name),
            device="cpu",
        )

    def name(self) -> str:
        return "sentence_transformer"

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(
            input,
            normalize_embeddings=True,
            batch_size=4,
            convert_to_numpy=True,
        ).tolist()


def get_chroma_client(
    retries: int = 10,
    delay: float = 2.0,
) -> ClientAPI:

    host = os.getenv("CHROMA_HOST", "chroma")
    port = int(os.getenv("CHROMA_PORT", "8000"))

    last_error = None

    for _ in range(retries):
        try:
            client = chromadb.HttpClient(
                host=host,
                port=port,
            )
            client.heartbeat()
            return client
        except Exception as e:
            last_error = e
            time.sleep(delay)

    raise RuntimeError(last_error)


embedding_function = LocalSentenceTransformerEmbeddingFunction(
    EMBEDDING_MODEL
)


def get_or_create_emv_collection():
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=os.getenv(
            "CHROMA_COLLECTION",
            "emv_knowledge_all_docs",
        ),
        embedding_function=embedding_function,
        metadata={
            "description": "EMV knowledge base"
        },
    )