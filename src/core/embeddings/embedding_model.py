from sentence_transformers import SentenceTransformer
from typing import List

# Load model once (singleton)
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> List[float]:
    """
    Convert a single text string into an embedding vector.
    """
    return _model.encode(text, normalize_embeddings=True).tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of text strings into embedding vectors.
    """
    return _model.encode(texts, normalize_embeddings=True).tolist()