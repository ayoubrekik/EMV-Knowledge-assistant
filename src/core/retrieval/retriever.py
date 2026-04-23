from typing import Any, Dict, List, Optional

from src.core.db.chroma_client import get_or_create_emv_collection


def retrieve_chunks(
    query: str,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top matching chunks from ChromaDB using the collection's
    configured embedding function.
    """

    collection = get_or_create_emv_collection()

    query_kwargs: Dict[str, Any] = {
        "query_texts": [query],
        "n_results": n_results,
    }

    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved_chunks: List[Dict[str, Any]] = []

    for chunk_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        retrieved_chunks.append(
            {
                "id": chunk_id,
                "text": document,
                "metadata": metadata or {},
                "distance": distance,
            }
        )

    return retrieved_chunks