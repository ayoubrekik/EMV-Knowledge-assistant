import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.db.chroma_client import get_or_create_emv_collection


def load_chunks(json_path: str | Path) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_document(item: Dict[str, Any]) -> str:
    return str(item.get("text") or "").strip()


def clean_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata", {})

    def safe_int(value):
        if value is None or value == "":
            return None
        try:
            return int(value)
        except Exception:
            return None

    # def safe_bool(value):
    #     if isinstance(value, bool):
    #         return value
    #     if value in [None, ""]:
    #         return False
    #     return bool(value)

    return {
        "doc_id": item.get("doc_id"),
        "doc_title" : item.get("doc_title"),
        "doc_version": item.get("doc_version"),
        "doc_date": item.get("doc_date"),
        "context_prefix": str(item.get("context_prefix") or ""),
        "chunk_index": str(item.get("chunk_index")) if item.get("chunk_index") is not None else None,
        "page_num": safe_int(item.get("page_start")),
    }


def iter_batches(items: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def ingest_chunks(json_path: str | Path, batch_size: int = 100):
    collection = get_or_create_emv_collection()
    chunks = load_chunks(json_path)

    print(f"Loaded {len(chunks)} items from {json_path}")
    total_inserted = 0

    for batch in iter_batches(chunks, batch_size):
        ids = []
        documents = []
        metadatas = []

        for item in batch:
            item_id = item.get("chunk_id")
            if not item_id:
                print("Skipped item: missing id")
                continue

            document = build_document(item)
            if not document:
                print(f"Skipped {item_id}: empty document")
                continue

            ids.append(str(item_id))
            documents.append(document)
            metadatas.append(clean_metadata(item))

        if not ids:
            continue

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        total_inserted += len(ids)
        print(f"Upserted {len(ids)} chunks")

    print(f"Finished ingestion. Total inserted: {total_inserted}")
    return total_inserted