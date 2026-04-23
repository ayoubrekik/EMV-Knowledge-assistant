import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.db.chroma_client import get_or_create_emv_collection


def load_chunks(json_path: str | Path) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_document(item: Dict[str, Any]) -> str:
    return str(item.get("document") or "").strip()


def clean_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata", {})

    def safe_int(value):
        if value is None or value == "":
            return None
        try:
            return int(value)
        except Exception:
            return None

    def safe_bool(value):
        if isinstance(value, bool):
            return value
        if value in [None, ""]:
            return False
        return bool(value)

    return {
        "doc_id": metadata.get("doc_id"),
        "doc_version": metadata.get("doc_version"),
        "doc_date": metadata.get("doc_date"),
        "section_id": metadata.get("section_id"),
        "parent_section_id": metadata.get("parent_section_id"),
        "title": metadata.get("title"),
        "section_number": metadata.get("section_number"),
        "level": safe_int(metadata.get("level")),
        "parent_titles": str(metadata.get("parent_titles") or ""),
        "chunk_index": str(metadata.get("chunk_index")) if metadata.get("chunk_index") is not None else None,
        "page_num": safe_int(metadata.get("page_num")),
        "type": metadata.get("type"),
        "table_id": metadata.get("table_id"),
        "table_title": metadata.get("table_title"),
        "token_count": safe_int(metadata.get("token_count")),
        "candidate_source": metadata.get("candidate_source"),
        "chunking_reason": metadata.get("chunking_reason"),
        "size_band": metadata.get("size_band"),
        "is_split": safe_bool(metadata.get("is_split")),
        "split_group_id": metadata.get("split_group_id"),
        "merge_group_size": safe_int(metadata.get("merge_group_size")) or 1,
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
            item_id = item.get("id")
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