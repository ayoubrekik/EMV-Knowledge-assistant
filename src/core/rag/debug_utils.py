import json
import os
from pathlib import Path
from .source_utils import source_from_doc
from .types import ScoredDoc


def save_debug_chunks_to_json(filename: str, scored_docs: list[ScoredDoc], query: str):
    debug_data = []

    for rank, (doc, distance) in enumerate(scored_docs, start=1):
        metadata = dict(doc.metadata or {})
        text = doc.page_content or ""
        source = source_from_doc(doc)

        debug_data.append({
            "rank": rank,
            "query": query,
            "distance": float(distance),
            "retrieval_method": metadata.get("retrieval_method"),
            "bm25_score": metadata.get("bm25_score"),
            "cross_encoder_score": metadata.get("cross_encoder_score"),
            "source": source,
            "metadata": {
                "chroma_id": metadata.get("chroma_id"),
                "chunk_id": metadata.get("chunk_id") or metadata.get("chroma_id"),
                "doc_id": metadata.get("doc_id"),
                "doc_title": metadata.get("doc_title"),
                "doc_version": metadata.get("doc_version"),
                "doc_date": metadata.get("doc_date"),
                "context_prefix": metadata.get("context_prefix"),
                "page_num": metadata.get("page_num"),
            },
            "text_preview": text[:2000],
        })

    debug_dir = Path(os.getenv("RAG_DEBUG_DIR", "/app/src"))
    debug_dir.mkdir(parents=True, exist_ok=True)
    file_path = debug_dir / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)


def save_final_context_to_txt(filename: str, context: str, query: str, input_type: str):
    debug_dir = Path(os.getenv("RAG_DEBUG_DIR", "/app/src"))
    debug_dir.mkdir(parents=True, exist_ok=True)
    file_path = debug_dir / filename

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"QUERY:\n{query}\n\n")
        f.write(f"INPUT TYPE:\n{input_type}\n\n")
        f.write("=" * 100 + "\n")
        f.write("FINAL CONTEXT SENT TO LLM\n")
        f.write("=" * 100 + "\n\n")
        f.write(context)
