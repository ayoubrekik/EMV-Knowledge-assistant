import re
from typing import Optional
from langchain_core.documents import Document
from .types import ScoredDoc


def normalize_text_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _first_present(*values):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def extract_context_prefix(text: str, metadata: Optional[dict] = None) -> str:
    metadata = metadata or {}
    prefix = str(metadata.get("context_prefix") or "").strip()
    if prefix:
        return prefix

    if not text:
        return ""

    head = text.split("\n\n", 1)[0].strip()
    if head.startswith("[Document:") or "[Section:" in head or "[Page:" in head:
        return head
    return ""


def parse_context_prefix(prefix: str) -> dict:
    def extract(label: str) -> Optional[str]:
        pattern = rf"\[{re.escape(label)}:\s*(.*?)\]"
        match = re.search(pattern, prefix or "", flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    part_match = re.search(r"\[(Part\s+\d+\s*/\s*\d+)\]", prefix or "", flags=re.IGNORECASE)

    return {
        "document_id": extract("Document"),
        "section": extract("Section"),
        "page": extract("Page"),
        "content_type": extract("Type"),
        "part": part_match.group(1).strip() if part_match else None,
    }


def source_from_doc(doc: Document) -> dict:
    metadata = dict(doc.metadata or {})
    text = doc.page_content or ""
    prefix = extract_context_prefix(text, metadata)
    parsed = parse_context_prefix(prefix)

    doc_id = _first_present(metadata.get("doc_id"), parsed.get("document_id"), "Unknown document")
    doc_title = _first_present(metadata.get("doc_title"), metadata.get("document_title"), doc_id, "Unknown title")
    page = _first_present(metadata.get("page_num"), metadata.get("page_start"), parsed.get("page"), "Unknown page")

    return {
        "doc_id": str(doc_id),
        "doc_title": str(doc_title),
        "doc_version": metadata.get("doc_version"),
        "doc_date": metadata.get("doc_date"),
        "document_id": parsed.get("document_id"),
        "section": parsed.get("section") or "Unknown section",
        "page": str(page),
        "content_type": parsed.get("content_type") or metadata.get("content_type") or "Unknown",
        "part": parsed.get("part"),
        "context_prefix": prefix,
        "chunk_id": metadata.get("chunk_id") or metadata.get("chroma_id"),
        "chroma_id": metadata.get("chroma_id"),
        "retrieval_method": metadata.get("retrieval_method"),
        "bm25_score": metadata.get("bm25_score"),
        "cross_encoder_score": metadata.get("cross_encoder_score"),
    }


def citation_from_source(source: dict) -> str:
    return (
        f"[Source: {source.get('doc_id', 'Unknown document')} | "
        f"{source.get('section', 'Unknown section')} | "
        f"{source.get('doc_title', 'Unknown title')} | "
        f"page {source.get('page', 'Unknown page')}]"
    )


def document_key(doc: Document):
    metadata = doc.metadata or {}
    source = source_from_doc(doc)

    chroma_id = metadata.get("chroma_id")
    if chroma_id:
        return ("chroma_id", chroma_id)

    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return ("chunk_id", chunk_id)

    return (
        "fallback",
        source.get("doc_id"),
        source.get("section"),
        source.get("page"),
        source.get("content_type"),
        normalize_text_for_dedup(doc.page_content or "")[:500],
    )


def deduplicate_scored_docs(scored_docs: list[ScoredDoc]) -> list[ScoredDoc]:
    deduped = []
    seen = set()
    for doc, distance in scored_docs:
        key = document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((doc, distance))
    return deduped



def build_sources_and_chunks(scored_docs: list[ScoredDoc]):
    sources = []
    retrieved_chunks = []
    seen = set()

    for rank, (doc, distance) in enumerate(scored_docs, start=1):
        source = source_from_doc(doc)
        key = (
            source.get("doc_id"),
            source.get("doc_title"),
            source.get("section"),
            source.get("page"),
            source.get("content_type"),
        )

        if key not in seen:
            sources.append({
                "doc_id": source.get("doc_id"),
                "doc_title": source.get("doc_title"),
                "doc_version": source.get("doc_version"),
                "doc_date": source.get("doc_date"),
                "section": source.get("section"),
                "page": source.get("page"),
                "content_type": source.get("content_type"),
                "citation": citation_from_source(source),
            })
            seen.add(key)

        retrieved_chunks.append({
            "rank": rank,
            "distance": distance,
            "cross_encoder_score": source.get("cross_encoder_score"),
            "retrieval_method": source.get("retrieval_method"),
            "bm25_score": source.get("bm25_score"),
            "text_preview": (doc.page_content or "")[:700],
            "source": source,
            "metadata": doc.metadata or {},
        })

    return sources, retrieved_chunks
