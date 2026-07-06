import re
import hashlib
from typing import Optional, Tuple
import os
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .debug_utils import save_debug_chunks_to_json
from .source_utils import deduplicate_scored_docs
from .types import ScoredDoc

from src.core.db.chroma_client import embedding_function, resolve_hf_snapshot

cross_encoder_reranker = None
_bm25_index: Optional[BM25Okapi] = None
_bm25_docs: Optional[list[Document]] = None


# Keep section numbers like "6.3.7" intact
SECTION_REF_PATTERN = re.compile(r"\b\d+(?:\.\d+){1,4}\b", re.IGNORECASE)
PART_SECTION_REF_PATTERN = re.compile(
    r"\bpart\s*[ivxlcdm]+\s*\.?\s*\d+(?:\.\d+){1,4}\b",
    re.IGNORECASE,
)

def _normalize_section_ref_text(text: str) -> str:
    """
    Normalize section references so these become comparable:

        PartIII.10.1
        Part III 10.1
        part iii.10.1

    into:

        partiii.10.1
    """

    text = (text or "").lower()

    # Part III 10.1 -> partiii.10.1
    text = re.sub(
        r"\bpart\s+([ivxlcdm]+)\s*\.?\s*(\d+(?:\.\d+){1,4})",
        r"part\1.\2",
        text,
        flags=re.IGNORECASE,
    )

    # PartIII 10.1 -> partiii.10.1
    text = re.sub(
        r"\bpart([ivxlcdm]+)\s+(\d+(?:\.\d+){1,4})",
        r"part\1.\2",
        text,
        flags=re.IGNORECASE,
    )

    # PartIII.10.1 -> partiii.10.1
    text = re.sub(
        r"\bpart\s*([ivxlcdm]+)\s*\.\s*(\d+(?:\.\d+){1,4})",
        r"part\1.\2",
        text,
        flags=re.IGNORECASE,
    )

    return text


def extract_section_refs(query: str) -> list[str]:
    """
    Extract explicit section references from the user query.

    Example:
        "what is Part III 10.1?"
        -> ["partiii.10.1", "10.1"]
    """

    normalized_query = _normalize_section_ref_text(query)

    part_refs = re.findall(
        r"\bpart[ivxlcdm]+\.\d+(?:\.\d+){1,4}\b",
        normalized_query,
        flags=re.IGNORECASE,
    )

    numeric_refs = SECTION_REF_PATTERN.findall(normalized_query)

    refs = []

    for ref in part_refs + numeric_refs:
        ref = ref.lower()
        if ref not in refs:
            refs.append(ref)

    return refs

def _tokenize(text: str) -> list[str]:
    text = _normalize_section_ref_text(text).lower()

    part_section_refs = re.findall(
        r"\bpart[ivxlcdm]+\.\d+(?:\.\d+){1,4}\b",
        text,
        flags=re.IGNORECASE,
    )

    numeric_section_refs = SECTION_REF_PATTERN.findall(text)

    other_tokens = re.findall(r"[0-9a-f]{2,8}|[a-z]{2,}", text)

    return part_section_refs + numeric_section_refs + other_tokens


def _get_doc_key(doc: Document) -> str:
    """
    New metadata does not contain chunk_id.
    So we create a stable key using:
    - chroma_id if available
    - otherwise doc_id + page_num + context_prefix + content hash
    """

    metadata = doc.metadata or {}

    if metadata.get("chroma_id"):
        return str(metadata["chroma_id"])

    if metadata.get("chunk_id"):
        return str(metadata["chunk_id"])

    raw_key = "|".join(
        [
            str(metadata.get("doc_id", "")),
            str(metadata.get("doc_version", "")),
            str(metadata.get("doc_date", "")),
            str(metadata.get("page_num", "")),
            str(metadata.get("context_prefix", "")),
            doc.page_content or "",
        ]
    )

    return hashlib.md5(raw_key.encode("utf-8")).hexdigest()


def _normalize_metadata(metadata: Optional[dict], chroma_id: Optional[str] = None) -> dict:
    """
    Normalize metadata for the new structure:

    persisted metadata:
        doc_title
        doc_id
        context_prefix
        doc_version
        page_num
        doc_date

    runtime metadata added here:
        chroma_id
        chunk_id
        retrieval_method
        scores
    """

    metadata = dict(metadata or {})

    if chroma_id:
        metadata["chroma_id"] = chroma_id

    # Keep chunk_id as a runtime alias for compatibility with your debug/dedup code.
    # It is not required to exist in Chroma metadata.
    metadata["chunk_id"] = metadata.get("chunk_id") or metadata.get("chroma_id")

    return metadata


def _doc_text_for_search(doc: Document) -> str:
    """
    Search/rerank using both context_prefix and content.

    This is important because your new metadata stores structural context in:
        context_prefix

    Example:
        Book 3 > Section 6.3.7 > ...
    """

    metadata = doc.metadata or {}
    context_prefix = metadata.get("context_prefix") or ""
    content = doc.page_content or ""

    if context_prefix:
        return f"{context_prefix}\n\n{content}"

    return content


def get_cross_encoder_reranker():
    global cross_encoder_reranker

    reranker_model = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    if cross_encoder_reranker is None:
        cross_encoder_reranker = CrossEncoder(
            resolve_hf_snapshot(reranker_model),
            device="cpu",
        )

    return cross_encoder_reranker


def cross_encoder_rerank(
    scored_docs: list[ScoredDoc],
    query: str,
    final_k: int,
) -> list[ScoredDoc]:
    if not scored_docs:
        return []

    reranker = get_cross_encoder_reranker()

    pairs = [
        [query, _doc_text_for_search(doc)]
        for doc, _ in scored_docs
    ]

    scores = reranker.predict(pairs)

    ranked = []

    for (doc, distance), score in zip(scored_docs, scores):
        metadata = dict(doc.metadata or {})
        metadata["cross_encoder_score"] = float(score)
        doc.metadata = metadata

        ranked.append((doc, distance, float(score)))

    ranked.sort(key=lambda item: item[2], reverse=True)

    return [(doc, distance) for doc, distance, _ in ranked[:final_k]]


def get_all_chroma_documents(vectorstore) -> list[Document]:
    collection = getattr(vectorstore, "_collection", None)

    if collection is None:
        return []

    results = collection.get(include=["documents", "metadatas"])

    texts = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    ids = results.get("ids") or []

    docs = []

    for i, text in enumerate(texts):
        chroma_id = ids[i] if i < len(ids) else None
        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}

        metadata = _normalize_metadata(metadata, chroma_id=chroma_id)

        docs.append(
            Document(
                page_content=text or "",
                metadata=metadata,
            )
        )

    return docs


def build_bm25_index(vectorstore) -> Tuple[BM25Okapi, list[Document]]:
    global _bm25_index, _bm25_docs

    if _bm25_index is None:
        _bm25_docs = get_all_chroma_documents(vectorstore)

        tokenized = [
            _tokenize(_doc_text_for_search(doc))
            for doc in _bm25_docs
        ]

        _bm25_index = BM25Okapi(tokenized)

    return _bm25_index, _bm25_docs or []


def reset_bm25_index():
    """
    Call this after re-indexing or uploading new documents.
    Otherwise BM25 may keep old cached documents.
    """

    global _bm25_index, _bm25_docs

    _bm25_index = None
    _bm25_docs = None


def bm25_search(
    vectorstore,
    query: str,
    k: int = 20,
) -> list[ScoredDoc]:
    index, docs = build_bm25_index(vectorstore)

    query_tokens = _tokenize(query)

    if not query_tokens or not docs:
        return []

    scores = index.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:k]

    results = []

    for idx in top_indices:
        if scores[idx] <= 0:
            continue

        doc = docs[idx]

        metadata = dict(doc.metadata or {})
        metadata["retrieval_method"] = "bm25"
        metadata["bm25_score"] = float(scores[idx])

        results.append(
            (
                Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                ),
                float(scores[idx]),
            )
        )

    return results


def semantic_search(
    vectorstore,
    query: str,
    k: int = 40,
    where: Optional[dict] = None,
) -> list[ScoredDoc]:
    collection = getattr(vectorstore, "_collection", None)

    if collection is None:
        return []

    query_embedding = embedding_function([query])[0]

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }

    # You can still filter using your new metadata:
    # example:
    # where={"doc_id": "Book_3"}
    # where={"doc_title": "Book 3"}
    # where={"doc_version": "4.4"}
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    output = []

    for text, metadata, distance, chroma_id in zip(docs, metadatas, distances, ids):
        metadata = _normalize_metadata(metadata, chroma_id=chroma_id)
        metadata["retrieval_method"] = "semantic"

        output.append(
            (
                Document(
                    page_content=text or "",
                    metadata=metadata,
                ),
                float(distance),
            )
        )

    return output

def exact_section_lookup(vectorstore, section_refs: list[str]) -> list[ScoredDoc]:
    """
    New metadata has no section_numbers field.

    Exact section search is now done inside context_prefix.

    Example context_prefix:
        [Document: EMV_v4.4_Book_1_ICC_to_Terminal_Interface]
        [Section: PartIII.10.1 Files, Commands, and Application Selection > Files > File Structure]
        [Page: 36]
        [Type: text]
    """

    if not section_refs:
        return []

    docs = get_all_chroma_documents(vectorstore)

    if not docs:
        return []

    output = []
    seen_keys = set()

    normalized_refs = [
        _normalize_section_ref_text(ref).lower()
        for ref in section_refs
    ]

    for doc in docs:
        metadata = dict(doc.metadata or {})

        context_prefix = str(metadata.get("context_prefix") or "")
        normalized_context = _normalize_section_ref_text(context_prefix).lower()

        matched = False

        for ref in normalized_refs:
            if ref.startswith("part"):
                # Match full PartIII.10.1 format
                pattern = re.compile(
                    rf"(?<![a-z0-9]){re.escape(ref)}(?![a-z0-9.])",
                    re.IGNORECASE,
                )
            else:
                # Match numeric section like 10.1 or 6.3.7
                # Avoid matching 10.1 inside 10.10 or 10.1.2
                pattern = re.compile(
                    rf"(?<!\d){re.escape(ref)}(?![\d.])",
                    re.IGNORECASE,
                )

            if pattern.search(normalized_context):
                matched = True
                break

        if not matched:
            continue

        metadata["retrieval_method"] = "exact_context_prefix_match"

        new_doc = Document(
            page_content=doc.page_content,
            metadata=metadata,
        )

        key = _get_doc_key(new_doc)

        if key in seen_keys:
            continue

        seen_keys.add(key)

        output.append((new_doc, 0.0))

    return output


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredDoc]],
    k: int = 60,
) -> list[ScoredDoc]:
    fused_scores: dict[str, float] = {}
    doc_lookup: dict[str, Document] = {}

    for result_list in result_lists:
        for rank, (doc, _) in enumerate(result_list, start=1):
            key = _get_doc_key(doc)

            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_lookup[key] = doc

    ranked_keys = sorted(
        fused_scores,
        key=lambda key_: fused_scores[key_],
        reverse=True,
    )

    # Distance placeholder:
    # negative fused score because downstream may treat lower as better
    return [
        (doc_lookup[key], -fused_scores[key])
        for key in ranked_keys
    ]


def retrieve_documents(
    vectorstore,
    query: str,
    final_k: int = 15,
    semantic_k: int = 40,
    bm25_k: int = 20,
    where: Optional[dict] = None,
) -> list[ScoredDoc]:
    """
    Hybrid retrieval:
    1. Detect explicit section refs like "6.3.7"
    2. Search exact matches in context_prefix
    3. Run semantic search
    4. Run BM25 search
    5. Merge using RRF
    6. Rerank using CrossEncoder

    New metadata supported:
        doc_title
        doc_id
        context_prefix
        doc_version
        page_num
        doc_date
    """

    section_refs = extract_section_refs(query)

    exact_docs = exact_section_lookup(
        vectorstore=vectorstore,
        section_refs=section_refs,
    ) if section_refs else []

    semantic_docs = semantic_search(
        vectorstore=vectorstore,
        query=query,
        k=semantic_k,
        where=where,
    )

    bm25_docs = bm25_search(
        vectorstore=vectorstore,
        query=query,
        k=bm25_k,
    )

    fused_docs = reciprocal_rank_fusion(
        [
            bm25_docs,
            semantic_docs,
        ]
    )

    exact_keys = {
        _get_doc_key(doc)
        for doc, _ in exact_docs
    }

    combined = exact_docs + [
        scored_doc
        for scored_doc in fused_docs
        if _get_doc_key(scored_doc[0]) not in exact_keys
    ]

    combined = deduplicate_scored_docs(combined)

    save_debug_chunks_to_json(
        "before_reranking.json",
        combined,
        query,
    )

    reranked_docs = cross_encoder_rerank(
        combined,
        query,
        final_k=final_k,
    )

    reranked_docs = deduplicate_scored_docs(reranked_docs)

    save_debug_chunks_to_json(
        "after_reranking.json",
        reranked_docs,
        query,
    )

    return reranked_docs