import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple
from dotenv import load_dotenv
import os
import hashlib

from datetime import datetime

from src.core.db.models import ChatSession, ChatMessage

from uuid import UUID, uuid4
from src.core.db.models import RagMetadata, RagSource


from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from rank_bm25 import BM25Okapi
import numpy as np

from sentence_transformers import CrossEncoder

from src.core.rag.chat_title import generate_title
from src.core.rag.langchain_chroma import get_langchain_chroma
from src.core.rag.langchain_llm import get_llm
from src.core.rag.hex_preprocessor import inject_hex_breakdown, get_hex_breakdown
from src.core.rag.emv_registry_decoder import decode_from_registry
load_dotenv()
cross_encoder_reranker = None
_bm25_index: Optional[BM25Okapi] = None
_bm25_docs: Optional[List[Document]] = None

chat_histories: Dict[str, List] = {}


InputType = Literal[
    "emv_question",
    "hex_decode_question",
    "contextual_follow_up",
    "source_lookup",
    "noise",
]


ROUTER_PROMPT = """
You are an input router for an EMV specification RAG chatbot.

Return exactly one label:

- emv_question:
  Any question that could reasonably be answered from EMV specification books or payment-terminal documentation.
  This includes questions about payment flow, terminal behavior, card interaction,
  displayed messages, cardholder/attendant actions, commands, data elements,
  tags, security, authentication, risk management, transaction processing,
  application selection, or acceptance behavior.
  If the question is technical and related to electronic payment, classify it as emv_question.
 
 -hex_decode_question:
  The user asks to decode, interpret, parse, explain, or break down a hexadecimal value
  according to an EMV byte/bit table.
  Examples:
  "Decode AIP 4080"
  "Decode TVR 8000080000"
  "Decode CSU 80000000"
  "Interpret 95 value 8000080000"
  "What does 4080 mean according to AIP?"

- contextual_follow_up:
  A follow-up that depends on a previous EMV answer, such as:
  "explain more", "I did not understand", "why", "how", "clarify that".
  Only use this if the previous conversation was EMV-related.

- source_lookup:
  The user asks where something is found, asks for a source, page, section,
  table, reference, or citation in the EMV books.

- noise:
  Greetings, thanks, small talk, empty input, random text, frontend commands,
  project implementation questions, or questions clearly unrelated to EMV/payment specifications.

Rules:
- Return only one label.
- Do not answer the question.
- Do not rewrite the question.
- If unsure between emv_question and noise, choose emv_question.
- If the user asks a factual/technical question and it could belong to payment systems, choose emv_question.
- If the question contains a hexadecimal value and asks to decode or interpret it using EMV byte/bit meaning, choose hex_decode_question.
- hex_decode_question has priority over emv_question.
- Only choose noise when it is clearly unrelated to EMV/payment specifications.
- Only choose contextual_follow_up when the previous relevant assistant answer was EMV-related.
- If the previous assistant answer was a refusal or unrelated-topic response, and the latest input says “I did not understand”, “explain”, or “clarify”, classify it as noise, not contextual_follow_up.


Chat history:
{chat_history}

Latest input:
{question}

Category:
"""


REWRITE_PROMPT = """
You are a query rewriting assistant for an EMV technical RAG chatbot.

Rewrite the user's latest follow-up into a standalone EMV-related question.

Rules:
- Use chat history only to resolve references like "it", "this", "that", "more", "explain more".
- Preserve exact technical identifiers, tags, command names, section numbers, and acronyms.
- Do not answer the question.
- Do not add information that is not implied by the conversation.
- Do not invent a new topic.
- Return only the rewritten standalone question.

Chat history:
{chat_history}

Latest follow-up:
{question}

Standalone question:
"""

HEX_RAG_PROMPT = """
You are an EMV hexadecimal decoding formatter.

The hexadecimal decoding has ALREADY been performed by the system.

The section:
MATCHED ACTIVE BIT DEFINITIONS

contains the FINAL authoritative decoded meanings.

You must NEVER:
- inspect other tables
- search for additional meanings
- infer meanings
- reinterpret bytes
- decode bits yourself
- use inactive bytes
- use any definition not explicitly listed in
  MATCHED ACTIVE BIT DEFINITIONS

You must ONLY reformat the provided decoded definitions.
If MATCHED ACTIVE BIT DEFINITIONS contains RFU or Reserved for future use,
display it exactly as provided.

Otherwise NEVER mention RFU, reserved bits,
unused bits, or unsupported bits.
If a byte is not listed in MATCHED ACTIVE BIT,
it must NOT appear in the decoded meaning section.

IMPORTANT:
Byte breakdown may contain bytes with value 00.
These bytes are inactive and MUST NOT be interpreted.

--- FORMATTING RULES ---

1. The answer must be clean, structured, and highly readable.

2. Use:
- section titles
- bullet points
- spacing between sections
- aligned formatting when possible

3. The "Decoded meaning" section must clearly separate each decoded bit.

4. The "Overall summary" must be concise and technical.

5. Do not generate long paragraphs.

6. Every decoded meaning MUST come ONLY from
MATCHED ACTIVE BIT DEFINITIONS.

If a bit appears in MATCHED ACTIVE BIT DEFINITIONS,
it MUST appear in the final answer exactly as provided.

The answer MUST be valid Markdown.
Use blank lines between sections.
Use bullet lists exactly as shown.
Do not output compact paragraphs.
Each bullet point must appear on its own line.
--- OUTPUT FORMAT ---

Decoding <VALUE> (<N> bytes)

Byte breakdown:

- Byte 1: <HEX> = <BINARY> → <ACTIVE_BITS>
- Byte 2: <HEX> = <BINARY> → none
- Byte 3: <HEX> = <BINARY> → <ACTIVE_BITS>

Decoded meaning:

- Byte <N> <bX> → <meaning>
- Byte <N> <bY> → <meaning>

Overall summary:

<very short summary strictly based on matched definitions>

Citations:

<copy the provided sources>
Context:
{context}

Question:
{question}

Answer:
"""

EMV_RAG_PROMPT = """
You are an EMV specification assistant. Answer the question using ONLY the
retrieved context below.

--- CONTEXT INTERPRETATION RULES ---
1. Some sources contain EMV data tables. Tables may appear as Markdown
   (| col | col |) or as semicolon-separated rows. Each row may describe one
   data element, tag, command, parameter, condition, or specification rule.
2. If a tag, concept, command, parameter, or technical term appears anywhere
   in a row, paragraph, list, or fragmented text, treat it as valid evidence.
3. Information may be distributed across multiple sources. Combine related
   evidence into one coherent answer when appropriate.
4. Say "I could not find this in the retrieved EMV sources." if the
   requested information does not appear anywhere in the context.
5. Don't add any technica term from your knowledge, just explain the retrieved context.

--- ANSWERING RULES ---
5. Lead with the direct answer first.
6. Use concise technical wording based strictly on the retrieved sources.
7. If the context contains tables, interpret them carefully and extract
   relevant rows, even when formatting is imperfect.
8. If the context contains normal paragraphs or procedural text, summarize
   them clearly while preserving the technical meaning.
9. Do not invent facts, assumptions, or missing values.
10. Do not mention chunk IDs, distances, vector scores, embeddings, or
    retrieval internals.
11. If a source appears truncated or partially malformed, use the visible
    information and explicitly mention when some parts appear incomplete.
12. Always end with one or more citations in EXACTLY this format:
    [Source: <doc_id> | <section_number> | <title> | page <page_num>]
13. If multiple sources contributed, include multiple citations.
14. Don't use your external knowledge in guessing missing information 

The answer MUST be valid Markdown.
Use blank lines between sections.
Use bullet lists exactly as shown.
Do not output compact paragraphs.
Each bullet point must appear on its own line.
Context:
{context}

Question:
{question}

Answer:
"""

# ─────────────────────────────────────────────────────────────────────────────
# 3. SOURCE LOOKUP PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_LOOKUP_PROMPT = """
You are an EMV specification assistant. The user is asking where specific
information is located in the EMV books.

--- RULES ---
1. Use ONLY the retrieved context to answer.
2. Provide the exact location details available in the source metadata:
   book (doc_id), section number, section title, and page number.
3. If a table is the primary source, also include the table title and table ID.
4. If the same information appears in multiple sources, list all of them.
5. Do not summarize or explain the content unless the user also asked for it.
6. If the location cannot be determined from the context, say:
   "The exact location could not be determined from the retrieved sources."
7. Always end with one or more citations in EXACTLY this format:
   [Source: <doc_id> | <section_number> | <title> | page <page_num>]

--- OUTPUT FORMAT ---
The answer MUST be valid Markdown.
Use blank lines between sections.
Use bullet lists exactly as shown.
Do not output compact paragraphs.
Each bullet point must appear on its own line.
Answer in this structure:
- Book: <doc_id>
- Section: <section_number> — <title>
- Page: <page_num>
- Table (if applicable): <table_title>

Context:
{context}

Question:
{question}

Answer:
"""

FOLLOWUP_PROMPT = """
You are an EMV specification assistant continuing a technical conversation.
The user is asking a follow-up question based on a previous answer.

--- RULES ---
1. Use the retrieved context AND the previous answer (visible in chat history)
   to respond.
2. Do not repeat information the user already received unless clarification
   requires it.
3. If the follow-up asks to elaborate on a specific point, focus only on that
   point — do not restate the entire previous answer.
4. If the follow-up cannot be answered from the context, say:
   "I could not find further detail on this in the retrieved EMV sources."
5. Use concise technical wording. Do not invent facts.
6. Do not mention chunk IDs, distances, vector scores, or retrieval internals.
7. Always end with one or more citations in EXACTLY this format:
   [Source: <doc_id> | <section_number> | <title> | page <page_num>]
   If no new sources were used, repeat the citation from the previous answer.
The answer MUST be valid Markdown.
Use blank lines between sections.
Use bullet lists exactly as shown.
Do not output compact paragraphs.
Each bullet point must appear on its own line.
Context:
{context}

Question:
{question}

Answer:
"""
ScoredDoc = Tuple[Document, float]

def get_cross_encoder_reranker():
    global cross_encoder_reranker

    if cross_encoder_reranker is None:
        cross_encoder_reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return cross_encoder_reranker


def get_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]


def format_history_for_prompt(history: List, max_messages: int = 6) -> str:
    if not history:
        return "No previous conversation."

    recent_history = history[-max_messages:]
    lines = []

    for message in recent_history:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"Assistant: {message.content}")

    return "\n".join(lines)


def normalize_label(label: str) -> InputType:
    raw = (label or "").strip().lower()

    allowed = list(InputType.__args__)  # type: ignore[attr-defined]

    if raw in allowed:
        return raw  # type: ignore[return-value]

    for item in allowed:
        if item in raw:
            return item  # type: ignore[return-value]

    return "noise"


def classify_input(llm, question: str, history: List) -> InputType:
    if not question or not question.strip():
        return "noise"

    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    messages = prompt.invoke({
        "chat_history": format_history_for_prompt(history),
        "question": question,
    })

    result = llm.invoke(messages)
    return normalize_label(result.content)

def rewrite_question(llm, question: str, history: List) -> str:
    prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
    messages = prompt.invoke({
        "chat_history": format_history_for_prompt(history),
        "question": question,
    })
    rewritten = llm.invoke(messages)
    return rewritten.content.strip()

def normalize_text_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def document_key(doc: Document):
    metadata = doc.metadata or {}

    chroma_id = metadata.get("chroma_id")
    if chroma_id:
        return ("chroma_id", chroma_id)

    return (
        "fallback",
        metadata.get("doc_id"),
        metadata.get("section_number"),
        metadata.get("page_num"),
        normalize_text_for_dedup(doc.page_content or "")[:500],
    )

def deduplicate_scored_docs(scored_docs: List[ScoredDoc]) -> List[ScoredDoc]:
    deduped = []
    seen = set()

    for doc, distance in scored_docs:
        key = document_key(doc)

        if key in seen:
            continue

        seen.add(key)
        deduped.append((doc, distance))

    return deduped


def merge_scored_docs(groups: Iterable[Iterable[ScoredDoc]]) -> List[ScoredDoc]:
    merged: List[ScoredDoc] = []
    seen = set()

    for group in groups:
        for doc, distance in group:
            key = document_key(doc)
            if key in seen:
                continue
            seen.add(key)
            merged.append((doc, distance))

    return merged


def cross_encoder_rerank(
    scored_docs: List[ScoredDoc],
    query: str,
    final_k: int,
) -> List[ScoredDoc]:
    """
    Rerank retrieved semantic candidates using a CrossEncoder.
    The CrossEncoder reads the query and the full candidate text together,
    then predicts a relevance score.
    """
    if not scored_docs:
        return []

    reranker = get_cross_encoder_reranker()

    pairs = [
        [query, doc.page_content or ""]
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

    return [
        (doc, distance)
        for doc, distance, _ in ranked[:final_k]
    ]


# ----------------------------------------------------------------
def save_debug_chunks_to_json(
    filename: str,
    scored_docs: List[ScoredDoc],
    query: str,
):
    debug_data = []


    for rank, (doc, distance) in enumerate(scored_docs, start=1):
        metadata = doc.metadata or {}
        text = doc.page_content or ""


        debug_data.append({
            "rank": rank,
            "distance": float(distance),
            "cross_encoder_score": metadata.get("cross_encoder_score"),
            "metadata": {
                "chroma_id": metadata.get("chroma_id"),
                "doc_id": metadata.get("doc_id"),
                "section_number": metadata.get("section_number"),
                "title": metadata.get("title"),
                "page_num": metadata.get("page_num"),
                "table_title": metadata.get("table_title"),
                "table_id": metadata.get("table_id"),
                "chunk_id": metadata.get("chunk_id"),
            },
            "text_preview": text[:2000],
        })

    debug_dir = Path("/app/src")
    debug_dir.mkdir(parents=True, exist_ok=True)

    file_path = debug_dir / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)



def save_final_context_to_txt(
    filename: str,
    context: str,
    query: str,
    input_type: str,
):
    debug_dir = Path("/app/src")
    debug_dir.mkdir(parents=True, exist_ok=True)

    file_path = debug_dir / filename

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"QUERY:\n{query}\n\n")
        f.write(f"INPUT TYPE:\n{input_type}\n\n")
        f.write("=" * 100 + "\n")
        f.write("FINAL CONTEXT SENT TO LLM\n")
        f.write("=" * 100 + "\n\n")
        f.write(context)
# ----------------------------------------------------------------


def retrieve_documents(
    vectorstore,
    query: str,
    final_k: int = 15,
    semantic_k: int = 40,
    bm25_k: int = 20,
) -> List[ScoredDoc]:
    """
    Hybrid retrieval pipeline:
    1. Chroma semantic search retrieves conceptually relevant chunks.
    2. BM25 retrieves exact lexical matches such as tags, acronyms, and rare terms.
    3. Results are merged and deduplicated.
    4. CrossEncoder reranks the combined candidate pool.
    """

    semantic_docs = semantic_search(
        vectorstore=vectorstore,
        query=query,
        k=semantic_k,
    )

    bm25_docs = bm25_search(
        vectorstore=vectorstore,
        query=query,
        k=bm25_k,
        score_threshold=0.5,
    )

    merged_docs = merge_scored_docs([
        bm25_docs,
        semantic_docs,
    ])
    merged_docs = deduplicate_scored_docs(merged_docs)
    save_debug_chunks_to_json(
        filename="before_reranking.json",
        scored_docs=merged_docs,
        query=query,
    )

    reranked_docs = cross_encoder_rerank(
        scored_docs=merged_docs,
        query=query,
        final_k=final_k,
    )
    reranked_docs = deduplicate_scored_docs(reranked_docs)

    save_debug_chunks_to_json(
        filename="after_reranking.json",
        scored_docs=reranked_docs,
        query=query,
    )

    return reranked_docs



def get_all_chroma_documents(vectorstore) -> List[Document]:
    """
    Load all stored chunks from Chroma so BM25 can index the same text corpus.
    """
    collection = getattr(vectorstore, "_collection", None)

    if collection is None:
        return []

    results = collection.get(
        include=["documents", "metadatas"]
    )

    texts = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    ids = results.get("ids") or []

    docs = []

    for i, text in enumerate(texts):
        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}

        if i < len(ids):
            metadata = {**metadata, "chroma_id": ids[i]}

        docs.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return docs


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[0-9a-f]{2,6}|[a-z]{2,}", text)


def build_bm25_index(vectorstore) -> Tuple[BM25Okapi, List[Document]]:
    global _bm25_index, _bm25_docs

    if _bm25_index is None:
        _bm25_docs = get_all_chroma_documents(vectorstore)
        tokenized = [_tokenize(doc.page_content or "") for doc in _bm25_docs]
        _bm25_index = BM25Okapi(tokenized)

    return _bm25_index, _bm25_docs or []


def bm25_search(
    vectorstore,
    query: str,
    k: int = 20,
    score_threshold: float = 0.5,
) -> List[ScoredDoc]:
    index, docs = build_bm25_index(vectorstore)

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = index.get_scores(query_tokens)

    top_score = float(scores.max())
    if top_score <= 0:
        return []

    cutoff = top_score * score_threshold
    candidate_indices = np.where(scores >= cutoff)[0]

    candidate_indices = sorted(
        candidate_indices,
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    results = []

    for idx in candidate_indices:
        doc = docs[idx]
        metadata = dict(doc.metadata or {})
        metadata["retrieval_method"] = "bm25"
        metadata["bm25_score"] = float(scores[idx])

        results.append((
            Document(page_content=doc.page_content, metadata=metadata),
            float(scores[idx])
        ))

    return results

def semantic_search(vectorstore, query: str, k: int = 40) -> List[ScoredDoc]:
    collection = getattr(vectorstore, "_collection", None)

    if collection is None:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    output = []

    for text, metadata, distance, chroma_id in zip(docs, metadatas, distances, ids):
        metadata = dict(metadata or {})
        metadata["chroma_id"] = chroma_id
        metadata["retrieval_method"] = "semantic"

        output.append((
            Document(page_content=text, metadata=metadata),
            float(distance)
        ))

    return output

# ================================================================
# CONTEXT CONSTRUCTION — rewritten
# ================================================================

def find_relevant_window(text: str, query: str, window: int = 1600) -> str:
    """
    Instead of blindly taking text[:window], locate the region of the chunk
    that is most relevant to the query and return a window centered there.

    Strategy:
    - Extract candidate tokens from the query (hex tags, words ≥ 3 chars).
    - Slide a window across the text and score each position by how many
      query tokens appear inside it.
    - Return the highest-scoring window.
    - Falls back to head truncation when nothing matches.

    This ensures that a tag like '9F12' buried mid-chunk is not lost.
    """
    if len(text) <= window:
        return text

    # Extract tokens: EMV hex tags + meaningful words
    tokens = re.findall(r"[0-9A-Fa-f]{2,6}|[A-Za-z]{3,}", query)
    tokens = list({t.upper() for t in tokens if t.strip()})

    if not tokens:
        return text[:window]

    text_upper = text.upper()
    best_pos = 0
    best_score = -1

    # Coarse scan: step by 1/8 of window for speed
    step = max(1, window // 8)

    for start in range(0, max(1, len(text) - window + 1), step):
        region = text_upper[start: start + window]
        score = sum(1 for tok in tokens if tok in region)
        if score > best_score:
            best_score = score
            best_pos = start

    if best_score == 0:
        # No token found anywhere — return head
        return text[:window]

    # Fine-tune: search ±step around best_pos
    fine_start = max(0, best_pos - step)
    fine_end = min(len(text) - window + 1, best_pos + step + 1)

    for start in range(fine_start, fine_end):
        region = text_upper[start: start + window]
        score = sum(1 for tok in tokens if tok in region)
        if score > best_score:
            best_score = score
            best_pos = start

    # Center the window on the best position found
    center = best_pos + window // 2
    half = window // 2
    start = max(0, center - half)
    end = min(len(text), start + window)
    start = max(0, end - window)

    return text[start:end]


def normalize_table_chunk(text: str) -> str:
    """
    Convert serialized table text into a clean Markdown table.

    Handles:
    Table X: ...
    Columns: A | B | C || row1 ; row1 ; row1 || row2 ; row2 ; row2
    """

    if "Columns:" not in text or "||" not in text:
        return text

    before_columns, after_columns = text.split("Columns:", 1)

    table_title = before_columns.strip()

    parts = re.split(r"\s*\|\|\s*", after_columns.strip())

    if not parts:
        return text

    columns_part = parts[0].strip()
    row_parts = parts[1:]

    columns = [
        col.strip()
        for col in columns_part.split("|")
        if col.strip()
    ]

    if not columns or not row_parts:
        return text

    normalized_rows = []

    for row in row_parts:
        cells = [
            cell.strip().strip("'")
            for cell in re.split(r"\s*;\s*", row.strip())
        ]

        # Pad missing cells
        while len(cells) < len(columns):
            cells.append("")

        # Trim extra cells
        cells = cells[:len(columns)]

        normalized_rows.append(cells)

    markdown_lines = []

    if table_title:
        markdown_lines.append(table_title)
        markdown_lines.append("")

    markdown_lines.append("| " + " | ".join(columns) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for cells in normalized_rows:
        markdown_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(markdown_lines)
BIT_COLS = ["b8", "b7", "b6", "b5", "b4", "b3", "b2", "b1"]


def extract_byte_number(text: str) -> Optional[int]:
    match = re.search(r"\bByte\s+(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def enrich_bit_table(text: str) -> str:
    if not all(bit in text.lower() for bit in BIT_COLS):
        return text

    byte_num = extract_byte_number(text)
    byte_label = f"Byte {byte_num}" if byte_num is not None else "Byte ?"

    mappings = []
    rows = re.split(r"\n|\|\|", text)

    for row in rows:
        parts = [p.strip() for p in row.split(";")]

        if len(parts) < 9:
            continue

        bit_values = parts[:8]
        meaning = " ; ".join(parts[8:]).strip()

        if not meaning:
            continue

        for bit, value in zip(BIT_COLS, bit_values):
            if value == "1":
                mappings.append(
                    f"- Definition: if {byte_label} {bit} = 1, then {meaning}"
                )
            elif value == "0":
                mappings.append(
                    f"- Constraint: {byte_label} {bit} must be 0 ({meaning})"
                )

    if not mappings:
        return text

    normalized = f"\n\nBYTE_MAPPING: BYTE_{byte_num if byte_num is not None else 'UNKNOWN'}\n"
    normalized += f"Normalized byte-specific bit mappings for {byte_label}:\n"
    normalized += "\n".join(mappings)

    return text + normalized
    
def format_context(
    scored_docs: List[ScoredDoc],
    query: str = "",
    max_chars_per_chunk: int = 3000,
) -> str:
    """
    Build the LLM context string from reranked documents.

    Improvements over the original:
    1. Accepts `query` so find_relevant_window can locate the relevant region.
    2. Converts serialized PDF table chunks into clean Markdown tables via
       normalize_table_chunk before windowing, so the LLM sees structured rows.
    3. Uses find_relevant_window instead of head truncation, ensuring tags
       buried mid-chunk are not silently dropped.
    4. Removes the raw Distance field from the header (LLM doesn't need it).
    5. Separates sources with a clear horizontal rule for readability.
    """
    context_parts = []

    for rank, (doc, distance) in enumerate(scored_docs, start=1):
        metadata = doc.metadata or {}
        raw_text = doc.page_content or ""

        # Step 1: convert serialized tables into Markdown
        # normalized_text = normalize_table_chunk(raw_text)

        enriched_text = enrich_bit_table(raw_text)
        # Step 2: normalize only if enrichment did nothing
        if enriched_text == raw_text:
            final_text = normalize_table_chunk(raw_text)
        else:
            final_text = enriched_text

        # Step 2: extract the most query-relevant window
        windowed_text = find_relevant_window(
            final_text,
            query=query,
            window=max_chars_per_chunk,
        )

        header = (
            f"[Source {rank}]\n"
            f"Book:    {metadata.get('doc_id', 'Unknown')}\n"
            f"Section: {metadata.get('section_number', 'Unknown')}\n"
            f"Title:   {metadata.get('title', 'Unknown')}\n"
            f"Page:    {metadata.get('page_num', 'Unknown')}\n"
            f"Table:   {metadata.get('table_title') or metadata.get('table_id') or 'N/A'}\n"
        )

        context_parts.append(header + "\nContent:\n" + windowed_text)

    return "\n\n---\n\n".join(context_parts)


# ================================================================
# SOURCE / CHUNK HELPERS  (unchanged logic, kept for compatibility)
# ================================================================

def extract_active_bits(hex_breakdown: str) -> List[Dict[str, str]]:
    """
    Extract active bits from decoded hex breakdown.

    Returns:
    [
        {"byte": 1, "bit": "b8"},
        {"byte": 3, "bit": "b4"}
    ]
    """
    active_bits = []

    current_byte = None

    for line in hex_breakdown.splitlines():
        byte_match = re.search(r"Byte\s+(\d+)", line, re.IGNORECASE)
        if byte_match:
            current_byte = int(byte_match.group(1))

        active_match = re.search(
            r"ACTIVE SET BITS ONLY\s*:\s*(.+)",
            line,
            re.IGNORECASE
        )

        if active_match and current_byte is not None:
            value = active_match.group(1).strip()

            if value.lower() == "none":
                continue

            bits = re.findall(r"(b[1-8])\s*=\s*1", value, re.IGNORECASE)

            for bit in bits:
                active_bits.append({
                    "byte": current_byte,
                    "bit": bit.lower()
                })

    return active_bits

def extract_matching_bit_definitions(
    context: str,
    active_bits: List[Dict[str, str]]
) -> str:
    """
    Search inside BYTE_MAPPING sections and keep only definitions
    matching active Byte + Bit pairs.
    """
    matched_lines = []

    for active in active_bits:
        byte_num = active["byte"]
        bit = active["bit"]

        pattern = re.compile(
            rf"-\s*(Definition|Constraint):\s*"
            rf"(?:if\s*)?Byte\s+{byte_num}\s+{bit}\s*"
            rf"(?:=\s*1,?\s*then|must be 0)?\s*(.+)",
            re.IGNORECASE
        )

        for line in context.splitlines():
            match = pattern.search(line)
            if match:
                meaning = match.group(2).strip()
                matched_lines.append(
                    f"- Byte {byte_num} {bit} → {meaning}"
                )

    if not matched_lines:
        return "No active bit definitions were matched from the retrieved context."

    return "\n".join(dict.fromkeys(matched_lines))

def format_source(metadata: dict):
    return {
        "doc_id": metadata.get("doc_id", "Unknown document"),
        "section_number": metadata.get("section_number", "Unknown section"),
        "title": metadata.get("title", "Unknown title"),
        "page": metadata.get("page_num", "Unknown page"),
        "table": metadata.get("table_title") or metadata.get("table_id"),
    }


def build_sources_and_chunks(scored_docs: List[ScoredDoc]):
    sources = []
    retrieved_chunks = []
    seen = set()

    for rank, (doc, distance) in enumerate(scored_docs, start=1):
        metadata = doc.metadata or {}
        source = format_source(metadata)
        source_key = (
            source["doc_id"],
            source["section_number"],
            source["title"],
            source["page"],
            source.get("table"),
        )

        if source_key not in seen:
            sources.append(source)
            seen.add(source_key)

        retrieved_chunks.append({
            "rank": rank,
            "distance": distance,
            "cross_encoder_score": metadata.get("cross_encoder_score"),
            "text_preview": (doc.page_content or "")[:700],
            "metadata": metadata,
        })

    return sources, retrieved_chunks


def build_non_rag_answer(llm, question: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
You are an EMV specification assistant.

Rules:
- If the message is a greeting, thanks, or polite conversation,
  respond briefly and naturally.
- If the message is unrelated to EMV specifications,
  payment systems, smart cards, APDU commands, EMV tags,
  or the EMV books, politely refuse.
- Do NOT answer unrelated factual questions.
- Keep responses short.

User message:
{question}

Answer:
""")

    messages = prompt.invoke({"question": question})
    response = llm.invoke(messages)
    return response.content or ""



def resolve_standalone_question(llm, question: str, history: List, input_type: InputType) -> str:
    if input_type == "contextual_follow_up":
        return rewrite_question(llm, question, history)
    return question



def stream_answer(llm, context: str, question: str, input_type: InputType):

    if input_type == "hex_decode_question":
        template = HEX_RAG_PROMPT

    elif input_type == "source_lookup":
        template = SOURCE_LOOKUP_PROMPT

    elif input_type == "contextual_follow_up":
        template = FOLLOWUP_PROMPT

    else:
        template = EMV_RAG_PROMPT

    prompt = ChatPromptTemplate.from_template(template)

    messages = prompt.invoke({
        "context": context,
        "question": question,
    })

    return llm.stream(messages)

# ================================================================
# MAIN ENTRY POINTS
# ================================================================




def clear_conversation(session_id: str = "default"):
    chat_histories[session_id] = []


def sse_event(event_type: str, data):
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def get_or_create_chat_session(db, session_id: str, user_id: str, question: str):
    session_uuid = UUID(session_id)
    user_uuid = UUID(user_id)

    chat_session = db.query(ChatSession).filter(
        ChatSession.id == session_uuid,
        ChatSession.user_id == user_uuid
    ).first()

    if chat_session:
        return chat_session
    title = generate_title(question)
    chat_session = ChatSession(
        id=session_uuid,
        user_id=user_uuid,
        title=title,
    )
    
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)

    return chat_session



def stream_conversational_rag(
    db,
    question: str,
    session_id: str = "default",
    user_id: Optional[str] = None,
    k: int = 7,
):
    user_uuid = UUID(user_id)

    chat_session = get_or_create_chat_session(
        db=db,
        session_id=session_id,
        user_id=user_id,
        question=question,
    )

    user_message = ChatMessage(
        id=uuid4(),
        session_id=chat_session.id,
        user_id=user_uuid,
        role="user",
        content=question,
    )

    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    total_start = time.perf_counter()

    vectorstore = get_langchain_chroma()
    llm = get_llm()
    history = get_history(session_id)

    router_start = time.perf_counter()
    input_type = classify_input(llm, question, history)
    router_end = time.perf_counter()

    if input_type == "noise":
        generation_start = time.perf_counter()
        answer = build_non_rag_answer(llm, question)
        generation_end = time.perf_counter()
        total_end = time.perf_counter()

        assistant_message = ChatMessage(
            id=uuid4(),
            session_id=chat_session.id,
            user_id=user_uuid,
            role="assistant",
            content=answer,
        )

        db.add(assistant_message)
        db.flush()

        rag_metadata = RagMetadata(
            id=uuid4(),
            session_id=chat_session.id,
            user_id=user_uuid,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            original_question=question,
            rewritten_question=None,
            input_type=input_type,
            retrieved_chunks_count=0,
            best_distance=None,
            average_distance=None,
            worst_distance=None,
            router_time_seconds=router_end - router_start,
            retrieval_time_seconds=0,
            generation_time_seconds=generation_end - generation_start,
            total_time_seconds=total_end - total_start,
            model_name="mistral",
            embedding_model="sentence-transformers",
        )

        db.add(rag_metadata)
        chat_session.updated_at = datetime.utcnow()
        db.commit()

        yield sse_event("token", answer)

        yield sse_event("metadata", {
            "session_id": session_id,
            "user_id": user_id,
            "input_type": input_type,
            "original_question": question,
            "standalone_question": None,
            "answer": answer,
            "sources": [],
            "retrieved_chunks": [],
            "metrics": {
                "history_messages_count": len(history),
                "retrieved_chunks_count": 0,
                "best_distance": None,
                "worst_distance": None,
                "average_distance": None,
                "router_time_seconds": router_end - router_start,
                "rewrite_time_seconds": 0,
                "retrieval_time_seconds": 0,
                "generation_time_seconds": generation_end - generation_start,
                "total_time_seconds": total_end - total_start,
            },
        })

        yield sse_event("done", "[DONE]")
        return

    rewrite_start = time.perf_counter()
    standalone_question = resolve_standalone_question(
        llm,
        question,
        history,
        input_type
    )
    rewrite_end = time.perf_counter()
    
    if input_type == "hex_decode_question":
        registry_answer = decode_from_registry(standalone_question)

        if registry_answer:
            generation_start = time.perf_counter()
            generation_end = time.perf_counter()
            total_end = time.perf_counter()

            assistant_message = ChatMessage(
                id=uuid4(),
                session_id=chat_session.id,
                user_id=user_uuid,
                role="assistant",
                content=registry_answer,
            )

            db.add(assistant_message)
            db.flush()

            rag_metadata = RagMetadata(
                id=uuid4(),
                session_id=chat_session.id,
                user_id=user_uuid,
                user_message_id=user_message.id,
                assistant_message_id=assistant_message.id,
                original_question=question,
                rewritten_question=standalone_question,
                input_type=input_type,
                retrieved_chunks_count=0,
                best_distance=None,
                average_distance=None,
                worst_distance=None,
                router_time_seconds=router_end - router_start,
                retrieval_time_seconds=0,
                generation_time_seconds=generation_end - generation_start,
                total_time_seconds=total_end - total_start,
                model_name="local-registry",
                embedding_model="none",
            )

            db.add(rag_metadata)
            chat_session.updated_at = datetime.utcnow()
            db.commit()

            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=registry_answer))

            yield sse_event("token", registry_answer)

            yield sse_event("metadata", {
                "session_id": session_id,
                "user_id": user_id,
                "input_type": input_type,
                "original_question": question,
                "standalone_question": standalone_question,
                "answer": registry_answer,
                "sources": [{"doc_id": "emv_decode_registry.json"}],
                "retrieved_chunks": [],
                "hex_breakdown": None,
                "metrics": {
                    "history_messages_count": len(history),
                    "retrieved_chunks_count": 0,
                    "best_distance": None,
                    "worst_distance": None,
                    "average_distance": None,
                    "router_time_seconds": router_end - router_start,
                    "rewrite_time_seconds": rewrite_end - rewrite_start,
                    "retrieval_time_seconds": 0,
                    "generation_time_seconds": 0,
                    "total_time_seconds": total_end - total_start,
                },
            })

            yield sse_event("done", "[DONE]")
            return

    retrieval_start = time.perf_counter()
    scored_docs = retrieve_documents(
        vectorstore=vectorstore,
        query=standalone_question,
        final_k=k,
        semantic_k=40,
        bm25_k=20,
    )
    retrieval_end = time.perf_counter()

    context = format_context(scored_docs, query=standalone_question)

    if input_type == "hex_decode_question":
        hex_breakdown = get_hex_breakdown(standalone_question)

        context = inject_hex_breakdown(standalone_question, context)

        active_bits = extract_active_bits(hex_breakdown)

        matched_definitions = extract_matching_bit_definitions(
            context=context,
            active_bits=active_bits
        )

        decode_sources = []

        for doc, _ in scored_docs:
            meta = doc.metadata or {}

            source = (
                f"[Source: "
                f"{meta.get('doc_id')} | "
                f"{meta.get('section_number')} | "
                f"{meta.get('title')} | "
                f"page {meta.get('page_num')}]"
            )

            if source not in decode_sources:
                decode_sources.append(source)

        context = f"""
        HEX BREAKDOWN:
        {hex_breakdown}

        MATCHED ACTIVE BIT DEFINITIONS:
        {matched_definitions}

        SOURCES:
        {chr(10).join(decode_sources)}
        """
    
    else:
        hex_breakdown = None
    save_final_context_to_txt(
    filename="final_context_sent_to_llm.txt",
    context=context,
    query=standalone_question,
    input_type=input_type,
)
    generation_start = time.perf_counter()

    full_answer = ""

    for chunk in stream_answer(llm, context, standalone_question, input_type,):
        token = chunk.content or ""
        full_answer += token
        yield sse_event("token", token)

    generation_end = time.perf_counter()

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=full_answer))

    sources, retrieved_chunks = build_sources_and_chunks(scored_docs)
    distances = [distance for _, distance in scored_docs]

    best_distance = min(distances) if distances else None
    worst_distance = max(distances) if distances else None
    average_distance = sum(distances) / len(distances) if distances else None

    total_end = time.perf_counter()

    assistant_message = ChatMessage(
        id=uuid4(),
        session_id=chat_session.id,
        user_id=user_uuid,
        role="assistant",
        content=full_answer,
    )

    db.add(assistant_message)
    db.flush()

    rag_metadata = RagMetadata(
        id=uuid4(),
        session_id=chat_session.id,
        user_id=user_uuid,
        user_message_id=user_message.id,
        assistant_message_id=assistant_message.id,
        original_question=question,
        rewritten_question=standalone_question,
        input_type=input_type,
        retrieved_chunks_count=len(scored_docs),
        best_distance=best_distance,
        average_distance=average_distance,
        worst_distance=worst_distance,
        router_time_seconds=router_end - router_start,
        retrieval_time_seconds=retrieval_end - retrieval_start,
        generation_time_seconds=generation_end - generation_start,
        total_time_seconds=total_end - total_start,
        model_name=os.getenv("OLLAMA_MODEL"),
        embedding_model=os.getenv("EMBEDDING_MODEL"),
    )

    db.add(rag_metadata)
    db.flush()

    for rank, (doc, distance) in enumerate(scored_docs, start=1):
        meta = doc.metadata or {}

        rag_source = RagSource(
            id=uuid4(),
            rag_metadata_id=rag_metadata.id,
            rank=rank,
            chunk_id=meta.get("chunk_id"),
            section_id=meta.get("section_id"),
            doc_id=meta.get("doc_id"),
            title=meta.get("title"),
            section_number=meta.get("section_number"),
            page=meta.get("page"),
            distance=distance,
            text_preview=(doc.page_content or "")[:100],
        )

        db.add(rag_source)

    chat_session.updated_at = datetime.utcnow()
    db.commit()

    yield sse_event("metadata", {
        "session_id": session_id,
        "user_id": user_id,
        "input_type": input_type,
        "original_question": question,
        "standalone_question": standalone_question,
        "answer": full_answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "hex_breakdown": hex_breakdown,
        "metrics": {
            "history_messages_count": len(history),
            "retrieved_chunks_count": len(scored_docs),
            "best_distance": best_distance,
            "worst_distance": worst_distance,
            "average_distance": average_distance,
            "router_time_seconds": router_end - router_start,
            "rewrite_time_seconds": rewrite_end - rewrite_start,
            "retrieval_time_seconds": retrieval_end - retrieval_start,
            "generation_time_seconds": generation_end - generation_start,
            "total_time_seconds": total_end - total_start,
        },
    })

    yield sse_event("done", "[DONE]")


