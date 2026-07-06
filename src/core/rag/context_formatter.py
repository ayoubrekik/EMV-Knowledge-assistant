import re
from typing import Optional
from .source_utils import source_from_doc
from .types import ScoredDoc


def find_relevant_window(text: str, query: str, window: int = 3000) -> str:
    if len(text) <= window:
        return text

    tokens = re.findall(r"[0-9A-Fa-f]{2,8}|[A-Za-z]{3,}", query)
    tokens = list({t.upper() for t in tokens if t.strip()})
    if not tokens:
        return text[:window]

    text_upper = text.upper()
    best_pos = 0
    best_score = -1
    step = max(1, window // 8)

    for start in range(0, max(1, len(text) - window + 1), step):
        region = text_upper[start:start + window]
        score = sum(1 for tok in tokens if tok in region)
        if score > best_score:
            best_score = score
            best_pos = start

    if best_score == 0:
        return text[:window]

    fine_start = max(0, best_pos - step)
    fine_end = min(len(text) - window + 1, best_pos + step + 1)
    for start in range(fine_start, fine_end):
        region = text_upper[start:start + window]
        score = sum(1 for tok in tokens if tok in region)
        if score > best_score:
            best_score = score
            best_pos = start

    center = best_pos + window // 2
    half = window // 2
    start = max(0, center - half)
    end = min(len(text), start + window)
    start = max(0, end - window)
    return text[start:end]


BIT_COLS = ["b8", "b7", "b6", "b5", "b4", "b3", "b2", "b1"]


def extract_byte_number(text: str) -> Optional[int]:
    match = re.search(r"\bByte\s+(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def enrich_bit_table(text: str) -> str:
    lowered = text.lower()
    if not all(bit in lowered for bit in BIT_COLS):
        return text

    byte_num = extract_byte_number(text)
    byte_label = f"Byte {byte_num}" if byte_num is not None else "Byte ?"
    mappings = []

    rows = re.split(r"\n|\|\|", text)
    for row in rows:
        raw_parts = [p.strip() for p in row.split(";")]
        if len(raw_parts) < 9:
            raw_parts = [p.strip() for p in row.strip("|").split("|")]

        if len(raw_parts) < 9:
            continue

        bit_values = raw_parts[:8]
        meaning = " ; ".join(raw_parts[8:]).strip()
        if not meaning:
            continue

        for bit, value in zip(BIT_COLS, bit_values):
            value = value.strip()
            if value == "1":
                mappings.append(f"- Definition: if {byte_label} {bit} = 1, then {meaning}")
            elif value == "0":
                mappings.append(f"- Constraint: {byte_label} {bit} must be 0 ({meaning})")

    if not mappings:
        return text

    normalized = f"\n\nBYTE_MAPPING: BYTE_{byte_num if byte_num is not None else 'UNKNOWN'}\n"
    normalized += f"Normalized byte-specific bit mappings for {byte_label}:\n"
    normalized += "\n".join(dict.fromkeys(mappings))
    return text + normalized


def format_context(scored_docs: list[ScoredDoc], query: str = "", max_chars_per_chunk: int = 3000) -> str:
    context_parts = []

    for rank, (doc, distance) in enumerate(scored_docs, start=1):
        raw_text = doc.page_content or ""
        source = source_from_doc(doc)

        enriched_text = enrich_bit_table(raw_text)
        windowed_text = find_relevant_window(enriched_text, query=query, window=max_chars_per_chunk)

        header = (
            f"[Source {rank}]\n"
            f"Book: {source['doc_id']}\n"
            f"Document title: {source['doc_title']}\n"
            f"Section: {source['section']}\n"
            f"Page: {source['page']}\n"
            f"Type: {source['content_type']}\n"
        )

        if source.get("part"):
            header += f"Part: {source['part']}\n"

        context_parts.append(header + "\nContent:\n" + windowed_text)

    return "\n\n---\n\n".join(context_parts)
