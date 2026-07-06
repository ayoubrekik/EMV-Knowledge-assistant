"""
universal_pre_chunker_text_table_rules.py
=========================================

Universal pre-chunking pipeline for enriched technical PDF sections.

Main decisions use:
    total_tokens = n_tokens_text + n_tokens_tabs

Rules:
1. total < 10
   -> always accumulate forward into the next section.

2. 10 <= total < 80
   -> merge with the next section only if:
      - both sections are at the same heading level
      - merged total <= target_tokens = 256

3. 80 <= total <= 480
   -> good-size section.
      - If text is 80-480 and tables are 80-480: create separate chunks:
        one text chunk and one table chunk with the same context.
      - If one side is <80 and the other side is 80-480 and total <=480:
        create one mixed chunk.
      - Otherwise create a direct chunk.

4. total > 480
   -> split safely.
      - Do not cut normal rows from text_list.
      - Keep table caption + table markdown together as one table block.
      - If a single table block is itself > max_tokens, split it by markdown lines.
      - Overlap = 20% of body max, using complete text_list rows only.

5. Mix text and tables only when the final chunk is <= max_tokens.

Edit INPUT_PATH and OUTPUT_PATH below. No command-line arguments are required.
"""
from __future__ import annotations

import os
import json
import re
import uuid
import hashlib

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

HF_CACHE_DIR = "/root/.cache/huggingface"

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer

from src.core.db.chroma_client import resolve_hf_snapshot

# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_PATH = Path("/app/src/storage/token_stats/enriched_sections.json")
OUTPUT_PATH = Path("/app/src/storage/chunks/chunks.json")

SHOW_STATS = True

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ChunkingConfig:
    tiny_threshold: int = 10
    min_tokens: int = 80
    target_tokens: int = 256
    max_tokens: int = 480
    overlap_ratio: float = 0.20

# ============================================================
# TOKENIZER
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-large-en-v1.5",
)

tokenizer = AutoTokenizer.from_pretrained(
    resolve_hf_snapshot(EMBEDDING_MODEL),
    local_files_only=True,
)

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    )
# ============================================================
# 4. OUTPUT MODEL
# ============================================================

@dataclass
class Chunk:
    chunk_id: str
    document_id: str

    doc_id: str
    doc_title: str
    doc_version: str
    doc_date: str

    text: str
    context_prefix: str

    section_numbers: list[str]
    section_titles: list[str]
    section_path: str

    page_start: int | None
    page_end: int | None

    chunk_strategy: str
    content_type: str

    is_merged: bool
    merged_count: int

    is_split: bool
    split_part: int | None
    split_total: int | None

    has_tables: bool
    has_text: bool

    token_count: int
    source_heading_ids: list[str]

    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# 5. GENERIC HELPERS
# ============================================================

RowKind = Literal["text", "table"]
Row = tuple[RowKind, str, int]


def section_level(section_number: str) -> int:
    if not section_number:
        return 99

    value = str(section_number).strip()
    value = re.sub(r"^(part|chapter|section|annex)\s+", "", value, flags=re.I)
    value = value.replace(" ", "")

    if "." in value:
        return len([p for p in value.split(".") if p])
    return 1


def same_level(a: dict, b: dict) -> bool:
    return section_level(a.get("section_number", "")) == section_level(b.get("section_number", ""))

def is_parent_of(parent: dict, child: dict) -> bool:
    parent_num = str(parent.get("section_number", "") or "").strip()
    child_num = str(child.get("section_number", "") or "").strip()

    if not parent_num or not child_num:
        return False

    return child_num.startswith(parent_num + ".")

def get_text_tokens(section: dict) -> int:
    return int(section.get("n_tokens_text", 0) or 0)


def get_table_tokens(section: dict) -> int:
    return int(section.get("n_tokens_tabs", 0) or 0)


def get_total_tokens(section: dict) -> int:
    return get_text_tokens(section) + get_table_tokens(section)


def build_section_path(section: dict) -> str:
    section_number = str(section.get("section_number", "") or "").strip()
    title = str(section.get("title", "") or "").strip()
    parents = section.get("parent_titles", []) or []

    parts = [str(p).strip() for p in parents if str(p).strip()]
    if title:
        parts.append(title)

    path = " > ".join(parts)
    if section_number:
        return f"{section_number} {path}".strip()
    return path


def build_document_id(doc: dict, input_path: Path | None = None) -> str:
    if doc.get("document_id"):
        return str(doc["document_id"])

    doc_id = str(doc.get("doc_id") or (input_path.stem if input_path else "document"))
    doc_version = str(doc.get("doc_version", "") or "")
    doc_date = str(doc.get("doc_date", "") or "")

    value = f"{doc_id}_{doc_version}_{doc_date}".replace(" ", "_")
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "document"


def first_non_none(values: list[Any]) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def build_context_prefix(
    sections: list[dict],
    document_id: str,
    content_type: str | None = None,
    part_label: str | None = None,
) -> str:
    page_starts = [s.get("start_page") for s in sections if s.get("start_page") is not None]
    page_ends = [s.get("end_page") for s in sections if s.get("end_page") is not None]

    page_start = min(page_starts) if page_starts else None
    page_end = max(page_ends) if page_ends else None

    if page_start is None:
        page_text = ""
    elif page_start == page_end:
        page_text = str(page_start)
    else:
        page_text = f"{page_start}-{page_end}"

    section_text = " | ".join(build_section_path(s) for s in sections)

    prefix = (
        f"[Document: {document_id}]\n"
        f"[Section: {section_text}]\n"
        f"[Page: {page_text}]"
    )

    if content_type:
        prefix += f"\n[Type: {content_type}]"
    if part_label:
        prefix += f"\n[{part_label}]"

    return prefix



def make_chunk_id(
    document_id: str,
    sections: list[dict],
    content_type: str,
    content: str,
    part: int | None = None,
    suffix: str | None = None,
) -> str:
    base_items = []

    for s in sections:
        base_items.append(
            str(
                s.get("section_number")
                or s.get("heading_id")
                or s.get("title", "")[:30]
                or uuid.uuid4()
            )
        )

    raw = f"{document_id}_{'_'.join(base_items)}_{content_type}"

    if suffix:
        raw += f"_{suffix}"

    if part is not None:
        raw += f"_part_{part}"

    # Deterministic hash of the chunk content
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
    raw += f"_{content_hash}"

    raw = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")

    return raw[:160]

def get_chunk_level(chunk: dict | Chunk) -> int:
    section_number = (
        chunk.section_numbers[0]
        if isinstance(chunk, Chunk)
        else chunk.get("section_numbers", [""])[0]
    )

    section_number = section_number.split("|")[0].strip()
    return section_level(section_number)


def merge_following_same_level_chunks(
    chunks: list[Chunk],
    max_tokens: int = 480
) -> list[Chunk]:
    merged_chunks = []
    i = 0

    while i < len(chunks):
        current = chunks[i]

        if current.chunk_strategy != "merged_same_level":
            merged_chunks.append(current)
            i += 1
            continue

        group = [current]
        total_tokens = current.token_count
        current_level = get_chunk_level(current)

        j = i + 1

        while j < len(chunks):
            next_chunk = chunks[j]
            next_level = get_chunk_level(next_chunk)

            if next_level != current_level:
                break

            if total_tokens + next_chunk.token_count > max_tokens:
                break

            group.append(next_chunk)
            total_tokens += next_chunk.token_count
            j += 1

        if len(group) == 1:
            merged_chunks.append(current)
            i += 1
            continue

        merged = merge_chunk_group(group)
        merged_chunks.append(merged)

        i = j

    return merged_chunks

def merge_chunk_group(group: list[Chunk]) -> Chunk:
    first = group[0]
    last = group[-1]

    body_parts = []

    for chunk in group:
        body = chunk.text.replace(chunk.context_prefix, "").strip()
        if body:
            body_parts.append(body)

    section_numbers = []
    section_titles = []
    source_heading_ids = []

    for chunk in group:
        section_numbers.extend(chunk.section_numbers)
        section_titles.extend(chunk.section_titles)
        source_heading_ids.extend(chunk.source_heading_ids)

    section_numbers_text = " | ".join(section_numbers)
    section_titles_text = " | ".join(section_titles)

    context_prefix = (
        f"[Document: {first.document_id}]\n"
        f"[Section: {section_numbers_text} {section_titles_text}]\n"
        f"[Page: {first.page_start}-{last.page_end}]\n"
        f"[Type: {first.content_type}]"
    )

    full_text = context_prefix + "\n\n" + "\n\n".join(body_parts)

    return Chunk(
        chunk_id=f"{first.chunk_id}_extended",
        document_id=first.document_id,
        doc_id=first.doc_id,
        doc_title=first.doc_title,
        doc_version=first.doc_version,
        doc_date=first.doc_date,

        text=full_text,
        context_prefix=context_prefix,

        section_numbers=section_numbers,
        section_titles=section_titles,
        section_path=" | ".join(chunk.section_path for chunk in group),

        page_start=first.page_start,
        page_end=last.page_end,

        chunk_strategy="merged_same_level_extended",
        content_type=first.content_type,

        is_merged=True,
        merged_count=len(group),

        is_split=False,
        split_part=None,
        split_total=None,

        has_tables=any(chunk.has_tables for chunk in group),
        has_text=any(chunk.has_text for chunk in group),

        token_count=count_tokens(full_text),
        source_heading_ids=source_heading_ids,
    )
# ============================================================
# 6. ROW BUILDING
# ============================================================

def normalize_table_item(item: Any) -> str:
    if item is None:
        return ""

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        parts = []
        for key in ["previous_text", "caption", "markdown"]:
            value = str(item.get(key, "") or "").strip()
            if value:
                parts.append(value)
        return "\n\n".join(parts)

    return str(item).strip()


def build_text_rows(section: dict) -> list[Row]:
    rows: list[Row] = []
    title = str(section.get("title", "") or "").strip()
    text_list = section.get("text_list", []) or []

    for i, item in enumerate(text_list):
        text = str(item or "").strip()
        if not text:
            continue

        # Skip repeated heading/title row.
        if i == 0 and title and title in text:
            continue

        rows.append(("text", text, count_tokens(text)))

    return rows


def build_table_rows(section: dict) -> list[Row]:
    """
    Builds atomic table blocks.

    Generalized rule: ANY non-table text item (caption, "Table 46:",
    "TVR Byte 3:", a sub-heading, a footnote referencing the table that
    follows, etc.) that sits directly in front of a markdown table is
    glued to that table. This guarantees a label/caption can never be
    split into a different chunk than the table it introduces — no
    matter what the label's wording looks like.
    """
    table_list = section.get("table_list", []) or []
    rows: list[Row] = []
    i = 0
    n = len(table_list)

    def is_markdown_table(text: str) -> bool:
        return text.startswith("|")

    while i < n:
        current = normalize_table_item(table_list[i])
        if not current:
            i += 1
            continue

        if is_markdown_table(current):
            # A table with nothing pending in front of it.
            rows.append(("table", current, count_tokens(current)))
            i += 1
            continue

        # `current` is caption/label-like. Collect every consecutive
        # non-table item (covers multi-line captions too), then attach
        # the table that follows, if any.
        pending = [current]
        j = i + 1
        while j < n:
            nxt = normalize_table_item(table_list[j])
            if not nxt:
                j += 1
                continue
            if is_markdown_table(nxt):
                break
            pending.append(nxt)
            j += 1

        if j < n:
            table_md = normalize_table_item(table_list[j])
            block = "\n\n".join(pending) + "\n\n" + table_md
            rows.append(("table", block, count_tokens(block)))
            i = j + 1
        else:
            # No table follows (e.g. a trailing note after the last
            # table) — keep it as its own block instead of dropping it.
            block = "\n\n".join(pending)
            rows.append(("table", block, count_tokens(block)))
            i = j

    return rows


def rows_to_body(rows: list[Row]) -> str:
    return "\n\n".join(content for _, content, _ in rows if content.strip())


# ============================================================
# 7. CHUNK BUILDER
# ============================================================

def build_chunk(
    sections: list[dict],
    document_id: str,
    body: str,
    content_type: str,
    strategy: str,
    part: int | None = None,
    total_parts: int | None = None,
    suffix: str | None = None,
) -> Chunk:
    part_label = None
    if part is not None and total_parts is not None and total_parts > 1:
        part_label = f"Part {part}/{total_parts}"

    context_prefix = build_context_prefix(
        sections=sections,
        document_id=document_id,
        content_type=content_type,
        part_label=part_label,
    )

    full_text = context_prefix
    if body.strip():
        full_text += "\n\n" + body.strip()

    return Chunk(
        chunk_id=make_chunk_id(document_id, sections, content_type, full_text, part, suffix),
        document_id=document_id,

        doc_id=str(sections[0].get("doc_id", "") or ""),
        doc_title=str(sections[0].get("doc_title", "") or ""),
        doc_version=str(sections[0].get("doc_version", "") or ""),
        doc_date=str(sections[0].get("doc_date", "") or ""),

        text=full_text,
        context_prefix=context_prefix,

        section_numbers=[str(s.get("section_number", "") or "") for s in sections],
        section_titles=[str(s.get("title", "") or "") for s in sections],
        section_path=" | ".join(build_section_path(s) for s in sections),

        page_start=min([s.get("start_page") for s in sections if s.get("start_page") is not None], default=None),
        page_end=max([s.get("end_page") for s in sections if s.get("end_page") is not None], default=None),

        chunk_strategy=strategy,
        content_type=content_type,

        is_merged=len(sections) > 1,
        merged_count=len(sections),

        is_split=part is not None and total_parts is not None and total_parts > 1,
        split_part=part,
        split_total=total_parts,

        has_tables=content_type in {"table", "mixed"},
        has_text=content_type in {"text", "mixed"},

        token_count=count_tokens(full_text),
        source_heading_ids=[str(s.get("heading_id", "") or "") for s in sections],
    )


# ============================================================
# 8. SECTION SPLITTER
# ============================================================
def is_markdown_separator(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells)


def split_table_with_repeated_header(content: str, body_max: int) -> list[Row]:
    lines = [l for l in content.splitlines() if l.strip()]

    header_idx = None

    for i, line in enumerate(lines):
        if (
            line.strip().startswith("|")
            and i + 1 < len(lines)
            and is_markdown_separator(lines[i + 1])
        ):
            header_idx = i
            break

    if header_idx is None:
        return [("table", content, count_tokens(content))]

    caption = "\n".join(lines[:header_idx]).strip()
    header = lines[header_idx]
    separator = lines[header_idx + 1]
    data_rows = lines[header_idx + 2:]

    chunks = []
    current_rows = []

    def make_table(rows):
        parts = []
        if caption:
            parts.append(caption)
        parts.append(header)
        parts.append(separator)
        parts.extend(rows)
        return "\n".join(parts)

    for row in data_rows:
        candidate = make_table(current_rows + [row])

        if current_rows and count_tokens(candidate) > body_max:
            table_text = make_table(current_rows)
            chunks.append(("table", table_text, count_tokens(table_text)))
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        table_text = make_table(current_rows)
        chunks.append(("table", table_text, count_tokens(table_text)))

    return chunks



class SectionSplitter:
    def __init__(self, config: ChunkingConfig):
        self.cfg = config

    def available_body_tokens(self, section: dict, document_id: str, content_type: str) -> int:
        prefix = build_context_prefix([section], document_id, content_type=content_type, part_label="Part 1/99")
        prefix_tokens = count_tokens(prefix)
        return max(80, self.cfg.max_tokens - prefix_tokens)

    def split_rows(
        self,
        section: dict,
        document_id: str,
        rows: list[Row],
        content_type: str,
        strategy: str,
        small_rows_to_mix: list[Row] | None = None,
    ) -> list[Chunk]:
        """
        Split rows without cutting text rows. Table rows are atomic unless one
        single table row exceeds body capacity; then it is split by markdown lines.
        """
        if not rows and not small_rows_to_mix:
            return []

        body_max = self.available_body_tokens(section, document_id, content_type)
        expanded_rows = self.expand_oversized_rows(rows, body_max)

        parts: list[list[Row]] = []
        current: list[Row] = []
        current_tokens = 0

        for row in expanded_rows:
            _, _, row_tokens = row

            if current and current_tokens + row_tokens > body_max:
                parts.append(current)
                current = []
                current_tokens = 0

            current.append(row)
            current_tokens += row_tokens

        if current:
            parts.append(current)

        if not parts:
            parts = [[]]

        # Mix small rows only with the first split part if the final body fits.
        if small_rows_to_mix and parts:
            small_tokens = sum(t for _, _, t in small_rows_to_mix)
            first_tokens = sum(t for _, _, t in parts[0])
            if small_tokens + first_tokens <= body_max:
                parts[0] = small_rows_to_mix + parts[0]
                content_type = "mixed"

        chunks: list[Chunk] = []
        previous_overlap: list[Row] = []
        total = len(parts)

        for idx, part_rows in enumerate(parts, start=1):
            body_rows: list[Row] = []

            if previous_overlap:
                overlap_text = rows_to_body(previous_overlap)
                body_rows.append(("text", "[…continued]\n" + overlap_text, count_tokens(overlap_text)))

            body_rows.extend(part_rows)
            body = rows_to_body(body_rows)

            chunks.append(
                build_chunk(
                    sections=[section],
                    document_id=document_id,
                    body=body,
                    content_type=content_type,
                    strategy=strategy,
                    part=idx,
                    total_parts=total,
                )
            )

            previous_overlap = self.build_overlap_rows(part_rows, body_max)

        return chunks

    def expand_oversized_rows(self, rows: list[Row], body_max: int) -> list[Row]:
        expanded: list[Row] = []

        for kind, content, tokens in rows:
            if tokens <= body_max:
                expanded.append((kind, content, tokens))
                continue

            if kind == "text":
                # Preserve text rows as much as possible. If one text row is too large,
                # split by sentences as a fallback.
                expanded.extend(self.split_large_text_row(content, body_max))
            else:
                expanded.extend(self.split_large_table_row(content, body_max))

        return expanded

    def split_large_text_row(self, content: str, body_max: int) -> list[Row]:
        sentences = re.split(r"(?<=[.!?])\s+", content)
        return self.pack_lines_as_rows(sentences, body_max, kind="text")

    def split_large_table_row(self, content: str, body_max: int) -> list[Row]:
        return split_table_with_repeated_header(content, body_max)

    def pack_lines_as_rows(self, lines: list[str], body_max: int, kind: RowKind) -> list[Row]:
        result: list[Row] = []
        current: list[str] = []
        current_tokens = 0

        for line in lines:
            line = str(line).rstrip()
            if not line:
                continue
            line_tokens = count_tokens(line)

            # If one line alone is too large, split by tokens as last-resort.
            if line_tokens > body_max:
                if current:
                    text = "\n".join(current)
                    result.append((kind, text, count_tokens(text)))
                    current = []
                    current_tokens = 0
                result.extend(self.split_by_token_window(line, body_max, kind))
                continue

            if current and current_tokens + line_tokens > body_max:
                text = "\n".join(current)
                result.append((kind, text, count_tokens(text)))
                current = []
                current_tokens = 0

            current.append(line)
            current_tokens += line_tokens

        if current:
            text = "\n".join(current)
            result.append((kind, text, count_tokens(text)))

        return result

    def split_by_token_window(self, text: str, body_max: int, kind: RowKind) -> list[Row]:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        parts: list[Row] = []
        for start in range(0, len(ids), body_max):
            piece = tokenizer.decode(ids[start:start + body_max])
            parts.append((kind, piece, count_tokens(piece)))
        return parts

    def build_overlap_rows(self, rows: list[Row], body_max: int) -> list[Row]:
        target = max(1, int(body_max * self.cfg.overlap_ratio))
        overlap: list[Row] = []
        collected = 0

        for kind, content, tokens in reversed(rows):
            if kind != "text":
                continue

            # Always include at least one text row if available.
            if overlap and collected + tokens > target:
                break

            overlap.insert(0, (kind, content, tokens))
            collected += tokens

            if collected >= target:
                break

        return overlap


# ============================================================
# 9. SECTION PROCESSOR
# ============================================================

class SectionProcessor:
    def __init__(self, config: ChunkingConfig):
        self.cfg = config
        self.splitter = SectionSplitter(config)

    def process_section(self, section: dict, document_id: str, strategy: str = "direct") -> list[Chunk]:
        text_tokens = get_text_tokens(section)
        table_tokens = get_table_tokens(section)
        total = text_tokens + table_tokens

        text_rows = build_text_rows(section)
        table_rows = build_table_rows(section)

        # Empty fallback.
        if not text_rows and not table_rows:
            return [build_chunk([section], document_id, "", "text", strategy="direct_empty")]

        # Good-size total section.
        if self.cfg.min_tokens <= total <= self.cfg.max_tokens:
            return self.process_good_size_section(
                section, document_id, text_rows, table_rows, text_tokens, table_tokens, strategy
            )

        # Large section.
        if total > self.cfg.max_tokens:
            return self.process_large_section(
                section, document_id, text_rows, table_rows, text_tokens, table_tokens
            )

        # Small/tiny fallback, usually reached only for unmerged small sections.
        body = rows_to_body(text_rows + table_rows)
        content_type = self.infer_content_type(text_rows, table_rows)
        return [build_chunk([section], document_id, body, content_type, strategy="direct_small")]

    def process_good_size_section(
        self,
        section: dict,
        document_id: str,
        text_rows: list[Row],
        table_rows: list[Row],
        text_tokens: int,
        table_tokens: int,
        strategy: str,
    ) -> list[Chunk]:
        has_text = text_tokens > 0 and bool(text_rows)
        has_tables = table_tokens > 0 and bool(table_rows)

        # User rule: if text is 80-480 and table is 80-480, do not mix.
        if (
            has_text and has_tables
            and self.cfg.min_tokens <= text_tokens <= self.cfg.max_tokens
            and self.cfg.min_tokens <= table_tokens <= self.cfg.max_tokens
        ):
            return [
                build_chunk(
                    [section], document_id, rows_to_body(text_rows), "text", strategy="direct_text", suffix="text"
                ),
                build_chunk(
                    [section], document_id, rows_to_body(table_rows), "table", strategy="direct_table", suffix="table"
                ),
            ]

        # User rule: if one side <80 and the other side 80-480 and total <=480, mix.
        if has_text and has_tables and total_fits(text_tokens, table_tokens, self.cfg.max_tokens):
            if (
                (text_tokens < self.cfg.min_tokens and self.cfg.min_tokens <= table_tokens <= self.cfg.max_tokens)
                or (table_tokens < self.cfg.min_tokens and self.cfg.min_tokens <= text_tokens <= self.cfg.max_tokens)
            ):
                return [
                    build_chunk(
                        [section], document_id, rows_to_body(text_rows + table_rows), "mixed", strategy="direct_mixed"
                    )
                ]

        # Otherwise, total is good-size and may be emitted as direct mixed/text/table.
        body = rows_to_body(text_rows + table_rows)
        return [build_chunk([section], document_id, body, self.infer_content_type(text_rows, table_rows), strategy=strategy)]

    def process_large_section(
        self,
        section: dict,
        document_id: str,
        text_rows: list[Row],
        table_rows: list[Row],
        text_tokens: int,
        table_tokens: int,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        has_text = text_tokens > 0 and bool(text_rows)
        has_tables = table_tokens > 0 and bool(table_rows)

        # If both sides are individually good-size, separate them.
        if (
            has_text and has_tables
            and self.cfg.min_tokens <= text_tokens <= self.cfg.max_tokens
            and self.cfg.min_tokens <= table_tokens <= self.cfg.max_tokens
        ):
            chunks.append(build_chunk([section], document_id, rows_to_body(text_rows), "text", "direct_text", suffix="text"))
            chunks.append(build_chunk([section], document_id, rows_to_body(table_rows), "table", "direct_table", suffix="table"))
            return chunks

        # If one side is small and the other is large, mix the small side only into
        # the first split part if it fits.
        small_text = has_text and text_tokens < self.cfg.min_tokens
        small_table = has_tables and table_tokens < self.cfg.min_tokens

        if has_text and not has_tables:
            return self.splitter.split_rows(section, document_id, text_rows, "text", "split_text")

        if has_tables and not has_text:
            return self.splitter.split_rows(section, document_id, table_rows, "table", "split_table")

        if small_text and table_tokens > self.cfg.max_tokens:
            return self.splitter.split_rows(
                section, document_id, table_rows, "table", "split_table", small_rows_to_mix=text_rows
            )

        if small_table and text_tokens > self.cfg.max_tokens:
            return self.splitter.split_rows(
                section, document_id, text_rows, "text", "split_text", small_rows_to_mix=table_rows
            )

        # If text is good and table is large: text chunk + table split.
        if self.cfg.min_tokens <= text_tokens <= self.cfg.max_tokens and table_tokens > self.cfg.max_tokens:
            chunks.append(build_chunk([section], document_id, rows_to_body(text_rows), "text", "direct_text", suffix="text"))
            chunks.extend(self.splitter.split_rows(section, document_id, table_rows, "table", "split_table"))
            return chunks

        # If table is good and text is large: text split + table chunk.
        if self.cfg.min_tokens <= table_tokens <= self.cfg.max_tokens and text_tokens > self.cfg.max_tokens:
            chunks.extend(self.splitter.split_rows(section, document_id, text_rows, "text", "split_text"))
            chunks.append(build_chunk([section], document_id, rows_to_body(table_rows), "table", "direct_table", suffix="table"))
            return chunks

        # If both are large, split separately.
        if text_tokens > self.cfg.max_tokens and table_tokens > self.cfg.max_tokens:
            chunks.extend(self.splitter.split_rows(section, document_id, text_rows, "text", "split_text"))
            chunks.extend(self.splitter.split_rows(section, document_id, table_rows, "table", "split_table"))
            return chunks

        # Final safe fallback: split combined rows, but only because previous rules did not apply.
        return self.splitter.split_rows(section, document_id, text_rows + table_rows, "mixed", "split_mixed")

    @staticmethod
    def infer_content_type(text_rows: list[Row], table_rows: list[Row]) -> str:
        if text_rows and table_rows:
            return "mixed"
        if table_rows:
            return "table"
        return "text"


def total_fits(text_tokens: int, table_tokens: int, max_tokens: int) -> bool:
    return text_tokens + table_tokens <= max_tokens


# ============================================================
# 10. UNIVERSAL PRE-CHUNKER
# ============================================================
# ============================================================
# TOC / FRONT-MATTER FILTER
# ============================================================

_TOC_TITLE_PATTERNS = re.compile(
    r"""
    ^\s*
    (
        \btable\s+of\s+contents\b    |
        \bdocument\s+of\s+structure\b |
        \bcontents\b                 |
        \blist\s+of\s+figures\b      |
        \blist\s+of\s+tables\b       |
        \btables\b                   |
        \bfigures\b
    )
    .*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

def is_toc_section(section: dict) -> bool:
    """
    Returns True if this section is a table of contents, list of figures,
    list of tables, or similar front-matter navigation block that should
    be dropped before chunking.
    """
    title = str(section.get("title", "") or "").strip()
    if _TOC_TITLE_PATTERNS.match(title):
        return True

    # Also catch cases where the title is empty but the text content
    # looks like a TOC (lines are mostly "SomeTitle .... 12" patterns).
    text_list = section.get("text_list", []) or []
    if not title and text_list:
        toc_line_pattern = re.compile(r".{5,}\s*\.{3,}\s*\d+\s*$")
        sample = [str(t) for t in text_list[:20] if str(t).strip()]
        if len(sample) >= 3:
            matches = sum(1 for t in sample if toc_line_pattern.search(t))
            if matches / len(sample) >= 0.5:
                return True

    return False

class UniversalPreChunker:
    def __init__(self, config: ChunkingConfig):
        self.cfg = config
        self.processor = SectionProcessor(config)

    def chunk(self, doc: dict, input_path: Path | None = None) -> list[Chunk]:
        sections = doc.get("sections", []) or []
        document_id = build_document_id(doc, input_path=input_path)
        sections = [s for s in sections if not is_toc_section(s)]
        chunks: list[Chunk] = []
        i = 0

        while i < len(sections):
            current = sections[i]
            current_total = get_total_tokens(current)

            # Rule 1: total < 10 -> forced merge forward.
            if current_total < self.cfg.tiny_threshold:
                if i + 1 < len(sections):
                    next_section = sections[i + 1]
                    merged = self.merge_sections([current, next_section])

                    # Process merged content with normal text/table rules.
                    merged_chunks = self.processor.process_section(
                        merged,
                        document_id,
                        strategy="merged_forward",
                    )
                    chunks.extend(merged_chunks)
                    i += 2
                    continue

                # Last tiny section: emit safely.
                chunks.extend(self.processor.process_section(current, document_id, strategy="direct_tiny_last"))
                i += 1
                continue

            # Rule 2: 10-79 -> merge with next only if same level and total <=256.
            if self.cfg.tiny_threshold <= current_total < self.cfg.min_tokens:
                if i + 1 < len(sections):
                    next_section = sections[i + 1]
                    merged_total = current_total + get_total_tokens(next_section)

                    if (
                        (
                            same_level(current, next_section)
                            or is_parent_of(current, next_section)
                        )
                        and merged_total <= self.cfg.target_tokens
                    ):
                        merged = self.merge_sections([current, next_section])
                        chunks.extend(
                            self.processor.process_section(
                                merged,
                                document_id,
                                strategy="merged_same_level",
                            )
                        )
                        i += 2
                        continue

                # If it cannot be merged, keep it but mark as small.
                chunks.extend(self.processor.process_section(current, document_id, strategy="direct_small"))
                i += 1
                continue

            # Rule 3 and 4 handled by SectionProcessor.
            chunks.extend(self.processor.process_section(current, document_id, strategy="direct"))
            i += 1

        chunks = merge_following_same_level_chunks(
            chunks,
            max_tokens=self.cfg.max_tokens
        )

        self.add_navigation(chunks)
        return chunks

    def merge_sections(self, sections: list[dict]) -> dict:
        first = sections[0]
        merged = dict(first)

        merged["heading_id"] = "+".join(str(s.get("heading_id", "")) for s in sections)
        merged["section_number"] = " | ".join(str(s.get("section_number", "")) for s in sections)
        merged["title"] = " | ".join(str(s.get("title", "")) for s in sections)
        merged["parent_titles"] = first.get("parent_titles", [])

        merged["start_page"] = min([s.get("start_page") for s in sections if s.get("start_page") is not None], default=None)
        merged["end_page"] = max([s.get("end_page") for s in sections if s.get("end_page") is not None], default=None)

        merged["text_list"] = []
        merged["table_list"] = []
        for s in sections:
            merged["text_list"].extend(s.get("text_list", []) or [])
            merged["table_list"].extend(s.get("table_list", []) or [])

        merged["n_tokens_text"] = sum(get_text_tokens(s) for s in sections)
        merged["n_tokens_tabs"] = sum(get_table_tokens(s) for s in sections)
        merged["n_tokens_total"] = merged["n_tokens_text"] + merged["n_tokens_tabs"]
        merged["section_id"] = "+".join(str(s.get("section_id", "")) for s in sections)

        return merged


    @staticmethod
    def add_navigation(chunks: list[Chunk]) -> None:
        for idx, chunk in enumerate(chunks):
            chunk.prev_chunk_id = chunks[idx - 1].chunk_id if idx > 0 else None
            chunk.next_chunk_id = chunks[idx + 1].chunk_id if idx < len(chunks) - 1 else None


# ============================================================
# 11. IO + STATS
# ============================================================

def load_document(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    if isinstance(doc, list):
        return {"doc_id": path.stem, "document_id": path.stem, "sections": doc}

    if "sections" not in doc:
        raise ValueError("Input JSON must contain a 'sections' field or be a bare list of sections.")

    return doc


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)


def print_stats(chunks: list[Chunk], input_sections: int) -> None:
    from collections import Counter
    import statistics

    token_counts = [c.token_count for c in chunks]
    strategy_counts = Counter(c.chunk_strategy for c in chunks)
    type_counts = Counter(c.content_type for c in chunks)

    def band(n: int) -> str:
        if n < 10:
            return "0-10"
        if n < 80:
            return "10-80"
        if n <= 256:
            return "80-256"
        if n <= 480:
            return "257-480"
        return "480+"

    band_counts = Counter(band(n) for n in token_counts)

    print("\n" + "=" * 70)
    print("UNIVERSAL PRE-CHUNKING REPORT")
    print("=" * 70)
    print(f"Input sections : {input_sections}")
    print(f"Output chunks  : {len(chunks)}")
    if chunks:
        print(f"Compression    : {input_sections / len(chunks):.2f}x")

    if token_counts:
        print("\nToken stats:")
        print(f"Mean   : {statistics.mean(token_counts):.2f}")
        print(f"Median : {statistics.median(token_counts):.2f}")
        print(f"Min    : {min(token_counts)}")
        print(f"Max    : {max(token_counts)}")

    print("\nToken bands:")
    for key in ["0-10", "10-80", "80-256", "257-480", "480+"]:
        print(f"{key:<10} {band_counts[key]}")

    print("\nBy strategy:")
    for key, value in strategy_counts.most_common():
        print(f"{key:<24} {value}")

    print("\nBy content type:")
    for key, value in type_counts.most_common():
        print(f"{key:<24} {value}")

    oversized = [c for c in chunks if c.token_count > 480]
    if oversized:
        print(f"\nWARNING: {len(oversized)} chunks exceed 480 tokens.")
        print("This can happen when context_prefix is large or a table line is huge.")
        for c in oversized[:10]:
            print(f"- {c.chunk_id}: {c.token_count} tokens ({c.content_type}, {c.chunk_strategy})")

    print("=" * 70)


# ============================================================
# 12. RUN
# ============================================================

def main() -> None:
    config = ChunkingConfig(
        tiny_threshold=10,
        min_tokens=80,
        target_tokens=256,
        max_tokens=480,
        overlap_ratio=0.20,
    )

    doc = load_document(INPUT_PATH)
    chunker = UniversalPreChunker(config)
    chunks = chunker.chunk(doc, input_path=INPUT_PATH)
    save_chunks(chunks, OUTPUT_PATH)

    if SHOW_STATS:
        print_stats(chunks, len(doc.get("sections", [])))

    print("\nSaved chunks to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
