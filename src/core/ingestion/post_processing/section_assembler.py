from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ============================================================
# CONFIG
# ============================================================


INPUT_PATH = Path("/app/src/storage/post_processed_blocks.json")
OUTPUT_PATH = Path("/app/src/storage/post_processed.json")


# ============================================================
# TEXT / BBOX HELPERS
# ============================================================

def clean_inline_text(text: str) -> str:
    """Normalize whitespace without changing the semantic content."""
    return re.sub(r"\s+", " ", text or "").strip()


def get_page_num(obj: dict[str, Any]) -> int | None:
    """Support both page_num and page keys."""
    return obj.get("page_num", obj.get("page"))


def get_bbox_top(obj: dict[str, Any]) -> float | None:
    """
    Return the TOP coordinate of a block in PDF space.

    PDF origin is bottom-left; y increases upward. The 'top' of a block in
    reading order is therefore the LARGER y value (y0 in our data).
    """
    bbox = obj.get("bbox") or {}
    return bbox.get("y0", bbox.get("top"))


def get_bbox_bottom(obj: dict[str, Any]) -> float | None:
    """
    Return the BOTTOM coordinate of a block in PDF space.

    The 'bottom' of a block in reading order is the SMALLER y value (y1).
    """
    bbox = obj.get("bbox") or {}
    return bbox.get("y1", bbox.get("bottom"))


# ============================================================
# HEADING EXTRACTION — LOCAL NUMBER + PART CONTEXT
# ============================================================

def extract_local_section_number(title: str) -> tuple[str, str]:
    """
    Extract the *local* section identifier and its kind.

    Returns (local_number, kind) where kind is one of:
      "part"   – Roman numeral Part heading  (I, II, III …)
      "annex"  – Annex letter heading        (A, B, C …)
      "numeric"– Dotted numeric heading      (1, 1.1, 10.2.3 …)
      "annex_sub" – Sub-heading under an Annex  (B1, C2 …)
      ""       – Unrecognised, no number

    Examples
    --------
    "Part I General"              -> ("I",     "part")
    "Part I - General"            -> ("I",     "part")
    "1 Scope"                     -> ("1",     "numeric")
    "1.1 Changes in Version 4.4"  -> ("1.1",   "numeric")
    "10.1.1 Application …"        -> ("10.1.1","numeric")
    "Annex A Removed …"           -> ("A",     "annex")
    "Annex B Data Elements Table" -> ("B",     "annex")
    "B1 Data Elements by Name"    -> ("B1",    "annex_sub")
    "C3 Multi-Level Directory"    -> ("C3",    "annex_sub")
    """
    title = clean_inline_text(title)

    # Part heading  (Part I, Part III, Part IV …)
    m = re.match(
        r"^\s*(?:PART|Part|SECTION|Section)\s*[-–]?\s*([IVXLCDM]+)\b",
        title,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper(), "part"

    # Annex / Appendix heading  (Annex A …, Appendix B …)
    m = re.match(
        r"^\s*(?:ANNEX|Annex|APPENDIX|Appendix)\s+([A-Z])\b",
        title,
    )
    if m:
        return m.group(1).upper(), "annex"

    # Annex sub-heading  (B1, C2, A1.2 …) – single capital letter followed by digit(s)
    m = re.match(r"^\s*([A-Z]\d+(?:\.\d+)*)\s+\S", title)
    if m:
        return m.group(1), "annex_sub"

    # Purely numeric dotted number (1, 1.1, 10.2.3 …)
    m = re.match(r"^\s*(\d+(?:\.\d+)*)\s+\S", title)
    if m:
        return m.group(1), "numeric"

    return "", ""


def remove_local_number_from_title(title: str, local_number: str, kind: str) -> str:
    """Return a clean title with the leading section indicator stripped."""
    title = clean_inline_text(title)
    if not local_number:
        return title

    # Strip "Part X" or "Part X - " prefix
    if kind == "part":
        title = re.sub(
            r"^\s*(?:PART|Part|SECTION|Section)\s*[-–]?\s*[IVXLCDM]+\s*[-–]?\s*",
            "",
            title,
            flags=re.IGNORECASE,
        )
        return clean_inline_text(title)

    # Strip "Annex X" prefix
    if kind == "annex":
        title = re.sub(
            r"^\s*(?:ANNEX|Annex|APPENDIX|Appendix)\s+[A-Z]\s*",
            "",
            title,
        )
        return clean_inline_text(title)

    # Strip leading number / annex-sub code
    escaped = re.escape(local_number)
    title = re.sub(rf"^\s*{escaped}\s+", "", title)
    return clean_inline_text(title)


# ============================================================
# FULL SECTION-NUMBER BUILDER
# ============================================================

# Mapping from roman-numeral string to "PartX" label
_ROMAN_TO_PART: dict[str, str] = {
    "I":    "PartI",
    "II":   "PartII",
    "III":  "PartIII",
    "IV":   "PartIV",
    "V":    "PartV",
    "VI":   "PartVI",
    "VII":  "PartVII",
    "VIII": "PartVIII",
    "IX":   "PartIX",
    "X":    "PartX",
}


def build_headings_from_blocks(text_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract headings from section_header blocks.

    Stores both the local_number (bare identifier) and kind on each heading
    so the hierarchy pass can resolve parents correctly.
    """
    headings: list[dict[str, Any]] = []

    for block in text_blocks:
        if block.get("type") != "section_header":
            continue

        raw_title = clean_inline_text(block.get("text", ""))
        local_number, kind = extract_local_section_number(raw_title)

        headings.append(
            {
                "doc_id":       block.get("doc_id", ""),
                "doc_title" :   block.get("doc_title", ""),
                "doc_version":  block.get("doc_version", ""),
                "doc_date":     block.get("doc_date", ""),
                "page_num":     get_page_num(block),
                "title":        remove_local_number_from_title(raw_title, local_number, kind),
                "raw_title":    raw_title,
                "local_number": local_number,  # bare identifier, used for parent lookup
                "kind":         kind,
                "bbox":         block.get("bbox", {}),
            }
        )

    return headings


def add_heading_ids_and_parents(headings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add heading_id, parent_heading_id, and section_number (full path) to each heading.

    Sort order
    ----------
    PDF y-coordinates increase *upward* (origin at bottom-left).
    To get top-to-bottom reading order we sort:
      1. page_num  ASCENDING
      2. bbox y0   DESCENDING  (larger y0 = higher on page = earlier in reading order)
    """
    headings_sorted = sorted(
        headings,
        key=lambda h: (
            h.get("page_num", 0),
            -(get_bbox_top(h) if get_bbox_top(h) is not None else 0),  # descending y0
        ),
    )

    enriched: list[dict[str, Any]] = []

    # ── Registry of recent headings by (kind, local_number) ──────────────────
    # part      : {"I": heading, "III": heading …}
    # annex     : {"A": heading, "B": heading …}
    # annex_sub : {"B1": heading, "C2": heading …}
    # numeric   : {"1": h, "1.1": h, "10": h, "10.1": h …}

    by_local: dict[str, dict[str, Any]] = {}   # local_number -> heading (most recent)
    current_part_heading: dict[str, Any] | None = None
    current_annex_heading: dict[str, Any] | None = None

    for idx, heading in enumerate(headings_sorted, start=1):
        h = dict(heading)
        local_number = h.get("local_number", "")
        kind = h.get("kind", "")

        # ── heading_id ────────────────────────────────────────────────────────
        doc_id = clean_inline_text(h.get("doc_id", "DOC")).replace(" ", "_") or "DOC"
        doc_version = clean_inline_text(h.get("doc_version", "")).replace(" ", "_")
        h["heading_id"] = (
            f"{doc_id}_{doc_version}"
            f"_p{int(h.get('page_num') or 0):04d}_h{idx:04d}"
        )

        # ── parent resolution ─────────────────────────────────────────────────
        parent_heading: dict[str, Any] | None = None

        if kind == "part":
            # Top-level: no parent. Update part context.
            parent_heading = None
            current_part_heading = h
            current_annex_heading = None

        elif kind == "annex":
            # Parent = current Part (e.g. Part IV)
            parent_heading = current_part_heading
            current_annex_heading = h

        elif kind == "annex_sub":
            # "B1", "C3" etc. — parent is the Annex whose letter matches prefix
            annex_letter = local_number[0]  # e.g. "B" from "B1"
            parent_heading = by_local.get(annex_letter) or current_annex_heading

        elif kind == "numeric":
            if "." in local_number:
                # Dotted: parent is the shorter number  e.g. "1.1" -> "1", "10.1.1" -> "10.1"
                parent_local = local_number.rsplit(".", 1)[0]
                parent_heading = by_local.get(parent_local)
            else:
                # Top-level chapter  e.g. "1", "10" — parent is current Part
                parent_heading = current_part_heading

        h["parent_heading_id"] = parent_heading["heading_id"] if parent_heading else None

        # ── full section_number (path from root) ─────────────────────────────
        h["section_number"] = _compute_full_section_number(h, parent_heading)

        # ── register for future parent lookups ───────────────────────────────
        if local_number:
            by_local[local_number] = h

        # Update context trackers
        if kind == "part":
            current_part_heading = h
        if kind == "annex":
            current_annex_heading = h

        enriched.append(h)

    return enriched


def _compute_full_section_number(
    heading: dict[str, Any],
    parent_heading: dict[str, Any] | None,
) -> str:
    """
    Derive the full dot-separated section number that includes the Part prefix.

    Examples
    --------
    Part I                        ->  PartI
    1 Scope       (under Part I)  ->  PartI.1
    1.1 Changes   (under 1)       ->  PartI.1.1
    Annex B       (under Part IV) ->  PartIV.AnnexB
    B1 …          (under Annex B) ->  PartIV.AnnexB.B1
    """
    local_number = heading.get("local_number", "")
    kind = heading.get("kind", "")

    if not local_number:
        return ""

    if kind == "part":
        return _ROMAN_TO_PART.get(local_number, f"Part{local_number}")

    if kind == "annex":
        part_prefix = (
            parent_heading["section_number"]
            if parent_heading and parent_heading.get("section_number")
            else "Part"
        )
        return f"{part_prefix}.Annex{local_number}"

    # annex_sub or numeric
    if parent_heading and parent_heading.get("section_number"):
        # Append only the leaf segment of the local number.
        # e.g. local_number="1.1", parent already encoded "PartI.1"
        #      -> append "1" (the last dotted segment) -> "PartI.1.1"
        leaf = local_number.rsplit(".", 1)[-1] if "." in local_number else local_number
        return f"{parent_heading['section_number']}.{leaf}"

    # Fallback — no parent resolved, use bare local number
    return local_number


# ============================================================
# SPAN COMPUTATION
# ============================================================

def add_heading_spans(
    headings: list[dict[str, Any]],
    last_page_num: int,
) -> list[dict[str, Any]]:
    """
    Add start / end span coordinates to each heading.

    Sorting note (same as add_heading_ids_and_parents):
    PDF y-coordinates increase upward, so reading-order sort requires
    DESCENDING y0 within the same page.

    Each heading's content span runs from just below its own bottom edge
    (start_top = y1 of this heading) down to just above the next heading's
    top edge (end_top = y0 of next heading).
    """
    hs = sorted(
        headings,
        key=lambda h: (
            h.get("page_num", 0),
            -(get_bbox_top(h) if get_bbox_top(h) is not None else 0),  # descending y0
        ),
    )

    for i, heading in enumerate(hs):
        heading["start_page"] = heading.get("page_num")
        heading["start_top"]  = get_bbox_bottom(heading)   # y1 of this heading

        next_heading = hs[i + 1] if i + 1 < len(hs) else None

        if next_heading:
            heading["end_page"] = next_heading.get("page_num")
            heading["end_top"]  = get_bbox_top(next_heading)  # y0 of next heading
        else:
            heading["end_page"] = last_page_num
            heading["end_top"]  = None

    return hs


# ============================================================
# PARENT TITLE CHAIN
# ============================================================

def build_parent_titles(
    heading: dict[str, Any],
    heading_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    """Return parent titles from root down to direct parent."""
    titles: list[str] = []
    parent_id = heading.get("parent_heading_id")

    while parent_id:
        parent = heading_by_id.get(parent_id)
        if not parent:
            break
        title = clean_inline_text(parent.get("title", ""))
        if title:
            titles.append(title)
        parent_id = parent.get("parent_heading_id")

    return list(reversed(titles))


# ============================================================
# SPAN MEMBERSHIP
# ============================================================

def block_belongs_to_heading_span(
    block: dict[str, Any],
    heading: dict[str, Any],
) -> bool:
    """
    Check whether a text block falls within a heading's content span.

    Coordinate convention (PDF bottom-left origin, y upward):
    - start_top = y1 of heading (bottom edge of heading text, START of content area)
    - end_top   = y0 of next heading (top edge of next heading, END of content area)

    A block is *included* when:
      - its bottom edge (y1) is NOT above start_top  (block is at or below heading)
      - its top edge   (y0) is NOT below end_top     (block is at or above next heading)
    """
    page_num = get_page_num(block)
    top      = get_bbox_top(block)      # y0
    bottom   = get_bbox_bottom(block)   # y1

    start_page = heading.get("start_page")
    end_page   = heading.get("end_page")
    start_top  = heading.get("start_top")   # y1 of heading (content starts here)
    end_top    = heading.get("end_top")     # y0 of next heading (content ends here)

    if page_num is None or start_page is None or end_page is None:
        return False

    if page_num < start_page or page_num > end_page:
        return False

    if start_page == end_page:
        # Exclude blocks whose BOTTOM is above start_top (i.e. sits above the heading itself)
        if start_top is not None and bottom is not None and bottom > start_top:
            return False
        # Exclude blocks whose TOP is below end_top (i.e. sits below the next heading)
        if end_top is not None and top is not None and top < end_top:
            return False
        return True

    if page_num == start_page:
        if start_top is not None and bottom is not None and bottom > start_top:
            return False
        return True

    if page_num == end_page:
        if end_top is not None and top is not None and top < end_top:
            return False
        return True

    # Middle pages — always included
    return True


# ============================================================
# Table Handling
# ============================================================

def is_table_caption(block: dict[str, Any]) -> bool:
    if block.get("type") != "caption":
        return False

    text = clean_inline_text(block.get("text", ""))

    return bool(
        re.match(r"^\s*Table\s+\d+\b.*", text, re.IGNORECASE)
    )


def table_caption_for_table(
    table: dict[str, Any],
    text_blocks: list[dict[str, Any]],
    max_gap: float = 60.0,
) -> dict[str, Any] | None:

    table_page = table.get("page_num", table.get("page"))
    table_bbox = table.get("bbox") or {}

    table_top = max(table_bbox.get("y0", 0), table_bbox.get("y1", 0))
    table_bottom = min(table_bbox.get("y0", 0), table_bbox.get("y1", 0))

    candidates = []

    for block in text_blocks:
        if get_page_num(block) != table_page:
            continue

        if not is_table_caption(block):
            continue

        cap_top = get_bbox_top(block)
        cap_bottom = get_bbox_bottom(block)

        if cap_top is None or cap_bottom is None:
            continue

        # caption above table
        gap_above = cap_bottom - table_top

        # caption below table
        gap_below = table_bottom - cap_top

        if 0 <= gap_above <= max_gap:
            candidates.append((gap_above, block))

        if 0 <= gap_below <= max_gap:
            candidates.append((gap_below, block))

    if not candidates:
        return None

    return min(candidates, key=lambda x: x[0])[1]


def previous_text_block_before_table(
    table: dict[str, Any],
    text_blocks: list[dict[str, Any]],
) -> dict[str, Any] | None:

    table_page = table.get("page_num", table.get("page"))
    table_bbox = table.get("bbox") or {}
    table_top = max(table_bbox.get("y0", 0), table_bbox.get("y1", 0))

    candidates = []

    for block in text_blocks:
        if get_page_num(block) != table_page:
            continue

        if block.get("type") in {"caption", "page_header", "page_footer"}:
            continue

        block_bottom = get_bbox_bottom(block)

        if block_bottom is None:
            continue

        # block visually above table
        if block_bottom >= table_top:
            candidates.append(block)

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda b: abs(get_bbox_bottom(b) - table_top)
    )


def normalize_table(table: dict[str, Any], text_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    caption_block = table_caption_for_table(table, text_blocks)
    previous_block = previous_text_block_before_table(table, text_blocks)

    caption_text = caption_block.get("text", "") if caption_block else ""
    previous_text = previous_block.get("text", "") if previous_block else ""

    return {
        "page_num": table.get("page_num", table.get("page")),
        "bbox": table.get("bbox", {}),
        "caption": caption_text,
        "previous_text": previous_text,
        "markdown": table.get("markdown", ""),
        "raw_matrix": table.get("raw_matrix", []),
        "rows": table.get("rows"),
        "cols": table.get("cols"),
    }


def table_belongs_to_heading_span(table: dict[str, Any], heading: dict[str, Any]) -> bool:
    fake_block = {
        "page_num": table.get("page_num", table.get("page")),
        "bbox": table.get("bbox", {}),
    }

    return block_belongs_to_heading_span(fake_block, heading)

# ============================================================
# SECTION ASSEMBLY
# ============================================================

def heading_to_section_id(heading: dict[str, Any]) -> str:
    """Generate a stable section id from document metadata and heading info."""
    doc_id = clean_inline_text(heading.get("doc_id", "DOC")).replace(" ", "_") or "DOC"
    version = clean_inline_text(heading.get("doc_version", "")).replace(" ", "_")
    page_num = heading.get("page_num", "")
    section_number = clean_inline_text(heading.get("section_number", ""))

    if section_number:
        safe_section = re.sub(r"[^A-Za-z0-9.]+", "_", section_number).strip("_")
    else:
        safe_section = (
            clean_inline_text(heading.get("heading_id", "section")).replace(" ", "_")
        )

    return f"{doc_id}_{version}_p{page_num}_{safe_section}"


def normalize_text_block(block: dict[str, Any]) -> dict[str, Any]:
    """Keep each original block as one row-like item inside text_blocks."""
    text = block.get("text", "")
    return {
        "doc_id":       block.get("doc_id", ""),
        "doc_title":    block.get("doc_title", ""),
        "doc_version":  block.get("doc_version", ""),
        "doc_date":     block.get("doc_date", ""),
        "page_num":     get_page_num(block),
        "type":         block.get("type", ""),
        "text":         text,
        "bbox":         block.get("bbox", {}),
        "char_count":   block.get("char_count", len(text)),
    }


def assemble_sections(
    text_blocks: list[dict[str, Any]],
    headings: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Assemble text blocks under heading spans.

    Output per section:
    - text_list : list of text strings, one item per block
    - full_text : newline-joined text_list
    """
    heading_by_id = {
        h["heading_id"]: h
        for h in headings
        if h.get("heading_id")
    }

    sections: list[dict[str, Any]] = []
    
    for heading in headings:
        heading_raw_title   = clean_inline_text(heading.get("raw_title", heading.get("title", "")))
        heading_clean_title = clean_inline_text(heading.get("title", ""))

        section_blocks: list[dict[str, Any]] = []

        section_tables = [
            normalize_table(table, text_blocks)
            for table in tables
            if table_belongs_to_heading_span(table, heading)
        ]
        for block in text_blocks:
            if not block_belongs_to_heading_span(block, heading):
                continue

            block_text = clean_inline_text(block.get("text", ""))

            # Skip next/other section titles from current section
            if block.get("type") == "section_header" and block_text != heading_raw_title:
                continue

            section_blocks.append(normalize_text_block(block))

        text_list = [
            clean_inline_text(b.get("text", ""))
            for b in section_blocks
            if clean_inline_text(b.get("text", ""))
        ]

        section_title_for_text = clean_inline_text(heading.get("raw_title", "")) or heading_clean_title

        if section_title_for_text:
            text_list = [section_title_for_text] + [
                txt for txt in text_list
                if clean_inline_text(txt) != section_title_for_text
            ]

        table_list = []

        for table in section_tables:
            if table.get("previous_text"):
                table_list.append(clean_inline_text(table["previous_text"]))

            if table.get("caption"):
                table_list.append(clean_inline_text(table["caption"]))

            if table.get("markdown"):
                table_list.append(table["markdown"])

        # --------------------------------------------------
        # Remove duplicates from text_list
        # --------------------------------------------------

        table_entries = {
            clean_inline_text(x)
            for x in table_list
            if clean_inline_text(x)
        }

        text_list = [
            txt
            for txt in text_list
            if clean_inline_text(txt) not in table_entries
        ]

        section_obj = {
            "heading_id":        heading.get("heading_id"),
            "parent_heading_id": heading.get("parent_heading_id"),
            "doc_id":            heading.get("doc_id"),
            "doc_title":         heading.get("doc_title"),
            "doc_version":       heading.get("doc_version"),
            "doc_date":          heading.get("doc_date"),
            "section_number":    heading.get("section_number", ""),
            "title":             heading.get("title", ""),
            "parent_titles":     build_parent_titles(heading, heading_by_id),
            "start_page":        heading.get("start_page"),
            "end_page":          heading.get("end_page"),
            "start_top":         heading.get("start_top"),
            "end_top":           heading.get("end_top"),
            # "text_blocks": section_blocks,  # uncomment to include raw blocks
            "text_list":         text_list,
            "table_list":        table_list,
            #"full_text":         "\n".join(text_list + table_list).strip(),
            #"tables": section_tables,

        }

        section_obj["section_id"] = heading_to_section_id(heading)

        sections.append(section_obj)

    return sections


# ============================================================
# RUNNER
# ============================================================

def main() -> None:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        blocks_data = json.load(f)

    text_blocks = blocks_data.get("text_blocks", [])
    tables = blocks_data.get("tables", [])
    last_page_num = max(
        [get_page_num(block) or 1 for block in text_blocks]
        or [1]
    )

    headings = build_headings_from_blocks(text_blocks)
    headings = add_heading_ids_and_parents(headings)
    headings = add_heading_spans(headings, last_page_num)

    sections = assemble_sections(
        text_blocks=text_blocks,
        headings=headings,
        tables=tables,
    )

    final_output = {
        "doc_id":        text_blocks[0].get("doc_id", "") if text_blocks else "",
        "doc_title": text_blocks[0].get("doc_title", "") if text_blocks else "",
        "doc_version":   text_blocks[0].get("doc_version", "") if text_blocks else "",
        "doc_date":      text_blocks[0].get("doc_date", "") if text_blocks else "",
        "headings_count": len(headings),
        "sections_count": len(sections),
        # "headings": headings,  # uncomment to include raw headings
        "sections": sections,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"Headings : {len(headings)}")
    print(f"Sections : {len(sections)}")
    print(f"Saved to : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
