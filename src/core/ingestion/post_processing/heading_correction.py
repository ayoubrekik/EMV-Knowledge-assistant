from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from ..extraction.extraction import TextBlock, BoundingBox


@dataclass
class DefinitionEntry:
    page_number: int
    term: str
    definition: str
    list_items: list[str] = field(default_factory=list)
    term_bbox: Optional[BoundingBox] = None
    definition_bbox: Optional[BoundingBox] = None

    def to_text(self) -> str:
        lines = [f"{self.term}: {self.definition}".strip()]
        for item in self.list_items:
            lines.append(f"  • {item}")
        return "\n".join(lines)


class LayoutAwareBlockCorrector:
    """
    Post-processes Docling text blocks.

    Current behavior:
    - Keeps original layout/order structure.
    - Validates headings.
    - If a block is classified as section_header but does not match
      a real heading pattern, it is changed to text.
    """

    HEADING_TYPES = {"section_header"}

    HEADING_PATTERNS = [
        r"^\s*(PART|Part|SECTION|Section)\s+[IVXLCDM\d]+\b.*$",
        r"^\s*\d+(\.\d+)*\s+.+$",
        r"^\s*(ANNEX|Annex|APPENDIX|Appendix)\s+[A-Z\d]+\b.*$",
        r"^\s*[A-Z]\d+(\.\d+)*\s+.+$",
        r"^\s*[A-Z]\s+\d+(\.\d+)*\s+.+$",
        r"^\s*[A-Z](\.\d+)+\s+.+$",
        
    ]

    def correct(
        self,
        text_blocks: list[TextBlock],
    ) -> tuple[list[TextBlock], list[DefinitionEntry]]:

        corrected_blocks: list[TextBlock] = []
        pages: dict[int, list[TextBlock]] = defaultdict(list)

        for block in text_blocks:
            pages[block.page_number].append(block)

        toc_like_pages = set()

        for page_num in sorted(pages):
            page_blocks = self._sort_blocks(pages[page_num])

            if self._is_toc_lof_lot_page(page_blocks):
                toc_like_pages.add(page_num)

            corrected_blocks.extend(page_blocks)

        corrected_blocks = self._validate_headings(
            corrected_blocks,
            skip_pages=toc_like_pages,
        )

        corrected_blocks = self._remove_page_headers_footers(corrected_blocks)

        return corrected_blocks, []

    def _validate_headings(
        self,
        blocks: list[TextBlock],
        skip_pages: set[int] | None = None,
    ) -> list[TextBlock]:

        skip_pages = skip_pages or set()

        for block in blocks:
            if block.page_number in skip_pages:
                continue

            if block.block_type in self.HEADING_TYPES:
                if not self._matches_heading_pattern(block.text):
                    block.block_type = "text"

        return blocks

    def _matches_heading_pattern(self, text: str) -> bool:
        return any(
            re.match(pattern, text)
            for pattern in self.HEADING_PATTERNS
        )
    
    def _remove_page_headers_footers(
            self,
            blocks: list[TextBlock],
        ) -> list[TextBlock]:

            return [
                block
                for block in blocks
                if block.block_type not in {"page_header", "page_footer", "page_number"}
                and not (
                    block.block_type == "footnote" 
                    and (
                        block.text.strip().startswith("©") 
                        or len(block.text.split()) < 4
                    )
                )
            ]

    def _is_toc_lof_lot_page(self, blocks: list[TextBlock]) -> bool:
        page_text = "\n".join(
            block.text.strip()
            for block in blocks
            if block.text and block.text.strip()
        ).lower()

        patterns = [
            r"\btable\s+of\s+contents\b",
            r"\bdocument\s+of\s+structure\b",
            r"\bcontents\b",
            r"\blist\s+of\s+figures\b",
            r"\blist\s+of\s+tables\b",
            r"\btables\b",
            r"\bfigures\b",
        ]

        return any(re.search(pattern, page_text) for pattern in patterns)

    @staticmethod
    def _sort_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
        return sorted(
            blocks,
            key=lambda b: (b.page_number, -b.bbox.y0, b.bbox.x0),
        )

class UnnumberedHeadingBlockCorrector(LayoutAwareBlockCorrector):
    """
    Corrector for documents where headings are not numbered.

    Example:
    - Why Do People Use Python?
    - Software quality
    - Developer productivity
    - Numbers
    - Strings
    """

    def _looks_like_unnumbered_heading(self, block: TextBlock) -> bool:
        text = block.text.strip()
        words = text.split()

        if not text:
            return False

        if block.block_type in {
            "caption",
            "table",
            "table_cell",
            "page_header",
            "page_footer",
            "page_number",
            "footnote",
        }:
            return False

        # Avoid paragraphs
        if len(words) > 10:
            return False

        if len(text) > 90:
            return False

        # Avoid normal sentences
        if text.endswith((".", ",", ";", ":")):
            return False

        # Avoid bullet/list items
        if text.startswith(("•", "-", "*", "–")):
            return False

        # Avoid code-like lines
        if text.startswith((">>>", "...")):
            return False

        # Accept simple title-like blocks
        return True

    def _validate_headings(
        self,
        blocks: list[TextBlock],
        skip_pages: set[int] | None = None,
    ) -> list[TextBlock]:

        skip_pages = skip_pages or set()

        for block in blocks:
            if block.page_number in skip_pages:
                continue

            text = block.text.strip()

            # Case 1: Docling already detected a heading
            if block.block_type in self.HEADING_TYPES:
                if (
                    self._matches_heading_pattern(text)
                    or self._looks_like_unnumbered_heading(block)
                ):
                    block.block_type = "section_header"
                else:
                    block.block_type = "text"

        return blocks

def detect_numbered_document(text_blocks: list[TextBlock]) -> bool:
    """
    Detect document style using only Docling's classification.

    True  -> most Docling section_header blocks are numbered
    False -> most Docling section_header blocks are unnumbered
    """

    heading_patterns = [
        r"^\s*(PART|Part|SECTION|Section)\s+[IVXLCDM\d]+\b.*$",
        r"^\s*\d+(\.\d+)*\s+.+$",
        r"^\s*(ANNEX|Annex|APPENDIX|Appendix)\s+[A-Z\d]+\b.*$",
        r"^\s*[A-Z]\d+(\.\d+)*\s+.+$",
        r"^\s*[A-Z]\s+\d+(\.\d+)*\s+.+$",
        r"^\s*[A-Z](\.\d+)+\s+.+$",
    ]

    docling_headings = [
        block.text.strip()
        for block in text_blocks
        if block.block_type == "section_header" and block.text.strip()
    ]

    if not docling_headings:
        return False

    numbered_count = sum(
        1
        for text in docling_headings
        if any(re.match(pattern, text) for pattern in heading_patterns)
    )

    numbered_ratio = numbered_count / len(docling_headings)

    return numbered_ratio >= 0.60