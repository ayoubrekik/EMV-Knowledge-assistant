from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """Normalised (x0, y0, x1, y1) in PDF-space points."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class TextBlock:
    page_number: int
    text: str
    bbox: BoundingBox
    block_type: str = "paragraph"
    doc_id: str = ""
    doc_title: str = ""
    doc_version: str = ""
    doc_date: str = ""
    font_sizes: list[float] = field(default_factory=list)
    font_names: list[str] = field(default_factory=list)
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class TableCell:
    row: int
    col: int
    text: str
    is_header: bool = False
    row_span: int = 1
    col_span: int = 1


@dataclass
class TableData:
    """One extracted table."""
    page_number: int
    table_index: int            # 0-based index among tables on this page
    bbox: BoundingBox
    cells: list[TableCell]
    headers: list[str]          # first-row text, or empty
    markdown: str               # pre-rendered markdown representation
    raw_matrix: list[list[str]] # raw 2-D cell matrix (may contain None)
    caption: str = ""
    source: str = "docling"     # docling | pdfplumber_fallback

    @property
    def num_rows(self) -> int:
        return max((c.row for c in self.cells), default=0) + 1

    @property
    def num_cols(self) -> int:
        return max((c.col for c in self.cells), default=0) + 1


@dataclass
class ImageData:
    """One extracted image / figure."""
    page_number: int
    image_index: int            # 0-based index among images on this page
    bbox: BoundingBox
    image_bytes: bytes          # raw PNG bytes
    image_hash: str             # sha256 for deduplication
    width_px: int
    height_px: int
    color_space: str            # RGB | GRAY | CMYK | unknown
    format: str                 # PNG | JPEG | …
    xref: int                   # PyMuPDF internal cross-reference id
    caption: str = ""           # filled by preprocessing layer (VLM)
    is_diagram: bool = False    # heuristic: True when image looks like a figure


@dataclass
class SectionNode:
    """One node in the logical document tree (heading or leaf paragraph)."""
    level: int                  # 1 = top-level chapter, 2 = section, … 0 = body text
    title: str
    page_number: int
    bbox: BoundingBox | None
    children: list["SectionNode"] = field(default_factory=list)
    text_chunk: str = ""        # body text directly under this heading (before sub-sections)

    def to_path(self) -> str:
        """Returns a breadcrumb like '2. Methods > 2.3 Results'."""
        return self.title  # caller stitches the full path from the tree walk


@dataclass
class DocumentStructure:
    """High-level outline + metadata about the PDF."""
    title: str
    authors: list[str]
    num_pages: int
    language: str
    is_scanned: bool
    section_tree: list[SectionNode]
    raw_headings: list[dict[str, Any]]  # flat list for simpler downstream use
    source: str = "docling"


@dataclass
class ExtractionResult:
    """Aggregated output of the full extraction pipeline for one PDF."""
    source_path: str
    text_blocks: list[TextBlock]
    tables: list[TableData]
    images: list[ImageData]
    structure: DocumentStructure
    extraction_errors: list[str] = field(default_factory=list)

    # Convenience helpers
    @property
    def page_count(self) -> int:
        return self.structure.num_pages

    @property
    def has_tables(self) -> bool:
        return len(self.tables) > 0

    @property
    def has_images(self) -> bool:
        return len(self.images) > 0


# ─────────────────────────────────────────────
# 2. TEXT EXTRACTOR  (pdfplumber)
# ─────────────────────────────────────────────

class DoclingContentExtractor:
    """
    Extracts both text blocks and tables from one Docling conversion.

    This avoids:
    - converting the same PDF twice
    - extracting tables as normal text
    """



    def _page_has_figure_caption(self, doc, page_num: int) -> bool:
        import re

        caption_pattern = re.compile(
            r"^\s*(figure|fig\.?)\s+\d+([-.–]\d+)*\s*([:—-]\s*.+)?$",
            re.IGNORECASE,
        )

        for element in getattr(doc, "texts", []):
            text = getattr(element, "text", "").strip()

            if not text:
                continue

            if hasattr(element, "prov") and element.prov:
                element_page = int(getattr(element.prov[0], "page_no", 1))
            else:
                element_page = 1

            if element_page != page_num:
                continue

            # Caption must be a standalone line
            if "\n" in text:
                continue

            if caption_pattern.match(text):
                print(f"Detected figure caption: {text}")
                return True

        return False

    def _extract_picture_text_refs(self, doc) -> set[str]:
        picture_text_refs: set[str] = set()

        for pic in getattr(doc, "pictures", []):
            for child in getattr(pic, "children", []):
                cref = getattr(child, "cref", None)
                if cref:
                    picture_text_refs.add(str(cref))

        return picture_text_refs

    def _extract_picture_regions(self, doc) -> dict[int, list[BoundingBox]]:
        regions: dict[int, list[BoundingBox]] = {}

        picture_items = []

        if hasattr(doc, "pictures"):
            picture_items.extend(doc.pictures)

        if hasattr(doc, "figures"):
            picture_items.extend(doc.figures)

        for pic in picture_items:
            if not hasattr(pic, "prov") or not pic.prov:
                continue

            prov = pic.prov[0]
            page_num = int(getattr(prov, "page_no", 1))

            bb = getattr(prov, "bbox", None)
            if bb is None:
                continue

            bbox = BoundingBox(
                x0=float(bb.l),
                y0=float(bb.t),
                x1=float(bb.r),
                y1=float(bb.b),
            )

            regions.setdefault(page_num, []).append(bbox)

        return regions

                
    def extract(
        self,
        pdf_path: Path,
        image_regions_by_page: dict[int, list[BoundingBox]] | None = None,
        cancel_callback=None,
    ) -> tuple[list[TextBlock], list[TableData], list[str]]:

        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        blocks: list[TextBlock] = []
        tables: list[TableData] = []
        errors: list[str] = []
        if cancel_callback:
            cancel_callback()
        # IMPORTANT: convert None to empty dict before using setdefault
        image_regions_by_page = image_regions_by_page or {}

        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )

            result = converter.convert(str(pdf_path))
            doc = result.document
            # ADD HERE
            print("Pictures:", len(getattr(doc, "pictures", [])))
            print("Figures :", len(getattr(doc, "figures", [])))

            for i, pic in enumerate(getattr(doc, "pictures", [])):
                print(f"Picture {i}")
                print(type(pic))
                print(pic)

            doc_metadata = self._extract_doc_metadata(doc)

            picture_text_refs = self._extract_picture_text_refs(doc)


            docling_picture_regions = self._extract_picture_regions(doc)

            for page_num, regions in docling_picture_regions.items():
                image_regions_by_page.setdefault(page_num, []).extend(regions)

            total_pages = len(doc.pages) if hasattr(doc, "pages") else 0

            tables = self._extract_tables(doc)

            blocks = self._extract_text_blocks(
                doc,
                total_pages,
                doc_metadata,
                image_regions_by_page,
                picture_text_refs,
                cancel_callback=cancel_callback,
            )

        except Exception as exc:
            msg = f"[DoclingContentExtractor] failed: {exc}"
            logger.error(msg)
            errors.append(msg)

        return blocks, tables, errors

    def _extract_text_blocks(
        self,
        doc,
        total_pages: int,
        doc_metadata: dict,
        image_regions_by_page: dict[int, list[BoundingBox]],
        picture_text_refs: set[str],
        cancel_callback=None,
    ) -> list[TextBlock]:

        def bbox_overlap_ratio(a: BoundingBox, b: BoundingBox) -> float:
            ix0 = max(a.x0, b.x0)
            iy0 = max(a.y0, b.y0)
            ix1 = min(a.x1, b.x1)
            iy1 = min(a.y1, b.y1)

            iw = max(0.0, ix1 - ix0)
            ih = max(0.0, iy1 - iy0)

            inter_area = iw * ih

            if a.area <= 0:
                return 0.0

            return inter_area / a.area

        blocks: list[TextBlock] = []

        current_page = None
        page_start_time = None

        import time

        for element in doc.texts:
            element_ref = str(getattr(element, "self_ref", ""))

            if cancel_callback:
                cancel_callback()

            text = getattr(element, "text", "").strip()
            if not text:
                continue

            label = str(getattr(element, "label", "")).lower()

            if self._is_table_related_label(label):
                continue

            page_num = 1
            bbox = BoundingBox(0.0, 0.0, 0.0, 0.0)

            if hasattr(element, "prov") and element.prov:
                prov = element.prov[0]
                page_num = int(getattr(prov, "page_no", 1))

                bb = getattr(prov, "bbox", None)
                if bb is not None:
                    bbox = BoundingBox(
                        x0=float(bb.l),
                        y0=float(bb.t),
                        x1=float(bb.r),
                        y1=float(bb.b),
                    )

            page_has_figure_caption = self._page_has_figure_caption(doc, page_num)

            if element_ref in picture_text_refs and page_has_figure_caption:
                continue

            image_regions = image_regions_by_page.get(page_num, [])

            if page_has_figure_caption:
                if any(
                    bbox_overlap_ratio(bbox, img_bbox) >= 0.50
                    for img_bbox in image_regions
                ):
                    continue
            # ----------------------------------------------------

            if current_page is None:
                current_page = page_num
                page_start_time = time.time()

            elif page_num != current_page:
                elapsed = time.time() - page_start_time

                if total_pages:
                    print(
                        f"Page {current_page}/{total_pages} extracted "
                        f"in {elapsed:.2f}s"
                    )
                else:
                    print(
                        f"Page {current_page} extracted "
                        f"in {elapsed:.2f}s"
                    )

                current_page = page_num
                page_start_time = time.time()

            block_type = self._map_docling_label(label, text)

            blocks.append(
                TextBlock(
                    page_number=page_num,
                    text=self._clean_text(text),
                    bbox=bbox,
                    block_type=block_type,
                    doc_id=doc_metadata.get("doc_id", ""),
                    doc_title=doc_metadata.get("doc_title", ""),
                    doc_version=doc_metadata.get("doc_version", ""),
                    doc_date=doc_metadata.get("doc_date", ""),
                    font_sizes=[],
                    font_names=[],
                )
            )

        if current_page is not None and page_start_time is not None:
            elapsed = time.time() - page_start_time

            if total_pages:
                print(
                    f"Page {current_page}/{total_pages} extracted "
                    f"in {elapsed:.2f}s"
                )
            else:
                print(
                    f"Page {current_page} extracted "
                    f"in {elapsed:.2f}s"
                )

        return blocks  
          
    def _extract_doc_metadata(self, doc) -> dict:
        first_page_texts = []

        for element in doc.texts:
            text = getattr(element, "text", "").strip()
            if not text:
                continue

            page_num = 1
            if hasattr(element, "prov") and element.prov:
                page_num = int(getattr(element.prov[0], "page_no", 1))

            if page_num == 1:
                first_page_texts.append(text)

        doc_id = ""
        doc_version = ""
        doc_date = ""
        doc_title = ""

        # Book number
        book_index = None
        version_index = None

        for i, text in enumerate(first_page_texts):
            if re.search(r"\bBook\s+[\w\-]+\b", text, re.IGNORECASE):
                doc_id = text.strip()
                book_index = i
                break

        # Version + date
        for i, text in enumerate(first_page_texts):
            version_match = re.search(
                r"\b(?:Version|v)\s*([0-9]+(?:\.[0-9]+)*)\b",
                text,
                re.IGNORECASE,
            )
            if version_match:
                doc_version = version_match.group(1).strip()
                version_index = i
                break

        for text in first_page_texts:
            date_match = re.search(
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
                text,
                re.IGNORECASE,
            )
            if date_match:
                doc_date = date_match.group(0).strip()
                break

        # Title = text between Book number and Version
        if book_index is not None and version_index is not None and book_index < version_index:
            title_candidates = first_page_texts[book_index + 1:version_index]
            title_candidates = [
                t.strip()
                for t in title_candidates
                if t.strip()
            ]
            doc_title = " ".join(title_candidates)

        return {
            "doc_id": doc_id,
            "doc_version": doc_version,
            "doc_date": doc_date,
            "doc_title": doc_title,
        }

    def _extract_tables(self, doc) -> list[TableData]:

        tables: list[TableData] = []
        table_index_by_page: dict[int, int] = {}

        for table in doc.tables:
            page_num = 1
            bbox_raw = None

            if hasattr(table, "prov") and table.prov:
                prov = table.prov[0]
                page_num = int(getattr(prov, "page_no", 1))

                bb = getattr(prov, "bbox", None)
                if bb is not None:
                    bbox_raw = BoundingBox(
                        x0=float(bb.l),
                        y0=float(bb.t),
                        x1=float(bb.r),
                        y1=float(bb.b),
                    )

            if bbox_raw is None:
                bbox_raw = BoundingBox(0.0, 0.0, 0.0, 0.0)

            idx = table_index_by_page.get(page_num, 0)
            table_index_by_page[page_num] = idx + 1

            cells: list[TableCell] = []
            raw_matrix: list[list[str]] = []
            headers: list[str] = []

            grid = (
                table.data.grid
                if hasattr(table, "data") and hasattr(table.data, "grid")
                else []
            )

            for row_idx, row in enumerate(grid):
                raw_row: list[str] = []

                for col_idx, cell in enumerate(row):
                    cell_text = (
                        cell.text.strip()
                        if hasattr(cell, "text") and cell.text
                        else ""
                    )

                    is_hdr = (
                        getattr(cell, "column_header", False)
                        or getattr(cell, "row_header", False)
                    )

                    row_span = getattr(cell, "row_span", 1) or 1
                    col_span = getattr(cell, "col_span", 1) or 1

                    cells.append(
                        TableCell(
                            row=row_idx,
                            col=col_idx,
                            text=cell_text,
                            is_header=is_hdr,
                            row_span=row_span,
                            col_span=col_span,
                        )
                    )

                    raw_row.append(cell_text)

                    if row_idx == 0:
                        headers.append(cell_text)

                raw_matrix.append(raw_row)

            markdown = self._matrix_to_markdown(raw_matrix, headers)

            tables.append(
                TableData(
                    page_number=page_num,
                    table_index=idx,
                    bbox=bbox_raw,
                    cells=cells,
                    headers=headers,
                    markdown=markdown,
                    raw_matrix=raw_matrix,
                    source="docling",
                )
            )

        return tables

    @staticmethod
    def _is_table_related_label(label: str) -> bool:
        return (
            "table" in label
            or label in {
                "table",
                "table_cell",
                "table_row",
                "table_header",
            }
        )

    @staticmethod
    def _map_docling_label(label: str, text: str) -> str:
        # Only override captions heuristically
        if len(text) < 200 and re.match(
            r"^(figure|fig\.?|table|chart|diagram|image|photo)\s*[\d\w]",
            text.strip(),
            re.IGNORECASE,
        ):
            return "caption"

        # Preserve original Docling label
        return label

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        return text.strip()

    @staticmethod
    def _matrix_to_markdown(
        matrix: list[list[str]],
        headers: list[str]
    ) -> str:
        if not matrix:
            return ""

        rows = []

        for i, row in enumerate(matrix):
            rows.append("| " + " | ".join(str(c) for c in row) + " |")

            if i == 0 and headers:
                rows.append("|" + "|".join("---" for _ in row) + "|")

        return "\n".join(rows)


# ─────────────────────────────────────────────
# 4. IMAGE EXTRACTOR  (PyMuPDF)
# ─────────────────────────────────────────────

class ImageExtractor:
    """
    Extracts embedded images and figure candidates using PyMuPDF.

    Deduplication
    -------------
    Images are sha256-hashed; exact duplicates (e.g. the same logo on every
    page) are emitted only once — on their first occurrence.

    Filtering
    ---------
    Images smaller than MIN_AREA_PX2 pixels squared are skipped (icons,
    decorative elements, bullet graphics).

    Diagram heuristic
    -----------------
    An image is flagged as a diagram candidate when its colour entropy is low
    (few distinct colours), which is typical of line-art, schematics, and
    charts. This flag guides the preprocessing layer to use a diagram-specific
    VLM prompt.
    """

    MIN_AREA_PX2: int = 200 * 200        # ignore images smaller than this
    DIAGRAM_MAX_UNIQUE_COLORS: int = 64  # low colour count → likely diagram

    
    def extract(self, pdf_path: Path, cancel_callback=None) -> tuple[list[ImageData], list[str]]:
        """Return (images, errors)."""
        import fitz  # PyMuPDF

        images: list[ImageData] = []
        errors: list[str] = []
        seen_hashes: set[str] = set()

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            msg = f"[ImageExtractor] failed to open PDF: {exc}"
            logger.error(msg)
            return images, [msg]

        try:
            for page_num, page in enumerate(doc, start=1):
                if cancel_callback:
                    cancel_callback()
                try:
                    page_images = self._extract_page_images(
                        doc=doc,
                        page=page,
                        page_num=page_num,
                        seen_hashes=seen_hashes,
                    )
                    images.extend(page_images)
                except Exception as exc:  # noqa: BLE001
                    msg = f"[ImageExtractor] page {page_num}: {exc}"
                    logger.warning(msg)
                    errors.append(msg)
        finally:
            doc.close()

        return images, errors

    # ── private ──────────────────────────────

    def _extract_page_images(
        self,
        doc,
        page,
        page_num: int,
        seen_hashes: set[str],
    ) -> list[ImageData]:
        import fitz

        results: list[ImageData] = []
        image_list = page.get_images(full=True)
        img_index = 0

        for img_info in image_list:
            xref = img_info[0]  # cross-reference id

            # Get image bytes from the PDF cross-reference table
            try:
                base_image = doc.extract_image(xref)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not extract xref %d: %s", xref, exc)
                continue

            img_bytes = base_image["image"]
            img_format = base_image.get("ext", "png").upper()
            width_px = base_image.get("width", 0)
            height_px = base_image.get("height", 0)
            colorspace = base_image.get("colorspace", 1)
            color_space_name = self._colorspace_name(colorspace)

            # Size filter
            if width_px * height_px < self.MIN_AREA_PX2:
                continue

            # Deduplication
            img_hash = hashlib.sha256(img_bytes).hexdigest()
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            # Bounding box on the page (PDF-space points)
            bbox = self._get_image_bbox(page, xref, width_px, height_px)

            # Normalise to PNG for downstream uniform handling
            png_bytes = self._to_png(img_bytes, img_format)

            # Diagram heuristic
            is_diagram = self._looks_like_diagram(png_bytes)

            results.append(ImageData(
                page_number=page_num,
                image_index=img_index,
                bbox=bbox,
                image_bytes=png_bytes,
                image_hash=img_hash,
                width_px=width_px,
                height_px=height_px,
                color_space=color_space_name,
                format="PNG",
                xref=xref,
                is_diagram=is_diagram,
            ))
            img_index += 1

        return results

    def _get_image_bbox(self, page, xref: int, w: int, h: int) -> BoundingBox:
        """
        Try to find the image's bounding box on the page.
        Falls back to a zero bbox if the image position cannot be determined.
        """
        try:
            rects = page.get_image_rects(xref)
            if rects:
                r = rects[0]
                return BoundingBox(
                    x0=float(r.x0), y0=float(r.y0),
                    x1=float(r.x1), y1=float(r.y1),
                )
        except Exception:  # noqa: BLE001
            pass

        # Fallback: try get_image_bbox (older PyMuPDF API)
        try:
            info_list = page.get_image_info(xrefs=True)
            for info in info_list:
                if info.get("xref") == xref:
                    bb = info.get("bbox")
                    if bb:
                        return BoundingBox(
                            x0=float(bb[0]), y0=float(bb[1]),
                            x1=float(bb[2]), y1=float(bb[3]),
                        )
        except Exception:  # noqa: BLE001
            pass

        return BoundingBox(0.0, 0.0, float(w), float(h))

    @staticmethod
    def _to_png(img_bytes: bytes, fmt: str) -> bytes:
        """
        Convert image bytes to PNG for uniform downstream handling.
        Returns original bytes unchanged if Pillow is unavailable or
        the image is already PNG.
        """
        if fmt == "PNG":
            return img_bytes
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(img_bytes))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:  # noqa: BLE001
            return img_bytes

    def _looks_like_diagram(self, png_bytes: bytes) -> bool:
        """
        Heuristic: images with few unique colours are likely diagrams /
        schematics / charts rather than photographs.
        """
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(png_bytes)).convert("RGB")
            # Sample a 128×128 thumbnail for speed
            img.thumbnail((128, 128))
            pixels = list(img.getdata())
            unique = len(set(pixels))
            return unique <= self.DIAGRAM_MAX_UNIQUE_COLORS
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _colorspace_name(cs: int) -> str:
        return {1: "GRAY", 3: "RGB", 4: "CMYK"}.get(cs, "unknown")


# ─────────────────────────────────────────────
# 5. STRUCTURE EXTRACTOR  (Docling primary, pdfplumber font-heuristic fallback)
# ─────────────────────────────────────────────

class StructureExtractor:
    """
    Extracts the logical document structure: title, authors, headings, and
    section hierarchy.

    Primary: Docling's converter produces a labelled document model with
    explicit heading levels, body text, captions, etc.

    Fallback: when Docling is unavailable, we use a font-size heuristic over
    pdfplumber characters to approximate the heading hierarchy.
    """

    def extract(
        self, pdf_path: Path, text_blocks: list[TextBlock]
    ) -> tuple[DocumentStructure, list[str]]:
        """Return (structure, errors)."""
        errors: list[str] = []

        try:
            structure, docling_errors = self._extract_with_docling(pdf_path)
            errors.extend(docling_errors)
            return structure, errors
        except ImportError:
            logger.warning(
                "[StructureExtractor] Docling not installed. Using font-heuristic fallback."
            )
            errors.append("Docling not installed; used font-heuristic fallback.")
        except Exception as exc:  # noqa: BLE001
            msg = f"[StructureExtractor] Docling failed: {exc}. Falling back."
            logger.warning(msg)
            errors.append(msg)

        structure, fb_errors = self._extract_with_font_heuristic(
            pdf_path, text_blocks
        )
        errors.extend(fb_errors)
        return structure, errors

    # ── Docling path ──────────────────────────

    def _extract_with_docling(
        self, pdf_path: Path
    ) -> tuple[DocumentStructure, list[str]]:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        errors: list[str] = []

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False  # handled by TableExtractor

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(pdf_path))
        doc = result.document

        # Docling exposes a texts list with labels
        # Label types include: title, section_header, paragraph, caption, etc.
        title = ""
        authors: list[str] = []
        raw_headings: list[dict] = []
        heading_stack: list[SectionNode] = []
        root_nodes: list[SectionNode] = []

        for element in doc.texts:
            label = str(getattr(element, "label", "")).lower()
            text = element.text.strip() if hasattr(element, "text") else ""
            if not text:
                continue

            # Page provenance
            page_num = 1
            bbox_raw = None
            if hasattr(element, "prov") and element.prov:
                prov = element.prov[0]
                page_num = getattr(prov, "page_no", 1)
                bb = getattr(prov, "bbox", None)
                if bb is not None:
                    bbox_raw = BoundingBox(
                        x0=float(bb.l), y0=float(bb.t),
                        x1=float(bb.r), y1=float(bb.b),
                    )

            # Document title and authors
            if label == "title" and not title:
                title = text
                continue
            if label in ("author", "authors"):
                authors.append(text)
                continue

            # Section headings
            if "section_header" in label or "heading" in label:
                # Infer level from Docling's level attribute if present,
                # otherwise from heading_level attribute
                level = getattr(element, "level", None)
                if level is None:
                    level = getattr(element, "heading_level", 1)
                if level is None or level < 1:
                    level = 1

                node = SectionNode(
                    level=int(level),
                    title=text,
                    page_number=page_num,
                    bbox=bbox_raw,
                )
                raw_headings.append({
                    "level": int(level),
                    "title": text,
                    "page_number": page_num,
                })
                self._insert_into_tree(root_nodes, heading_stack, node)

        # Fallback title from first heading
        if not title and raw_headings:
            title = raw_headings[0]["title"]

        # Detect page count from result
        num_pages = getattr(result, "pages", None)
        if num_pages is None:
            num_pages = len(doc.pages) if hasattr(doc, "pages") else 0

        return DocumentStructure(
            title=title,
            authors=authors,
            num_pages=int(num_pages) if num_pages else 0,
            language=self._detect_language(doc),
            is_scanned=False,
            section_tree=root_nodes,
            raw_headings=raw_headings,
            source="docling",
        ), errors

    @staticmethod
    def _detect_language(doc) -> str:
        try:
            lang = getattr(doc, "language", None)
            if lang:
                return str(lang)
        except Exception:  # noqa: BLE001
            pass
        return "unknown"

    @staticmethod
    def _insert_into_tree(
        root_nodes: list[SectionNode],
        stack: list[SectionNode],
        node: SectionNode,
    ) -> None:
        """Insert a heading node into the tree using a parent-stack approach."""
        # Pop stack until we find a node at a shallower level
        while stack and stack[-1].level >= node.level:
            stack.pop()

        if stack:
            stack[-1].children.append(node)
        else:
            root_nodes.append(node)

        stack.append(node)

    # ── Font-heuristic fallback ───────────────

    def _extract_with_font_heuristic(
        self, pdf_path: Path, text_blocks: list[TextBlock]
    ) -> tuple[DocumentStructure, list[str]]:
        import pdfplumber

        errors: list[str] = []

        # Determine total page count
        num_pages = 0
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                num_pages = len(pdf.pages)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[StructureExtractor/heuristic] page count: {exc}")

        # Gather all font sizes across all blocks that look like headings
        heading_blocks = [
            b for b in text_blocks
            if b.block_type == "header" and b.font_sizes
        ]

        # Determine font-size thresholds for level 1, 2, 3
        all_sizes = sorted(
            {max(b.font_sizes) for b in heading_blocks}, reverse=True
        )
        def size_to_level(size: float) -> int:
            if not all_sizes:
                return 1
            if size >= all_sizes[0]:
                return 1
            if len(all_sizes) > 1 and size >= all_sizes[1]:
                return 2
            return 3

        raw_headings: list[dict] = []
        root_nodes: list[SectionNode] = []
        stack: list[SectionNode] = []
        title = ""

        for block in heading_blocks:
            max_fs = max(block.font_sizes)
            level = size_to_level(max_fs)

            if not title and level == 1:
                title = block.text
                # Don't add document title to the section tree
                continue

            node = SectionNode(
                level=level,
                title=block.text,
                page_number=block.page_number,
                bbox=block.bbox,
            )
            raw_headings.append({
                "level": level,
                "title": block.text,
                "page_number": block.page_number,
            })
            self._insert_into_tree(root_nodes, stack, node)

        return DocumentStructure(
            title=title,
            authors=[],
            num_pages=num_pages,
            language="unknown",
            is_scanned=False,
            section_tree=root_nodes,
            raw_headings=raw_headings,
            source="font_heuristic",
        ), errors


# ─────────────────────────────────────────────
# 6. SCANNED PDF DETECTOR
# ─────────────────────────────────────────────

class ScannedPageDetector:
    """
    Classifies each page as digital (has selectable text) or scanned
    (image-only). Returns a summary for the whole document.

    Threshold: if a non-blank page has fewer than MIN_CHARS_FOR_DIGITAL
    extractable characters, it is considered scanned.
    """

    MIN_CHARS_FOR_DIGITAL: int = 50

    def classify(self, pdf_path: Path) -> dict[str, Any]:
        import pdfplumber

        page_types: list[str] = []
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    n_chars = len(text.strip())
                    page_types.append(
                        "digital" if n_chars >= self.MIN_CHARS_FOR_DIGITAL else "scanned"
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ScannedPageDetector] %s", exc)

        total = len(page_types)
        n_scanned = page_types.count("scanned")
        n_digital = page_types.count("digital")

        return {
            "is_fully_scanned": total > 0 and n_scanned == total,
            "is_mixed": 0 < n_scanned < total,
            "is_fully_digital": total > 0 and n_digital == total,
            "scanned_pages": n_scanned,
            "digital_pages": n_digital,
            "total_pages": total,
            "page_types": page_types,
        }


# ─────────────────────────────────────────────
# 7. ORCHESTRATOR
# ─────────────────────────────────────────────

class PDFExtractionPipeline:
    """
    Orchestrates the four extractors in a safe, parallel-friendly way.

    Usage
    -----
        pipeline = PDFExtractionPipeline()
        result: ExtractionResult = pipeline.run("path/to/document.pdf")

    Each extractor runs independently; an exception in one does NOT
    abort the others. All errors are collected into result.extraction_errors.

    Parallel execution
    ------------------
    Set use_threads=True to run text, table, and image extraction in
    parallel using ThreadPoolExecutor (I/O-bound, safe for GIL).
    Structure extraction always runs after text extraction since the
    font-heuristic fallback depends on text_blocks.
    """

    def __init__(self, use_threads: bool = False, cancel_callback=None) -> None:
        self.use_threads = use_threads
        self.cancel_callback = cancel_callback
        self.content_extractor = DoclingContentExtractor()
        self.image_extractor = ImageExtractor()
        self.structure_extractor = StructureExtractor()
        self.scanned_detector = ScannedPageDetector()

    def check_cancelled(self):
        if self.cancel_callback:
            self.cancel_callback()

    def run(self, pdf_path: str | Path) -> ExtractionResult:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        all_errors: list[str] = []

        logger.info("Starting extraction: %s", pdf_path.name)

        scan_info = self.scanned_detector.classify(pdf_path)

        if scan_info["is_fully_scanned"]:
            logger.warning(
                "%s appears to be fully scanned. OCR is disabled — "
                "text extraction will be empty.",
                pdf_path.name,
            )
            all_errors.append(
                "Document appears to be fully scanned. Enable OCR for text extraction."
            )

        self.check_cancelled()

        images, im_err = self.image_extractor.extract(
            pdf_path,
            cancel_callback=self.check_cancelled,
        )
        all_errors.extend(im_err)

        image_regions_by_page: dict[int, list[BoundingBox]] = {}

        for img in images:
            image_regions_by_page.setdefault(
                img.page_number,
                []
            ).append(img.bbox)

        self.check_cancelled()

        text_blocks, tables, content_errors = self.content_extractor.extract(
            pdf_path,
            image_regions_by_page=image_regions_by_page,
            cancel_callback=self.check_cancelled,
        )
        all_errors.extend(content_errors)

        self.check_cancelled()

        structure, st_err = self.structure_extractor.extract(
            pdf_path,
            text_blocks,
        )
        all_errors.extend(st_err)

        structure.is_scanned = scan_info["is_fully_scanned"]

        if structure.num_pages == 0:
            structure.num_pages = scan_info["total_pages"]

        logger.info(
            "Extraction complete: %d text blocks, %d tables, %d images, "
            "%d headings, %d errors",
            len(text_blocks),
            len(tables),
            len(images),
            len(structure.raw_headings),
            len(all_errors),
        )

        return ExtractionResult(
            source_path=str(pdf_path),
            text_blocks=text_blocks,
            tables=tables,
            images=images,
            structure=structure,
            extraction_errors=all_errors,
        )

        
    def _run_parallel(
        self, pdf_path: Path, all_errors: list[str]
    ) -> tuple[list[TextBlock], list[TableData], list[ImageData]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        text_blocks: list[TextBlock] = []
        tables: list[TableData] = []
        images: list[ImageData] = []

        futures = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures["text"]  = pool.submit(self.text_extractor.extract,  pdf_path)
            futures["table"] = pool.submit(self.table_extractor.extract, pdf_path)
            futures["image"] = pool.submit(self.image_extractor.extract, pdf_path)

            for name, future in futures.items():
                try:
                    result, errs = future.result()
                    all_errors.extend(errs)
                    if name == "text":
                        text_blocks = result
                    elif name == "table":
                        tables = result
                    elif name == "image":
                        images = result
                except Exception as exc:  # noqa: BLE001
                    msg = f"[{name.capitalize()}Extractor] fatal: {exc}"
                    logger.error(msg)
                    all_errors.append(msg)

        return text_blocks, tables, images


# ─────────────────────────────────────────────
# 8. CLI ENTRY POINT
# ─────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> None:
    import argparse, json

    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Universal PDF extraction layer"
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--no-threads", action="store_true",
        help="Run extractors sequentially instead of in parallel"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print a JSON summary instead of full output"
    )
    args = parser.parse_args()

    pipeline = PDFExtractionPipeline(use_threads=not args.no_threads)
    result = pipeline.run(args.pdf)

    if args.summary:
        summary = {
            "source": result.source_path,
            "pages": result.page_count,
            "text_blocks": len(result.text_blocks),
            "tables": len(result.tables),
            "images": len(result.images),
            "headings": len(result.structure.raw_headings),
            "title": result.structure.title,
            "is_scanned": result.structure.is_scanned,
            "structure_source": result.structure.source,
            "errors": result.extraction_errors,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*60}")
        print(f"  {Path(result.source_path).name}")
        print(f"{'='*60}")
        print(f"  Title       : {result.structure.title or '—'}")
        print(f"  Pages       : {result.page_count}")
        print(f"  Is scanned  : {result.structure.is_scanned}")
        print(f"  Text blocks : {len(result.text_blocks)}")
        print(f"  Tables      : {len(result.tables)}")
        print(f"  Images      : {len(result.images)}")
        print(f"  Headings    : {len(result.structure.raw_headings)}")
        print(f"  Structure   : {result.structure.source}")

        if result.structure.raw_headings:
            print("\n  Document outline:")
            for h in result.structure.raw_headings[:15]:
                indent = "  " * h["level"]
                print(f"    {indent}[H{h['level']}] p.{h['page_number']}  {h['title']}")
            if len(result.structure.raw_headings) > 15:
                print(f"    … ({len(result.structure.raw_headings) - 15} more)")

        if result.tables:
            print("\n  Tables found:")
            for t in result.tables:
                print(
                    f"    p.{t.page_number} table[{t.table_index}] "
                    f"{t.num_rows}×{t.num_cols}  ({t.source})"
                )

        if result.images:
            print("\n  Images found:")
            for img in result.images[:10]:
                flag = "📊 diagram" if img.is_diagram else "🖼  photo"
                print(
                    f"    p.{img.page_number} img[{img.image_index}] "
                    f"{img.width_px}×{img.height_px}px  {flag}"
                )

        if result.extraction_errors:
            print(f"\n  ⚠  {len(result.extraction_errors)} extraction warning(s):")
            for e in result.extraction_errors:
                print(f"    • {e}")

    return result


if __name__ == "__main__":
    main()
