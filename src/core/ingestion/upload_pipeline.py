from pathlib import Path
import re
from src.core.ingestion.extraction.extraction import (
    PDFExtractionPipeline,
    TextBlock,
    BoundingBox,
)

from src.core.ingestion.post_processing.heading_correction import (
    LayoutAwareBlockCorrector,
)

from src.core.ingestion.post_processing.section_assembler import (
    get_page_num,
    build_headings_from_blocks,
    add_heading_ids_and_parents,
    add_heading_spans,
    assemble_sections,
)

from src.core.ingestion.chunking.chunker import (
    ChunkingConfig,
    UniversalPreChunker,
    save_chunks,
    count_tokens,
)


CHUNKS_OUTPUT_PATH = Path("src/storage/chunks/chunks.json")


def build_text_content(section: dict) -> str:
    return "\n".join(section.get("text_list", []))


def build_table_content(section: dict) -> str:
    table_texts = []

    for table in section.get("table_list", []):
        if isinstance(table, dict):
            table_texts.append(
                "\n".join([
                    table.get("previous_text", ""),
                    table.get("caption", ""),
                    table.get("markdown", "")
                ])
            )
        elif isinstance(table, str):
            table_texts.append(table)

    return "\n\n".join(table_texts)



def safe_name(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def build_doc_key(metadata: dict) -> str:
    return safe_name(
        f"{metadata.get('doc_id', '')}_"
        f"{metadata.get('doc_version', '')}_"
        f"{metadata.get('doc_date', '')}"
    )


def build_chunks_path_from_metadata(metadata: dict) -> Path:
    return Path("src/storage/chunks") / f"{build_doc_key(metadata)}_chunks.json"

class ProcessingCancelled(Exception):
    pass


def run_uploaded_pdf_pipeline(
    pdf_path: Path,
    progress_callback=None,
    cancel_callback=None,
):

    def progress(step: str, percent: int, message: str):
        if progress_callback:
            progress_callback(step, percent, message)
    def check_cancelled():
        if cancel_callback and cancel_callback():
            raise ProcessingCancelled("Processing cancelled by user.")

    progress("Extraction", 20, "Extracting text, tables and images...<br>This step may take a while.")

    extraction_pipeline = PDFExtractionPipeline(
        use_threads=False,
        cancel_callback=check_cancelled,
    )

    extraction_result = extraction_pipeline.run(pdf_path)

    progress("Extraction done", 35, "Extraction completed.")

    raw_data = {
        "text_blocks": [
            {
                "doc_id": block.doc_id,
                "doc_version": block.doc_version,
                "doc_title": block.doc_title,
                "doc_date": block.doc_date,
                "page_num": block.page_number,
                "type": block.block_type,
                "text": block.text,
                "bbox": block.bbox.to_dict(),
                "char_count": block.char_count,
            }
            for block in extraction_result.text_blocks
        ],
        "tables": [
            {
                "page": table.page_number,
                "table_index": table.table_index,
                "rows": table.num_rows,
                "cols": table.num_cols,
                "bbox": table.bbox.to_dict(),
                "markdown": table.markdown,
                "raw_matrix": table.raw_matrix,
            }
            for table in extraction_result.tables
        ],
        "errors": extraction_result.extraction_errors,
    }

    progress("Post-processing", 45, "Cleaning and correcting extracted blocks...")

    text_blocks = []

    for row in raw_data["text_blocks"]:
        bbox = row["bbox"]

        text_blocks.append(
            TextBlock(
                page_number=row["page_num"],
                text=row["text"],
                bbox=BoundingBox(
                    x0=bbox["x0"],
                    y0=bbox["y0"],
                    x1=bbox["x1"],
                    y1=bbox["y1"],
                ),
                block_type=row["type"],
                doc_id=row.get("doc_id", ""),
                doc_title=row.get("doc_title", ""),
                doc_version=row.get("doc_version", ""),
                doc_date=row.get("doc_date", ""),
            )
        )

    corrector = LayoutAwareBlockCorrector()
    corrected_blocks, _ = corrector.correct(text_blocks)

    progress("Post-processing done", 55, "Extracted blocks corrected.")

    processed_text_blocks = [
        {
            "doc_id": block.doc_id,
            "doc_title": block.doc_title,
            "doc_version": block.doc_version,
            "doc_date": block.doc_date,
            "page_num": block.page_number,
            "type": block.block_type,
            "text": block.text,
            "bbox": block.bbox.to_dict(),
            "char_count": block.char_count,
        }
        for block in corrected_blocks
    ]

    progress("Section building", 65, "Building document sections...")

    tables = raw_data["tables"]

    last_page_num = max(
        [get_page_num(block) or 1 for block in processed_text_blocks]
        or [1]
    )

    headings = build_headings_from_blocks(processed_text_blocks)
    headings = add_heading_ids_and_parents(headings)
    headings = add_heading_spans(headings, last_page_num)

    sections = assemble_sections(
        text_blocks=processed_text_blocks,
        headings=headings,
        tables=tables,
    )

    progress("Section building done", 75, f"{len(sections)} sections created.")

    enriched_doc = {
        "doc_id": processed_text_blocks[0].get("doc_id", "") if processed_text_blocks else pdf_path.stem,
        "doc_title" : processed_text_blocks[0].get("doc_title", "") if processed_text_blocks else "",
        "doc_version": processed_text_blocks[0].get("doc_version", "") if processed_text_blocks else "",
        "doc_date": processed_text_blocks[0].get("doc_date", "") if processed_text_blocks else "",
        "document_id": pdf_path.stem,
        "sections": sections,
    }

    progress("Token counting", 80, "Counting section tokens...")

    for section in enriched_doc["sections"]:
        text_content = build_text_content(section)
        table_content = build_table_content(section)

        section["n_tokens_text"] = count_tokens(text_content)
        section["n_tokens_tabs"] = count_tokens(table_content)
        section["n_tokens_total"] = (
            section["n_tokens_text"] + section["n_tokens_tabs"]
        )

    progress("Token counting done", 85, "Token counts added.")

    progress("Chunking", 90, "Creating chunks...")

    config = ChunkingConfig(
        tiny_threshold=10,
        min_tokens=80,
        target_tokens=256,
        max_tokens=480,
        overlap_ratio=0.20,
    )

    chunker = UniversalPreChunker(config)
    chunks = chunker.chunk(enriched_doc, input_path=pdf_path)

    progress("Chunking done", 95, f"{len(chunks)} chunks created.")

    progress("Saving", 98, "Saving final chunks...")

    CHUNKS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunks_output_path = build_chunks_path_from_metadata(enriched_doc)
    save_chunks(chunks, chunks_output_path)

    progress("Saving done", 100, "Chunks saved successfully.")

    return chunks