import pdfplumber
import re
from src.core.db.chroma_client import get_or_create_emv_collection
from pathlib import Path

def extract_first_page_metadata(pdf_path: Path) -> dict:
    with pdfplumber.open(str(pdf_path)) as pdf:
        text = pdf.pages[0].extract_text() or ""

    doc_id = ""
    doc_version = ""
    doc_date = ""

    book_match = re.search(r"\bBook\s+[\w\-]+\b", text, re.IGNORECASE)
    if book_match:
        doc_id = book_match.group(0).strip()

    version_match = re.search(
        r"\b(?:Version|v)\s*([0-9]+(?:\.[0-9]+)*)\b",
        text,
        re.IGNORECASE,
    )
    if version_match:
        doc_version = version_match.group(1).strip()

    date_match = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        text,
        re.IGNORECASE,
    )
    if date_match:
        doc_date = date_match.group(0).strip()

    return {
        "doc_id": doc_id,
        "doc_version": doc_version,
        "doc_date": doc_date,
    }


def document_exists_in_db(metadata: dict) -> bool:
    collection = get_or_create_emv_collection()

    result = collection.get(
        where={
            "$and": [
                {"doc_id": {"$eq": metadata.get("doc_id", "")}},
                {"doc_version": {"$eq": metadata.get("doc_version", "")}},
                {"doc_date": {"$eq": metadata.get("doc_date", "")}},
            ]
        },
        limit=1,
        include=["metadatas"],
    )

    return len(result.get("ids", [])) > 0