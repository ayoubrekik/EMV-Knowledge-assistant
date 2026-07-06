from pathlib import Path
import json
import re


INPUT_PATH = Path("/app/src/storage/post_processed.json")
OUTPUT_PATH = Path("/app/src/storage/enriched_sections.json")


def clean_inline_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def make_document_id(section: dict) -> str:
    doc_id = clean_inline_text(
        section.get("doc_id", "")
    ).replace(" ", "_")

    version = clean_inline_text(
        section.get("doc_version", "")
    ).replace(" ", "_")

    date = clean_inline_text(
        section.get("doc_date", "")
    ).replace(" ", "_")

    return f"{doc_id}_{version}_{date}"


def estimate_tokens(items: list[str]) -> int:
    # Join items and strip whitespaces
    text = "\n".join(items).strip()

    if not text:
        # Prevent zero-division or unnecessary math on empty text
        return 0

    # Adjusted for technical text (Markdown tables, hex codes, pipes)
    # BGE models use WordPiece tokenization where 1 token ≈ 3.8 characters
    return round(len(text) / 3.8)


def build_context_injection(section: dict) -> str:

    path_parts = []

    if section.get("parent_titles"):
        path_parts.extend(section["parent_titles"])

    if section.get("title"):
        path_parts.append(section["title"])

    full_path = " > ".join(path_parts)

    return (
        f"[Document: {section['document_id']}]\n"
        f"[Section: {section.get('section_number','')} {full_path}]\n"
        f"[Page: {section.get('start_page')}]"
    )


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

sections = data["sections"]

for section in sections:

    document_id = make_document_id(section)

    text_list = section.get("text_list", [])
    table_list = section.get("table_list", [])

    section["document_id"] = document_id

    section["n_tokens_text"] = estimate_tokens(
        text_list
    )

    section["n_tokens_tabs"] = estimate_tokens(
        table_list
    )

    section["context_injection"] = build_context_injection(
        {
            **section,
            "document_id": document_id,
        }
    )

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        data,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("Sections:", len(sections))
print("Saved:", OUTPUT_PATH)