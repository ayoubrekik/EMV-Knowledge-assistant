import re
import pandas as pd
from typing import Optional


TAG_RE = re.compile(r"^'?[0-9A-Fa-f]{2,8}'?$")


def normalize_tag(value: str) -> str:
    return str(value).strip().strip("'\"`").upper()


def detect_tag_column(df: pd.DataFrame) -> Optional[str]:
    # Priority 1: column name
    for col in df.columns:
        if "tag" in str(col).lower():
            return col

    # Priority 2: values look like EMV tags
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20)
        if sample.empty:
            continue

        values = sample.map(lambda x: normalize_tag(x))
        match_ratio = values.map(lambda x: bool(re.fullmatch(r"[0-9A-F]{2,8}", x))).mean()

        if match_ratio >= 0.8:
            return col

    return None


def parse_markdown_table(markdown_table: str) -> Optional[pd.DataFrame]:
    lines = [l.strip() for l in markdown_table.strip().splitlines() if l.strip()]

    table_lines = [l for l in lines if "|" in l]

    if len(table_lines) < 2:
        return None

    def is_separator(line: str) -> bool:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        return all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells)

    table_lines = [l for l in table_lines if not is_separator(l)]

    if len(table_lines) < 2:
        return None

    def split_row(line: str):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    headers = split_row(table_lines[0])
    rows = [split_row(l) for l in table_lines[1:]]

    fixed_rows = []
    for r in rows:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        elif len(r) > len(headers):
            r = r[:len(headers)]
        fixed_rows.append(r)

    return pd.DataFrame(fixed_rows, columns=headers)


def reformat_table_for_llm(markdown_table: str, queried_tag: str) -> str:
    df = parse_markdown_table(markdown_table)

    if df is None or df.empty:
        return ""

    tag_col = detect_tag_column(df)
    if tag_col is None:
        return ""

    queried_norm = normalize_tag(queried_tag)

    df["_tag_norm"] = df[tag_col].apply(normalize_tag)
    matching = df[df["_tag_norm"] == queried_norm].drop(columns=["_tag_norm"])

    if matching.empty:
        return ""

    other_cols = [c for c in matching.columns if c != tag_col]

    output_lines = [
        f"Exact rows matching tag {queried_norm}:"
    ]

    for i, (_, row) in enumerate(matching.iterrows(), 1):
        output_lines.append(f"\nRow {i}:")
        output_lines.append(f"- Tag: {queried_norm}")

        for col in other_cols:
            value = str(row[col]).strip()
            if value:
                output_lines.append(f"- {col}: {value}")

    return "\n".join(output_lines)

def build_tag_lookup_context(scored_docs, tag: str) -> str:
    blocks = []

    for i, scored_doc in enumerate(scored_docs, start=1):
        doc = scored_doc[0]   # because scored_doc = (Document, score)

        content = doc.page_content
        meta = doc.metadata

        reformatted = reformat_table_for_llm(content, tag)

        if not reformatted:
            continue

        header = (
            f"[Source {i}]\n"
            f"Document: {meta.get('doc_id', 'Unknown')}\n"
            f"Document title: {meta.get('doc_title', 'Unknown')}\n"
            f"Section: {meta.get('section', meta.get('section_path', meta.get('context_prefix', 'Unknown')))}\n"
            f"Page: {meta.get('page', meta.get('page_num', 'Unknown'))}\n"
            f"Type: {meta.get('type', 'Unknown')}\n"
        )

        blocks.append(header + "\n" + reformatted)

    if not blocks:
        return f"No exact table rows were found for tag {tag}."

    return "\n\n---\n\n".join(blocks)

def extract_emv_tag(question: str) -> Optional[str]:
    """
    Extract the first EMV tag appearing in the question.

    Examples:
        "What is tag DF53?"          -> DF53
        "Description of 9F27"        -> 9F27
        "What does BF4E mean?"       -> BF4E
        "Meaning of 84"              -> 84
    """

    matches = re.findall(r"\b[0-9A-Fa-f]{2,8}\b", question)

    if not matches:
        return None

    # Prefer longer tags first (9F27 before 9F)
    matches = sorted(matches, key=len, reverse=True)

    return matches[0].upper()