from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False
        )
    )

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

INPUT_PATH = Path("/app/src/storage/enriched_sections.json")
OUT_DIR = Path("/app/src/storage/token_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

sections = data["sections"]

rows = []

for s in sections:
    text_content = build_text_content(s)
    table_content = build_table_content(s)

    n_text = count_tokens(text_content)
    n_tabs = count_tokens(table_content)

    s["n_tokens_text"] = n_text
    s["n_tokens_tabs"] = n_tabs
    s["n_tokens_total"] = n_text + n_tabs

    rows.append({
        "section_id": s.get("section_id", ""),
        "section_number": s.get("section_number", ""),
        "title": s.get("title", ""),
        "start_page": s.get("start_page"),
        "end_page": s.get("end_page"),
        "n_tokens_text": n_text,
        "n_tokens_tabs": n_tabs,
        "n_tokens_total": n_text + n_tabs,
        "has_tables": n_tabs > 0,
    })

df = pd.DataFrame(rows)

summary = df[
    ["n_tokens_text", "n_tokens_tabs", "n_tokens_total"]
].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

bins = [8, 50, 100, 256, 500, 700, 1000, 1000000]
labels = ["8-50", "51-100", "101-256", "257-500", "501-700", "701-1000", "1000+"]

df["text_size_band"] = pd.cut(df["n_tokens_text"], bins=bins, labels=labels, include_lowest=True)
df["tabs_size_band"] = pd.cut(df["n_tokens_tabs"], bins=bins, labels=labels, include_lowest=True)
df["total_size_band"] = pd.cut(df["n_tokens_total"], bins=bins, labels=labels, include_lowest=True)

text_band_counts = df["text_size_band"].value_counts().sort_index()
tabs_band_counts = df["tabs_size_band"].value_counts().sort_index()
total_band_counts = df["total_size_band"].value_counts().sort_index()

very_small_text = df[df["n_tokens_text"] <= 80].sort_values("n_tokens_text")
large_text = df[df["n_tokens_text"] > 1000].sort_values("n_tokens_text", ascending=False)

very_small_tabs = df[(df["n_tokens_tabs"] > 0) & (df["n_tokens_tabs"] <= 80)].sort_values("n_tokens_tabs")
large_tabs = df[df["n_tokens_tabs"] > 1000].sort_values("n_tokens_tabs", ascending=False)

top_large_total = df.sort_values("n_tokens_total", ascending=False).head(30)
table_sections = df[df["has_tables"]].sort_values("n_tokens_tabs", ascending=False)

excel_path = OUT_DIR / "section_token_stats.xlsx"
updated_json_path = OUT_DIR / "enriched_sections.json"

with open(updated_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

with pd.ExcelWriter(excel_path) as writer:
    df.to_excel(writer, sheet_name="sections", index=False)
    summary.to_excel(writer, sheet_name="summary")

    text_band_counts.reset_index().to_excel(writer, sheet_name="text_size_bands", index=False)
    tabs_band_counts.reset_index().to_excel(writer, sheet_name="tabs_size_bands", index=False)
    total_band_counts.reset_index().to_excel(writer, sheet_name="total_size_bands", index=False)

    very_small_text.to_excel(writer, sheet_name="very_small_text", index=False)
    large_text.to_excel(writer, sheet_name="large_text", index=False)
    very_small_tabs.to_excel(writer, sheet_name="very_small_tabs", index=False)
    large_tabs.to_excel(writer, sheet_name="large_tabs", index=False)
    top_large_total.to_excel(writer, sheet_name="top_large_total", index=False)
    table_sections.to_excel(writer, sheet_name="table_sections", index=False)

plt.figure(figsize=(10, 6))
plt.hist(df["n_tokens_text"], bins=40)
plt.title("Distribution of text tokens per section")
plt.xlabel("Text tokens")
plt.ylabel("Number of sections")
plt.tight_layout()
plt.savefig(OUT_DIR / "text_tokens_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(df["n_tokens_tabs"], bins=40)
plt.title("Distribution of table tokens per section")
plt.xlabel("Table tokens")
plt.ylabel("Number of sections")
plt.tight_layout()
plt.savefig(OUT_DIR / "table_tokens_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(df["n_tokens_total"], bins=40)
plt.title("Distribution of total tokens per section")
plt.xlabel("Total tokens")
plt.ylabel("Number of sections")
plt.tight_layout()
plt.savefig(OUT_DIR / "total_tokens_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
text_band_counts.plot(kind="bar")
plt.title("Text token size bands")
plt.xlabel("Text token band")
plt.ylabel("Number of sections")
plt.tight_layout()
plt.savefig(OUT_DIR / "text_size_bands.png")
plt.close()

plt.figure(figsize=(10, 6))
tabs_band_counts.plot(kind="bar")
plt.title("Table token size bands")
plt.xlabel("Table token band")
plt.ylabel("Number of sections")
plt.tight_layout()
plt.savefig(OUT_DIR / "table_size_bands.png")
plt.close()

plt.figure(figsize=(10, 6))
total_band_counts.plot(kind="bar")
plt.title("Total token size bands")
plt.xlabel("Total token band")
plt.ylabel("Number of sections")
plt.tight_layout()
plt.savefig(OUT_DIR / "total_size_bands.png")
plt.close()

plt.figure(figsize=(12, 8))
df.sort_values("n_tokens_text", ascending=False).head(30).sort_values("n_tokens_text").plot(
    x="section_number",
    y="n_tokens_text",
    kind="barh",
    legend=False,
)
plt.title("Top 30 largest text sections")
plt.xlabel("Text tokens")
plt.ylabel("Section")
plt.tight_layout()
plt.savefig(OUT_DIR / "top_30_largest_text_sections.png")
plt.close()

plt.figure(figsize=(12, 8))
df[df["n_tokens_tabs"] > 0].sort_values("n_tokens_tabs", ascending=False).head(30).sort_values("n_tokens_tabs").plot(
    x="section_number",
    y="n_tokens_tabs",
    kind="barh",
    legend=False,
)
plt.title("Top 30 largest table sections")
plt.xlabel("Table tokens")
plt.ylabel("Section")
plt.tight_layout()
plt.savefig(OUT_DIR / "top_30_largest_table_sections.png")
plt.close()

print("\nSUMMARY")
print(summary)

print("\nTEXT SIZE BANDS")
print(text_band_counts)

print("\nTABLE SIZE BANDS")
print(tabs_band_counts)

print("\nTOTAL SIZE BANDS")
print(total_band_counts)

print("\nSaved Excel:", excel_path)
print("Saved updated JSON:", updated_json_path)
print("Saved charts in:", OUT_DIR)