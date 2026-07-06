from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

CHUNKS_PATH = Path("/app/src/storage/chunks/chunks.json")
OUT_DIR = Path("/app/src/storage/chunk_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

rows = []

for c in chunks:
    rows.append({
        "chunk_id": c.get("chunk_id", ""),
        "document_id": c.get("document_id", ""),
        "section_path": c.get("section_path", ""),
        "section_numbers": " | ".join(c.get("section_numbers", [])),
        "section_titles": " | ".join(c.get("section_titles", [])),
        "page_start": c.get("page_start"),
        "page_end": c.get("page_end"),
        "content_type": c.get("content_type", ""),
        "chunk_strategy": c.get("chunk_strategy", ""),
        "token_count": c.get("token_count", 0),
        "has_text": c.get("has_text", False),
        "has_tables": c.get("has_tables", False),
        "is_merged": c.get("is_merged", False),
        "merged_count": c.get("merged_count", 1),
        "is_split": c.get("is_split", False),
        "split_part": c.get("split_part"),
        "split_total": c.get("split_total"),
    })

df = pd.DataFrame(rows)

bins = [0, 10, 80, 256, 480, 10_000_000]
labels = ["0-10", "10-80", "80-256", "257-480", "480+"]

df["token_band"] = pd.cut(
    df["token_count"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
)

summary = df["token_count"].describe(
    percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
)

band_counts = df["token_band"].value_counts().sort_index()
strategy_counts = df["chunk_strategy"].value_counts()
type_counts = df["content_type"].value_counts()

large_chunks = df[df["token_count"] > 480].sort_values("token_count", ascending=False)
small_chunks = df[df["token_count"] < 80].sort_values("token_count")
merged_chunks = df[df["is_merged"] == True]
split_chunks = df[df["is_split"] == True]

excel_path = OUT_DIR / "chunk_statistics.xlsx"

with pd.ExcelWriter(excel_path) as writer:
    df.to_excel(writer, sheet_name="chunks", index=False)
    summary.to_frame("value").to_excel(writer, sheet_name="summary")
    band_counts.to_frame("count").to_excel(writer, sheet_name="token_bands")
    strategy_counts.to_frame("count").to_excel(writer, sheet_name="strategies")
    type_counts.to_frame("count").to_excel(writer, sheet_name="content_types")
    large_chunks.to_excel(writer, sheet_name="large_chunks_480_plus", index=False)
    small_chunks.to_excel(writer, sheet_name="small_chunks_under_80", index=False)
    merged_chunks.to_excel(writer, sheet_name="merged_chunks", index=False)
    split_chunks.to_excel(writer, sheet_name="split_chunks", index=False)

plt.figure(figsize=(10, 6))
plt.hist(df["token_count"], bins=40)
plt.title("Distribution of chunk token counts")
plt.xlabel("Tokens")
plt.ylabel("Number of chunks")
plt.tight_layout()
plt.savefig(OUT_DIR / "chunk_token_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
band_counts.plot(kind="bar")
plt.title("Chunk token bands")
plt.xlabel("Token band")
plt.ylabel("Number of chunks")
plt.tight_layout()
plt.savefig(OUT_DIR / "chunk_token_bands.png")
plt.close()

plt.figure(figsize=(10, 6))
strategy_counts.plot(kind="bar")
plt.title("Chunks by strategy")
plt.xlabel("Strategy")
plt.ylabel("Number of chunks")
plt.tight_layout()
plt.savefig(OUT_DIR / "chunks_by_strategy.png")
plt.close()

plt.figure(figsize=(10, 6))
type_counts.plot(kind="bar")
plt.title("Chunks by content type")
plt.xlabel("Content type")
plt.ylabel("Number of chunks")
plt.tight_layout()
plt.savefig(OUT_DIR / "chunks_by_content_type.png")
plt.close()

print("\nCHUNK TOKEN SUMMARY")
print(summary)

print("\nTOKEN BANDS")
print(band_counts)

print("\nBY STRATEGY")
print(strategy_counts)

print("\nBY CONTENT TYPE")
print(type_counts)

print("\nChunks > 480:", len(large_chunks))
print("Chunks < 80:", len(small_chunks))
print("Merged chunks:", len(merged_chunks))
print("Split chunks:", len(split_chunks))

print("\nSaved Excel:", excel_path)
print("Saved charts in:", OUT_DIR)