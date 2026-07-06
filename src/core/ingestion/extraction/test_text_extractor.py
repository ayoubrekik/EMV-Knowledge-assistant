from pathlib import Path
from extraction import DoclingContentExtractor
import json
from collections import Counter

pdf_path = Path("/app/src/data/raw/TVR_test.pdf")
output_path = Path("/app/src/storage/TVR_test.json")

extractor = DoclingContentExtractor()
text_blocks, tables, errors = extractor.extract(pdf_path)

print("=" * 60)
print("TOTAL TEXT BLOCKS:", len(text_blocks))
print("TOTAL TABLES:", len(tables))
print("TOTAL ERRORS:", len(errors))
print("=" * 60)

label_counts = Counter(block.block_type for block in text_blocks)
print("\nLABELS FOUND:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

print("\nFIRST 10 TEXT BLOCKS:")
for i, block in enumerate(text_blocks[:10]):
    print(f"\nBLOCK #{i+1}")
    print("Page:", block.page_number)
    print("Type:", block.block_type)
    print("Text:", block.text[:300])

print("\nFIRST 5 TABLES:")
for i, table in enumerate(tables[:5]):
    print(f"\nTABLE #{i+1}")
    print("Page:", table.page_number)
    print("Rows:", table.num_rows)
    print("Cols:", table.num_cols)
    print(table.markdown[:500])

output = {
    "text_blocks": [
        {
            "doc_id": block.doc_id,
            "doc_version": block.doc_version,
            "doc_date": block.doc_date,
            "page_num": block.page_number,
            "type": block.block_type,
            "text": block.text,
            "bbox": block.bbox.to_dict(),
            "char_count": block.char_count,
        }
        for block in text_blocks
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
        for table in tables
    ],
    "errors": errors,
}

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved to: {output_path}")

if errors:
    print("\nERRORS:")
    for error in errors:
        print("-", error)