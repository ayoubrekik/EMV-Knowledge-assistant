from pathlib import Path
import json
import copy

from ..extraction.extraction import TextBlock, BoundingBox
from .heading_correction import LayoutAwareBlockCorrector

input_path = Path("/app/src/storage/TVR_test.json")
output_path = Path("/app/src/storage/post_processed_blocks.json")

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

output = copy.deepcopy(data)

text_blocks = []

for r in data["text_blocks"]:
    bbox = r["bbox"]

    text_blocks.append(
        TextBlock(
            page_number=r.get("page_num", r.get("page")),
            text=r["text"],
            bbox=BoundingBox(
                x0=bbox["x0"],
                y0=bbox["y0"],
                x1=bbox["x1"],
                y1=bbox["y1"],
            ),
            block_type=r["type"],
            doc_id=r.get("doc_id", ""),
            doc_version=r.get("doc_version", ""),
            doc_date=r.get("doc_date", ""),
        )
    )

before = {
    (b.page_number, b.text, b.bbox.x0, b.bbox.y0): b.block_type
    for b in text_blocks
}

corrector = LayoutAwareBlockCorrector()
corrected_blocks, _ = corrector.correct(text_blocks)

changed_blocks = []

for b in corrected_blocks:
    key = (b.page_number, b.text, b.bbox.x0, b.bbox.y0)
    old_type = before.get(key)

    if old_type and old_type != b.block_type:
        changed_blocks.append({
            "doc_id": b.doc_id,
            "doc_version": b.doc_version,
            "doc_date": b.doc_date,
            "page_num": b.page_number,
            "old_type": old_type,
            "new_type": b.block_type,
            "text": b.text[:200],
        })

output["text_blocks"] = [
    {
        "doc_id": b.doc_id,
        "doc_version": b.doc_version,
        "doc_date": b.doc_date,
        "page_num": b.page_number,
        "type": b.block_type,
        "text": b.text,
        "bbox": b.bbox.to_dict(),
        "char_count": b.char_count,
    }
    for b in corrected_blocks
]

output["post_processing"] = {
    "changed_blocks_count": len(changed_blocks),
    "changed_blocks": changed_blocks,
}

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("Changed blocks:", len(changed_blocks))
print(f"Saved to: {output_path}")