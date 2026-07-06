from pathlib import Path
import json
from docling.document_converter import DocumentConverter

pdf_path = Path("/app/src/data/raw/EMV_v4.4_Book_1_ICC_to_Terminal_Interface.pdf")
output_path = Path("/app/src/storage/docling_page_1.json")
import json

def get_unique_types(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Assumes your JSON file is a list of objects: [{"type": "A"}, {"type": "B"}]
    # Using a set comprehension automatically removes duplicates
    unique_types = {item['type'] for item in data if 'type' in item}
    
    return sorted(list(unique_types))
# Usage
file_name = "/app/src/storage/text_blocks_docling.json" 
types = get_unique_types(file_name)

print(f"Found {len(types)} unique types:")
for t in types:
    print(f"- {t}")