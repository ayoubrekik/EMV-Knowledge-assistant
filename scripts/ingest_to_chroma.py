from src.core.db.ingest_chunks import ingest_chunks
from pathlib import Path
if __name__ == "__main__":
    BOOK_PATHS = [
    "src/data/processed/book1/done/chroma_ready.json",
    "src/data/processed/book2/done/chroma_ready.json",
    "src/data/processed/book3/done/chroma_ready.json",
    "src/data/processed/book4/done/chroma_ready.json",
]

    for path in BOOK_PATHS:
        if Path(path).exists():
            ingest_chunks(path, batch_size=100)
        else:
            print(f"Missing file: {path}")