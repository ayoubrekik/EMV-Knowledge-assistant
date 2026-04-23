from fastapi import FastAPI
from src.core.db.chroma_client import get_chroma_client,get_or_create_emv_collection

app = FastAPI(title="RAG API")

@app.get("/")
def root():
    return {"status": "ok", "message": "FastAPI is running"}

@app.get("/health")
def health():
    return {"ok": True}

@app.on_event("startup")
def startup_event():
    collection = get_or_create_emv_collection()
    print(f"Chroma collection ready: {collection.name} and {collection.metadata}")