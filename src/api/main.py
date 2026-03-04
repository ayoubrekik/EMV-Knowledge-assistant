from fastapi import FastAPI

app = FastAPI(title="RAG API")

@app.get("/")
def root():
    return {"status": "ok", "message": "FastAPI is running"}

@app.get("/health")
def health():
    return {"ok": True}