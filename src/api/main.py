from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse,StreamingResponse
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.core.rag.conversational_rag_service import stream_conversational_rag
from src.core.db.chroma_client import get_chroma_client,get_or_create_emv_collection

app = FastAPI(title="RAG API")

app.mount("/static", StaticFiles(directory="src/static"), name="static")


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
# @app.get("/")
# def root():
#     current_file = Path(__file__).resolve()
#     print(f"DEBUG: Current file path: {current_file}")
#     return {"status": "ok", "message": "FastAPI is running"}
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    current_file = Path(__file__).resolve()
    template_dir = current_file.parent.parent / "templates"
    templates = Jinja2Templates(directory=str(template_dir))
    return templates.TemplateResponse(request, "chat.html", {"request": request})

@app.get("/health")
def health():
    return {"ok": True}

@app.on_event("startup")
def startup_event():
    collection = get_or_create_emv_collection()
    print(f"Chroma collection ready: {collection.name} and {collection.metadata}")


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def event_generator():
        for token in stream_conversational_rag(
            question=req.question,
            session_id=req.session_id,
        ):
            yield f"data: {token}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )