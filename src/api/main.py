from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse,StreamingResponse
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.core.rag.conversational_rag_service import stream_conversational_rag
from src.core.db.chroma_client import get_chroma_client,get_or_create_emv_collection
from src.api.routes.test_db import router as test_db_router
from src.api.routes.auth import router as auth_router

from src.core.auth import get_current_user
from src.core.db.models import User
from src.core.auth import require_admin

app = FastAPI(title="RAG API")
app.include_router(auth_router)
app.include_router(test_db_router)
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
    return templates.TemplateResponse(request, "login.html", {"request": request})

@app.get("/health")
def health():
    return {"ok": True}

@app.on_event("startup")
def startup_event():
    collection = get_or_create_emv_collection()
    print(f"Chroma collection ready: {collection.name} and {collection.metadata}")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    current_file = Path(__file__).resolve()
    template_dir = current_file.parent.parent / "templates"
    templates = Jinja2Templates(directory=str(template_dir))

    return templates.TemplateResponse(
        request,
        "login.html",
        {"request": request}
    )
@app.get("/user", response_class=HTMLResponse)
def user_chat(request: Request):
    current_file = Path(__file__).resolve()
    template_dir = current_file.parent.parent / "templates"
    templates = Jinja2Templates(directory=str(template_dir))

    return templates.TemplateResponse(
        request,
        "user_chat.html",
        {"request": request}
    )
@app.get("/admin", response_class=HTMLResponse)
def admin_chat(request: Request):
    current_file = Path(__file__).resolve()
    template_dir = current_file.parent.parent / "templates"
    templates = Jinja2Templates(directory=str(template_dir))

    return templates.TemplateResponse(
        request,
        "admin_chat.html",
        {"request": request}
    )
@app.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    return StreamingResponse(
        stream_conversational_rag(
            question=req.question,
            session_id=req.session_id,
            user_id=str(current_user.id),
        ),
        media_type="text/event-stream",
    )