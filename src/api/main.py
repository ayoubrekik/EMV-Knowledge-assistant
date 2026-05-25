from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse,StreamingResponse
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sqlalchemy.orm import Session
from uuid import UUID

from src.core.rag.conversational_rag_service import stream_conversational_rag
from src.core.db.chroma_client import get_chroma_client,get_or_create_emv_collection
from src.core.db.deps import get_db
from src.api.routes.test_db import router as test_db_router
from src.api.routes.auth import router as auth_router

from src.core.auth import get_current_user
from src.core.db.models import User, ChatSession, ChatMessage, RagMetadata, RagSource
from src.core.auth import require_admin

from fastapi import HTTPException

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


#############################################################
@app.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    print("QUESTION RECEIVED:", req.question)
    print("SESSION RECEIVED:", req.session_id)
    print("USER:", current_user.username, current_user.role)

    return StreamingResponse(
        stream_conversational_rag(
            question=req.question,
            session_id=req.session_id,
            db=db,
            user_id=str(current_user.id),
        ),
        media_type="text/event-stream",
    )
############################################################


@app.get("/chat/sessions")
def get_chat_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == current_user.id
    ).order_by(ChatSession.updated_at.desc()).all()

    return [
        {
            "id": str(session.id),
            "title": session.title or "New chat",
            "created_at": str(session.created_at),
            "updated_at": str(session.updated_at),
        }
        for session in sessions
    ]


@app.get("/chat/sessions/{session_id}/messages")
def get_chat_messages(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_uuid = UUID(session_id)

    session = db.query(ChatSession).filter(
        ChatSession.id == session_uuid,
        ChatSession.user_id == current_user.id
    ).first()

    if not session:
        return {"messages": []}

    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_uuid,
        ChatMessage.user_id == current_user.id
    ).order_by(ChatMessage.created_at.asc()).all()

    result = []

    for message in messages:
        item = {
            "id": str(message.id),
            "role": message.role,
            "content": message.content,
            "created_at": str(message.created_at),
            "metadata": None,
        }

        if message.role == "assistant":
            rag_metadata = db.query(RagMetadata).filter(
                RagMetadata.assistant_message_id == message.id
            ).first()

            if rag_metadata:
                sources = db.query(RagSource).filter(
                    RagSource.rag_metadata_id == rag_metadata.id
                ).order_by(RagSource.rank.asc()).all()

                item["metadata"] = {
                    "input_type": rag_metadata.input_type,
                    "original_question": rag_metadata.original_question,
                    "standalone_question": rag_metadata.rewritten_question,
                    "sources": [
                        {
                            "doc_id": source.doc_id,
                            "section_number": source.section_number,
                            "title": source.title,
                            "page": source.page,
                            "distance": source.distance,
                            "text_preview": source.text_preview,
                        }
                        for source in sources
                    ],
                    "metrics": {
                        "retrieved_chunks_count": rag_metadata.retrieved_chunks_count,
                        "best_distance": rag_metadata.best_distance,
                        "average_distance": rag_metadata.average_distance,
                        "worst_distance": rag_metadata.worst_distance,
                        "router_time_seconds": rag_metadata.router_time_seconds,
                        "retrieval_time_seconds": rag_metadata.retrieval_time_seconds,
                        "generation_time_seconds": rag_metadata.generation_time_seconds,
                        "total_time_seconds": rag_metadata.total_time_seconds,
                    }
                }

        result.append(item)

    return {
        "session_id": str(session.id),
        "title": session.title,
        "messages": result,
    }




@app.delete("/chat/sessions/{session_id}")
def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_uuid = UUID(session_id)

    session = db.query(ChatSession).filter(
        ChatSession.id == session_uuid,
        ChatSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    db.delete(session)
    db.commit()

    return {"message": "Session deleted"}