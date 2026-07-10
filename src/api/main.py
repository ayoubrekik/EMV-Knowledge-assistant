from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File, BackgroundTasks, Form
import shutil
from fastapi.responses import HTMLResponse,StreamingResponse
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re
from sqlalchemy.orm import Session
from uuid import UUID
from typing import Optional
from src.core.rag.conversational_rag_service import stream_conversational_rag

from src.core.db.chroma_client import get_chroma_client,get_or_create_emv_collection

from src.core.db.check_existance import extract_first_page_metadata, document_exists_in_db

from src.core.db.deps import get_db
from src.api.routes.test_db import router as test_db_router
from src.api.routes.auth import router as auth_router
from src.api.routes.voice import router as voice_router
from src.api.routes.auth_register import router as auth_register_router
from src.api.routes.manage_users import router as manage_users_router

from src.core.ingestion.upload_pipeline import run_uploaded_pdf_pipeline, build_chunks_path_from_metadata


from src.core.auth import get_current_user
from src.core.db.models import User, ChatSession, ChatMessage, RagMetadata, RagSource
from src.core.auth import require_admin

from fastapi import HTTPException

from src.core.db.ingest_chunks import ingest_chunks

from src.core.db.database import Base, engine
from src.core.db.models import *

app = FastAPI(title="RAG API")
app.include_router(auth_router)
app.include_router(test_db_router)
app.include_router(voice_router)
app.include_router(auth_register_router)
app.include_router(manage_users_router)

app.mount("/static", StaticFiles(directory="src/static"), name="static")

UPLOAD_DIR = Path("src/storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSING_STATUS = {}
CANCEL_FLAGS = {}

class EmbedChunksRequest(BaseModel):
    chunks_path: str = "/app/src/storage/chunks/chunks.json"

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    regenerate: bool = False
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
    Base.metadata.create_all(bind=engine) # Create Postgrestables if they don't exist
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

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    current_file = Path(__file__).resolve()
    template_dir = current_file.parent.parent / "templates"
    templates = Jinja2Templates(directory=str(template_dir))

    return templates.TemplateResponse(
        request,
        "register.html",
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
    temp = 0.5 if req.regenerate else 0.2
    print("Temprature:", temp)

    return StreamingResponse(
        stream_conversational_rag(
            question=req.question,
            session_id=req.session_id,
            db=db,
            user_id=str(current_user.id),
            temp=temp,
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




@app.get("/settings/manage-users", response_class=HTMLResponse)
def manage_users_page(request: Request):
    current_file = Path(__file__).resolve()
    template_dir = current_file.parent.parent / "templates"
    templates = Jinja2Templates(directory=str(template_dir))

    return templates.TemplateResponse(
        request,
        "manage_users.html",
        {"request": request}
    )

@app.get("/settings/add-document", response_class=HTMLResponse)
def add_document_page(request: Request):
    current_file = Path(__file__).resolve()
    template_dir = current_file.parent.parent / "templates"
    templates = Jinja2Templates(directory=str(template_dir))
    return templates.TemplateResponse(
        request,
        "add_document.html",
        {"request": request}
    )

@app.get("/settings/delete-document")
def delete_document_page(request: Request):
    current_file = Path(__file__).resolve()
    template_dir = current_file.parent.parent / "templates"
    templates = Jinja2Templates(directory=str(template_dir))

    return templates.TemplateResponse(
        request,
        "delete_document.html",
        {"request": request}
    )


# Delete conversation
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
        raise HTTPException(status_code=404, detail="Chat session not found")

    rag_metadata_ids = [
        row.id for row in db.query(RagMetadata.id).filter(
            RagMetadata.session_id == session_uuid
        ).all()
    ]

    if rag_metadata_ids:
        db.query(RagSource).filter(
            RagSource.rag_metadata_id.in_(rag_metadata_ids)
        ).delete(synchronize_session=False)

    db.query(RagMetadata).filter(
        RagMetadata.session_id == session_uuid
    ).delete(synchronize_session=False)

    db.query(ChatMessage).filter(
        ChatMessage.session_id == session_uuid
    ).delete(synchronize_session=False)

    db.delete(session)
    db.commit()

    return {"success": True, "message": "Chat session deleted"}


    
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
                        "best_relevance": rag_metadata.best_relevance,
                        "average_relevance": rag_metadata.average_relevance,
                        "worst_relevance": rag_metadata.worst_relevance,
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




### Delete


class DeleteBookRequest(BaseModel):
    doc_id: str
    doc_version: str
    doc_date: str



@app.post("/settings/delete-book")
def delete_book(
    req: DeleteBookRequest,
    current_user: User = Depends(require_admin),
):

    collection = get_or_create_emv_collection()

    existing = collection.get(
        where={
            "$and": [
                {"doc_id": req.doc_id},
                {"doc_version": req.doc_version},
                {"doc_date": req.doc_date},
            ]
        }
    )

    ids = existing.get("ids", [])

    if not ids:
        return {
            "success": False,
            "deleted_chunks": 0,
            "message": "No matching chunks found."
        }

    collection.delete(ids=ids)

    return {
        "success": True,
        "deleted_chunks": len(ids),
        "doc_id": req.doc_id,
        "doc_version": req.doc_version,
        "doc_date": req.doc_date,
    }

### Upload a file 


@app.post("/documents/upload")
def upload_document(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    doc_version: Optional[str] = Form(None),
    doc_date: Optional[str] = Form(None),
    current_user: User = Depends(require_admin),
):
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    safe_filename = Path(pdf_file.filename).name
    saved_path = UPLOAD_DIR / safe_filename

    for old_pdf in UPLOAD_DIR.glob("*.pdf"):
        old_pdf.unlink()

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(pdf_file.file, buffer)

    metadata = extract_first_page_metadata(saved_path)

    if doc_id:
        metadata["doc_id"] = doc_id.strip()

    if doc_version:
        metadata["doc_version"] = doc_version.strip()

    if doc_date:
        metadata["doc_date"] = doc_date.strip()

    missing_fields = []

    if not metadata.get("doc_id"):
        missing_fields.append("Document ID")

    if not metadata.get("doc_version"):
        missing_fields.append("Document Version")

    if not metadata.get("doc_date"):
        missing_fields.append("Document Date")

    if missing_fields:
        return {
            "success": False,
            "status": "missing_metadata",
            "missing_fields": missing_fields,
            "message": "Some document metadata could not be extracted automatically."
        }

    if document_exists_in_db(metadata):
        return {
            "success": False,
            "already_exists": True,
            "source": "db",
            "message": "Document already exists in ChromaDB.",
            "metadata": metadata,
        }

    chunks_path = build_chunks_path_from_metadata(metadata)

    if chunks_path.exists():
        return {
            "success": True,
            "already_chunked": True,
            "needs_embedding": True,
            "source": "storage",
            "message": "Chunks already exist in storage. You can now embed and save them in ChromaDB.",
            "chunks_path": str(chunks_path),
            "metadata": metadata,
        }
        
    import uuid
    job_id = str(uuid.uuid4())

    PROCESSING_STATUS[job_id] = {
        "status": "running",
        "step": "Upload done",
        "progress": 10,
        "message": "PDF uploaded successfully.",
        "filename": safe_filename,
        "chunks_count": 0,
        "error": None,
    }
    CANCEL_FLAGS[job_id] = False

    background_tasks.add_task(process_uploaded_document, job_id, saved_path, metadata)

    return {
        "success": True,
        "job_id": job_id,
        "message": "Upload started."
    }

def process_uploaded_document(job_id: str, saved_path: Path, metadata: dict):
    def update_progress(step: str, progress: int, message: str):
        PROCESSING_STATUS[job_id].update({
            "step": step,
            "progress": progress,
            "message": message,
        })

    try:
        chunks = run_uploaded_pdf_pipeline(
            saved_path,
            metadata=metadata,
            progress_callback=update_progress,
            cancel_callback=lambda: CANCEL_FLAGS.get(job_id, False),
        )

        chunks_path = build_chunks_path_from_metadata({
            "doc_id": chunks[0].doc_id,
            "doc_version": chunks[0].doc_version,
            "doc_date": chunks[0].doc_date,
        }) if chunks else None

        PROCESSING_STATUS[job_id].update({
            "status": "done",
            "step": "Finished",
            "progress": 100,
            "message": "Document processed successfully.",
            "chunks_count": len(chunks),
            "chunks_path": str(chunks_path) if chunks_path else None,
        })

    except Exception as e:
        PROCESSING_STATUS[job_id].update({
            "status": "error",
            "step": "Failed",
            "message": str(e),
            "error": str(e),
        })
@app.get("/documents/status/{job_id}")
def get_document_status(
    job_id: str,
    current_user: User = Depends(require_admin),
):
    status = PROCESSING_STATUS.get(job_id)

    if not status:
        raise HTTPException(status_code=404, detail="Job not found.")

    return status

@app.post("/documents/cancel/{job_id}")
def cancel_document(
    job_id: str,
    current_user: User = Depends(require_admin),
):
    if job_id not in PROCESSING_STATUS:
        raise HTTPException(status_code=404, detail="Job not found.")

    CANCEL_FLAGS[job_id] = True

    PROCESSING_STATUS[job_id].update({
        "status": "cancelled",
        "step": "Cancelled",
        "progress": 0,
        "message": "Document processing cancelled.",
    })

    return {"success": True}
    

@app.post("/settings/embed-chunks")
def embed_chunks_endpoint(
    req: EmbedChunksRequest,
    current_user: User = Depends(require_admin),
):
    chunks_path = Path(req.chunks_path)

    if not chunks_path.exists():
        return {
            "success": False,
            "message": f"Chunks file not found: {chunks_path}"
        }

    try:
        inserted = ingest_chunks(chunks_path)

        return {
            "success": True,
            "message": "Chunks embedded and saved successfully.",
            "inserted_chunks": inserted,
            "chunks_path": str(chunks_path)
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

@app.get("/settings/books")
def list_books(
    current_user: User = Depends(require_admin),
):
    collection = get_or_create_emv_collection()

    data = collection.get(
        include=["metadatas"]
    )

    metadatas = data.get("metadatas", [])

    books_map = {}

    for meta in metadatas:
        doc_id = meta.get("doc_id", "")
        doc_title = meta.get("doc_title", "")
        doc_version = meta.get("doc_version", "")
        doc_date = meta.get("doc_date", "")

        if not doc_id:
            continue

        key = (doc_id, doc_version, doc_date)

        if key not in books_map:
            books_map[key] = {
                "doc_id": doc_id,
                "doc_title": doc_title,
                "doc_version": doc_version,
                "doc_date": doc_date,
                "chunks_count": 0,
            }

        books_map[key]["chunks_count"] += 1

    books = list(books_map.values())

    books.sort(
        key=lambda x: (
            x["doc_id"],
            x["doc_version"],
            x["doc_date"],
        )
    )

    return {
        "success": True,
        "books_count": len(books),
        "books": books,
    }


