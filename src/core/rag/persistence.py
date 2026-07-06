from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
from src.core.db.models import ChatMessage, ChatSession, RagMetadata, RagSource
from src.core.rag.chat_title import generate_title
from .source_utils import source_from_doc
from .types import InputType, ScoredDoc
import math 

def relevance_percent(score):
    if score is None:
        return None
    return round(100 / (1 + math.exp(-float(score))), 1)
    
def get_or_create_chat_session(db, session_id: str, user_id: str, question: str):
    session_uuid = UUID(session_id)
    user_uuid = UUID(user_id)

    chat_session = db.query(ChatSession).filter(
        ChatSession.id == session_uuid,
        ChatSession.user_id == user_uuid,
    ).first()

    if chat_session:
        return chat_session

    title = generate_title(question)
    chat_session = ChatSession(id=session_uuid, user_id=user_uuid, title=title)
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session


def create_user_message(db, chat_session, user_uuid, question: str, input_type: str | None = None):
    user_message = ChatMessage(
        id=uuid4(),
        session_id=chat_session.id,
        user_id=user_uuid,
        role="user",
        content=question,
        input_type=input_type,
    )

    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    return user_message


def create_assistant_message(db, chat_session, user_uuid, answer: str):
    assistant_message = ChatMessage(
        id=uuid4(),
        session_id=chat_session.id,
        user_id=user_uuid,
        role="assistant",
        content=answer,
    )
    db.add(assistant_message)
    db.flush()
    return assistant_message


def create_and_save_rag_metadata(
    db,
    chat_session,
    user_uuid,
    user_message,
    assistant_message,
    question: str,
    standalone_question: Optional[str],
    input_type: InputType,
    scored_docs: list[ScoredDoc],
    router_time: float,
    retrieval_time: float,
    generation_time: float,
    total_time: float,
    model_name: Optional[str],
    embedding_model: Optional[str],
):
    scores = [
        doc.metadata.get("cross_encoder_score")
        for doc, _ in scored_docs
        if doc.metadata.get("cross_encoder_score") is not None
    ]

    best_score = max(scores) if scores else None
    worst_score = min(scores) if scores else None
    average_score = sum(scores) / len(scores) if scores else None

    best_relevance = relevance_percent(best_score)
    average_relevance = relevance_percent(average_score)
    worst_relevance = relevance_percent(worst_score)

    rag_metadata = RagMetadata(
        id=uuid4(),
        session_id=chat_session.id,
        user_id=user_uuid,
        user_message_id=user_message.id,
        assistant_message_id=assistant_message.id,
        original_question=question,
        rewritten_question=standalone_question,
        input_type=input_type,
        retrieved_chunks_count=len(scored_docs),
        best_relevance=best_relevance,
        average_relevance=average_relevance,
        worst_relevance=worst_relevance,
        router_time_seconds=router_time,
        retrieval_time_seconds=retrieval_time,
        generation_time_seconds=generation_time,
        total_time_seconds=total_time,
        model_name=model_name,
        embedding_model=embedding_model,
    )
    db.add(rag_metadata)
    db.flush()
    return rag_metadata, best_relevance, worst_relevance, average_relevance


def save_rag_sources(db, rag_metadata_id, scored_docs: list[ScoredDoc]):
    for rank, (doc, distance) in enumerate(scored_docs, start=1):
        source = source_from_doc(doc)

        kwargs = dict(
            id=uuid4(),
            rag_metadata_id=rag_metadata_id,
            rank=rank,
            chunk_id=source.get("chunk_id"),
            doc_id=source.get("doc_id"),
            title=source.get("doc_title"),
            page=source.get("page"),
            distance=distance,
            text_preview=(doc.page_content or "")[:100],
        )

        try:
            rag_source = RagSource(**kwargs, section_number=source.get("section"))
        except TypeError:
            rag_source = RagSource(**kwargs)

        db.add(rag_source)


def touch_session(chat_session):
    chat_session.updated_at = datetime.utcnow()
