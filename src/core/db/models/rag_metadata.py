from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    DateTime,
    Integer,
    Double,
    ForeignKey,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.core.db.database import Base

class RagMetadata(Base):
    __tablename__ = "rag_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True)

    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    user_message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
    )

    assistant_message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
    )

    original_question = Column(Text, nullable=False)
    rewritten_question = Column(Text, nullable=True)

    input_type = Column(String(50), nullable=True)

    retrieved_chunks_count = Column(Integer, default=0)

    best_relevance = Column(Double, nullable=True)
    average_relevance = Column(Double, nullable=True)
    worst_relevance = Column(Double, nullable=True)

    router_time_seconds = Column(Double, nullable=True)
    retrieval_time_seconds = Column(Double, nullable=True)
    generation_time_seconds = Column(Double, nullable=True)
    total_time_seconds = Column(Double, nullable=True)

    model_name = Column(String(100), nullable=True)
    embedding_model = Column(String(150), nullable=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())

    session = relationship("ChatSession", back_populates="rag_metadata")
    user = relationship("User", back_populates="rag_metadata")

    user_message = relationship(
        "ChatMessage",
        foreign_keys=[user_message_id],
        back_populates="rag_as_user_message",
    )

    assistant_message = relationship(
        "ChatMessage",
        foreign_keys=[assistant_message_id],
        back_populates="rag_as_assistant_message",
    )

    sources = relationship(
        "RagSource",
        back_populates="rag_metadata",
        cascade="all, delete",
    )
