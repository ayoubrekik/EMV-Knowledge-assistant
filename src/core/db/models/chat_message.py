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

class ChatMessage(Base):
    __tablename__ = "chat_messages"

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

    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)

    created_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="chat_messages_role_check",
        ),
    )

    session = relationship("ChatSession", back_populates="messages")
    user = relationship("User", back_populates="messages")

    rag_as_user_message = relationship(
        "RagMetadata",
        foreign_keys="RagMetadata.user_message_id",
        back_populates="user_message",
    )

    rag_as_assistant_message = relationship(
        "RagMetadata",
        foreign_keys="RagMetadata.assistant_message_id",
        back_populates="assistant_message",
    )
