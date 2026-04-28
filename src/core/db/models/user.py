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

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True)

    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)

    role = Column(String(20), nullable=False, default="user")
    is_active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint("role IN ('user', 'admin')", name="users_role_check"),
    )

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete")
    messages = relationship("ChatMessage", back_populates="user", cascade="all, delete")
    rag_metadata = relationship("RagMetadata", back_populates="user", cascade="all, delete")
