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

class RagSource(Base):
    __tablename__ = "rag_sources"

    id = Column(UUID(as_uuid=True), primary_key=True)

    rag_metadata_id = Column(
        UUID(as_uuid=True),
        ForeignKey("rag_metadata.id", ondelete="CASCADE"),
        nullable=False,
    )

    rank = Column(Integer, nullable=False)

    chunk_id = Column(String(255), nullable=True)
    section_id = Column(String(255), nullable=True)
    doc_id = Column(String(100), nullable=True)

    title = Column(Text, nullable=True)
    section_number = Column(String(100), nullable=True)
    page = Column(Integer, nullable=True)

    distance = Column(Double, nullable=True)

    text_preview = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now())

    rag_metadata = relationship("RagMetadata", back_populates="sources")