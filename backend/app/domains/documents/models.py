from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, ForeignKey, JSON
import uuid
from typing import Optional
from app.db.base import Base, UUIDMixin, TimestampMixin

class Document(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "documents"

    canonical_url: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    source_domain: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)

class DocumentVersion(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "document_versions"

    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content_hash: Mapped[str] = mapped_column(String, index=True, nullable=False)
    extracted_text: Mapped[str] = mapped_column(String, nullable=False)
    raw_metadata: Mapped[Optional[dict]] = mapped_column(JSON)
