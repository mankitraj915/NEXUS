import uuid

from sqlalchemy import JSON, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDMixin


class Document(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "documents"

    canonical_url: Mapped[str] = mapped_column(String, index=True, nullable=False)
    source_domain: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "project_id", "canonical_url", name="uix_project_canonical_url"
        ),
    )


class DocumentVersion(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "document_versions"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    content_hash: Mapped[str] = mapped_column(String, index=True, nullable=False)
    extracted_text: Mapped[str] = mapped_column(String, nullable=False)
    raw_metadata: Mapped[dict | None] = mapped_column(JSON)
