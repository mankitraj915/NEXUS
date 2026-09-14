import uuid

from sqlalchemy import (
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ProjectStats(Base):
    __tablename__ = "search_project_stats"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    total_documents: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)


class DocumentStats(Base):
    __tablename__ = "search_document_stats"

    document_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), primary_key=True
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=False
    )
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(50), default="PENDING")


class Term(Base):
    __tablename__ = "search_terms"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    term: Mapped[str] = mapped_column(String, index=True, nullable=False)
    document_frequency: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (UniqueConstraint("project_id", "term", name="uix_project_term"),)


class Posting(Base):
    __tablename__ = "search_postings"

    term_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("search_terms.id", ondelete="CASCADE")
    )
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE")
    )
    term_frequency: Mapped[int] = mapped_column(Integer, nullable=False)
    positions: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)

    __table_args__ = (PrimaryKeyConstraint("term_id", "document_version_id"),)
