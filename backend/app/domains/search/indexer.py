import logging
import uuid
from collections import defaultdict

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.documents.models import Document, DocumentVersion
from app.domains.embeddings.models import Chunk
from app.domains.search.analyzer import analyze_text
from app.domains.search.models import DocumentStats, Posting, ProjectStats, Term

logger = logging.getLogger(__name__)


async def deindex_document_version(db: AsyncSession, version_id: uuid.UUID) -> None:
    """
    Removes a superseded version from the lexical and dense indexes so that only the
    latest version of a document is searchable and corpus statistics stay consistent.
    The DocumentVersion row itself is kept as history.
    """
    doc_stats = (
        (
            await db.execute(
                select(DocumentStats).where(
                    DocumentStats.document_version_id == version_id
                )
            )
        )
        .scalars()
        .first()
    )

    if doc_stats and doc_stats.status == "INDEXED":
        await db.execute(
            update(ProjectStats)
            .where(ProjectStats.project_id == doc_stats.project_id)
            .values(
                total_documents=ProjectStats.total_documents - 1,
                total_tokens=ProjectStats.total_tokens - doc_stats.token_count,
            )
        )
        term_ids = select(Posting.term_id).where(
            Posting.document_version_id == version_id
        )
        await db.execute(
            update(Term)
            .where(Term.id.in_(term_ids))
            .values(document_frequency=Term.document_frequency - 1)
        )

    await db.execute(delete(Posting).where(Posting.document_version_id == version_id))
    await db.execute(
        delete(DocumentStats).where(DocumentStats.document_version_id == version_id)
    )
    # Chunk embeddings are removed by the ON DELETE CASCADE foreign key.
    await db.execute(delete(Chunk).where(Chunk.document_version_id == version_id))
    await db.commit()


async def index_document_version(db: AsyncSession, version_id: uuid.UUID) -> None:
    stmt = (
        select(DocumentVersion, Document.project_id)
        .join(Document)
        .where(DocumentVersion.id == version_id)
    )
    result = await db.execute(stmt)
    row = result.first()
    if not row:
        logger.warning(f"DocumentVersion {version_id} not found.")
        return

    version, project_id = row[0], row[1]

    stmt_stats = select(DocumentStats).where(
        DocumentStats.document_version_id == version_id
    )
    doc_stats = (await db.execute(stmt_stats)).scalars().first()
    if doc_stats and doc_stats.status == "INDEXED":
        return

    if not doc_stats:
        doc_stats = DocumentStats(
            document_version_id=version_id, project_id=project_id, status="PENDING"
        )
        db.add(doc_stats)
        await db.commit()
        await db.refresh(doc_stats)

    try:
        tokens = analyze_text(version.extracted_text)
        token_count = len(tokens)

        tf: dict[str, int] = defaultdict(int)
        positions: dict[str, list[int]] = defaultdict(list)
        for pos, token in enumerate(tokens):
            tf[token] += 1
            positions[token].append(pos)

        # Upsert ProjectStats
        proj_stmt = (
            insert(ProjectStats)
            .values(project_id=project_id, total_documents=1, total_tokens=token_count)
            .on_conflict_do_update(
                index_elements=["project_id"],
                set_={
                    "total_documents": ProjectStats.total_documents + 1,
                    "total_tokens": ProjectStats.total_tokens + token_count,
                },
            )
        )
        await db.execute(proj_stmt)

        # Upsert Terms and create Posting rows
        if tf:
            term_values = [
                {"project_id": project_id, "term": term_str, "document_frequency": 1}
                for term_str in tf
            ]
            term_stmt = (
                insert(Term)
                .values(term_values)
                .on_conflict_do_update(
                    index_elements=["project_id", "term"],
                    set_={"document_frequency": Term.document_frequency + 1},
                )
                .returning(Term.id, Term.term)
            )

            result = await db.execute(term_stmt)
            term_rows = result.all()

            postings_data = [
                {
                    "term_id": row.id,
                    "document_version_id": version_id,
                    "term_frequency": tf[row.term],
                    "positions": positions[row.term],
                }
                for row in term_rows
            ]

            if postings_data:
                await db.execute(insert(Posting).values(postings_data))

        doc_stats.token_count = token_count
        doc_stats.status = "INDEXED"

        await db.commit()

    except Exception as e:
        logger.error(f"Failed to index {version_id}: {e}")
        doc_stats.status = "FAILED"
        await db.commit()
