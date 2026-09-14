import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.documents.models import DocumentVersion
from app.domains.embeddings.chunker import chunk_text
from app.domains.embeddings.models import Chunk, ChunkEmbedding
from app.domains.embeddings.provider import LocalSentenceTransformerProvider

logger = logging.getLogger(__name__)

# Initialize the provider as a global instance so model weights stay in memory
provider = LocalSentenceTransformerProvider()


async def process_document_version(db: AsyncSession, version_id: uuid.UUID):
    """
    Called after DocumentVersion is saved. Segments the document into chunks deterministically.
    """
    stmt_ver = select(DocumentVersion).where(DocumentVersion.id == version_id)
    version = (await db.execute(stmt_ver)).scalars().first()
    if not version:
        return

    # Check if already chunked
    stmt_chunk = select(Chunk).where(Chunk.document_version_id == version_id).limit(1)
    existing = (await db.execute(stmt_chunk)).scalars().first()
    if existing:
        return

    chunks_text = chunk_text(version.extracted_text)

    for idx, text in enumerate(chunks_text):
        chunk = Chunk(
            document_version_id=version_id, chunk_index=idx, text=text, status="PENDING"
        )
        db.add(chunk)

    await db.commit()


async def process_pending_chunks(db: AsyncSession, batch_size: int = 32) -> int:
    """
    Finds PENDING chunks, generates embeddings in batch, and stores them in pgvector.
    """
    stmt = select(Chunk).where(Chunk.status == "PENDING").limit(batch_size)
    chunks = (await db.execute(stmt)).scalars().all()

    if not chunks:
        return 0

    texts_to_embed = [c.text for c in chunks]

    try:
        embeddings = provider.embed_documents(texts_to_embed)

        # Bulk insert embeddings
        db.add_all(
            [
                ChunkEmbedding(
                    chunk_id=chunk.id,
                    model_name=provider.model_name,
                    model_version=provider.model_version,
                    embedding=emb,
                )
                for chunk, emb in zip(chunks, embeddings)
            ]
        )
        for chunk in chunks:
            chunk.status = "EMBEDDED"
        await db.commit()
        return len(chunks)

    except Exception as e:
        logger.error(f"Failed to embed batch: {e}")
        await db.rollback()

        # Mark as FAILED so we don't infinitely retry in this MVP
        for chunk in chunks:
            chunk.status = "FAILED"
            db.add(chunk)
        await db.commit()
        return 0
