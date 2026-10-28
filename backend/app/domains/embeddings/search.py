import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.documents.models import Document, DocumentVersion
from app.domains.embeddings.models import Chunk, ChunkEmbedding
from app.domains.embeddings.pipeline import provider
from app.domains.search.schemas import SearchResultItem


async def dense_search(
    db: AsyncSession, project_id: uuid.UUID, query: str, top_k: int = 10
) -> list[SearchResultItem]:
    query_embedding = provider.embed_query(query)
    if not query_embedding:
        return []

    distance_expr = ChunkEmbedding.embedding.cosine_distance(query_embedding).label(
        "distance"
    )

    stmt = (
        select(Document, DocumentVersion, Chunk, distance_expr)
        .select_from(ChunkEmbedding)
        .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(Document.project_id == project_id)
        .order_by(distance_expr, Chunk.id)
        .limit(top_k * 5)
    )

    result = await db.execute(stmt)
    rows = result.all()

    doc_best_chunk: dict[uuid.UUID, float] = {}
    doc_meta: dict[uuid.UUID, dict[str, object]] = {}

    for doc, version, chunk, distance in rows:
        similarity = 1.0 - distance

        if doc.id not in doc_best_chunk or similarity > doc_best_chunk[doc.id]:
            doc_best_chunk[doc.id] = similarity

            title = (
                version.raw_metadata.get("title", "Untitled")
                if version.raw_metadata
                else "Untitled"
            )
            doc_meta[doc.id] = {
                "url": doc.canonical_url,
                "title": title,
                "snippet": chunk.text,
            }

    sorted_docs = sorted(doc_best_chunk.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, sim in sorted_docs[:top_k]:
        meta = doc_meta[doc_id]
        results.append(
            SearchResultItem(
                document_id=str(doc_id),
                url=str(meta["url"]),
                title=str(meta["title"]),
                score=sim,
                snippet=str(meta["snippet"]),
            )
        )

    return results
