from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.domains.embeddings.models import Chunk, ChunkEmbedding
from app.domains.documents.models import Document, DocumentVersion
from app.domains.embeddings.pipeline import provider
from app.domains.search.schemas import SearchResponse, SearchResultItem
import uuid

async def dense_search(db: AsyncSession, project_id: uuid.UUID, query: str, top_k: int = 10) -> list[SearchResultItem]:
    query_embedding = provider.embed_query(query)
    if not query_embedding:
        return []

    distance_expr = ChunkEmbedding.embedding.cosine_distance(query_embedding).label("distance")

    stmt = (
        select(
            Document,
            DocumentVersion,
            Chunk,
            distance_expr
        )
        .select_from(ChunkEmbedding)
        .join(Chunk, ChunkEmbedding.chunk_id == Chunk.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(Document.project_id == project_id)
        .order_by(distance_expr)
        .limit(top_k * 5)
    )
    
    result = await db.execute(stmt)
    rows = result.all()
    
    doc_best_chunk = {}
    doc_meta = {}
    
    for doc, version, chunk, distance in rows:
        similarity = 1.0 - distance
        
        if doc.id not in doc_best_chunk or similarity > doc_best_chunk[doc.id]:
            doc_best_chunk[doc.id] = similarity
            
            title = version.raw_metadata.get("title", "Untitled") if version.raw_metadata else "Untitled"
            doc_meta[doc.id] = {
                "url": doc.canonical_url,
                "title": title,
                "snippet": chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                "doc": doc,
                "version": version,
                "chunk": chunk,
                "distance": distance
            }
            
    sorted_docs = sorted(doc_best_chunk.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for doc_id, sim in sorted_docs[:top_k]:
        meta = doc_meta[doc_id]
        doc = meta["doc"]
        doc_ver = meta["version"]
        chunk = meta["chunk"]
        distance = meta["distance"]
        results.append(
            SearchResultItem(
                document_id=str(doc.id),
                url=doc.canonical_url,
                title=meta["title"],
                score=1.0 - distance,
                snippet=chunk.text
            )
        )
        
    return results
