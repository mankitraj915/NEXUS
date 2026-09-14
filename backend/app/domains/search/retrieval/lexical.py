import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.search.bm25 import search_project
from app.domains.search.retrieval.models import Candidate


async def retrieve_lexical(
    db: AsyncSession, project_id: uuid.UUID, query: str, top_k: int = 10
) -> list[Candidate]:
    """
    Executes BM25 search and wraps the results in the normalized Candidate representation.
    """
    bm25_results = await search_project(db, project_id, query, top_k=top_k)

    candidates = []
    for i, res in enumerate(bm25_results):
        candidates.append(
            Candidate(
                document_id=res.document_id,
                bm25_rank=i + 1,
                bm25_score=res.score,
                bm25_passage=res.snippet,
                snippet=res.snippet,
                url=res.url,
                title=res.title,
            )
        )
    return candidates
