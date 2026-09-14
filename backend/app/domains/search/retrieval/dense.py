import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.embeddings.search import dense_search
from app.domains.search.retrieval.models import Candidate


async def retrieve_dense(
    db: AsyncSession, project_id: uuid.UUID, query: str, top_k: int = 10
) -> list[Candidate]:
    """
    Executes dense embeddings search and wraps the results in the normalized Candidate representation.
    """
    # NOTE: Our current dense_search returns SearchResultItem which has score and snippet,
    # but does NOT currently return the chunk_id explicitly.
    # The snippet *is* the dense chunk text.
    dense_results = await dense_search(db, project_id, query, top_k=top_k)

    candidates = []
    for i, res in enumerate(dense_results):
        candidates.append(
            Candidate(
                document_id=res.document_id,
                dense_rank=i + 1,
                dense_score=res.score,
                dense_passage=res.snippet,
                snippet=res.snippet,
                url=res.url,
                title=res.title,
            )
        )
    return candidates
