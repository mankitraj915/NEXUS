import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.domains.projects.models import Project
from app.domains.search.retrieval.orchestrator import execute_search
from app.domains.search.schemas import SearchMode, SearchResponse, SearchResultItem

router = APIRouter(tags=["Search"])


async def get_project(db: AsyncSession, project_id: uuid.UUID) -> Project | None:
    stmt = select(Project).where(Project.id == project_id)
    return (await db.execute(stmt)).scalars().first()


@router.get("", response_model=SearchResponse)
async def search(
    project_id: uuid.UUID,
    q: str = Query(..., min_length=1),
    mode: str = Query(
        SearchMode.HYBRID_FIXED.value,
        description="Search mode: hybrid_fixed, lexical, dense (or experimental_* modes)",
    ),
    top_k: int = Query(10, ge=1, le=100),
    retrieval_k: int = Query(
        100, ge=10, le=1000, description="Depth of initial retrieval for hybrid"
    ),
    rerank_k: int = Query(
        25,
        ge=1,
        le=100,
        description="Unused by stable modes; cross-encoder reranking was rejected (ADR-005)",
    ),
    rrf_k: int = Query(
        10,
        ge=1,
        le=200,
        description="RRF constant (k=10 is the benchmarked production baseline)",
    ),
    db: AsyncSession = Depends(get_db),
):
    if not q.strip():
        raise HTTPException(status_code=422, detail="Query must not be blank")

    project = await get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not SearchMode.is_valid_mode(mode):
        stable = [
            m.value for m in SearchMode if not m.value.startswith("experimental_")
        ]
        experimental = [
            m.value for m in SearchMode if m.value.startswith("experimental_")
        ]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid search mode '{mode}'. Stable modes: {stable}. Experimental modes: {experimental}.",
        )

    normalized_mode = SearchMode.normalize_mode(mode)
    candidates = await execute_search(
        db=db,
        project_id=project_id,
        query=q,
        mode=normalized_mode,
        top_k=top_k,
        retrieval_k=retrieval_k,
        rerank_k=rerank_k,
        rrf_k=rrf_k,
    )

    # Surface the score that produced the ranking (higher is better) and the 1-based rank
    results = []
    for rank, c in enumerate(candidates, start=1):
        score = next(
            (
                s
                for s in (c.reranker_score, c.rrf_score, c.bm25_score, c.dense_score)
                if s is not None
            ),
            0.0,
        )

        results.append(
            SearchResultItem(
                document_id=c.document_id,
                url=c.url or "",
                title=c.title or "Untitled",
                score=score,
                rank=rank,
                snippet=c.snippet or "",
            )
        )

    return SearchResponse(query=q, mode=normalized_mode, results=results)
