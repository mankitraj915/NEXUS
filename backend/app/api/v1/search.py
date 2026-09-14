from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from app.db.session import get_db
from app.domains.projects.models import Project
from app.domains.search.schemas import SearchResponse, SearchResultItem
from app.domains.search.retrieval.orchestrator import execute_search

router = APIRouter(tags=["Search"])

async def get_project(db: AsyncSession, project_id: uuid.UUID):
    stmt = select(Project).where(Project.id == project_id)
    return (await db.execute(stmt)).scalars().first()

@router.get("", response_model=SearchResponse)
async def search(
    project_id: uuid.UUID,
    q: str = Query(..., min_length=1),
    mode: str = Query("hybrid_fixed", description="Search mode: lexical, dense, hybrid_fixed, adaptive, hybrid_no_rerank, hybrid_rerank"),
    top_k: int = Query(10, ge=1, le=100),
    retrieval_k: int = Query(100, ge=10, le=1000, description="Depth of initial retrieval for hybrid"),
    rerank_k: int = Query(25, ge=1, le=100, description="Number of candidates to neural rerank"),
    rrf_k: int = Query(60, ge=1, le=200, description="RRF constant"),
    db: AsyncSession = Depends(get_db)
):
    project = await get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    valid_modes = {"lexical", "dense", "hybrid", "hybrid_fixed", "adaptive", "hybrid_no_rerank", "hybrid_rerank"}
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid search mode. Must be one of {valid_modes}.")
        
    candidates = await execute_search(
        db=db,
        project_id=project_id,
        query=q,
        mode=mode,
        top_k=top_k,
        retrieval_k=retrieval_k,
        rerank_k=rerank_k,
        rrf_k=rrf_k
    )
    
    # Map candidates back to SearchResultItem for the API
    results = []
    for c in candidates:
        # Determine best score to surface in the basic API
        score = c.final_rank if c.final_rank else c.rrf_score if c.rrf_score else c.bm25_score or c.dense_score or 0.0
        
        results.append(SearchResultItem(
            document_id=c.document_id,
            url=c.url or "",
            title=c.title or "Untitled",
            score=score,
            snippet=c.snippet or ""
        ))
        
    return SearchResponse(query=q, results=results)
