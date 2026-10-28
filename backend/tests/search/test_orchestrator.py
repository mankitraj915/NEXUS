import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.domains.documents.service import save_document
from app.domains.embeddings.pipeline import process_pending_chunks
from app.domains.projects.models import Project
from app.domains.search.retrieval import orchestrator
from app.domains.search.retrieval.models import Candidate
from app.domains.users.models import User

DOCS = [
    ("https://kb.example.com/bm25", "BM25 ranks documents by term statistics."),
    ("https://kb.example.com/dense", "Dense retrieval compares embedding vectors."),
    ("https://kb.example.com/rrf", "Rank fusion merges lexical and dense rankings."),
]


@pytest.mark.asyncio
async def test_hybrid_uses_both_retrievers_on_a_fresh_session(db_session):
    user = User(email="orchestrator@nexus.com", hashed_password="pw")
    db_session.add(user)
    await db_session.flush()
    project = Project(name="Orchestrator", user_id=user.id)
    db_session.add(project)
    await db_session.commit()
    for url, text in DOCS:
        await save_document(db_session, project.id, url, "kb.example.com", text, {})
    await process_pending_chunks(db_session)

    # The search is the first statement on a session that must open a new connection.
    # Running both branches as concurrent tasks on one session used to make the dense
    # branch fail here, and hybrid silently returned BM25-only results.
    engine = create_async_engine(db_session.bind.url, poolclass=NullPool)
    try:
        async with AsyncSession(engine) as fresh:
            results = await orchestrator.execute_search(
                fresh, project.id, "dense rankings", mode="hybrid_fixed"
            )
    finally:
        await engine.dispose()

    assert any(c.bm25_rank is not None for c in results)
    assert any(c.dense_rank is not None for c in results)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["retrieve_lexical", "retrieve_dense"])
async def test_hybrid_surfaces_retriever_failure(monkeypatch, failing):
    async def ok(db, project_id, query, top_k=10):
        return [Candidate(document_id="d1", bm25_rank=1, dense_rank=1)]

    async def broken(db, project_id, query, top_k=10):
        raise RuntimeError("retriever down")

    monkeypatch.setattr(orchestrator, "retrieve_lexical", ok)
    monkeypatch.setattr(orchestrator, "retrieve_dense", ok)
    monkeypatch.setattr(orchestrator, failing, broken)

    # A failed branch must not be passed off as a hybrid ranking.
    with pytest.raises(RuntimeError, match="retriever down"):
        await orchestrator.execute_search(
            None,  # type: ignore[arg-type]
            uuid.uuid4(),
            "query",
            mode="hybrid_fixed",
        )
