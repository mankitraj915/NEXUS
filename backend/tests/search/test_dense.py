import asyncio
import time
import uuid
from typing import cast

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.domains.documents.service import save_document
from app.domains.embeddings import search
from app.domains.embeddings.models import Chunk
from app.domains.embeddings.pipeline import process_pending_chunks, provider
from app.domains.projects.models import Project
from app.domains.users.models import User


async def test_query_embedding_does_not_block_the_event_loop(monkeypatch):
    def slow_embed(text: str) -> list[float]:
        time.sleep(0.3)  # stands in for a CPU-bound encode or the first model load
        return []  # dense_search returns before touching the database

    monkeypatch.setattr(provider, "embed_query", slow_embed)

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        results = await search.dense_search(
            cast(AsyncSession, None), uuid.uuid4(), "query"
        )
    finally:
        task.cancel()

    assert results == []
    # Inline encoding froze the loop for the whole call, so other requests (and this
    # ticker) could not run at all.
    assert ticks >= 5


@pytest.mark.asyncio
async def test_concurrent_embedding_workers_do_not_claim_the_same_chunks(
    db_session, monkeypatch
):
    user = User(email="pipeline@nexus.com", hashed_password="pw")
    db_session.add(user)
    await db_session.flush()
    project = Project(name="Pipeline", user_id=user.id)
    db_session.add(project)
    await db_session.commit()
    for i in range(3):
        await save_document(
            db_session,
            project.id,
            f"https://kb.example.com/{i}",
            "kb.example.com",
            f"Document number {i}.",
            {},
        )

    def slow_embed(texts: list[str]) -> list[list[float]]:
        time.sleep(0.2)  # keeps both workers inside the batch at the same time
        return [[1.0] + [0.0] * 383 for _ in texts]

    monkeypatch.setattr(provider, "embed_documents", slow_embed)

    # Two crawl jobs embed on their own sessions. If both took the same PENDING chunks,
    # the second insert would hit the unique chunk_id and mark the chunks FAILED.
    engine = create_async_engine(db_session.bind.url, poolclass=NullPool)
    try:
        async with AsyncSession(engine) as a, AsyncSession(engine) as b:
            processed = await asyncio.gather(
                process_pending_chunks(a), process_pending_chunks(b)
            )
    finally:
        await engine.dispose()

    statuses = (await db_session.execute(select(Chunk.status))).scalars().all()
    assert sorted(processed) == [0, 3]
    assert statuses == ["EMBEDDED"] * 3
