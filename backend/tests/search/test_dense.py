import asyncio
import time
import uuid
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.embeddings import search
from app.domains.embeddings.pipeline import provider


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
