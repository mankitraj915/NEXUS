import asyncio
import os
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.documents.models import Document, DocumentVersion
from app.domains.embeddings.models import Chunk
from app.domains.users.models import User

from .mock_server import start_mock_server


@pytest.fixture(scope="module", autouse=True)
def mock_server():
    server = start_mock_server(8081)
    os.environ["ALLOW_LOCAL_URLS"] = "true"
    yield
    server.shutdown()
    os.environ["ALLOW_LOCAL_URLS"] = "false"


@pytest.mark.asyncio
async def test_crawler_end_to_end(db_session: AsyncSession, client: AsyncClient):
    # 0. Create User
    user = User(email="crawler@example.com", hashed_password="pw")
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    # 1. Create Project
    resp = await client.post(
        "/api/v1/projects/",
        json={
            "name": "Crawl Project",
            "description": "Integration test project",
            "user_id": str(user.id),
        },
    )
    project_id = resp.json()["id"]

    # 2. Create Crawl Job
    resp = await client.post(
        "/api/v1/crawl/jobs",
        json={
            "project_id": project_id,
            "seed_url": "http://127.0.0.1:8081/",
            "max_pages": 5,
            "max_depth": 2,
            "respect_robots": True,
            "concurrency_limit": 2,
        },
    )
    assert resp.status_code == 201
    job_id = resp.json()["id"]

    # 3. Wait for crawl to complete (includes embedding; the first call loads the model)
    max_wait = 240
    completed = False
    for _ in range(max_wait):
        resp = await client.get(f"/api/v1/crawl/jobs/{job_id}")
        if resp.json()["status"] == "COMPLETED":
            completed = True
            break
        await asyncio.sleep(0.5)

    assert completed, "Crawler did not complete in time"

    # 4. Check stats
    resp = await client.get(f"/api/v1/crawl/jobs/{job_id}/stats")
    stats = resp.json()
    assert stats["fetched"] == 2  # / and /page1
    assert stats["failed"] == 1  # /admin (blocked by robots)

    # 5. Check documents
    stmt = select(Document).where(Document.project_id == uuid.UUID(project_id))
    docs = (await db_session.execute(stmt)).scalars().all()
    assert len(docs) == 2

    stmt = select(DocumentVersion)
    versions = (await db_session.execute(stmt)).scalars().all()
    assert len(versions) == 2

    # 6. Crawled pages are embedded, so dense and hybrid search can find them
    chunk_stmt = (
        select(Chunk.status)
        .join(DocumentVersion)
        .join(Document)
        .where(Document.project_id == uuid.UUID(project_id))
    )
    statuses = (await db_session.execute(chunk_stmt)).scalars().all()
    assert statuses and set(statuses) == {"EMBEDDED"}

    resp = await client.get(
        "/api/v1/search",
        params={"project_id": project_id, "q": "page 1", "mode": "dense"},
    )
    assert resp.status_code == 200
    assert "Page 1" in [r["title"] for r in resp.json()["results"]]
