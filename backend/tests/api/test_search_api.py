import uuid

import pytest
from httpx import AsyncClient

import app.api.v1.search as search_api
from app.domains.documents.service import save_document
from app.domains.embeddings.pipeline import process_pending_chunks
from app.domains.users.models import User

DOCS = [
    (
        "https://kb.example.com/bm25",
        "BM25 scoring",
        "BM25 ranks documents by term frequency and inverse document frequency.",
    ),
    (
        "https://kb.example.com/dense",
        "Dense retrieval",
        "Dense retrieval embeds text as vectors and compares cosine similarity.",
    ),
    (
        "https://kb.example.com/rrf",
        "Rank fusion",
        "Reciprocal rank fusion merges BM25 and dense rankings without score calibration.",
    ),
]


@pytest.fixture
async def project_id(client: AsyncClient, db_session) -> str:
    user = User(email="api@nexus.com", hashed_password="pw")
    db_session.add(user)
    await db_session.commit()
    resp = await client.post(
        "/api/v1/projects/", json={"name": "API", "user_id": str(user.id)}
    )
    assert resp.status_code == 201
    pid = resp.json()["id"]
    for url, title, text in DOCS:
        await save_document(
            db_session, uuid.UUID(pid), url, "kb.example.com", text, {"title": title}
        )
    await process_pending_chunks(db_session)
    return pid


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["hybrid_fixed", "lexical", "dense"])
async def test_results_are_ranked_unique_and_scored(
    client: AsyncClient, project_id, mode
):
    resp = await client.get(
        "/api/v1/search",
        params={"project_id": project_id, "q": "bm25 dense fusion", "mode": mode},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "score must be higher-is-better"
    assert len({r["document_id"] for r in results}) == len(results)


@pytest.mark.asyncio
async def test_hybrid_default_uses_benchmarked_rrf_k(
    client: AsyncClient, project_id, monkeypatch
):
    captured = {}
    real = search_api.execute_search

    async def spy(**kwargs):
        captured.update(kwargs)
        return await real(**kwargs)

    monkeypatch.setattr(search_api, "execute_search", spy)
    resp = await client.get(
        "/api/v1/search", params={"project_id": project_id, "q": "rank fusion"}
    )
    assert resp.status_code == 200
    assert captured["mode"] == "hybrid_fixed"
    assert captured["rrf_k"] == 10
    top = resp.json()["results"][0]
    # A document ranked first by both retrievers scores exactly 2 / (k + 1)
    assert top["score"] <= 2 / 11 + 1e-9


@pytest.mark.asyncio
async def test_search_input_validation(client: AsyncClient, project_id):
    base = {"project_id": project_id, "q": "bm25"}

    assert (
        await client.get("/api/v1/search", params={**base, "q": "   "})
    ).status_code == 422
    assert (
        await client.get("/api/v1/search", params={**base, "q": ""})
    ).status_code == 422
    assert (
        await client.get("/api/v1/search", params={**base, "mode": "bogus"})
    ).status_code == 400
    assert (
        await client.get("/api/v1/search", params={**base, "top_k": 0})
    ).status_code == 422
    assert (
        await client.get("/api/v1/search", params={**base, "top_k": 101})
    ).status_code == 422
    assert (
        await client.get("/api/v1/search", params={**base, "project_id": "not-a-uuid"})
    ).status_code == 422
    assert (
        await client.get(
            "/api/v1/search", params={**base, "project_id": str(uuid.uuid4())}
        )
    ).status_code == 404


@pytest.mark.asyncio
async def test_lexical_no_match_returns_empty_list(client: AsyncClient, project_id):
    resp = await client.get(
        "/api/v1/search",
        params={"project_id": project_id, "q": "photosynthesis", "mode": "lexical"},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


@pytest.mark.asyncio
async def test_create_project_validation(client: AsyncClient, db_session):
    unknown_user = {"name": "Orphan", "user_id": str(uuid.uuid4())}
    assert (
        await client.post("/api/v1/projects/", json=unknown_user)
    ).status_code == 404

    user = User(email="names@nexus.com", hashed_password="pw")
    db_session.add(user)
    await db_session.commit()
    for name in ("", "x" * 256):
        resp = await client.post(
            "/api/v1/projects/", json={"name": name, "user_id": str(user.id)}
        )
        assert resp.status_code == 422
