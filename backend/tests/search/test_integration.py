import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from app.domains.users.models import User

@pytest.mark.asyncio
async def test_search_end_to_end(db_session: AsyncSession, client: AsyncClient):
    user = User(email="searcher@example.com", hashed_password="pw")
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    resp = await client.post("/api/v1/projects/", json={
        "name": "Search Project",
        "description": "Integration test project",
        "user_id": str(user.id)
    })
    project_id = resp.json()["id"]
    
    resp2 = await client.post("/api/v1/projects/", json={
        "name": "Other Project",
        "description": "Should not leak",
        "user_id": str(user.id)
    })
    other_project_id = resp2.json()["id"]
    
    from app.domains.documents.service import save_document
            
    doc1 = await save_document(
        db_session, 
        project_id,
        "http://test.com/1",
        "test.com",
        "PostgreSQL 17 supports vector and AI/ML workloads natively with extensions.",
        {"title": "PostgreSQL Vector Database"}
    )
    
    doc2 = await save_document(
        db_session, 
        project_id,
        "http://test.com/2",
        "test.com",
        "PostgreSQL pgvector allows storing vector embeddings in postgres. Good for semantic search.",
        {"title": "pgvector Documentation"}
    )
    
    doc_other = await save_document(
        db_session, 
        other_project_id,
        "http://other.com/1",
        "other.com",
        "PostgreSQL vector database is leaked if seen here.",
        {"title": "Other Database"}
    )
    
    # Process embeddings!
    from app.domains.embeddings.pipeline import process_pending_chunks
    processed_count = await process_pending_chunks(db_session)
    assert processed_count == 3
    
    resp_search = await client.get(f"/api/v1/search?project_id={project_id}&q=postgresql+vector&mode=lexical")
    assert resp_search.status_code == 200
    data = resp_search.json()
    
    assert len(data["results"]) == 2
    
    for r in data["results"]:
        assert r["url"] != "http://other.com/1"
        
    # Test Dense Semantic Search
    resp_dense = await client.get(f"/api/v1/search?project_id={project_id}&q=postgresql+vector&mode=dense")
    assert resp_dense.status_code == 200
    data_dense = resp_dense.json()
    assert len(data_dense["results"]) == 2
    
    # Ensure isolation
    for r in data_dense["results"]:
        assert r["url"] != "http://other.com/1"
    
    resp_search_other = await client.get(f"/api/v1/search?project_id={other_project_id}&q=postgresql+vector")
    assert resp_search_other.status_code == 200
    data_other = resp_search_other.json()
    assert len(data_other["results"]) == 1
    assert data_other["results"][0]["document_id"] == str(doc_other.document_id)
