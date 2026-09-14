import pytest
from httpx import AsyncClient

from app.domains.users.models import User


@pytest.mark.asyncio
async def test_create_project(client: AsyncClient, db_session):
    # First, create a mock user in the DB
    user = User(email="test@example.com", hashed_password="hashed_pwd")
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    # Now create the project
    payload = {"name": "NEXUS Core", "user_id": str(user.id)}

    response = await client.post("/api/v1/projects/", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == "NEXUS Core"
    assert data["user_id"] == str(user.id)
    assert "id" in data
