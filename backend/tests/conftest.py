import socket

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.domains.benchmark.models
import app.domains.crawler.models
import app.domains.documents.models
import app.domains.embeddings.models
import app.domains.projects.models
import app.domains.search.models
import app.domains.users.models  # noqa: F401
from app.config.settings import settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app as fastapi_app


def is_db_reachable() -> bool:
    try:
        host = (
            "127.0.0.1"
            if settings.POSTGRES_SERVER == "localhost"
            else settings.POSTGRES_SERVER
        )
        with socket.create_connection((host, int(settings.POSTGRES_PORT)), timeout=0.2):
            return True
    except OSError:
        return False


from sqlalchemy.pool import NullPool

# Create a separate engine for testing with NullPool to prevent asyncpg connection conflicts
engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI, echo=False, poolclass=NullPool
)
TestingSessionLocal = async_sessionmaker(
    autocommit=False, autoflush=False, bind=engine, expire_on_commit=False
)


TRUNCATE_SQL = text("""
    TRUNCATE TABLE 
        benchmark_mappings, 
        chunk_embeddings, 
        crawl_jobs, 
        crawl_targets, 
        document_chunks, 
        document_versions, 
        documents, 
        projects, 
        search_document_stats, 
        search_postings, 
        search_project_stats, 
        search_terms, 
        users 
    CASCADE;
""")


@pytest_asyncio.fixture
async def setup_db():
    if not is_db_reachable():
        pytest.skip(
            f"PostgreSQL/pgvector database is offline at {settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT} - skipping integration test"
        )

    try:
        async with engine.connect() as conn:
            await conn.execute(TRUNCATE_SQL)
            await conn.commit()
    except Exception as exc:
        # If tables don't exist yet, create them
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        except Exception:
            pytest.skip(f"PostgreSQL/pgvector database error: {exc}")

    yield

    try:
        async with engine.connect() as conn:
            await conn.execute(TRUNCATE_SQL)
            await conn.commit()
    except Exception:
        pass


@pytest_asyncio.fixture
async def db_session(setup_db):
    async with TestingSessionLocal() as session:
        yield session
        await session.close()


@pytest_asyncio.fixture
async def client(db_session):
    async def override_get_db():
        yield db_session

    fastapi_app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as ac:
        yield ac

    fastapi_app.dependency_overrides.clear()
