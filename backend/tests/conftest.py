import os
import socket

# Integration tests TRUNCATE every table, so they must never touch the development
# database (which holds the SciFact benchmark index). Point the whole application,
# including code paths that open their own sessions, at a dedicated test database
# before any app module reads settings.
_dev_db = os.environ.get("POSTGRES_DB", "nexus")
TEST_DB = os.environ.get("NEXUS_TEST_POSTGRES_DB", f"{_dev_db}_test")
if not TEST_DB.endswith("_test"):
    raise RuntimeError(
        f"Refusing to run tests against '{TEST_DB}': the test database name must end with '_test'."
    )
os.environ["POSTGRES_DB"] = TEST_DB

import asyncpg
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

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

assert settings.POSTGRES_DB == TEST_DB, (
    "settings were loaded before the test DB override"
)


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


# Separate engine with NullPool to prevent asyncpg connection conflicts across event loops
engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI, echo=False, poolclass=NullPool
)
TestingSessionLocal = async_sessionmaker(
    autocommit=False, autoflush=False, bind=engine, expire_on_commit=False
)


TRUNCATE_SQL = text(f"""
    TRUNCATE TABLE {", ".join(sorted(Base.metadata.tables))} CASCADE;
""")

_schema_ready = False


async def _ensure_test_database() -> None:
    """Create the test database, pgvector extension and schema once per session."""
    global _schema_ready
    if _schema_ready:
        return
    conn = await asyncpg.connect(
        host=settings.POSTGRES_SERVER,
        port=int(settings.POSTGRES_PORT),
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        database="postgres",
    )
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    finally:
        await conn.close()

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    _schema_ready = True


@pytest_asyncio.fixture
async def setup_db():
    if not is_db_reachable():
        pytest.skip(
            f"PostgreSQL/pgvector database is offline at {settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT} - skipping integration test"
        )

    try:
        await _ensure_test_database()
        async with engine.connect() as conn:
            await conn.execute(TRUNCATE_SQL)
            await conn.commit()
    except (OSError, asyncpg.PostgresError, SQLAlchemyError) as exc:
        pytest.skip(f"PostgreSQL/pgvector test database unavailable: {exc}")

    yield

    async with engine.connect() as conn:
        await conn.execute(TRUNCATE_SQL)
        await conn.commit()


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
