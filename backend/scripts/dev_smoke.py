"""Local development smoke test and optional demo seeder for NEXUS.

Verifies:
1. PostgreSQL connection & pgvector extension availability.
2. Database schema readiness (Alembic migrations applied).
3. Document ingestion, chunking, and embedding generation.
4. Lexical (BM25), Dense (pgvector), and Hybrid (RRF) retrieval execution.
5. Live API endpoint connectivity (if the FastAPI server is running).

Usage:
    # Run fast end-to-end smoke verification (cleans up test data)
    uv run python -m scripts.dev_smoke

    # Seed a persistent local demo project for testing via FastAPI (/api/v1/search)
    uv run python -m scripts.dev_smoke --seed
"""

import argparse
import asyncio
import sys
import uuid

import httpx
from sqlalchemy import select, text

from app.config.settings import settings
from app.db.session import AsyncSessionLocal
from app.domains.documents.models import Document
from app.domains.documents.service import save_document
from app.domains.embeddings.pipeline import process_pending_chunks
from app.domains.projects.models import Project
from app.domains.search.retrieval.orchestrator import execute_search
from app.domains.search.schemas import SearchMode
from app.domains.users.models import User

SAMPLE_DOCUMENTS = [
    {
        "url": "https://nexus.local/docs/bm25-retrieval",
        "title": "Robertson-Spärck Jones BM25 Retrieval in NEXUS",
        "text": (
            "NEXUS implements custom Robertson-Spärck Jones BM25 scoring with non-negative "
            "smoothed inverse document frequency (IDF) and positional term arrays stored in PostgreSQL. "
            "BM25 excels at exact keyword matching, technical identifiers, and lexical precision."
        ),
    },
    {
        "url": "https://nexus.local/docs/pgvector-dense",
        "title": "Dense Semantic Retrieval with pgvector",
        "text": (
            "Dense semantic retrieval in NEXUS generates 384-dimensional sentence embeddings "
            "using all-MiniLM-L6-v2 and stores them in PostgreSQL with the pgvector extension. "
            "Exact cosine distance scans recover semantically paraphrased claims and conceptual relevance."
        ),
    },
    {
        "url": "https://nexus.local/docs/reciprocal-rank-fusion",
        "title": "Reciprocal Rank Fusion (RRF) Hybrid Retrieval",
        "text": (
            "Reciprocal Rank Fusion combines ranked candidate lists from lexical BM25 and dense "
            "pgvector retrieval without requiring score calibration. A parameter sweep on the "
            "SciFact dev claims selected k=10 over the common default k=60."
        ),
    },
]

# Matches documents through both BM25 and dense retrieval, so the hybrid result shows fusion.
DEMO_QUERY = "lexical and vector hybrid search"


async def check_database() -> None:
    print("[1/5] Checking PostgreSQL connection & pgvector extension...")
    async with AsyncSessionLocal() as session:
        # Check connection
        res = await session.execute(text("SELECT version();"))
        ver = res.scalar()
        print(f"      PostgreSQL connected: {str(ver).split(',')[0]}")

        # Check pgvector
        res_ext = await session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        )
        ext_ver = res_ext.scalar()
        if not ext_ver:
            raise RuntimeError(
                "pgvector extension is NOT installed in database. "
                "Run 'uv run alembic upgrade head' after starting PostgreSQL."
            )
        print(f"      pgvector extension: v{ext_ver} (active)")


async def check_schema() -> None:
    print("[2/5] Checking schema tables & migration state...")
    async with AsyncSessionLocal() as session:
        # Verify core tables exist
        required_tables = [
            "users",
            "projects",
            "documents",
            "document_versions",
            "document_chunks",
            "chunk_embeddings",
            "search_terms",
            "search_postings",
            "search_project_stats",
        ]
        for tbl in required_tables:
            check_sql = text(
                "SELECT EXISTS ("
                "  SELECT FROM information_schema.tables "
                "  WHERE table_schema = 'public' AND table_name = :t"
                ");"
            )
            exists = (await session.execute(check_sql, {"t": tbl})).scalar()
            if not exists:
                raise RuntimeError(
                    f"Required table '{tbl}' does not exist. "
                    "Run 'uv run alembic upgrade head' to apply migrations."
                )

        # Verify alembic head
        res_alembic = await session.execute(
            text("SELECT version_num FROM alembic_version;")
        )
        current_rev = res_alembic.scalar()
        print(f"      Schema verified at revision: {current_rev}")


async def run_pipeline_and_search(persist_seed: bool) -> uuid.UUID:
    print("[3/5] Testing ingestion, chunking, and embedding generation...")
    async with AsyncSessionLocal() as session:
        # Get or create user
        stmt_user = select(User).where(User.email == "demo@nexus.local")
        user = (await session.execute(stmt_user)).scalars().first()
        if not user:
            user = User(email="demo@nexus.local", hashed_password="dev_local_hash")
            session.add(user)
            await session.commit()
            await session.refresh(user)

        # Get or create project
        proj_name = (
            "NEXUS Local Demo" if persist_seed else f"Smoke Test {uuid.uuid4().hex[:8]}"
        )
        project = None
        if persist_seed:
            stmt_proj = select(Project).where(
                Project.user_id == user.id, Project.name == proj_name
            )
            project = (await session.execute(stmt_proj)).scalars().first()

        if not project:
            project = Project(name=proj_name, user_id=user.id)
            session.add(project)
            await session.commit()
            await session.refresh(project)

        project_id = project.id

        # Ingest sample documents
        for doc in SAMPLE_DOCUMENTS:
            await save_document(
                db=session,
                project_id=project_id,
                canonical_url=doc["url"],
                source_domain="nexus.local",
                text=doc["text"],
                metadata={"title": doc["title"]},
            )

        # Ensure chunks & embeddings are fully processed
        pending = await process_pending_chunks(session, batch_size=32)
        print(
            f"      Ingested {len(SAMPLE_DOCUMENTS)} documents, processed {pending} new embedding chunks."
        )

    print("[4/5] Testing Lexical, Dense, and Hybrid (RRF) search...")
    async with AsyncSessionLocal() as session:
        # Test 1: Lexical BM25 search
        lex_results = await execute_search(
            db=session,
            project_id=project_id,
            query="Robertson-Spärck Jones BM25",
            mode=SearchMode.LEXICAL.value,
            top_k=3,
        )
        assert len(lex_results) > 0, "Lexical BM25 search returned 0 results"
        print(
            f"      [PASS] Lexical search (BM25): Top result '{lex_results[0].title}' (score={lex_results[0].bm25_score:.4f})"
        )

        # Test 2: Dense semantic search
        dense_results = await execute_search(
            db=session,
            project_id=project_id,
            query="vector similarity cosine embeddings",
            mode=SearchMode.DENSE.value,
            top_k=3,
        )
        assert len(dense_results) > 0, "Dense pgvector search returned 0 results"
        print(
            f"      [PASS] Dense search (pgvector): Top result '{dense_results[0].title}' (score={dense_results[0].dense_score:.4f})"
        )

        # Test 3: Hybrid RRF search
        hybrid_results = await execute_search(
            db=session,
            project_id=project_id,
            query=DEMO_QUERY,
            mode=SearchMode.HYBRID_FIXED.value,
            top_k=3,
            rrf_k=10,
        )
        assert len(hybrid_results) > 0, "Hybrid RRF search returned 0 results"
        assert any(
            c.bm25_rank is not None and c.dense_rank is not None for c in hybrid_results
        ), "Hybrid search did not fuse BM25 and dense rankings"
        print(f"      [PASS] Hybrid search (RRF k=10), '{DEMO_QUERY}':")
        for c in hybrid_results:
            print(
                f"        {c.final_rank}. {c.title}  rrf={c.rrf_score:.4f}  "
                f"bm25_rank={c.bm25_rank}  dense_rank={c.dense_rank}"
            )

        # Clean up if not seeding
        if not persist_seed:
            stmt_del_docs = select(Document).where(Document.project_id == project_id)
            docs = (await session.execute(stmt_del_docs)).scalars().all()
            for d in docs:
                await session.delete(d)
            await session.delete(project)
            await session.commit()
            print("      Cleaned up temporary smoke test project.")

    return project_id


async def check_api_server(
    project_id: uuid.UUID, is_seeded: bool, api_url: str
) -> None:
    print(f"[5/5] Checking live FastAPI HTTP server ({api_url})...")
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            # Confirm the port is actually serving NEXUS, not another local service
            resp_schema = await client.get(
                f"{api_url}{settings.API_V1_STR}/openapi.json"
            )
            title = (
                resp_schema.json().get("info", {}).get("title")
                if resp_schema.status_code == 200
                else None
            )
            if title != settings.PROJECT_NAME:
                print(
                    f"      [WARN] {api_url} is not serving the NEXUS API "
                    f"(OpenAPI title: {title!r}). Start NEXUS on a free port and pass --api-url."
                )
                return

            resp_health = await client.get(f"{api_url}/health")
            if resp_health.status_code == 200:
                print(f"      [PASS] GET /health -> 200 OK {resp_health.json()}")
            else:
                print(
                    f"      [WARN] GET /health returned status {resp_health.status_code}"
                )

            resp_docs = await client.get(f"{api_url}/docs")
            if resp_docs.status_code == 200:
                print("      [PASS] GET /docs -> 200 OK (Swagger UI available)")

            if is_seeded:
                resp_search = await client.get(
                    f"{api_url}{settings.API_V1_STR}/search",
                    params={
                        "project_id": str(project_id),
                        "q": DEMO_QUERY,
                        "mode": "hybrid_fixed",
                    },
                )
                if resp_search.status_code != 200:
                    raise RuntimeError(
                        f"GET /api/v1/search returned {resp_search.status_code}: {resp_search.text}"
                    )
                hits = resp_search.json().get("results", [])
                print(
                    f"      [PASS] GET /api/v1/search -> 200 OK ({len(hits)} results returned)"
                )
    except (httpx.ConnectError, httpx.ConnectTimeout):
        print(f"      [INFO] No server is running at {api_url} (API check skipped).")
        port = httpx.URL(api_url).port or 8000
        print(
            f"             Start the server with: uv run uvicorn app.main:app --port {port}"
        )
    except ValueError:
        print(
            f"      [WARN] {api_url} returned a non-JSON response; it is not the NEXUS API."
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS dev smoke test and seeder")
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Keep sample data in database as a persistent demo project for API testing",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Base URL of a running NEXUS API to check (default: http://127.0.0.1:8000)",
    )
    args = parser.parse_args()
    api_url = args.api_url.rstrip("/")

    print("=" * 65)
    print("NEXUS Runtime & Local Environment Smoke Verification")
    print(
        f"Database Target: {settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
    print("=" * 65)

    try:
        await check_database()
        await check_schema()
        project_id = await run_pipeline_and_search(persist_seed=args.seed)
        await check_api_server(project_id, is_seeded=args.seed, api_url=api_url)

        print("-" * 65)
        print("STATUS: ALL SMOKE CHECKS PASSED")
        if args.seed:
            search_url = f"{api_url}{settings.API_V1_STR}/search?project_id={project_id}&q={DEMO_QUERY.replace(' ', '+')}"
            print("\nPersistent Demo Project Seeded!")
            print(f"Project ID: {project_id}")
            print("\nSearch it once the API is running:")
            print(f'  curl "{search_url}"')
            print(f'  Invoke-RestMethod -Uri "{search_url}"')
        print("=" * 65)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"\n[FAIL] Smoke verification error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
