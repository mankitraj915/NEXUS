# NEXUS Backend

Hybrid search engine backend. See the [root README](../README.md) for the overview,
architecture and benchmark results.

## Structure

```
app/
├── api/v1/              # FastAPI routes (projects, crawl, search)
├── config/              # Pydantic settings
├── db/                  # SQLAlchemy base, session, Alembic
└── domains/
    ├── benchmark/       # SciFact benchmark mapping model
    ├── crawler/         # Web crawler (BFS, SSRF, robots.txt)
    ├── documents/       # Document storage, versioning, dedup
    ├── embeddings/      # Chunking, embedding, pgvector search
    ├── projects/        # Multi-tenant project model
    ├── search/          # BM25, inverted index, query analysis
    │   ├── query/       # Feature extraction, classification, planning
    │   └── retrieval/   # Orchestrator, RRF fusion, reranker, router
    └── users/           # User model (auth not implemented)

scripts/
├── benchmark/           # SciFact evaluation harness
├── dev_smoke.py         # End-to-end smoke test and demo seeder
└── router_training/     # Learned router experiments

tests/
├── api/                 # API integration tests
├── benchmark/           # Metric, fusion, isolation, routing tests
├── crawler/             # Crawler unit + integration tests
└── search/              # Analyzer, invariants, storage consistency, integration
```

## Quick start

```bash
# 1. Start PostgreSQL + pgvector (from the repository root; host port 5433)
docker compose up -d

# 2. Install dependencies and configure the environment (from backend/)
cd backend
uv sync --all-groups
cp .env.example .env

# 3. Apply database migrations
uv run alembic upgrade head

# 4. Ingest a 3-document demo project, run lexical/dense/hybrid searches and keep the
#    project. Prints the project ID. The first run downloads all-MiniLM-L6-v2 (~90 MB).
uv run python -m scripts.dev_smoke --seed

# 5. Start the API
uv run uvicorn app.main:app --port 8000
```

Search the seeded project (replace `<PROJECT_ID>` with the ID printed in step 4):

```bash
curl -G "http://127.0.0.1:8000/api/v1/search" \
  --data-urlencode "project_id=<PROJECT_ID>" \
  --data-urlencode "q=hybrid search"
```

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/search?project_id=<PROJECT_ID>&q=hybrid+search"
```

`mode` is `hybrid_fixed` (default, RRF k=10), `lexical` or `dense`. Each result carries
`score` (higher is better: the RRF, BM25 or cosine score, depending on mode) and `rank`.
Interactive API documentation is at `http://127.0.0.1:8000/docs`. If port 8000 is taken,
start uvicorn on another port and pass `--api-url` to `scripts.dev_smoke`.

## Checks

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy .
uv run python -m scripts.benchmark verify   # IR math and RRF invariants, < 1 s, no DB
uv run pytest tests/                         # integration tests skip if Postgres is offline
```

Integration tests run against `<POSTGRES_DB>_test` (default `nexus_test`), which they
create on first use and truncate between tests; the development database is never
touched. Override the name with `NEXUS_TEST_POSTGRES_DB` (it must end in `_test`).
