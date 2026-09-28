# NEXUS Backend

Hybrid search engine backend. See the [root README](../README.md) for an overview.

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
└── router_training/     # Learned router experiments

tests/
├── api/                 # API integration tests
├── benchmark/           # Metric, fusion, isolation, routing tests
├── crawler/             # Crawler unit + integration tests
└── search/              # Analyzer, invariant, integration tests
```

## Quick start

```bash
# 1. Start PostgreSQL + pgvector (service: postgres on host port 5433)
docker compose up -d

# 2. Install dependencies
uv sync --all-groups

# 3. Configure environment
cp .env.example .env

# 4. Apply database migrations
uv run alembic upgrade head

# 5. Verify local installation & seed demo project (< 2s)
uv run python -m scripts.dev_smoke --seed

# 6. Verify retrieval invariants & IR math (< 0.2s, no DB required)
uv run python -m scripts.benchmark verify

# 7. Run full test suite (45 tests)
uv run pytest tests/ -v

# 8. Start development server
uv run uvicorn app.main:app --port 8000

# 9. Lint & Typecheck (repository-wide)
uv run ruff check .
uv run mypy .
```

### Testing Search

Once the server is running on port 8000, query the seeded demo project:

**PowerShell:**
```powershell
# Hybrid RRF Search (default k=10)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/search?project_id=9cd45a55-1687-444f-9f2d-9ff7c99dd507&q=hybrid+search&mode=hybrid_fixed"

# Lexical BM25 Search
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/search?project_id=9cd45a55-1687-444f-9f2d-9ff7c99dd507&q=bm25&mode=lexical"

# Dense pgvector Semantic Search
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/search?project_id=9cd45a55-1687-444f-9f2d-9ff7c99dd507&q=semantic+embeddings&mode=dense"
```

**curl:**
```bash
curl -G "http://127.0.0.1:8000/api/v1/search" \
  --data-urlencode "project_id=9cd45a55-1687-444f-9f2d-9ff7c99dd507" \
  --data-urlencode "q=hybrid search" \
  --data-urlencode "mode=hybrid_fixed"
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.



