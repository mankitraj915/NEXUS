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
# Start PostgreSQL + pgvector
docker compose up -d

# Install dependencies
uv sync --all-groups

# Configure
cp .env.example .env

# Migrate
uv run alembic upgrade head

# Run unit tests (no DB required)
uv run pytest tests/search/test_invariants.py tests/search/test_analyzer.py tests/benchmark/test_fusion.py tests/benchmark/test_metrics.py tests/benchmark/test_query_routing.py -v

# Run all tests (requires live DB)
uv run pytest tests/ -v

# Start server
uv run uvicorn app.main:app --reload --port 8000

# Lint
uv run ruff check app/ tests/
uv run mypy app/
```
