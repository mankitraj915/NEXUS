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

# Verify retrieval invariants & IR math (< 0.2s, no DB required)
uv run python -m scripts.benchmark verify

# Run test suite
uv run pytest tests/ -v

# Start server
uv run uvicorn app.main:app --reload --port 8000

# Lint & Typecheck (repository-wide)
uv run ruff check .
uv run mypy .
```


