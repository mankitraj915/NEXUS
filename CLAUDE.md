# NEXUS

Hybrid search engine: custom BM25 inverted index + all-MiniLM-L6-v2 embeddings in
PostgreSQL/pgvector, fused with Reciprocal Rank Fusion (k=10). FastAPI backend in
`backend/`; Postgres runs from the root `docker-compose.yml` on host port 5433.

## Commands (run from `backend/`)

- Fast checks, no DB or model downloads: `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy .`, `uv run python -m scripts.benchmark verify`, `uv run pytest tests/`
  (integration tests skip when Postgres is offline).
- Integration tests use the database `<POSTGRES_DB>_test` (created automatically) and
  TRUNCATE it. Never point them at the dev database; it holds the SciFact benchmark index.
- Benchmark: `uv run python -m scripts.benchmark prepare --dataset scifact` (~15 min CPU),
  then `uv run python -m scripts.benchmark evaluate --dataset scifact --split test`.

## Conventions

- Code is `ruff format`ted; keep mypy clean.
- Only `hybrid_fixed`, `lexical` and `dense` are stable search modes; `experimental_*`
  modes (adaptive, learned router, multi-query, cross-encoder rerank) were measured and
  rejected or left inconclusive — see `docs/adr/`.
- The BEIR SciFact "test" split is the official SciFact dev set, and RRF k=10 was tuned on
  it. Report results on it as tuned, and never present unmeasured numbers.
