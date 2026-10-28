# NEXUS

Hybrid search engine: custom BM25 inverted index + all-MiniLM-L6-v2 chunk embeddings in
PostgreSQL/pgvector, fused with Reciprocal Rank Fusion (k=10). FastAPI backend in
`backend/`; Postgres runs from the root `docker-compose.yml` on host port 5433.
Must stay runnable on an 8 GB RAM, CPU-only laptop with no paid services.

## Commands (run from `backend/`)

- Fast checks, no DB or model downloads (same as the CI `quality` job): `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run mypy .`, `uv run python -m scripts.benchmark verify`,
  `uv run pytest tests/` (integration tests skip when Postgres is offline).
- With Postgres up (`docker compose up -d` at the repo root): `uv run alembic upgrade head`,
  `uv run pytest tests/ -rs` (confirm nothing skipped), `uv run python -m scripts.dev_smoke --seed`.
- Release checks: `/verify-release`. Benchmark reproduction: `/reproduce-benchmark`.

## Invariants

- Integration tests use `<POSTGRES_DB>_test` (auto-created, TRUNCATEd; `tests/conftest.py`
  refuses names not ending in `_test`). Never point them at the dev database: it holds the
  5,183-document SciFact benchmark index.
- One live version per document: saving changed content de-indexes the previous version
  (postings, stats, chunks, term DF) before indexing the new one.
- API `score` is the score that produced the ranking (higher is better); `rank` is 1-based.
- Only `hybrid_fixed`, `lexical` and `dense` are stable search modes; `experimental_*`
  modes (adaptive, learned router, multi-query, cross-encoder rerank) were measured and
  rejected or left inconclusive — see `docs/adr/`.
- `import app.main` must not import torch; heavy models load lazily.

## Results and claims

- The BEIR SciFact "test" split (300 queries) is the official SciFact dev set, and RRF k=10
  was tuned on it: report it as tuned. The 809-query "train" split is the untuned check.
- Measured numbers live in the generated `backend/docs/benchmark-evaluation-{test,train}.{md,json}`.
  Cite those; never present unmeasured numbers. Latency is load-dependent and indicative only.

## Pitfalls

- The first dense/hybrid query in a process takes seconds (model load).
- Port 8000 may be taken by another local project: run uvicorn on another port and pass
  `--api-url` to `scripts.dev_smoke`.
- `backend/.cache/` (dataset, benchmark manifest) and `.agents/` are local and gitignored.
