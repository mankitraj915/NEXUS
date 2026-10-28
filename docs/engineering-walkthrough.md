# Engineering Walkthrough

This document traces how NEXUS evolved from an initial prototype to a benchmark-evaluated hybrid search engine. The chronology is reconstructed from the git history and code structure.

---

## Phase 1 — Foundation

**Commit**: `5353652` — *initial baseline snapshot before Stage 1*

The initial system established:

- **FastAPI application** with versioned API (`/api/v1/`)
- **PostgreSQL database** with SQLAlchemy 2.0 async ORM and Alembic migrations
- **Domain-driven structure**: `app/domains/` with separate packages for projects, users, documents, crawler, search, embeddings, and benchmark
- **Web crawler**: BFS crawl engine with a URL frontier, robots.txt compliance, SSRF protection (blocking private/loopback IPs), HTML parsing (BeautifulSoup), link extraction, and URL normalization
- **Document storage**: Content-hash deduplication, version tracking, metadata extraction
- **Custom BM25 implementation**: Text analyzer (NFKC normalization, Snowball stemming, 174 stopwords), inverted index with positional posting lists, BM25 scoring (k1=1.5, b=0.75)
- **Dense retrieval**: `all-MiniLM-L6-v2` sentence-transformer, 500-char chunking with 100-char overlap, pgvector cosine distance
- **Hybrid retrieval**: Reciprocal Rank Fusion, search orchestrator with parallel dispatch
- **Benchmark harness**: SciFact dataset loader, evaluation metrics (NDCG, Recall, Precision, MRR), experiment tracking

At this point, the system was functional but had a critical isolation gap: documents were globally unique by URL rather than scoped per project.

---

## Phase 2 — Multi-tenant document isolation

**Commit**: `031c3e0` — *fix: scope document uniqueness by project and add migration*

**Problem**: The original `Document` model had a unique constraint on `canonical_url` alone. This meant if Project A and Project B crawled the same URL, one of them would silently share the other's document record — breaking project isolation.

**Fix**: 
- Added `UniqueConstraint('project_id', 'canonical_url')` to the Document model
- Created Alembic migration `7a1b8c2d3e4f` to add the multi-tenant constraint
- Updated `save_document()` to query by `(project_id, canonical_url)` instead of `canonical_url` alone

This was a data model change with database migration — not just a code fix.

---

## Phase 3 — Isolation test infrastructure

**Commit**: `065f3ec` — *test: implement real tenant isolation and IR invariant tests*

Built a comprehensive test suite covering:

1. **Analyzer invariants**: NFKC normalization, stemming determinism, stopword removal
2. **BM25 mathematical properties**: Zero-TF produces zero score, monotonicity with TF, diminishing returns, document length normalization, non-negative IDF
3. **RRF correctness**: Formula verification (`1/(k+rank)`), missing retriever handling, deterministic ordering
4. **Query feature extraction bounds**: All features within [0,1], deterministic
5. **API contract validation**: Mode enum completeness, normalization, backwards compatibility
6. **Project isolation** (integration): Cross-project URL coexistence, same-project deduplication, lexical retrieval isolation, statistics isolation

The test infrastructure uses `pytest-asyncio` with real PostgreSQL connections, table truncation between tests, and a `is_db_reachable()` guard that skips integration tests when the database is offline.

---

## Phase 4 — Benchmark hardening

**Commit**: `9cbbb46` — *fix: make benchmark split explicit and guard project selection*

After encountering a benchmark anomaly (NDCG@10 = 0.4464 instead of ~0.72), the benchmark harness was hardened:

- Made the `--split` argument required and explicit (`test` or `train`)
- Added pre-flight query count assertions (300 for test, 809 for train)
- Added project existence verification before evaluation
- Added corpus completeness check (`COUNT(documents) == 5183`)
- Added `BenchmarkMapping` count verification

The anomaly was caused by evaluating against a project with incomplete data. The fix ensures the benchmark cannot silently produce invalid results.

---

## Phase 5 — API and static analysis cleanup

**Commit**: `dd45d46` — *fix: synchronize search API modes and remove machine-specific paths*

- Synchronized `SearchMode` enum to cover all orchestrator modes (stable + experimental)
- Added `SearchMode.is_valid_mode()` and `SearchMode.normalize_mode()` for safe API input handling
- Removed hardcoded machine-specific filesystem paths
- Added Ruff lint rule exceptions with documented rationale (`B008` for FastAPI Depends, `BLE001` for fallback routing, `S110` for cache loading)

**Commit**: `702db32` — *fix: repair static analysis configuration and clean Ruff/Mypy baseline*

- Fixed Ruff and mypy configuration in `pyproject.toml`
- Achieved zero Ruff errors and zero mypy errors across the codebase

---

## Phase 6 — Routing experiments

Across the development history, three experimental approaches were implemented and evaluated:

### Learned query routing

- Implemented query feature extraction (8 features: token count, unique ratio, identifier density, quoted term density, term rarity, lexical specificity, NL score, complexity)
- Implemented query classification (IDENTIFIER_HEAVY, NATURAL_LANGUAGE_QUESTION, TECHNICAL, etc.)
- Built an exhaustive action-outcome matrix: 6 retrieval configurations × 300 queries (BEIR test = official SciFact dev)
- Trained Decision Tree and Random Forest classifiers with GroupKFold cross-validation
- Performed regret analysis: mean NDCG regret = −0.0446, 19.7% of queries degraded >0.05

**Outcome**: Rejected. Quality regression was unacceptable.

### Cross-encoder reranking

- Implemented `LocalCrossEncoderReranker` with lazy model loading (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Tested reranking at depths 10, 25, 50
- NDCG@10 decreased at every depth while latency increased dramatically

**Outcome**: Rejected. No quality improvement, significant latency cost.

### LLM query transformation

- Implemented `LLMQueryTransformer` with semantic drift filtering, variant caching, and multi-query RRF (Strategy A and B)
- Implemented `OpenAICompatibleProvider` with configuration validation and secret protection
- Tested on SciFact with 1,109 queries
- 99.5% of queries failed due to API quota exhaustion

**Outcome**: Inconclusive — infrastructure failure, not algorithmic failure.

---

## Phase 7 — Integration verification

**Commit**: `b91f9eb` — *test: verify live database tenant isolation and canonical 300-query benchmark*

Final integration tests verifying:
- End-to-end search (ingest → index → embed → search → results)
- Dense and lexical retrieval returning correct results
- Cross-project isolation enforced at the API level
- Configuration test suite for LLM provider (missing key detection, env override, secret logging prevention)

---

## Current state

The system is an empirically evaluated hybrid search engine with:
- A measured production baseline (Hybrid RRF $k=10$: NDCG@10 = 0.7188, Recall@100 = 0.9577)
- An empirical hyperparameter sweep on $k \in \{10, 30, 60\}$ documenting the rank damping mechanism
- Three documented experimental rejections (Learned Router, Cross-Encoder, LLM Query Expansion)
- 45 automated tests protecting mathematical invariants, analyzer determinism, and multi-tenant isolation
- Zero static analysis violations across all 80 source files (`uv run ruff check .` and `uv run mypy .`)
- Instant offline benchmark verification (`uv run python -m scripts.benchmark verify`) alongside full live-database evaluation on SciFact
