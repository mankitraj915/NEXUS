# Known Limitations

This document explicitly describes what NEXUS does not do and where the system's boundaries are.

## Dense retrieval uses exact scan

pgvector queries use exact cosine distance computation (`ORDER BY embedding <=> query_embedding`). There is no HNSW or IVFFlat index. This means dense retrieval latency scales linearly with corpus size. On the 5,183-document SciFact corpus this is acceptable (dense-only P95 of 190–221 ms measured in this release), but would become a bottleneck at tens of thousands of documents.

Adding an HNSW index is a straightforward pgvector operation, but it has not been implemented or measured, so it is not claimed.

## Benchmark scope

Evaluation has been performed on a single dataset (SciFact, 5,183 scientific abstracts). The measured quality metrics (NDCG@10 ≈ 0.72 for RRF) apply to this specific corpus and domain. Performance on other domains (legal, e-commerce, general web) has not been measured and may differ.

The corpus size is small by production standards. Behavior at 100K+ documents has not been tested.

## No authentication or authorization

The API does not implement authentication. Project isolation is enforced at the data layer (project_id filtering), but any caller can access any project via the API. This is appropriate for an experimental retrieval testbed but would require tenant authentication tokens in production.

## Single-node deployment

NEXUS runs as a single FastAPI process with a single PostgreSQL instance. There is no:
- Horizontal scaling
- Read replicas
- Connection pooling beyond SQLAlchemy's default
- Background job queue (crawl and embedding run inline or via `asyncio.create_task`)
- Rate limiting on the API

## Query embedding blocks the event loop

`dense_search()` calls the synchronous `embed_query()` inside the async request handler, so while one request embeds its query, other requests on the same API process wait. BM25 and dense retrieval for a single hybrid query also run one after the other, because they share one `AsyncSession`, which does not support concurrent operations. Moving embedding to a worker thread would allow overlap but has not been implemented or measured.

## Embedding model is fixed

The system uses `all-MiniLM-L6-v2` (384 dimensions). Changing the model requires re-embedding the entire corpus. There is no multi-model embedding support or model versioning beyond the `model_name` / `model_version` columns in `chunk_embeddings`.

## Chunking is character-based

Text chunking uses a fixed 500-character sliding window with 100-character overlap. This does not respect sentence boundaries, paragraph structure, or semantic coherence. A sentence-aware chunking strategy would likely improve dense retrieval quality but has not been measured.

## No query caching

Search results are computed fresh for every query. There is no result cache, embedding cache, or query deduplication.

## Crawler limitations

- Single-domain crawling only (does not follow cross-domain links)
- No JavaScript rendering (cannot crawl SPAs)
- No sitemap parsing
- Basic politeness (concurrency semaphore) but no per-domain rate limiting
- Robots.txt is cached in-memory and not refreshed

## Test infrastructure

- Integration tests require a live PostgreSQL + pgvector instance. They run in a separate `<POSTGRES_DB>_test` database that they create and truncate, so the development database is never touched.
- CI (`.github/workflows/ci.yml`) has two jobs: `quality` (Ruff lint and format, Mypy, harness invariants, unit tests; no database or model downloads) and `integration` (pgvector service container, migrations on a fresh database, full test suite, end-to-end smoke test; downloads `all-MiniLM-L6-v2`). The workflow has not yet run on GitHub because the repository has not been published.
- The benchmark takes roughly 15 minutes to prepare (ingestion + embedding generation) on a laptop CPU.
- The full SciFact benchmark is not run in CI.

## Runtime costs

- Importing the application takes a few seconds, mostly NLTK pulling in SciPy and scikit-learn on the way to its Snowball stemmer.
- The first dense or hybrid query in a new process loads `all-MiniLM-L6-v2` and takes several seconds; later queries do not.
- On Linux, the locked PyPI `torch` wheel pulls CUDA libraries even though NEXUS only uses the CPU (a large download, but no extra RAM).

## Technical debt

- The `User` model exists in `app/domains/users/models.py` but authentication and user management are not yet implemented
- The `CrawlJob` response schema is in `crawler/schemas.py` but crawl job listing and pagination are not implemented
- Embedding pipeline uses `process_pending_chunks()` called inline during document save — no external distributed background worker
