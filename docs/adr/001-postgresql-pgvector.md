# ADR-001: PostgreSQL + pgvector as unified storage

## Status
Accepted

## Context
NEXUS needs to store documents, inverted index data, and vector embeddings. The standard approach in the search/RAG ecosystem is to use separate specialized stores: Elasticsearch for lexical search, a vector database (Pinecone, Weaviate, Qdrant) for dense retrieval, and a relational database for metadata.

## Problem
Using three separate data stores increases operational complexity, introduces cross-store consistency issues, and makes project-scoped isolation harder to enforce. For a system focused on measuring retrieval behavior rather than serving production traffic, this complexity is not justified.

## Options considered
1. **PostgreSQL + pgvector** — single database for everything
2. **PostgreSQL + Elasticsearch + pgvector** — specialized lexical store
3. **PostgreSQL + external vector DB** — hosted vector search

## Decision
Use PostgreSQL with the pgvector extension as the sole data store.

## Evidence
- The custom BM25 implementation stores its inverted index (terms, postings, document stats) in PostgreSQL tables, giving full control over scoring parameters and indexing behavior
- pgvector stores 384-dimensional embeddings and supports cosine distance queries via the `<=>` operator
- Project isolation is enforced through standard SQL foreign keys and WHERE clauses — one consistent isolation model
- Alembic migrations manage the schema evolution (5 migrations total)
- On the SciFact corpus (5,183 documents), exact pgvector scan achieves P95 latency of ~213ms — acceptable for a benchmark-evaluated system

## Trade-offs
- **Pro**: Single database, single backup strategy, single consistency model, simple deployment
- **Pro**: Full SQL expressiveness for benchmark queries and data validation
- **Con**: Exact scan does not scale to large corpora — HNSW indexing would be needed
- **Con**: No dedicated full-text search features (phrase matching, field boosting) that Elasticsearch provides
- **Con**: pgvector's exact scan is slower than purpose-built vector databases at scale

## Consequences
- All data access goes through SQLAlchemy async sessions
- Schema changes are managed by Alembic migrations
- Dense retrieval latency is bounded by pgvector scan performance
- Scaling beyond ~100K documents would likely require adding HNSW indexes
