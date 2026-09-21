# ADR-002: Custom BM25 / Inverted Index

## Status
Accepted

## Context
NEXUS needs lexical retrieval to complement dense semantic search. Lexical retrieval is particularly important for exact-match queries, technical terminology, and identifiers that embedding models may not represent well.

## Problem
The standard approach is to delegate lexical search to Elasticsearch or a similar external search service. This provides proven performance and features but hides the retrieval internals — term frequency handling, IDF computation, document length normalization, and scoring parameters become opaque.

NEXUS was built to understand and measure retrieval behavior. Using an external search service would undermine this goal.

## Options considered
1. **Custom BM25 on PostgreSQL** — inverted index stored in SQL tables
2. **Elasticsearch** — external search engine
3. **Typesense / Meilisearch** — lightweight search engines
4. **PostgreSQL full-text search** — `tsvector` / `tsquery`

## Decision
Implement BM25 scoring from scratch with the inverted index stored in PostgreSQL tables.

## Evidence

The implementation in `search/bm25.py` uses the standard Robertson–Spärck Jones BM25 formula:

```
score(q, d) = Σ IDF(t) · (tf · (k1 + 1)) / (tf + k1 · (1 - b + b · |d| / avgdl))
```

Key implementation details:
- **k1 = 1.5, b = 0.75** — standard values, directly visible and modifiable
- **IDF smoothing**: `log((N - n + 0.5) / (n + 0.5) + 1)` — guarantees non-negative IDF
- **Positional posting lists**: Each posting stores term positions (integer array), enabling position-aware snippet extraction
- **Project-scoped statistics**: `ProjectStats` stores `total_documents` and `total_tokens` per project, ensuring avgdl is computed within the correct corpus

The text analyzer pipeline (`search/analyzer.py`) implements: NFKC normalization → lowercasing → regex tokenization → stopword removal (174 words) → Snowball stemming.

4 mathematical invariant tests verify BM25 properties:
- Zero TF produces zero score
- Scores are monotonic with TF (with diminishing returns)
- Shorter documents score higher for equal TF (b > 0)
- IDF remains non-negative even when a term appears in every document

## Trade-offs
- **Pro**: Full control over scoring parameters, observable and testable
- **Pro**: Positional data enables passage-level snippet extraction
- **Pro**: No external service dependency
- **Con**: Query performance limited by PostgreSQL table scan — not competitive with Lucene at scale
- **Con**: No phrase matching, proximity scoring, or field-level boosting
- **Con**: Index updates require SQL transactions

## Consequences
- Changing BM25 parameters (k1, b) requires code changes and re-benchmarking
- The inverted index grows with the corpus (terms table + postings table)
- Retrieval latency depends on the number of matching terms and postings
