---
paths:
  - "backend/app/domains/search/**"
  - "backend/app/domains/embeddings/**"
  - "backend/app/domains/documents/**"
---

# Retrieval code

- Any change that can alter rankings (tokenization, stemming, BM25 parameters, chunking,
  embedding model, fusion, tie-breaking) needs a test in `backend/tests/search/` and either a
  re-measurement on both SciFact splits (`/reproduce-benchmark`) or an explicit "not
  re-measured" note. Do not update published numbers without a new report.
- Every path that adds or removes postings must keep `Term.document_frequency` and
  `ProjectStats` consistent; reuse `index_document_version` / `deindex_document_version` in
  `search/indexer.py` rather than writing postings directly.
- Never run retrieval branches as concurrent tasks on one `AsyncSession` (it is not
  concurrency-safe; this once made hybrid silently BM25-only). Stable modes must let
  retriever errors propagate rather than fall back to one branch.
- Rankings must be deterministic: keep explicit tie-breakers (BM25 by document id, dense by
  chunk id) when touching ordering.
- Changing the embedding model, its dimension or chunking invalidates stored embeddings and
  the benchmark index (`prepare --force-prepare`) and needs a migration if the vector size changes.
- Keep it CPU and 8 GB friendly: lazy-load models, no models larger than MiniLM-class
  in the default path.
