# Dense Semantic Retrieval Architecture

Phase 4 introduces an independent Dense Semantic Retrieval pipeline utilizing `sentence-transformers` and `pgvector`. This enables search queries based on mathematical meaning instead of just explicit keyword overlaps.

## Architecture

Embeddings are strictly treated as **derived data**. The document is the authoritative source of truth.

1. **Chunking (`embeddings/chunker.py`)**: 
   When a new `DocumentVersion` is indexed, the text is split deterministically into semantic chunks (currently `chunk_size=500`, `overlap=100` characters). This ensures embeddings represent meaningful passages, avoiding the dilution that occurs when embedding entire articles at once.
2. **Embedding Pipeline (`embeddings/pipeline.py`)**:
   Pending chunks are swept in batches (e.g., 32 at a time). We use `all-MiniLM-L6-v2` via `SentenceTransformer`, generating L2-normalized 384-dimensional dense vectors.
3. **Storage (`embeddings/models.py`)**:
   Embeddings are saved to PostgreSQL using the `pgvector` extension.
4. **Search (`embeddings/search.py`)**:
   - The user's query is embedded.
   - We execute a strict project-isolated nearest-neighbor search (`cosine_distance` over `pgvector`).
   - The results are max-pooled by `DocumentVersion`: the document receives a score corresponding to its highest-matching chunk.

## API Usage

The search endpoint `GET /api/v1/search` now accepts a `mode` parameter:

```http
GET /api/v1/search?project_id=<id>&q=vector+database&mode=dense
```

By keeping `mode=dense` separate from `mode=lexical`, we ensure pristine, non-interfering environments for A/B testing and algorithmic evaluation.
