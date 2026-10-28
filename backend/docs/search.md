# NEXUS Search Architecture

Phase 3 introduced a custom-built Inverted Index and BM25 retrieval engine to NEXUS. We deliberately avoided black-box search solutions (like Elasticsearch or Postgres `tsvector`) to establish a solid foundation in Information Retrieval engineering.

## Text Analysis Pipeline

The Text Analyzer (`search/analyzer.py`) transforms raw text into an optimized sequence of indexable tokens:
1. **Unicode Normalization (NFKC)**: Standardizes ligatures and full-width characters.
2. **Case Normalization**: Lowercases all input.
3. **Tokenization**: Uses `\b[\w'-]+\b` to extract words while preserving internal hyphens (e.g., `state-of-the-art`) and apostrophes.
4. **Stopword Removal**: Filters out a hardcoded set of 150 highly frequent English terms to save index space and computation.
5. **Stemming**: Utilizes `nltk.stem.snowball.SnowballStemmer` to reduce words to their morphological roots (e.g., "running" -> "run").

## Inverted Index Schema

The index is stored structurally in PostgreSQL to maintain transactional consistency with documents:
- **`search_project_stats`**: Tracks the total documents (`N`) and average document length (`avgdl`) for BM25 parameters per project.
- **`search_document_stats`**: Tracks status (PENDING/INDEXED) and token counts (`|D|`) for specific document versions.
- **`search_terms`**: The lexicon. Tracks `document_frequency` (`n(q)`) uniquely per project.
- **`search_postings`**: The core index. Links a `term_id` and `document_version_id`, storing the `term_frequency` (`f(q,D)`) and a JSON array of `positions`.

## Indexing Lifecycle

1. `DocumentVersion` is created. If it replaces an earlier version of the same document, `deindex_document_version()` first removes the old version's postings, decrements term document frequencies and project statistics, and deletes its chunks.
2. `index_document_version(db, version_id)` is invoked synchronously.
3. Analyzer converts the raw text into a frequency map and positions list.
4. Using Postgres `ON CONFLICT DO UPDATE` (upserts), `ProjectStats` and `Terms` are atomically updated.
5. `Postings` are inserted in bulk.
6. The `DocumentStats` row is marked `INDEXED`.

## BM25 Retrieval

Queries execute in `search/bm25.py`:
1. The query is analyzed via the exact same Analyzer used during indexing.
2. `search_terms` is queried to fetch `document_frequency` for IDF calculation.
3. **Candidate Retrieval**: `search_postings` is queried *only* for the specific query terms, constrained strictly to the authenticated `project_id`. This guarantees **100% strict project isolation** without full table scans.
4. Candidates are scored dynamically using the standard Okapi BM25 formula (k1=1.5, b=0.75).
5. A min-heap (`heapq`) extracts the Top-K results dynamically in `O(N log K)` time.
6. Results are enriched with a snippet and returned via the `GET /api/v1/search` API.

## Rebuild Strategy (Future)

Because `search_document_stats` explicitly tracks the indexing state of every document, an index rebuild simply entails deleting all rows for a project in `search_project_stats`, `search_terms`, and `search_postings`, flipping all `DocumentStats` to `PENDING`, and running a background worker to reprocess the canonical `DocumentVersion.extracted_text`.
