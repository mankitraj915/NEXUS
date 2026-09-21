# ADR-006: Project-Scoped Document Isolation

## Status
Accepted

## Context
NEXUS supports multiple projects, each representing an independent document corpus. Two projects may crawl the same URL (e.g., documentation sites). The system must ensure that:
1. The same URL can exist in multiple projects independently
2. Searching one project never returns documents from another
3. Corpus statistics (document count, average document length) are computed per-project

## Problem
The original document model had a unique constraint on `canonical_url` alone. If Project A and Project B crawled `https://example.com/docs`, one project would silently share the other's document record, breaking isolation.

## Decision
Enforce project-scoped isolation at the database level with a composite unique constraint and per-project filtering on all query paths.

## Implementation

### Database constraints
- `UniqueConstraint('project_id', 'canonical_url', name='uix_project_canonical_url')` on `documents` table
- `UniqueConstraint('project_id', 'term', name='uix_project_term')` on `search_terms` table
- `ProjectStats` keyed by `project_id` — corpus statistics are per-project
- `DocumentStats.project_id` — token counts are per-project

### Query-level filtering
Every data access path includes `WHERE project_id = ?`:
- BM25 retrieval: `ProjectStats.project_id`, `Term.project_id`, `DocumentStats.project_id`
- Dense retrieval: `Document.project_id` in the embedding join query
- Document service: `Document.project_id` + `Document.canonical_url` for deduplication

### Migration
Alembic migration `7a1b8c2d3e4f` added the composite unique constraint to the existing `documents` table.

## Evidence

5 dedicated tests verify isolation:

| Test | What it verifies |
|---|---|
| `test_hermetic_candidate_metadata_project_isolation` | Candidate structures do not merge across projects (unit test, no DB) |
| `test_identical_url_coexistence_in_different_projects` | Same URL ingested into Project A and B creates distinct document IDs |
| `test_same_project_deduplication` | Re-crawling the same URL in the same project returns the existing version |
| `test_lexical_retrieval_isolation` | Querying Project A for Project B's unique term returns zero results |
| `test_statistics_isolation` | Project A's unique term does not appear in Project B's inverted index |

## Trade-offs
- **Pro**: Strong isolation guarantee at the data layer — no application-level bugs can leak cross-project data
- **Pro**: Standard SQL constraints — enforced by the database engine
- **Con**: Duplicate storage if the same URL is crawled by multiple projects (each gets its own copy)
- **Con**: No cross-project search capability (each project is fully isolated)

## Consequences
- Adding a "shared corpus" feature would require relaxing the isolation model
- All new query paths must include the `project_id` filter — this is enforced by test coverage
