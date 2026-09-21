# Benchmark Methodology

## Dataset

**SciFact** ([Allen AI](https://github.com/allenai/scifact)) — a dataset of scientific claims paired with evidence abstracts from research papers.

| Property | Value |
|---|---|
| Corpus | 5,183 scientific abstracts |
| Train queries | 809 |
| Test queries | 300 (canonical evaluation split) |
| Relevance labels | Expert-annotated claim-evidence pairs |
| Graded relevance | Binary (relevant / not relevant) |

The corpus and query splits are loaded via [`scripts/benchmark/datasets/scifact.py`](../backend/scripts/benchmark/datasets/scifact.py), which downloads and caches the HuggingFace dataset.

## Why SciFact

SciFact is a standard information retrieval benchmark with several properties useful for evaluating NEXUS:

- **Domain-specific vocabulary**: Scientific terminology tests BM25's lexical matching and stemming behavior
- **Moderate corpus size**: 5,183 documents allow exhaustive evaluation without requiring distributed infrastructure
- **Expert annotations**: Human-labeled relevance judgments (not synthetic)
- **Public and reproducible**: Available via HuggingFace Datasets

## Data preparation

The benchmark harness (`scripts/benchmark/__main__.py prepare`) performs:

1. Download SciFact from HuggingFace
2. Create a dedicated benchmark project in PostgreSQL
3. Ingest all 5,183 documents via `save_document()` (triggers BM25 indexing)
4. Generate chunk embeddings via `process_pending_chunks()` (triggers pgvector storage)
5. Create `BenchmarkMapping` records linking SciFact external IDs to NEXUS document UUIDs
6. Write a manifest file (`.cache/scifact_manifest.json`) with project ID and status
7. Verify: mapping count matches corpus count

### Pre-flight validation

Before evaluation, the harness verifies:

- **Query count**: 300 queries for test split, 809 for train split — aborts if mismatch
- **Project existence**: The benchmark project ID from the manifest must exist in the database
- **Corpus completeness**: `COUNT(documents) WHERE project_id = ?` must equal 5,183
- **Mapping integrity**: `COUNT(benchmark_mappings)` must match expected corpus size

These checks were added after a benchmark anomaly where an incompletely ingested project produced misleading results (NDCG@10 = 0.4464 instead of 0.7188).

## Evaluation protocol

For each experiment configuration:

1. Load the 300 test queries
2. For each query:
   - Execute `execute_search()` with the experiment's mode and parameters
   - Record wall-clock latency
   - Map retrieved NEXUS document IDs back to SciFact external IDs via `BenchmarkMapping`
   - Compute per-query metrics against ground-truth qrels
3. Aggregate metrics across all queries

## Metrics

| Metric | Definition | Purpose |
|---|---|---|
| **NDCG@10** | Normalized Discounted Cumulative Gain at rank 10 | Measures ranking quality — are relevant documents ranked higher? |
| **Recall@100** | Fraction of relevant documents retrieved in top 100 | Measures retrieval completeness — are relevant documents found at all? |
| **Precision@k** | Fraction of top-k results that are relevant | Measures result set quality |
| **MRR** | Mean Reciprocal Rank | Position of the first relevant result |
| **Jaccard@k** | Set overlap between two result lists | Measures agreement between retrievers |
| **Latency P50/P95/P99** | Latency percentiles across all queries | Measures performance distribution |

Implementations are in [`scripts/benchmark/metrics.py`](../backend/scripts/benchmark/metrics.py), with 5 dedicated unit tests in [`tests/benchmark/test_metrics.py`](../backend/tests/benchmark/test_metrics.py).

## Baseline experiments

The standard evaluation runs three configurations:

| Configuration | Mode | BM25 Depth | Dense Depth | RRF k |
|---|---|---:|---:|---:|
| BM25 (lexical only) | `lexical` | 100 | 0 | — |
| Dense (semantic only) | `dense` | 0 | 100 | — |
| Fixed Hybrid (production baseline) | `hybrid_fixed` | 100 | 100 | 10 |

## Reproducing the benchmark

```bash
cd backend

# 1. Ensure PostgreSQL + pgvector is running
docker compose up -d

# 2. Run migrations
uv run alembic upgrade head

# 3. Prepare corpus (downloads SciFact, ingests, generates embeddings)
#    Takes ~10–20 minutes depending on hardware
uv run python -m scripts.benchmark prepare --dataset scifact

# 4. Evaluate on the canonical 300-query test split
uv run python -m scripts.benchmark evaluate --dataset scifact --split test

# 5. Results are printed to stdout and written to docs/benchmark-evaluation-test.md
```

### Force re-preparation

If you need to re-ingest from scratch:

```bash
uv run python -m scripts.benchmark prepare --dataset scifact --force-prepare
```

### Evaluate on train split

```bash
uv run python -m scripts.benchmark evaluate --dataset scifact --split train
```

## Known limitations

- **Single-machine evaluation**: Latency measurements are hardware-dependent and include database query time. Results from different machines are not directly comparable.
- **Exact dense scan**: The pgvector query does not use HNSW indexing. Dense retrieval latency will increase with corpus size.
- **Binary relevance**: SciFact uses binary relevance labels. NDCG@k treats these as graded relevance with values 0/1, which reduces the discriminative power of the metric compared to multi-level relevance scales.
- **Corpus scope**: 5,183 documents is small by production search standards. Behavior at larger corpus sizes has not been measured.
