# Benchmark Methodology

## Dataset

**SciFact** ([Allen AI](https://github.com/allenai/scifact)): scientific claims paired with evidence abstracts, in the [BEIR](https://github.com/beir-cellar/beir) distribution.

| Property | Value |
|---|---|
| Corpus | 5,183 scientific abstracts (title + abstract text) |
| BEIR "test" qrels | 300 queries = the official SciFact **dev** claims |
| BEIR "train" qrels | 809 queries = the official SciFact train claims |
| Relevance labels | Expert-annotated claim–evidence abstracts, binary |
| Source | `scifact.zip` from the BEIR download server (UKP, TU Darmstadt) |

[`scripts/benchmark/datasets/scifact.py`](../backend/scripts/benchmark/datasets/scifact.py) downloads and caches the zip under `backend/.cache/scifact/`.

### Split roles (read before comparing numbers)

The BEIR "test" split is not an untouched test set for NEXUS. The claim IDs and texts are identical to the official SciFact dev set, and NEXUS's RRF constant (k=10) and the learned-router study were both selected or run on these 300 queries. Results there are **tuned**. The 809 BEIR "train" queries were not used to choose any NEXUS parameter, so they serve as the **untuned** check. Official SciFact test labels are not public.

## Data preparation

`python -m scripts.benchmark prepare --dataset scifact`:

1. Downloads and extracts the BEIR SciFact zip (if not cached).
2. Creates a dedicated benchmark project.
3. Ingests all 5,183 documents via `save_document()` (BM25 indexing and chunking).
4. Embeds all chunks via `process_pending_chunks()` (`all-MiniLM-L6-v2`).
5. Records `BenchmarkMapping` rows linking SciFact IDs to NEXUS document IDs.
6. Writes `.cache/scifact_manifest.json`.

If a manifest says READY but its project is missing or incomplete in the database (for example after the database was reset), `prepare` re-ingests instead of trusting it. Ingestion takes roughly 15 minutes on a laptop CPU.

### Pre-flight validation

Before evaluation the harness aborts unless:

- the split has the expected query count (300 test / 809 train);
- the manifest's project exists;
- the project contains exactly 5,183 documents and 5,183 mappings.

These checks were added after an incompletely ingested project once produced misleading results (NDCG@10 0.4464 instead of 0.7188).

## Evaluation protocol

`python -m scripts.benchmark evaluate --dataset scifact --split {test,train} [--with-rerank]`

For each configuration:

1. Run one unmeasured warm-up query, so model loading is not counted as latency.
2. For each query, call `execute_search()` with the configuration's mode and parameters, record wall-clock latency, map retrieved document IDs to SciFact IDs and score them against the qrels.
3. Average the per-query metrics.

Configurations: BM25 only, dense only, hybrid RRF with k=60 and with k=10 (both depth 100/100). With `--with-rerank`, also k=10 plus cross-encoder reranking of the top 25. Reranked runs return 25 results, so their Recall@100 is reported as n/a.

The run writes `backend/docs/benchmark-evaluation-<split>.md` and a `.json` sidecar. These record the timestamp, git commit (`-dirty` if the working tree had changes), CPU, Python/torch/sentence-transformers versions, and the Hugging Face revisions of both models.

## Metrics

| Metric | Definition |
|---|---|
| **NDCG@10** | Normalized DCG at rank 10 (binary gains): ranking quality |
| **Recall@k** | Fraction of relevant documents in the top k (k = 10, 100) |
| **MRR** | Mean reciprocal rank of the first relevant document |
| **Latency avg / P95** | Wall-clock per query from the benchmark client (query embedding + SQL + fusion) |

Implementations: [`scripts/benchmark/metrics.py`](../backend/scripts/benchmark/metrics.py), unit-tested in [`tests/benchmark/test_metrics.py`](../backend/tests/benchmark/test_metrics.py) and checked by `scripts.benchmark verify`.

## Results

The generated reports record commit `d7a9b27`. That is the application code that was measured; the harness that produced them (warm-up query, k=60 row, split labels) was committed together with the reports in `e64a899`, before the dirty-tree marker existed.

Measured 2026-10-28 on an Intel i5-10200H laptop (4 cores / 8 threads, 8 GB RAM, CPU only, torch 2.13 CPU, 4 intra-op threads), with `all-MiniLM-L6-v2@1110a243` and `ms-marco-MiniLM-L-6-v2@233902d2`.

**Untuned split (BEIR train, 809 queries)**

| Configuration | NDCG@10 | Recall@10 | Recall@100 | MRR | Avg | P95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (lexical) | 0.6690 | 0.7886 | 0.9149 | 0.6413 | 85 ms | 188 ms |
| Dense | 0.6759 | 0.8042 | 0.9306 | 0.6503 | 120 ms | 210 ms |
| Hybrid RRF k=60 | 0.7112 | 0.8312 | 0.9522 | 0.6835 | 204 ms | 355 ms |
| Hybrid RRF k=10 | **0.7180** | **0.8422** | **0.9522** | **0.6888** | 208 ms | 366 ms |

**Tuning split (BEIR test = official dev, 300 queries)**

| Configuration | NDCG@10 | Recall@10 | Recall@100 | MRR | Avg | P95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (lexical) | 0.6736 | 0.8105 | 0.9064 | 0.6388 | 90 ms | 172 ms |
| Dense | 0.6642 | 0.8096 | 0.9277 | 0.6287 | 143 ms | 233 ms |
| Hybrid RRF k=60 | 0.7043 | 0.8249 | 0.9577 | 0.6783 | 230 ms | 369 ms |
| Hybrid RRF k=10 | **0.7188** | **0.8568** | **0.9577** | **0.6850** | 272 ms | 557 ms |
| k=10 + rerank top 25 | 0.6752 | 0.8240 | n/a | 0.6386 | 1,273 ms | 1,871 ms |

On the rebuilt index, the tuning-split quality numbers match the values published before this release exactly for BM25, dense and both hybrid settings; reranking reproduced at 0.6752 vs 0.6751. An earlier k=30 run (NDCG@10 0.7137 on the tuning split) was not repeated.

### Interpreting latency

These runs were made while other containers competed for the machine's 8 GB of memory. The k=10 and k=60 hybrid configurations execute identical retrieval work (k only changes the fusion arithmetic), so the gap between their latencies, for example 366 ms vs 355 ms or 557 ms vs 369 ms P95, is noise. Use the latency columns to compare retriever families (single branch vs hybrid vs reranked), not neighbouring rows. Earlier write-ups report lower P95 values (e.g. 313 ms for k=10) measured under different load.

## Known limitations

- **Tuned split**: k=10 was chosen on the 300-query split; the 809-query split is the fair comparison. Per-query significance tests have not been run.
- **Single machine, uncontrolled load**: latency depends on hardware and background load; no concurrency or load testing.
- **Exact dense scan**: no HNSW/IVFFlat index; dense latency grows linearly with corpus size.
- **Binary relevance**: SciFact labels are binary, which limits how much NDCG can discriminate between rankings.
- **Corpus scope**: one domain, 5,183 documents.
