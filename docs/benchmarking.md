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

### Paired significance tests

Every query is run through every configuration, so configurations are compared per query rather than only by their means. The harness compares the default (hybrid RRF k=10) with each other configuration on NDCG@10 and reports:

- the mean per-query difference;
- a 95% percentile-bootstrap confidence interval for that mean;
- a two-sided paired randomization (sign-flip) test;
- win / tie / loss counts.

Both resampling procedures use 10,000 resamples with a fixed seed (`scripts/benchmark/significance.py`).

The analysis was fixed before the first run that produced these tests:

- **Primary comparisons.** On the untuned 809-query split, k=10 against BM25, against dense, and against k=60, on NDCG@10.
- **Correction.** Holm–Bonferroni at a family-wise α = 0.05 across those three.
- **Tuning split.** These comparisons are reported but not counted as evidence. k=10 was selected on that split, so its comparison against k=60 there is biased in k=10's favour.

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

The 809-query report records commit `be29d7a`, which is where the significance tests were added and the analysis above was fixed. The 300-query report records `9f905c2`, which adds only the 809-query report, so both splits measure the same code. Both trees were clean.

Measured 2026-10-28 (UTC) on an Intel i5-10200H laptop (4 cores / 8 threads, 8 GB RAM, CPU only, torch 2.13 CPU, 4 intra-op threads), with `all-MiniLM-L6-v2@1110a243` and `ms-marco-MiniLM-L-6-v2@233902d2`.

**Untuned split (BEIR train, 809 queries)**

| Configuration | NDCG@10 | Recall@10 | Recall@100 | MRR | Avg | P95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (lexical) | 0.6690 | 0.7886 | 0.9149 | 0.6413 | 60 ms | 121 ms |
| Dense | 0.6759 | 0.8042 | 0.9306 | 0.6503 | 110 ms | 190 ms |
| Hybrid RRF k=60 | 0.7112 | 0.8312 | 0.9522 | 0.6835 | 185 ms | 322 ms |
| Hybrid RRF k=10 | **0.7180** | **0.8422** | **0.9522** | **0.6888** | 184 ms | 295 ms |

**Tuning split (BEIR test = official dev, 300 queries)**

| Configuration | NDCG@10 | Recall@10 | Recall@100 | MRR | Avg | P95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (lexical) | 0.6736 | 0.8105 | 0.9064 | 0.6388 | 55 ms | 109 ms |
| Dense | 0.6642 | 0.8096 | 0.9277 | 0.6287 | 113 ms | 221 ms |
| Hybrid RRF k=60 | 0.7043 | 0.8249 | 0.9577 | 0.6783 | 186 ms | 317 ms |
| Hybrid RRF k=10 | **0.7188** | **0.8568** | **0.9577** | **0.6850** | 181 ms | 284 ms |
| k=10 + rerank top 25 | 0.6752 | 0.8240 | n/a | 0.6386 | 1,028 ms | 1,394 ms |

Every quality metric in both tables matches the previous run of the harness (commit `d7a9b27` before history was rewritten, now `0ea9343`) to four decimals. The reranking result also matches the original study (0.6752 vs 0.6751). An earlier k=30 run (NDCG@10 0.7137 on the tuning split) was not repeated.

**Paired tests, default (k=10) vs each configuration, NDCG@10**

| Split | Compared with | Mean Δ | 95% CI | p | W / T / L |
|---|---|---:|---:|---:|---:|
| Untuned (809) | BM25 | +0.0491 | [+0.0357, +0.0626] | ≤ 0.0001 | 182 / 556 / 71 |
| Untuned (809) | Dense | +0.0421 | [+0.0278, +0.0565] | ≤ 0.0001 | 190 / 516 / 103 |
| Untuned (809) | Hybrid k=60 | +0.0069 | [+0.0018, +0.0120] | 0.0083 | 61 / 707 / 41 |
| Tuning (300) | BM25 | +0.0452 | [+0.0239, +0.0675] | 0.0002 | 71 / 198 / 31 |
| Tuning (300) | Dense | +0.0546 | [+0.0309, +0.0783] | 0.0001 | 76 / 190 / 34 |
| Tuning (300) | Hybrid k=60 | +0.0145 | [+0.0029, +0.0266] | 0.0167 | 24 / 256 / 20 |
| Tuning (300) | k=10 + rerank top 25 | +0.0436 | [+0.0159, +0.0708] | 0.0022 | 74 / 178 / 48 |

**Primary comparisons.** The three untuned-split comparisons were pre-specified. Their p-values (≤ 0.0001, ≤ 0.0001, 0.0083) are all below the Holm–Bonferroni thresholds (0.0167, 0.025, 0.05), so all three improvements hold at a family-wise α = 0.05.

**k=10 vs k=60.** The difference is real but small. NDCG@10 is identical for 707 of the 809 queries, and the mean gain is under one hundredth.

**Tuning-split rows.** These are descriptive only. The reranking comparison was not used to choose anything on that split, but it remains a single-corpus result.

### Interpreting latency

These runs were made while other containers shared the machine's 8 GB of memory, and latency varies between runs. For example, the previous run measured 366 ms P95 for hybrid k=10 on the untuned split; this one measured 295 ms. The k=10 and k=60 hybrid configurations do identical retrieval work, because k only changes the fusion arithmetic, so the gap between their latencies is noise. Use the latency columns to compare retriever families (single branch vs hybrid vs reranked), not neighbouring rows.

## Known limitations

- **Tuned split**: k=10 was chosen on the 300-query split; the 809-query split is the fair comparison.
- **One corpus**: significance is over queries from one corpus. It says the gains are unlikely to be chance on SciFact, not that they transfer to other domains.
- **Single machine, uncontrolled load**: latency depends on hardware and background load; no concurrency or load testing.
- **Exact dense scan**: no HNSW/IVFFlat index; dense latency grows linearly with corpus size.
- **Binary relevance**: SciFact labels are binary, which limits how much NDCG can discriminate between rankings.
- **Corpus scope**: one domain, 5,183 documents.
