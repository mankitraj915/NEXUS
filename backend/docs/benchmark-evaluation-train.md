# NEXUS Benchmark Evaluation — SCIFACT (TRAIN split)

- **Split**: BEIR SciFact train = official SciFact train claims (809); not used for tuning.
- **Corpus**: 5183 documents
- **Queries**: 809 queries
- **Measured**: 2026-10-28 22:35 UTC at commit `be29d7a`
- **Hardware**: Intel(R) Core(TM) i5-10200H CPU @ 2.40GHz, 8 logical CPUs, CPU-only
- **Models**: dense `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a`
- **Latency**: wall-clock per query from the benchmark client (query embedding + SQL + fusion), after one unmeasured warm-up query.

| Configuration | NDCG@10 | Recall@10 | Recall@100 | MRR | Latency Avg | Latency P95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (lexical) | 0.6690 | 0.7886 | 0.9149 | 0.6413 | 60.18 ms | 120.57 ms |
| Dense (dense) | 0.6759 | 0.8042 | 0.9306 | 0.6503 | 110.37 ms | 189.93 ms |
| Hybrid RRF (100/100, k=60) | 0.7112 | 0.8312 | 0.9522 | 0.6835 | 184.75 ms | 321.65 ms |
| Fixed Baseline (100/100, k=10) | 0.7180 | 0.8422 | 0.9522 | 0.6888 | 183.54 ms | 294.54 ms |

## Paired comparison: Fixed Baseline (100/100, k=10) vs each configuration (NDCG@10)

Per-query differences (default minus other). 95% CI: percentile bootstrap of the mean difference; p: two-sided paired randomization test. Both use 10,000 resamples with a fixed seed. W / T / L: queries where the default scores higher / equal / lower.

| Compared with | Mean Δ NDCG@10 | 95% CI | p | W / T / L |
|---|---:|---:|---:|---:|
| BM25 (lexical) | +0.0491 | [+0.0357, +0.0626] | 0.0001 | 182 / 556 / 71 |
| Dense (dense) | +0.0421 | [+0.0278, +0.0565] | 0.0001 | 190 / 516 / 103 |
| Hybrid RRF (100/100, k=60) | +0.0069 | [+0.0018, +0.0120] | 0.0083 | 61 / 707 / 41 |
