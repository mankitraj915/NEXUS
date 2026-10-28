# NEXUS Benchmark Evaluation — SCIFACT (TEST split)

- **Split**: BEIR SciFact test = official SciFact dev claims (300). RRF k=10 was selected on this split, so the k=10 row is a tuned (optimistic) result.
- **Corpus**: 5183 documents
- **Queries**: 300 queries
- **Measured**: 2026-10-28 22:43 UTC at commit `9f905c2`
- **Hardware**: Intel(R) Core(TM) i5-10200H CPU @ 2.40GHz, 8 logical CPUs, CPU-only
- **Models**: dense `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a`
- **Latency**: wall-clock per query from the benchmark client (query embedding + SQL + fusion), after one unmeasured warm-up query.

| Configuration | NDCG@10 | Recall@10 | Recall@100 | MRR | Latency Avg | Latency P95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (lexical) | 0.6736 | 0.8105 | 0.9064 | 0.6388 | 55.42 ms | 108.95 ms |
| Dense (dense) | 0.6642 | 0.8096 | 0.9277 | 0.6287 | 112.65 ms | 221.04 ms |
| Hybrid RRF (100/100, k=60) | 0.7043 | 0.8249 | 0.9577 | 0.6783 | 186.35 ms | 317.43 ms |
| Fixed Baseline (100/100, k=10) | 0.7188 | 0.8568 | 0.9577 | 0.6850 | 180.75 ms | 284.17 ms |
| Baseline + CrossEncoder rerank (top 25) | 0.6752 | 0.8240 | n/a | 0.6386 | 1028.41 ms | 1393.66 ms |

## Paired comparison: Fixed Baseline (100/100, k=10) vs each configuration (NDCG@10)

Per-query differences (default minus other). 95% CI: percentile bootstrap of the mean difference; p: two-sided paired randomization test. Both use 10,000 resamples with a fixed seed. W / T / L: queries where the default scores higher / equal / lower.

| Compared with | Mean Δ NDCG@10 | 95% CI | p | W / T / L |
|---|---:|---:|---:|---:|
| BM25 (lexical) | +0.0452 | [+0.0239, +0.0675] | 0.0002 | 71 / 198 / 31 |
| Dense (dense) | +0.0546 | [+0.0309, +0.0783] | 0.0001 | 76 / 190 / 34 |
| Hybrid RRF (100/100, k=60) | +0.0145 | [+0.0029, +0.0266] | 0.0167 | 24 / 256 / 20 |
| Baseline + CrossEncoder rerank (top 25) | +0.0436 | [+0.0159, +0.0708] | 0.0022 | 74 / 178 / 48 |
