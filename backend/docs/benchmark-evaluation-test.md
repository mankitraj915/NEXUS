# NEXUS Benchmark Evaluation — SCIFACT (TEST split)

- **Split**: BEIR SciFact test = official SciFact dev claims (300). RRF k=10 was selected on this split, so the k=10 row is a tuned (optimistic) result.
- **Corpus**: 5183 documents
- **Queries**: 300 queries
- **Measured**: 2026-10-28 19:28 UTC at commit `d7a9b27`
- **Hardware**: Intel(R) Core(TM) i5-10200H CPU @ 2.40GHz, 8 logical CPUs, CPU-only
- **Models**: dense `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a`
- **Latency**: wall-clock per query from the benchmark client (query embedding + SQL + fusion), after one unmeasured warm-up query.

| Configuration | NDCG@10 | Recall@10 | Recall@100 | MRR | Latency Avg | Latency P95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (lexical) | 0.6736 | 0.8105 | 0.9064 | 0.6388 | 89.89 ms | 172.47 ms |
| Dense (dense) | 0.6642 | 0.8096 | 0.9277 | 0.6287 | 143.43 ms | 232.64 ms |
| Hybrid RRF (100/100, k=60) | 0.7043 | 0.8249 | 0.9577 | 0.6783 | 229.91 ms | 369.39 ms |
| Fixed Baseline (100/100, k=10) | 0.7188 | 0.8568 | 0.9577 | 0.6850 | 272.44 ms | 556.97 ms |
| Baseline + CrossEncoder rerank (top 25) | 0.6752 | 0.8240 | n/a | 0.6386 | 1272.61 ms | 1870.93 ms |
