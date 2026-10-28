# NEXUS Benchmark Evaluation — SCIFACT (TRAIN split)

- **Split**: BEIR SciFact train = official SciFact train claims (809); not used for tuning.
- **Corpus**: 5183 documents
- **Queries**: 809 queries
- **Measured**: 2026-10-28 19:37 UTC at commit `d7a9b27`
- **Hardware**: Intel(R) Core(TM) i5-10200H CPU @ 2.40GHz, 8 logical CPUs, CPU-only
- **Models**: dense `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a`
- **Latency**: wall-clock per query from the benchmark client (query embedding + SQL + fusion), after one unmeasured warm-up query.

| Configuration | NDCG@10 | Recall@10 | Recall@100 | MRR | Latency Avg | Latency P95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (lexical) | 0.6690 | 0.7886 | 0.9149 | 0.6413 | 85.30 ms | 188.09 ms |
| Dense (dense) | 0.6759 | 0.8042 | 0.9306 | 0.6503 | 119.78 ms | 209.69 ms |
| Hybrid RRF (100/100, k=60) | 0.7112 | 0.8312 | 0.9522 | 0.6835 | 203.51 ms | 355.19 ms |
| Fixed Baseline (100/100, k=10) | 0.7180 | 0.8422 | 0.9522 | 0.6888 | 208.02 ms | 366.26 ms |
