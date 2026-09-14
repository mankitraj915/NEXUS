# NEXUS Benchmark Evaluation — SCIFACT (TEST split)

- **Corpus**: 5183 documents
- **Queries**: 300 queries
- **Project ID**: `e8349fda-ba77-41b2-bed1-f48d93eb8dbf`

| Configuration | NDCG@10 | Recall@100 | Avg Candidates | Latency Avg | Latency P95 |
|---|---|---|---|---|---|
| BM25 (lexical) | 0.6736 | 0.9064 | 99.9 | 163.81ms | 411.80ms |
| Dense (dense) | 0.6642 | 0.9277 | 100.0 | 265.90ms | 566.23ms |
| Fixed Baseline (100/100, k=10) | 0.7188 | 0.9577 | 100.0 | 854.18ms | 2483.51ms |
