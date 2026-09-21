# NEXUS Benchmark Evaluation — SCIFACT (TEST split)

- **Corpus**: 5183 documents
- **Queries**: 300 queries
- **Project ID**: `a05727a1-0dec-4d39-8a3e-aab1be8f3706`

| Configuration | NDCG@10 | Recall@100 | Avg Candidates | Latency Avg | Latency P95 |
|---|---|---|---|---|---|
| BM25 (lexical) | 0.6736 | 0.9064 | 99.9 | 84.32ms | 201.40ms |
| Dense (dense) | 0.6642 | 0.9277 | 100.0 | 146.86ms | 272.41ms |
| Fixed Baseline (100/100, k=10) | 0.7188 | 0.9577 | 100.0 | 205.51ms | 344.81ms |
