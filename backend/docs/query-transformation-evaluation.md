# NEXUS Phase 8 Query Transformation Evaluation

## 1. Quality / Latency Frontier
| Configuration | NDCG@10 | Recall@100 | Avg Candidates | Latency Avg | Latency P95 |
|---|---|---|---|---|---|
| Fixed Baseline (100/100) | 0.4464 | 0.8941 | 100.0 | 383.98ms | 785.23ms |
| Multi-Query Hybrid (Strategy A) | 0.4390 | 0.8941 | 100.0 | 883.46ms | 1919.91ms |
| Multi-Query Hybrid (Strategy B) | 0.4505 | 0.8941 | 100.0 | 541.84ms | 1106.81ms |

## 2. Multi-Query Analysis
Detailed results for Rule-based expansion comparing fusion Strategy A (Global Union) vs Strategy B (Two-Stage).
