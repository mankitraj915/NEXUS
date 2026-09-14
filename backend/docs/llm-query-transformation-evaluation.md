# NEXUS Phase 8B Query Transformation Evaluation

## 1. Quality / Latency Frontier
| Configuration | NDCG@10 | Recall@100 | Latency P95 |
|---|---|---|---|
| A. Fixed RRF | 0.4464 | 0.8941 | 383.84ms |
| B. Rule-Based Multi-Query (Strategy A) | 0.4392 | 0.8941 | 928.76ms |
| C. LLM (1 variant / weight 1.0) | 0.4464 | 0.8941 | 427.59ms |
| D. LLM (2 variants / weight 1.0) | 0.4464 | 0.8941 | 1005.13ms |
| E. LLM (3 variants / weight 1.0) | 0.4464 | 0.8941 | 629.48ms |
| F. LLM (2 variants / weight 0.75) | 0.4464 | 0.8941 | 399.63ms |
| Drift Threshold 0.70 | 0.4464 | 0.8941 | N/A |
| Drift Threshold 0.75 | 0.4464 | 0.8941 | N/A |
| Drift Threshold 0.80 | 0.4464 | 0.8941 | N/A |
| Drift Threshold 0.85 | 0.4464 | 0.8941 | N/A |
