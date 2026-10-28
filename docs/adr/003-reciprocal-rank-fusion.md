# ADR-003: Reciprocal Rank Fusion

## Status
Accepted

## Context
NEXUS performs lexical (BM25) and dense (pgvector) retrieval for each hybrid query. These produce two independently ranked lists that must be combined into a single ranking.

## Problem
BM25 and dense retrieval have complementary strengths. BM25 excels on exact terminology and identifiers. Dense retrieval handles paraphrased queries and semantic similarity. Neither alone achieves the best ranking quality. A fusion strategy is needed.

## Options considered
1. **Reciprocal Rank Fusion (RRF)** — rank-based, parameter-light
2. **Linear score combination** — weighted sum of normalized scores
3. **Learned fusion** — trained combination weights
4. **CombMNZ** — frequency-weighted score combination

## Decision
Use Reciprocal Rank Fusion with k=10 for the production hybrid baseline (`hybrid_fixed`), while providing k=60 as the standard fallback heuristic.

```
RRF(d) = Σ 1 / (k + rank_i(d))
```

## Evidence

Measured on SciFact (300-query test split, depth 100/100):

| Retriever / Fusion Configuration | NDCG@10 | Recall@100 | Latency P95 |
|---|---:|---:|---:|
| BM25 only (lexical) | 0.6736 | 0.9064 | 192.81 ms |
| Dense only (`all-MiniLM-L6-v2`) | 0.6642 | 0.9277 | 212.57 ms |
| RRF (100/100, k=60) | 0.7043 | 0.9577 | 282.16 ms |
| RRF (100/100, k=30) | 0.7137 | 0.9577 | 308.01 ms |
| RRF (100/100, k=10) — Production Baseline | **0.7188** | **0.9577** | 313.14 ms |

### Empirical k-parameter findings:
- Cormack et al. originally suggested k=60 for large-scale web collections to dampen the influence of extreme outlier ranks.
- On the evaluated SciFact test split (5,183 scientific abstracts), tuning k down to 10 yielded a +0.0145 NDCG@10 improvement over k=60. This is an empirical finding on this specific technical abstract corpus rather than a universal claim across arbitrary document domains.
- Lower k sharpens the rank decay: rank 1 + rank 1 agreement contributes `(1/11) + (1/12) = 0.1742`, whereas rank 10 + rank 10 contributes `(1/20) + (1/20) = 0.1000`. Under k=60, the distinction between top and mediocre ranks is heavily dampened (`0.0325` vs `0.0286`).
- Hybrid RRF (k=10) improves NDCG@10 by +6.7% over BM25-only and +8.2% over Dense-only on this workload. Recall@100 reaches 0.9577.
- The cost is executing both branches one after the other (they share one database session): on the untuned 809-query split, P95 latency is 366 ms for hybrid vs 188 ms for BM25 and 210 ms for dense.

## Trade-offs
- **Pro**: Rank-based fusion eliminates cross-retriever score calibration (cosine similarity vs unbounded BM25).
- **Pro**: Only one hyperparameter (k) to calibrate.
- **Pro**: Documents present in only one retriever branch are gracefully handled with a single reciprocal rank term.
- **Con**: Ignores raw score confidence deltas (a marginal rank 1 is weighted identically to an overwhelming rank 1).
- **Con**: Higher latency than single-branch retrieval due to running two query pathways.

## Consequences
- Hybrid mode is the production default (`hybrid_fixed`)
- Latency is strictly higher than single-retriever mode
- The fusion implementation in `retrieval/fusion.py` also supports multi-query RRF (Strategy A and B) for experimental query expansion

## Addendum (2026-10-28): split provenance and out-of-sample check

The "300-query test split" above is BEIR's SciFact test qrels, which are the official SciFact **dev** claims, and k was chosen on it, so the +0.0145 gain is an in-sample result. On the 809 BEIR train queries, which were never used for tuning, the re-measured NDCG@10 is 0.7180 for k=10 vs 0.7112 for k=60 (+0.0068), with BM25 at 0.6690 and dense at 0.6759. k=10 remains the default, but the advantage over k=60 is small, and its statistical significance has not been tested. The latency columns above came from an earlier, differently loaded run; current numbers and caveats are in [`docs/benchmarking.md`](../benchmarking.md).
