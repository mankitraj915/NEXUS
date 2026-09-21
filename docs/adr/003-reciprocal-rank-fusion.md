# ADR-003: Reciprocal Rank Fusion

## Status
Accepted

## Context
NEXUS performs parallel lexical (BM25) and dense (pgvector) retrieval. These produce two independently ranked lists that must be combined into a single ranking.

## Problem
BM25 and dense retrieval have complementary strengths. BM25 excels on exact terminology and identifiers. Dense retrieval handles paraphrased queries and semantic similarity. Neither alone achieves the best ranking quality. A fusion strategy is needed.

## Options considered
1. **Reciprocal Rank Fusion (RRF)** — rank-based, parameter-light
2. **Linear score combination** — weighted sum of normalized scores
3. **Learned fusion** — trained combination weights
4. **CombMNZ** — frequency-weighted score combination

## Decision
Use Reciprocal Rank Fusion with k=60.

```
RRF(d) = Σ 1 / (k + rank_i(d))
```

## Evidence

Measured on SciFact (300-query test split):

| Retriever | NDCG@10 | Recall@100 |
|---|---:|---:|
| BM25 only | 0.6736 | 0.9064 |
| Dense only | 0.6642 | 0.9277 |
| RRF (k=60) | **0.7188** | **0.9577** |

RRF improves NDCG@10 by +6.7% over BM25-only and +8.2% over Dense-only. Recall@100 improves by +5.7% and +3.2% respectively.

The improvement is not free — hybrid mode executes both retrieval branches, adding ~100ms latency (313ms P95 vs. ~200ms for single-branch).

## Trade-offs
- **Pro**: RRF is rank-based, so it does not require score normalization across retrievers
- **Pro**: Only one parameter (k) to tune — less overfitting risk than learned weights
- **Pro**: Documents appearing in only one retriever list are naturally handled (single-term RRF contribution)
- **Con**: Does not leverage score magnitude — a highly confident BM25 match and a marginal one contribute the same if they share a rank
- **Con**: The k parameter (60) was not extensively tuned; the benchmark uses k=10 for the production baseline and k=60 for RRF computation

## Consequences
- Hybrid mode is the production default (`hybrid_fixed`)
- Latency is strictly higher than single-retriever mode
- The fusion implementation in `retrieval/fusion.py` also supports multi-query RRF (Strategy A and B) for experimental query expansion
