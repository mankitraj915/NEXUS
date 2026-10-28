# ADR-004: Learned Query Router — Rejected

## Status
Rejected

## Context
The fixed hybrid retrieval mode always executes both BM25 (depth 100) and dense (depth 100) retrieval. For many queries, one branch may be unnecessary — a keyword-heavy query could be served by BM25 alone at lower latency.

## Problem
Can a learned model predict which retrieval configuration to use based on query features, reducing latency without degrading quality?

## Method
1. Extracted 8 query features: token count, unique token ratio, identifier density, quoted term density, term rarity, lexical specificity, natural language score, complexity score
2. Defined 6 retrieval configurations (action space):
   - CONFIG_A: BM25=100, Dense=100 (fixed baseline)
   - CONFIG_B: BM25=100, Dense=25
   - CONFIG_C: BM25=25, Dense=100
   - CONFIG_D: BM25=100, Dense=0 (BM25-only)
   - CONFIG_E: BM25=0, Dense=100 (Dense-only)
   - CONFIG_F: BM25=25, Dense=25
3. Exhaustively evaluated all 6 configurations on the 300 BEIR-test queries (the official SciFact dev claims, the same split used to select RRF k=10), recording NDCG@10, Recall@100, and latency per query per configuration (`action_outcome_matrix.jsonl`)
4. Trained Decision Tree and Random Forest classifiers targeting a Quality-Constrained Latency Oracle (minimize latency while maintaining NDCG ≥ baseline − ε, with ε = 0.005)
5. Validated with GroupKFold (5 folds, grouped by query ID to prevent leakage)
6. Performed regret analysis against the fixed baseline

## Decision
Rejected. The quality regression was unacceptable.

## Evidence

| Metric | Fixed Baseline | Decision Tree |
|---|---:|---:|
| NDCG@10 | 0.7188 | 0.6742 |
| Mean Latency | ~157 ms | ~81 ms |
| Mean Regret (vs Fixed) | 0.0 | −0.0446 |
| Worst-case Regret | 0.0 | −1.0 |
| Queries losing >0.05 NDCG | 0% | 19.7% |
| Queries losing >0.10 NDCG | 0% | ~12% |

The decision tree achieved a 49% latency reduction but:
- Mean NDCG@10 dropped by 0.0446 (−6.2%)
- Nearly 20% of queries experienced meaningful quality degradation
- Worst-case regret was −1.0 (a query went from perfect ranking to completely wrong)

## Trade-offs
- **Pro**: Significant latency improvement (~50%)
- **Con**: NDCG degradation across 20% of queries
- **Con**: Catastrophic worst-case (regret = −1.0)
- **Con**: Model complexity adds a failure mode (pickle loading, feature extraction overhead)

## Consequences
- The learned router code remains in the repository (`retrieval/learned_router.py`, `scripts/router_training/`) as documented experimental evidence
- The `experimental_learned` search mode is available but not the production default
- The fixed hybrid baseline remains the recommended configuration
- The action-outcome matrix is preserved as a reusable dataset for future routing experiments
