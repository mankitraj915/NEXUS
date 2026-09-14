# NEXUS Phase 6 Adaptive Routing Evaluation

## 1. Overall Pareto Frontier
| Configuration | NDCG@10 | Recall@100 | Avg BM25 Depth | Avg Dense Depth | Latency Avg | Latency P95 |
|---|---|---|---|---|---|---|
| Fixed Baseline (100/100) | 0.7188 | 0.9577 | 100.0 | 100.0 | 194.17ms | 352.94ms |
| Lexical-Heavy Static (100/25) | 0.7188 | 0.9577 | 100.0 | 25.0 | 229.98ms | 475.07ms |
| Dense-Heavy Static (25/100) | 0.7188 | 0.9577 | 25.0 | 100.0 | 253.96ms | 514.43ms |
| Adaptive Routing | 0.7185 | 0.9570 | 96.5 | 99.5 | 177.55ms | 321.65ms |

## 2. Routing Regret Analysis
- **Adaptive Wins**: 1
- **Adaptive Ties**: 297
- **Adaptive Losses**: 2

### Top Losses
- **Query**: `Birth-weight is positively associated with breast cancer.` (Delta: -0.1185)
  - **Classes**: {'EXACT_TERM': False, 'IDENTIFIER_HEAVY': False, 'TECHNICAL': False, 'CONCEPTUAL': True, 'NATURAL_LANGUAGE_QUESTION': False, 'MIXED': False, 'AMBIGUOUS': False}
  - **Plan**: {'bm25_depth': 25, 'dense_depth': 100, 'rrf_k': 10, 'enable_lexical': True, 'enable_dense': True, 'query_expansion': False, 'multi_hop': False, 'reason': 'Conceptual/NL query with low lexical specificity', 'policy_version': '1.0.0'}
- **Query**: `Obesity is determined solely by environmental factors.` (Delta: -0.0299)
  - **Classes**: {'EXACT_TERM': False, 'IDENTIFIER_HEAVY': False, 'TECHNICAL': False, 'CONCEPTUAL': True, 'NATURAL_LANGUAGE_QUESTION': False, 'MIXED': False, 'AMBIGUOUS': False}
  - **Plan**: {'bm25_depth': 25, 'dense_depth': 100, 'rrf_k': 10, 'enable_lexical': True, 'enable_dense': True, 'query_expansion': False, 'multi_hop': False, 'reason': 'Conceptual/NL query with low lexical specificity', 'policy_version': '1.0.0'}

### Top Wins
- **Query**: `PPAR-RXRs are inhibited by PPAR ligands.` (Delta: 0.0438)
  - **Classes**: {'EXACT_TERM': False, 'IDENTIFIER_HEAVY': True, 'TECHNICAL': True, 'CONCEPTUAL': True, 'NATURAL_LANGUAGE_QUESTION': False, 'MIXED': True, 'AMBIGUOUS': False}
  - **Plan**: {'bm25_depth': 100, 'dense_depth': 25, 'rrf_k': 10, 'enable_lexical': True, 'enable_dense': True, 'query_expansion': False, 'multi_hop': False, 'reason': 'High lexical specificity and identifier density', 'policy_version': '1.0.0'}
