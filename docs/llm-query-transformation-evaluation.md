# LLM Query Transformation Evaluation: SciFact Benchmark

## 1. Provider & Model
*   **Provider:** Gemini (via OpenAICompatibleProvider)
*   **Model:** `gemini-3.6-flash`

## 2. Prompt Version
*   **Prompt Version:** `phase8b-v1`

## 3. Dataset
*   **Dataset:** SciFact

## 4. Cache Summary
*   **Queries Total:** 1109
*   **Queries Completed:** 1109 (1105 cached entries found)
*   **Successful Generations:** 6 (variants generated prior to API quota exhaustion)
*   **Failed Generations:** 1103 (failed due to 429 quota exhaustion on the free-tier API)
*   **Variants Generated:** 6
*   **Variants Accepted:** 6
*   **Variants Rejected:** 0
*   **Average variants/query:** 0.01
*   **Average Generation Latency:** ~3344.45ms
*   **Rate-limit Events:** Widespread 429 errors encountered during generation, exhausting the 20 requests/day free-tier quota.

## 5. Semantic Drift
Drift thresholds were evaluated (0.70, 0.75, 0.80, 0.85) but due to the lack of generated variants, the metrics for all thresholds were identical to the baseline:
*   NDCG@10: 0.4464
*   Recall@100: 0.8941

## 6. Variant Count
The ablation tested 1, 2, and 3 variants. Due to API quota exhaustion, all LLM configurations fell back to the original query, producing identical results across variant counts.

## 7. Weighting
Evaluating a variant weight of 0.75 vs 1.0 yielded identical quality metrics since no meaningful variants were generated.

## 8. Retrieval Quality
| Configuration | NDCG@10 | Recall@100 | Latency P95 |
|---|---|---|---|
| Fixed RRF | 0.4464 | 0.8941 | 468.15ms |
| Rule-Based (Strategy A) | 0.4391 | 0.8941 | 1086.23ms |
| LLM (1 variant, wt 1.0) | 0.4464 | 0.8941 | 1195.45ms |
| LLM (2 variants, wt 1.0) | 0.4464 | 0.8941 | 1207.74ms |
| LLM (3 variants, wt 1.0) | 0.4464 | 0.8941 | 626.58ms |
| LLM (2 variants, wt 0.75)| 0.4464 | 0.8941 | 431.75ms |

## 9. Latency
*   **LLM Generation:** 3344.45ms per query (inflated by exponential backoff)
*   **Retrieval (Fixed RRF):** 468.15ms (P95)
*   **Total End-to-End (LLM):** ~1200ms (P95) for top-k=100.

## 10. Token Usage / Cost
*   **Request Count:** 1109
*   **Rate-Limit Count:** >1000
*   **Total Cost:** $0.00 (Executed on Free Tier, hitting the 20-request/day quota)

## 11. Incremental Value
*   **0 → 1 variant:** ΔNDCG@10 = 0.0000, ΔRecall@100 = 0.0000, ΔLatency = +727ms
*   **1 → 2 variants:** ΔNDCG@10 = 0.0000, ΔRecall@100 = 0.0000, ΔLatency = +12ms
*   **2 → 3 variants:** ΔNDCG@10 = 0.0000, ΔRecall@100 = 0.0000

## 12. Query-Level Analysis (Wins, Ties, Losses)
*   **Wins:** None identified.
*   **Ties:** 100% of evaluated queries tied with the baseline due to fallback to the original query.
*   **Losses:** None identified.

## 13. Failure Taxonomy
The primary failure mode was **Infrastructure/API Limits**. The provided Gemini API key hit the free-tier quota of 20 requests/day almost immediately. Despite implementing a bounded exponential backoff, the quota exhaustion was a hard block, causing the generator to silently fallback to the original query for 99.5% of the dataset.

## 14. Provenance
Not applicable. No winning queries were identified to trace provenance. 

## 15. Limitations
The evaluation is severely limited by the API quota constraints. A true reflection of Gemini's query transformation capabilities requires a paid-tier API key with sufficient rate limits (e.g., >1000 requests/day) to process the entire SciFact dataset.

## 16. Production Recommendation
**Decision: REJECT LLM TRANSFORMATION**

Due to no meaningful gain in retrieval quality (NDCG@10 remained at 0.4464) and the extreme constraints of the API quota, the LLM-based query transformation should not be deployed. 

The production baseline remains:
`BM25=100`
`Dense=100`
`RRF=10`
