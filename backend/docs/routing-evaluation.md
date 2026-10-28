# NEXUS Phase 7.5 Validation Report

## 1. Executive Summary
This report details the full GroupKFold validation of the Learned Routing Models on the SciFact benchmark. 
The Decision Tree achieved significant latency savings but we must inspect the worst-case degradation.

## 2. Dataset
- **Queries Evaluated**: 300
- **Configurations**: 6 (A through F)
- **Leakage Safe**: GroupKFold by query_id (5 Folds)

## 3. Overall Performance

| Policy | NDCG@5 | NDCG@10 | Recall@100 | MRR | Latency (Avg) | Latency (P50) | Latency (P95) | Avg BM25 | Avg Dense |
|---|---|---|---|---|---|---|---|---|---|
| Fixed Baseline | 0.6942 | 0.7188 | 0.9577 | 0.6850 | 157.81ms | 147.23ms | 248.02ms | 100.0 | 100.0 |
| Rule-Based | 0.6922 | 0.7176 | 0.9553 | 0.6840 | 155.76ms | 143.01ms | 248.02ms | 76.2 | 99.5 |
| Decision Tree | 0.6535 | 0.6742 | 0.9064 | 0.6407 | 80.73ms | 70.13ms | 161.83ms | 96.5 | 1.8 |
| Random Forest | 0.6509 | 0.6736 | 0.9064 | 0.6388 | 76.45ms | 69.06ms | 140.52ms | 100.0 | 0.0 |
| Oracle A (Max Quality) | 0.7507 | 0.7739 | 0.9377 | 0.7430 | 91.11ms | 82.56ms | 170.82ms | 79.6 | 17.5 |
| Oracle B (Constrained) | 0.7249 | 0.7490 | 0.9286 | 0.7183 | 86.57ms | 79.48ms | 158.91ms | 86.3 | 9.3 |

## 4. Regret Analysis (Decision Tree vs Fixed)
- **Mean Regret**: -0.0446
- **Median Regret**: 0.0000
- **P90 Worst-case**: -0.3333
- **P95 Worst-case**: -0.3869
- **P99 Worst-case**: -0.6133
- **Absolute Worst**: -1.0000

**Loss Breakdown (Percentage of queries degrading by X):**
- Loss > 0.005: 22.3%
- Loss > 0.010: 22.3%
- Loss > 0.050: 19.7%
- Loss > 0.100: 17.7%

## 5. Cross-Fold Stability (Decision Tree)
| Fold | NDCG@10 | Latency (Avg) | Mean Regret |
|---|---|---|---|
| 1 | 0.6435 | 78.83ms | -0.0871 |
| 2 | 0.7156 | 80.30ms | -0.0263 |
| 3 | 0.7235 | 88.61ms | -0.0136 |
| 4 | 0.6414 | 81.00ms | -0.0503 |
| 5 | 0.6469 | 74.92ms | -0.0459 |

**NDCG Stats**: Mean=0.6742, Std=0.0372, Min=0.6414, Max=0.7235

## 6. Worst-Case Queries
**Query:** `The one-child policy has been successful in lowering population growth.`
- **Delta NDCG:** -1.0000 (1.0000 -> 0.0000)
- **Latency:** 108.62ms -> 45.69ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_B

**Query:** `High dietary calcium intakes are unnecessary for prevention of secondary hyperparathyroidism in subjects with 25(OH)D levels above 75 nmol/liter.`
- **Delta NDCG:** -1.0000 (1.0000 -> 0.0000)
- **Latency:** 255.68ms -> 129.51ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_F

**Query:** `Silencing of Bcl2 is important for the maintenance and progression of tumors.`
- **Delta NDCG:** -0.6309 (0.6309 -> 0.0000)
- **Latency:** 134.34ms -> 47.96ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_E

**Query:** `Fz/PCP-dependent Pk localizes to the anterior membrane of notochord cells during zebrafish neuralation.`
- **Delta NDCG:** -0.6131 (1.0000 -> 0.3869)
- **Latency:** 200.69ms -> 72.88ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_F

**Query:** `The treatment of cancer patients with co-IR blockade precipitates adverse autoimmune events.`
- **Delta NDCG:** -0.5693 (1.0000 -> 0.4307)
- **Latency:** 130.34ms -> 69.96ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_B

**Query:** `Golli-deficient T-cells prefer to differentiate into an anergic phenotype in the adaptive immune response when there are increased levels of Ca2+ in the cytosol.`
- **Delta NDCG:** -0.5693 (1.0000 -> 0.4307)
- **Latency:** 182.52ms -> 85.50ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_C

**Query:** `The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune events.`
- **Delta NDCG:** -0.5693 (1.0000 -> 0.4307)
- **Latency:** 151.72ms -> 85.08ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_F

**Query:** `Vitamin D deficiency is unrelated to birth weight.`
- **Delta NDCG:** -0.5693 (1.0000 -> 0.4307)
- **Latency:** 328.27ms -> 69.12ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_F

**Query:** `Fz/PCP-dependent Pk localizes to the anterior membrane of neuroectoderm cells during zebrafish neuralation`
- **Delta NDCG:** -0.5693 (1.0000 -> 0.4307)
- **Latency:** 177.47ms -> 118.83ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_B

**Query:** `Assembly of invadopodia is triggered by focal generation of phosphatidylinositol-3,4-biphosphate and the activation of the nonreceptor tyrosine kinase Src.`
- **Delta NDCG:** -0.5275 (0.6296 -> 0.1021)
- **Latency:** 166.05ms -> 72.86ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_F

**Query:** `Ethanol stress decreases the expression of IBP in bacteria.`
- **Delta NDCG:** -0.5000 (1.0000 -> 0.5000)
- **Latency:** 128.06ms -> 49.15ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_B

**Query:** `Integrating classroom-based collaborative learning with Web-based collaborative learning leads to subpar class performance`
- **Delta NDCG:** -0.5000 (1.0000 -> 0.5000)
- **Latency:** 112.48ms -> 43.64ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_F

**Query:** `Pleiotropic coupling of GLP-1R to intracellular effectors promotes distinct profiles of cellular signaling.`
- **Delta NDCG:** -0.5000 (1.0000 -> 0.5000)
- **Latency:** 126.62ms -> 67.09ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_F

**Query:** `Birth-weight is positively associated with breast cancer.`
- **Delta NDCG:** -0.4964 (0.7774 -> 0.2810)
- **Latency:** 124.60ms -> 56.95ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_B

**Query:** `PPAR-RXRs are inhibited by PPAR ligands.`
- **Delta NDCG:** -0.3869 (0.3869 -> 0.0000)
- **Latency:** 110.77ms -> 39.87ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_F

**Query:** `p16INK4A accumulation is  linked to an abnormal wound response caused by the microinvasive step of advanced Oral Potentially Malignant Lesions (OPMLs).`
- **Delta NDCG:** -0.3869 (0.3869 -> 0.0000)
- **Latency:** 231.33ms -> 147.52ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_E

**Query:** `Hypertension is frequently observed in type 1 diabetes patients.`
- **Delta NDCG:** -0.3869 (0.3869 -> 0.0000)
- **Latency:** 195.90ms -> 73.68ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_E

**Query:** `Vitamin D deficiency effects the term of delivery.`
- **Delta NDCG:** -0.3869 (0.3869 -> 0.0000)
- **Latency:** 255.20ms -> 71.41ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_F

**Query:** `CCL19 is absent within dLNs.`
- **Delta NDCG:** -0.3691 (1.0000 -> 0.6309)
- **Latency:** 92.17ms -> 31.89ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_F

**Query:** `Chenodeosycholic acid treatment reduces whole-body energy expenditure.`
- **Delta NDCG:** -0.3691 (1.0000 -> 0.6309)
- **Latency:** 112.27ms -> 47.67ms
- **Selected:** CONFIG_D (BM25=100, Dense=0) | **Oracle:** CONFIG_F



## 7. Probe Ablation
- Average Probe Latency Overhead: 73.78ms
(If probe overhead > latency savings on affected queries, it should be disabled.)

## 8. Feature Importance
- **token_count**: 0.5216
- **natural_language_score**: 0.2568
- **lexical_specificity_score**: 0.1333
- **unique_token_ratio**: 0.0884

## 9. Final Recommendation

Based on the full 300-query validation across 5 folds, we must conclude **Outcome B & Outcome E**:
> *Learned routing reduces latency but quality degradation is material. Reject as default production policy. Keep Fixed RRF.*

### Key Findings:
1. **Material Quality Loss:** The Decision Tree achieves a ~50% latency reduction (157ms -> 80ms), but it pays a heavy price. The average NDCG@10 drops from 0.7188 to 0.6742. This -0.0446 degradation is significant in the IR domain. Over 19.7% of queries experience a quality loss greater than 0.050.
2. **Unacceptable Worst-Case Behavior:** The P95 worst-case regret is -0.3869, and the absolute worst case is -1.000. In many cases (e.g., *"The one-child policy has been successful..."*), the router aggressively routes to `CONFIG_D (BM25=100, Dense=0)` to save latency, entirely missing the semantic matches that dense retrieval provides.
3. **Probe Overhead is Fatal:** The BM25 probe costs on average **73.78ms**. Since the fixed baseline latency is 157.81ms (the write-up assumed BM25 and dense ran concurrently; they shared one database session whose queries were serialized — see `docs/architecture.md`), inserting a 73ms sequential probe completely destroys any latency savings we could achieve by conditionally skipping dense retrieval.
4. **Oracle Gap:** A theoretical Oracle *can* reduce latency to 86ms while keeping NDCG at 0.7490. However, our simple query features (token count, NL score) are insufficient to learn this boundary safely.

### Answer to Final Question
> *If I remove the learned router tomorrow, what measurable capability do I lose?*

**Nothing of production value.** You lose a highly risky optimization that sacrifices too much relevance for speed. The fixed `BM25=100, Dense=100, RRF=10` pipeline remains the undisputed king of robust, high-quality retrieval for this domain. The learned router shall remain an offline experiment.
