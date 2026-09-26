# ADR-005: Cross-Encoder Reranking — Rejected

## Status
Rejected

## Context
After RRF fusion, the top-k candidates could be reranked using a cross-encoder model. Cross-encoders jointly encode the query and document, producing more accurate relevance scores than the independent bi-encoder embeddings used in dense retrieval.

## Problem
Can cross-encoder reranking improve the NDCG@10 of RRF results, and at what latency cost?

## Method
Implemented `LocalCrossEncoderReranker` using `cross-encoder/ms-marco-MiniLM-L-6-v2` (max length 512 tokens). The model is lazy-loaded to avoid startup latency when reranking is not used.

Tested reranking at three depths on SciFact:
- Top 10 RRF candidates
- Top 25 RRF candidates
- Top 50 RRF candidates

For each candidate, the reranker scores the (query, passage) pair, where the passage is the dense chunk text (preferred) or the BM25 snippet (fallback).

## Decision
Rejected. NDCG decreased at every depth while latency increased dramatically.

## Evidence

| Rerank Depth | NDCG@10 | P95 Latency |
|---:|---:|---:|
| No reranking (baseline) | 0.7188 | 313.14 ms |
| 10 | 0.6858 | 608.76 ms |
| 25 | 0.6751 | 1,475.43 ms |
| 50 | 0.6691 | 3,288.34 ms |

At every depth:
- NDCG@10 **decreased** relative to the RRF baseline
- Latency **increased** by 2–10×

## Analysis
The cross-encoder likely underperformed because:
1. The RRF ranking was already strong (NDCG@10 = 0.72) — the reranker had limited room to improve
2. The passages fed to the cross-encoder (500-char chunks) may not contain sufficient context for the cross-encoder to outperform the already-fused ranking
3. The cross-encoder model (`ms-marco-MiniLM-L-6-v2`) is a lightweight model — a larger cross-encoder might perform differently, but at even higher latency cost

This does not mean cross-encoders are generally ineffective. The conclusion applies to this specific model, corpus, and chunking strategy.

## Trade-offs
- **Pro**: Cross-encoders are theoretically more accurate than bi-encoder retrieval
- **Con**: NDCG decreased on this workload
- **Con**: Latency increase makes it impractical for interactive search
- **Con**: Model adds ~300MB memory footprint

## Consequences
- The reranker code is retained (`retrieval/reranker.py`) and the `hybrid_rerank` mode remains available for experimentation
- The production baseline does not use reranking
- Future experiments could test larger cross-encoder models or longer passage windows
