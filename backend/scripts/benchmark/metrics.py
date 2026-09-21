import math


def precision_at_k(retrieved_ids: list[str], qrels: dict[str, int], k: int, threshold: int = 1) -> float:
    """
    Calculate Precision@K.
    A document is considered relevant if its score in qrels >= threshold.
    """
    if not retrieved_ids:
        return 0.0
    k_retrieved = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in k_retrieved if qrels.get(doc_id, 0) >= threshold)
    return relevant_retrieved / k

def recall_at_k(retrieved_ids: list[str], qrels: dict[str, int], k: int, threshold: int = 1) -> float:
    """
    Calculate Recall@K.
    A document is considered relevant if its score in qrels >= threshold.
    """
    total_relevant = sum(1 for score in qrels.values() if score >= threshold)
    if total_relevant == 0:
        return 0.0
    if not retrieved_ids:
        return 0.0
    k_retrieved = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in k_retrieved if qrels.get(doc_id, 0) >= threshold)
    return relevant_retrieved / total_relevant

def mrr(retrieved_ids: list[str], qrels: dict[str, int], threshold: int = 1) -> float:
    """
    Calculate Mean Reciprocal Rank (for a single query).
    Returns 1 / rank of the FIRST relevant document (score >= threshold).
    Returns 0.0 if no relevant document is found.
    """
    for idx, doc_id in enumerate(retrieved_ids):
        if qrels.get(doc_id, 0) >= threshold:
            return 1.0 / (idx + 1)
    return 0.0

def dcg_at_k(retrieved_ids: list[str], qrels: dict[str, int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at K using graded relevance.
    Formula: sum((2^rel - 1) / log2(rank + 1))
    """
    k_retrieved = retrieved_ids[:k]
    dcg = 0.0
    for idx, doc_id in enumerate(k_retrieved):
        rel = qrels.get(doc_id, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(idx + 2)
    return dcg

def ndcg_at_k(retrieved_ids: list[str], qrels: dict[str, int], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    Uses graded relevance directly from qrels.
    """
    actual_dcg = dcg_at_k(retrieved_ids, qrels, k)
    # Ideal DCG: sort all qrels by relevance descending
    ideal_ranking = sorted([doc_id for doc_id in qrels], key=lambda d: qrels[d], reverse=True)
    ideal_dcg = dcg_at_k(ideal_ranking, qrels, k)
    
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg

def jaccard_at_k(list_a: list[str], list_b: list[str], k: int) -> float:
    """
    Calculate Jaccard Similarity between top-K sets of two lists.
    |A ∩ B| / |A ∪ B|
    """
    set_a = set(list_a[:k])
    set_b = set(list_b[:k])
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union
