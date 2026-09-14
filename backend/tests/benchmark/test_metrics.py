import pytest

from scripts.benchmark.metrics import (
    jaccard_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    qrels = {"d1": 1, "d2": 1, "d4": 1}
    retrieved = ["d1", "d3", "d2"]

    # K=1 -> [d1] -> 1/1 = 1.0
    assert precision_at_k(retrieved, qrels, k=1) == 1.0

    # K=2 -> [d1, d3] -> 1/2 = 0.5
    assert precision_at_k(retrieved, qrels, k=2) == 0.5

    # K=3 -> [d1, d3, d2] -> 2/3
    assert precision_at_k(retrieved, qrels, k=3) == pytest.approx(2 / 3)


def test_recall_at_k():
    qrels = {"d1": 1, "d2": 1, "d4": 1}  # Total relevant = 3
    retrieved = ["d1", "d3", "d2"]

    # K=1 -> [d1] -> 1/3
    assert recall_at_k(retrieved, qrels, k=1) == pytest.approx(1 / 3)

    # K=2 -> [d1, d3] -> 1/3
    assert recall_at_k(retrieved, qrels, k=2) == pytest.approx(1 / 3)

    # K=3 -> [d1, d3, d2] -> 2/3
    assert recall_at_k(retrieved, qrels, k=3) == pytest.approx(2 / 3)


def test_mrr():
    qrels = {"d1": 1, "d2": 1}

    # Relevant is at rank 2
    retrieved = ["d3", "d1", "d4"]
    assert mrr(retrieved, qrels) == 0.5

    # Relevant is at rank 1
    retrieved = ["d2", "d3"]
    assert mrr(retrieved, qrels) == 1.0

    # No relevant retrieved
    retrieved = ["d3", "d4"]
    assert mrr(retrieved, qrels) == 0.0


def test_ndcg_at_k():
    # Ideal ranking: d1(3), d2(2), d3(1), d4(0)
    qrels = {"d1": 3, "d2": 2, "d3": 1}

    # Actual ranking exactly ideal
    retrieved = ["d1", "d2", "d3"]
    assert ndcg_at_k(retrieved, qrels, k=3) == 1.0

    # Bad ranking (no relevant)
    retrieved = ["d4", "d5"]
    assert ndcg_at_k(retrieved, qrels, k=2) == 0.0


def test_jaccard_at_k():
    list_a = ["d1", "d2", "d3"]
    list_b = ["d1", "d4", "d5"]

    # K=1 -> {d1} and {d1} -> 1/1 = 1.0
    assert jaccard_at_k(list_a, list_b, k=1) == 1.0

    # K=2 -> {d1, d2} and {d1, d4} -> intersection: {d1}(1), union: {d1, d2, d4}(3) -> 1/3
    assert jaccard_at_k(list_a, list_b, k=2) == pytest.approx(1 / 3)
