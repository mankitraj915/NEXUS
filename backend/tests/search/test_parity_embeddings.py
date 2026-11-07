import numpy as np
import pytest

from scripts.benchmark.parity import (
    cosine_summary,
    embedding_parity,
    fp32_passes_g2,
    sample_chunks,
)


def test_identical_vectors_summarize_to_one():
    a = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    summary = cosine_summary(a, a.copy())
    assert summary["mean"] == pytest.approx(1.0)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["p1"] == pytest.approx(1.0)


def test_orthogonal_vectors_give_a_min_of_zero():
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    summary = cosine_summary(a, b)
    assert summary["min"] == pytest.approx(0.0)
    assert summary["mean"] == pytest.approx(0.5)


def test_summary_does_not_assume_unit_vectors():
    summary = cosine_summary(np.array([[2.0, 0.0]]), np.array([[5.0, 0.0]]))
    assert summary["min"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("mean", "minimum", "expected"),
    [
        (0.9999, 0.999, True),
        (0.99989, 0.999, False),
        (0.9999, 0.99899, False),
    ],
)
def test_g2_thresholds_apply_to_both_sets(mean, minimum, expected):
    good = {"mean": 1.0, "min": 1.0, "p1": 1.0}
    bad = {"mean": mean, "min": minimum, "p1": minimum}
    assert fp32_passes_g2(bad, good) is expected
    assert fp32_passes_g2(good, bad) is expected


def test_sample_uses_production_chunking_and_is_seeded():
    docs = ["x" * 900, "y" * 300]
    total, chunks = sample_chunks(docs, sample=2, seed=0)
    assert total == 3  # 500/100 chunking: 0-500, 400-900, and the short document
    assert chunks == sample_chunks(docs, sample=2, seed=0)[1]
    assert len(chunks) == 2


def test_sample_fails_when_the_corpus_is_too_small():
    with pytest.raises(ValueError):
        sample_chunks(["short"], sample=2, seed=0)


class FakeProvider:
    """Embeds each text as a fixed direction, so the expected cosines are known."""

    def __init__(self, vector):
        self.vector = vector

    def embed_documents(self, texts):
        return [self.vector for _ in texts]

    def embed_query(self, text):
        return self.vector


def test_embedding_parity_scores_each_candidate_against_the_baseline():
    result = embedding_parity(
        chunks=["c1", "c2"],
        queries=["q1"],
        baseline=FakeProvider([1.0, 0.0]),
        candidates={"same": FakeProvider([1.0, 0.0]), "off": FakeProvider([0.0, 1.0])},
    )
    assert result["candidates"]["same"]["chunks"]["min"] == pytest.approx(1.0)
    assert result["candidates"]["same"]["queries"]["min"] == pytest.approx(1.0)
    assert result["candidates"]["off"]["chunks"]["mean"] == pytest.approx(0.0)
    assert set(result["ms_per_100"]) == {"baseline", "same", "off"}
    assert set(result["ms_per_100"]["same"]) == {"chunks", "queries"}
