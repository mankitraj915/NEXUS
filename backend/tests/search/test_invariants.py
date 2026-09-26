import math

from app.domains.search.analyzer import analyze_text
from app.domains.search.query.features import extract_features
from app.domains.search.retrieval.fusion import reciprocal_rank_fusion
from app.domains.search.retrieval.models import Candidate
from app.domains.search.schemas import SearchMode

# ==============================================================================
# 1. Analyzer Invariant Tests
# ==============================================================================


def test_analyzer_nfkc_normalization():
    """Verify NFKC normalization transforms compatibility characters (e.g. ligatures, full-width)."""
    # Ligature 'ﬁ' -> 'fi'
    text_ligature = "scientific ﬁndings"
    tokens = analyze_text(text_ligature)
    assert "find" in tokens or "finding" in tokens

    # Full-width characters 'ＡＩ' -> 'ai'
    text_fullwidth = "ＡＩ search"
    tokens_fw = analyze_text(text_fullwidth)
    assert "ai" in tokens_fw


def test_analyzer_stemming_and_stopwords():
    """Verify stopword stripping and Porter stemming invariants."""
    raw = "The retrieval engines are retrieving relevant documents effectively"
    tokens = analyze_text(raw)
    # Stopwords like 'the', 'are' must be stripped
    assert "the" not in tokens
    assert "are" not in tokens

    # Both 'retrieval' and 'retrieving' should stem to 'retriev'
    assert "retriev" in tokens


def test_analyzer_deterministic_tokenization():
    """Verify tokenization is strictly deterministic across multiple runs."""
    text = "Hybrid search with reciprocal rank fusion (RRF) & BM25."
    run1 = analyze_text(text)
    run2 = analyze_text(text)
    assert run1 == run2
    assert run1 == ["hybrid", "search", "reciproc", "rank", "fusion", "rrf", "bm25"]


# ==============================================================================
# 2. BM25 Mathematics Invariant Tests
# ==============================================================================


def bm25_score_term(
    tf: int,
    doc_len: int,
    avgdl: float,
    N: int,
    n_q: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Standard Robertson-Spärck Jones probabilistic BM25 formula matching backend/app/domains/search/bm25.py."""
    if tf <= 0 or n_q <= 0 or N <= 0:
        return 0.0
    idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
    numerator = tf * (k1 + 1.0)
    denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
    return idf * (numerator / denominator)


def test_bm25_zero_term_frequency():
    """TF=0 must contribute exactly 0.0 to BM25 score."""
    score = bm25_score_term(tf=0, doc_len=100, avgdl=100.0, N=1000, n_q=10)
    assert score == 0.0


def test_bm25_monotonicity_with_term_frequency():
    """BM25 score must increase monotonically with term frequency (with diminishing returns)."""
    score_tf1 = bm25_score_term(tf=1, doc_len=100, avgdl=100.0, N=1000, n_q=50)
    score_tf2 = bm25_score_term(tf=2, doc_len=100, avgdl=100.0, N=1000, n_q=50)
    score_tf5 = bm25_score_term(tf=5, doc_len=100, avgdl=100.0, N=1000, n_q=50)

    assert score_tf1 > 0.0
    assert score_tf2 > score_tf1
    assert score_tf5 > score_tf2

    # Diminishing returns: delta(1->2) > delta(4->5)
    delta_1_2 = score_tf2 - score_tf1
    score_tf4 = bm25_score_term(tf=4, doc_len=100, avgdl=100.0, N=1000, n_q=50)
    delta_4_5 = score_tf5 - score_tf4
    assert delta_1_2 > delta_4_5


def test_bm25_document_length_normalization():
    """For identical TF, shorter documents must score higher than long documents (b > 0)."""
    score_short = bm25_score_term(tf=2, doc_len=50, avgdl=150.0, N=1000, n_q=50)
    score_avg = bm25_score_term(tf=2, doc_len=150, avgdl=150.0, N=1000, n_q=50)
    score_long = bm25_score_term(tf=2, doc_len=500, avgdl=150.0, N=1000, n_q=50)

    assert score_short > score_avg > score_long


def test_bm25_idf_non_negativity_when_all_docs_match():
    """Even when term appears in all documents (n_q = N), smoothed IDF must remain > 0."""
    score = bm25_score_term(tf=1, doc_len=100, avgdl=100.0, N=100, n_q=100)
    assert score > 0.0


# ==============================================================================
# 3. RRF (Reciprocal Rank Fusion) Invariant Tests
# ==============================================================================


def test_rrf_rank_contribution_formula():
    """Verify RRF formula strictly matches sum(1 / (k + rank))."""
    k = 60
    cand_lex = Candidate(document_id="docA", bm25_rank=1, bm25_score=10.0)
    cand_dense = Candidate(document_id="docA", dense_rank=2, dense_score=0.9)

    fused = reciprocal_rank_fusion([cand_lex], [cand_dense], k=k)
    assert len(fused) == 1
    expected_score = (1.0 / (k + 1)) + (1.0 / (k + 2))
    assert fused[0].rrf_score is not None
    assert math.isclose(fused[0].rrf_score, expected_score, rel_tol=1e-6)


def test_rrf_missing_retriever_results():
    """A candidate present in only one retriever list must be included without error."""
    cand_bm25_only = Candidate(document_id="doc_bm25", bm25_rank=1, bm25_score=8.0)
    cand_dense_only = Candidate(document_id="doc_dense", dense_rank=1, dense_score=0.95)

    fused = reciprocal_rank_fusion([cand_bm25_only], [cand_dense_only], k=60)
    assert len(fused) == 2
    doc_ids = {c.document_id for c in fused}
    assert doc_ids == {"doc_bm25", "doc_dense"}


def test_rrf_deterministic_ordering():
    """RRF ordering must be strictly monotonic and stable."""
    bm25 = [
        Candidate(document_id=f"doc_{i}", bm25_rank=i, bm25_score=100.0 - i)
        for i in range(1, 6)
    ]
    dense = [
        Candidate(document_id=f"doc_{6 - i}", dense_rank=i, dense_score=1.0 - (i * 0.1))
        for i in range(1, 6)
    ]

    fused = reciprocal_rank_fusion(bm25, dense, k=60)
    assert all(c.rrf_score is not None for c in fused)
    scores: list[float] = [c.rrf_score for c in fused if c.rrf_score is not None]
    # Scores must be sorted descending
    assert scores == sorted(scores, reverse=True)


# ==============================================================================
# 4. Query Feature Extraction Invariants
# ==============================================================================


def test_query_feature_bounds_and_determinism():
    """Extracted query features must lie within valid mathematical domains."""
    query = "What is the capital of France?"
    features = extract_features(query, db_rarity_score=0.45)

    assert features.token_count > 0
    assert 0.0 <= features.unique_token_ratio <= 1.0
    assert 0.0 <= features.identifier_density <= 1.0
    assert 0.0 <= features.quoted_term_density <= 1.0
    assert 0.0 <= features.term_rarity_score <= 1.0
    assert 0.0 <= features.lexical_specificity_score <= 1.0
    assert 0.0 <= features.natural_language_score <= 1.0
    assert 0.0 <= features.complexity_score <= 1.0
    assert (
        features.natural_language_score > 0.3
    )  # Query has 'What is', so high NL score


# ==============================================================================
# 5. API SearchMode Contract Invariants
# ==============================================================================


def test_api_search_mode_validation():
    """Verify SearchMode enum recognizes stable, alias, and experimental modes."""
    # Stable modes
    assert SearchMode.is_valid_mode("hybrid_fixed") is True
    assert SearchMode.is_valid_mode("lexical") is True
    assert SearchMode.is_valid_mode("dense") is True

    # Backwards compatibility alias
    assert SearchMode.is_valid_mode("hybrid") is True
    assert SearchMode.normalize_mode("hybrid") == "hybrid_fixed"

    # Experimental modes
    assert SearchMode.is_valid_mode("experimental_adaptive") is True
    assert SearchMode.is_valid_mode("adaptive") is True
    assert SearchMode.normalize_mode("experimental_adaptive") == "adaptive"

    # Invalid modes
    assert SearchMode.is_valid_mode("invalid_mode_xyz") is False
    assert SearchMode.is_valid_mode("") is False
    assert SearchMode.is_valid_mode("magic_vector") is False
