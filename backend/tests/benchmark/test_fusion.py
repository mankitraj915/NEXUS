from app.domains.search.retrieval.fusion import reciprocal_rank_fusion
from app.domains.search.retrieval.models import Candidate


def test_rrf():
    c1 = Candidate(
        document_id="doc1",
        bm25_rank=1,
        bm25_score=10.0,
        url="A",
        title="A",
        snippet="A",
    )
    c2 = Candidate(
        document_id="doc2", bm25_rank=2, bm25_score=5.0, url="B", title="B", snippet="B"
    )

    c1_dense = Candidate(document_id="doc1", dense_rank=2, dense_score=0.9)
    c3_dense = Candidate(document_id="doc3", dense_rank=1, dense_score=0.95)

    bm25_list = [c1, c2]
    dense_list = [c3_dense, c1_dense]

    # RRF (k=60)
    # doc1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01612 = 0.0325
    # doc3: 1/(60+1) = 1/61 = 0.01639
    # doc2: 1/(60+2) = 1/62 = 0.01612

    fused = reciprocal_rank_fusion(bm25_list, dense_list, k=60)

    assert len(fused) == 3
    assert fused[0].document_id == "doc1"
    assert fused[1].document_id == "doc3"
    assert fused[2].document_id == "doc2"

    assert fused[0].rrf_score is not None
    assert fused[1].rrf_score is not None
    assert fused[2].rrf_score is not None
    assert fused[0].rrf_score > fused[1].rrf_score
    assert fused[1].rrf_score > fused[2].rrf_score
