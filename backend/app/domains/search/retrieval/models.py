from pydantic import BaseModel, Field


class VariantProvenance(BaseModel):
    variant_id: str
    source: str  # 'bm25' or 'dense'
    rank: int
    score: float
    passage: str | None = None
    chunk_id: str | None = None


class Candidate(BaseModel):
    document_id: str
    external_id: str | None = None

    # BM25 Provenance (Legacy fields for single-query)
    bm25_rank: int | None = None
    bm25_score: float | None = None
    bm25_passage: str | None = None

    # Dense Provenance (Legacy fields for single-query)
    dense_rank: int | None = None
    dense_score: float | None = None
    dense_chunk_id: str | None = None
    dense_passage: str | None = None

    # Multi-Query Provenance
    provenance: list[VariantProvenance] = Field(default_factory=list)

    # Fusion & Reranking
    rrf_score: float | None = None
    reranker_score: float | None = None
    final_rank: int | None = None

    # Display Snippet
    snippet: str | None = None

    # Document Metadata (fetched at the end)
    url: str | None = None
    title: str | None = None
