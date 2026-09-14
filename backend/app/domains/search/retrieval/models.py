from pydantic import BaseModel, Field
from typing import Optional, List

class VariantProvenance(BaseModel):
    variant_id: str
    source: str # 'bm25' or 'dense'
    rank: int
    score: float
    passage: Optional[str] = None
    chunk_id: Optional[str] = None

class Candidate(BaseModel):
    document_id: str
    external_id: Optional[str] = None
    
    # BM25 Provenance (Legacy fields for single-query)
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_passage: Optional[str] = None
    
    # Dense Provenance (Legacy fields for single-query)
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    dense_chunk_id: Optional[str] = None
    dense_passage: Optional[str] = None
    
    # Multi-Query Provenance
    provenance: List[VariantProvenance] = Field(default_factory=list)
    
    # Fusion & Reranking
    rrf_score: Optional[float] = None
    reranker_score: Optional[float] = None
    final_rank: Optional[int] = None
    
    # Display Snippet
    snippet: Optional[str] = None
    
    # Document Metadata (fetched at the end)
    url: Optional[str] = None
    title: Optional[str] = None
