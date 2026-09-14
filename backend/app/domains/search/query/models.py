from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class QueryFeatures(BaseModel):
    # Continuous scores [0, 1]
    token_count: int = 0
    unique_token_ratio: float = 0.0
    identifier_density: float = 0.0
    quoted_term_density: float = 0.0
    term_rarity_score: float = 0.0
    lexical_specificity_score: float = 0.0
    natural_language_score: float = 0.0
    complexity_score: float = 0.0

class QueryClasses(BaseModel):
    # Boolean flags
    EXACT_TERM: bool = False
    IDENTIFIER_HEAVY: bool = False
    TECHNICAL: bool = False
    CONCEPTUAL: bool = False
    NATURAL_LANGUAGE_QUESTION: bool = False
    MIXED: bool = False
    AMBIGUOUS: bool = False

class RetrievalPlan(BaseModel):
    bm25_depth: int
    dense_depth: int
    rrf_k: int
    enable_lexical: bool = True
    enable_dense: bool = True
    query_expansion: bool = False
    multi_hop: bool = False
    reason: str
    policy_version: str = "1.0.0"

class QueryAnalysis(BaseModel):
    raw_query: str
    normalized_query: str
    features: QueryFeatures
    classes: QueryClasses
    plan: Optional[RetrievalPlan] = None

from datetime import datetime

class QueryVariant(BaseModel):
    variant_id: str
    query: str
    variant_type: str
    generator: str
    generator_version: str
    confidence: Optional[float] = None
    is_original: bool
    created_at: datetime
