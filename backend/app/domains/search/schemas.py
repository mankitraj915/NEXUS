from enum import Enum

from pydantic import BaseModel


class SearchMode(str, Enum):
    # Stable Production Modes
    HYBRID_FIXED = "hybrid_fixed"
    LEXICAL = "lexical"
    DENSE = "dense"

    # Backwards-compatible alias for production callers
    HYBRID = "hybrid"

    # Explicitly Experimental / Research Modes
    EXPERIMENTAL_ADAPTIVE = "experimental_adaptive"
    EXPERIMENTAL_LEARNED = "experimental_learned"
    EXPERIMENTAL_MULTI_QUERY_RULE = "experimental_multi_query_rule"
    EXPERIMENTAL_MULTI_QUERY_LLM = "experimental_multi_query_llm"
    EXPERIMENTAL_HYBRID_RERANK = "experimental_hybrid_rerank"

    @classmethod
    def is_valid_mode(cls, mode: str) -> bool:
        """Check if mode string is a recognized stable or experimental mode."""
        valid = {m.value for m in cls}
        # Accept legacy un-prefixed experimental strings for backwards compatibility
        valid.update(
            {
                "adaptive",
                "learned",
                "multi_query_rule",
                "multi_query_llm",
                "hybrid_no_rerank",
                "hybrid_rerank",
            }
        )
        return mode in valid

    @classmethod
    def normalize_mode(cls, mode: str) -> str:
        """Map external mode strings to internal orchestrator execution modes."""
        if mode in ("hybrid", "hybrid_fixed", "hybrid_no_rerank"):
            return "hybrid_fixed"
        if mode in ("experimental_hybrid_rerank", "hybrid_rerank"):
            return "hybrid_rerank"
        if mode in ("experimental_adaptive", "adaptive"):
            return "adaptive"
        if mode in ("experimental_learned", "learned"):
            return "learned"
        if mode in ("experimental_multi_query_rule", "multi_query_rule"):
            return "multi_query_rule"
        if mode in ("experimental_multi_query_llm", "multi_query_llm"):
            return "multi_query_llm"
        return mode


class SearchResultItem(BaseModel):
    document_id: str
    url: str
    title: str
    score: float
    snippet: str
    rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str | None = None
    results: list[SearchResultItem]
