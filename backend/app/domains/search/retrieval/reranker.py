import logging

import torch
from sentence_transformers import CrossEncoder

from app.domains.search.retrieval.models import Candidate

logger = logging.getLogger(__name__)


class RerankerProvider:
    def rerank(
        self, query: str, candidates: list[Candidate], top_k: int = 10
    ) -> list[Candidate]:
        raise NotImplementedError


class LocalCrossEncoderReranker(RerankerProvider):
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # We load the model lazily to avoid heavy imports slowing down startup if reranking isn't used.
        self.model = None
        self.max_length = max_length

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading CrossEncoder: {self.model_name} on {self.device}")
            self.model = CrossEncoder(
                self.model_name, max_length=self.max_length, device=self.device
            )

    def rerank(
        self, query: str, candidates: list[Candidate], top_k: int = 10
    ) -> list[Candidate]:
        if not candidates:
            return []

        self._load_model()
        assert self.model is not None

        # Policy: prefer dense_passage, fallback to bm25_passage
        pairs = []
        for c in candidates:
            passage = c.dense_passage if c.dense_passage else c.bm25_passage
            if not passage:
                passage = c.snippet if c.snippet else ""
            pairs.append((query, passage))

        # CrossEncoder scores
        scores = self.model.predict(pairs, show_progress_bar=False)

        for i, c in enumerate(candidates):
            c.reranker_score = float(scores[i])

        # Sort descending by reranker_score
        candidates.sort(key=lambda x: x.reranker_score or 0.0, reverse=True)

        # Assign final ranks
        for i, c in enumerate(candidates):
            c.final_rank = i + 1

        return candidates[:top_k]
