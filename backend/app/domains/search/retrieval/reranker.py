import logging
import threading
from typing import TYPE_CHECKING

from app.domains.search.retrieval.models import Candidate

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

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
        # torch and the model are loaded lazily so that processes which never rerank
        # (the stable search modes) do not pay their import time or memory.
        self.model: CrossEncoder | None = None
        self.max_length = max_length
        # rerank() runs in worker threads; concurrent first requests must load once.
        self._load_lock = threading.Lock()

    def _load_model(self) -> "CrossEncoder":
        with self._load_lock:
            if self.model is None:
                import torch
                from sentence_transformers import CrossEncoder

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading CrossEncoder: {self.model_name} on {device}")
                self.model = CrossEncoder(
                    self.model_name, max_length=self.max_length, device=device
                )
            return self.model

    def rerank(
        self, query: str, candidates: list[Candidate], top_k: int = 10
    ) -> list[Candidate]:
        if not candidates:
            return []

        model = self._load_model()

        pairs = []
        for c in candidates:
            passage = c.dense_passage if c.dense_passage else c.bm25_passage
            if not passage:
                passage = c.snippet if c.snippet else ""
            pairs.append((query, passage))

        scores = model.predict(pairs, show_progress_bar=False)

        for i, c in enumerate(candidates):
            c.reranker_score = float(scores[i])

        candidates.sort(key=lambda x: x.reranker_score or 0.0, reverse=True)

        for i, c in enumerate(candidates):
            c.final_rank = i + 1

        return candidates[:top_k]
