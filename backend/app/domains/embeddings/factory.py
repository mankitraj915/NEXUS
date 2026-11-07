import functools
from pathlib import Path

from app.config.settings import settings
from app.domains.embeddings.provider import (
    EmbeddingProvider,
    LocalSentenceTransformerProvider,
)


@functools.lru_cache(maxsize=1)
def get_provider() -> EmbeddingProvider:
    """The process-wide embedding provider. Cached so model weights stay in memory."""
    if settings.EMBEDDING_BACKEND == "torch":
        return LocalSentenceTransformerProvider()

    from app.domains.embeddings.onnx_provider import OnnxMiniLMProvider

    return OnnxMiniLMProvider(
        Path(settings.EMBEDDING_MODEL_DIR), variant=settings.EMBEDDING_ONNX_VARIANT
    )
