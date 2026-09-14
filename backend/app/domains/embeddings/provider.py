from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class LocalSentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name_or_path: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name_or_path
        self._model = SentenceTransformer(model_name_or_path)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        if not text:
            return []
        embedding = self._model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return embedding.tolist()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return "1.0"

    @property
    def dimension(self) -> int:
        return 384
