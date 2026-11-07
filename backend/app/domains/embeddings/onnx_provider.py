import threading
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.domains.embeddings.provider import EmbeddingProvider

MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILES = {"fp32": "onnx/model.onnx", "int8": "onnx/model_quint8_avx2.onnx"}
TOKENIZER_FILE = "tokenizer.json"

# The model's sentence_bert_config.json; tokenizer.json itself says 128.
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 32


def mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Average each row's token vectors, counting only tokens the mask marks as real."""
    mask = attention_mask[..., np.newaxis].astype(token_embeddings.dtype)
    return (token_embeddings * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.clip(
        np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None
    )


class OnnxMiniLMProvider(EmbeddingProvider):
    def __init__(self, model_dir: Path, variant: Literal["fp32", "int8"] = "fp32"):
        self._model_dir = Path(model_dir)
        self._variant = variant
        self._session: Any = None
        self._tokenizer: Any = None
        self._needs_token_types = False
        # Queries embed in worker threads; concurrent first requests must load once.
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        with self._load_lock:
            if self._session is not None:
                return
            model_path = self._model_dir / MODEL_FILES[self._variant]
            tokenizer_path = self._model_dir / TOKENIZER_FILE
            for path in (model_path, tokenizer_path):
                if not path.exists():
                    raise FileNotFoundError(
                        f"{path} is missing: run `uv run python -m scripts.download_onnx_model`"
                    )

            import onnxruntime as ort
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
            tokenizer.enable_padding()  # to the longest in the batch, not the file's fixed 128

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(model_path), options, providers=["CPUExecutionProvider"]
            )
            self._needs_token_types = "token_type_ids" in {
                model_input.name for model_input in session.get_inputs()
            }
            self._tokenizer = tokenizer
            self._session = session

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        inputs = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": attention_mask,
        }
        if self._needs_token_types:
            inputs["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )
        token_embeddings = self._session.run(None, inputs)[0]
        return _l2_normalize(mean_pool(token_embeddings, attention_mask))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        # Similar lengths share a batch, so less time is spent on padding.
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        vectors: list[list[float]] = [[] for _ in texts]
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            encoded = self._encode_batch([texts[i] for i in batch])
            for i, vector in zip(batch, encoded.tolist()):
                vectors[i] = vector
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text:
            return []
        self._load()
        return self._encode_batch([text])[0].tolist()

    @property
    def model_name(self) -> str:
        return "all-MiniLM-L6-v2"

    @property
    def model_version(self) -> str:
        return f"onnx-{self._variant}@{MODEL_REVISION[:12]}"

    @property
    def dimension(self) -> int:
        return 384
