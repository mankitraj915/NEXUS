"""Fetch the pinned ONNX export of all-MiniLM-L6-v2 and verify every file's checksum.

Run from backend/: uv run python -m scripts.download_onnx_model
"""

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

from app.config.settings import settings
from app.domains.embeddings.onnx_provider import MODEL_REVISION

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"

# Relative path in the repo at MODEL_REVISION -> SHA-256.
FILES: dict[str, str] = {
    "onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    # The repo's other 8-bit files need ARM64 or AVX-512; this one needs only AVX2.
    "onnx/model_quint8_avx2.onnx": "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
}


class ChecksumMismatch(Exception):
    """A downloaded model file does not match its pinned SHA-256."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_model(target: Path) -> None:
    for relative_path, expected in FILES.items():
        path = target / relative_path
        if path.exists() and sha256(path.read_bytes()) == expected:
            continue
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=relative_path,
            revision=MODEL_REVISION,
            local_dir=target,
            force_download=path.exists(),
        )
        actual = sha256(path.read_bytes())
        if actual != expected:
            path.unlink()
            raise ChecksumMismatch(
                f"{relative_path}: expected {expected}, downloaded {actual}"
            )


def main() -> None:
    target = Path(settings.EMBEDDING_MODEL_DIR)
    ensure_model(target)
    print(f"All checksums OK: {len(FILES)} files in {target} at {MODEL_REVISION}")


if __name__ == "__main__":
    main()
