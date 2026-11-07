from pathlib import Path

import numpy as np
import pytest

from app.config.settings import settings
from app.domains.embeddings.factory import get_provider
from app.domains.embeddings.onnx_provider import (
    MODEL_FILES,
    MODEL_REVISION,
    OnnxMiniLMProvider,
    mean_pool,
)
from app.domains.embeddings.provider import LocalSentenceTransformerProvider
from scripts import download_onnx_model

MODEL_DIR = Path(settings.EMBEDDING_MODEL_DIR)


@pytest.fixture(scope="module")
def provider():
    return OnnxMiniLMProvider(MODEL_DIR)


def test_mean_pool_ignores_padding():
    tokens = np.array([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = np.array([[1, 1, 0]])
    np.testing.assert_array_equal(mean_pool(tokens, mask), [[2.0, 2.0]])


def test_mean_pool_handles_each_row_of_a_batch_separately():
    tokens = np.array([[[2.0], [4.0]], [[6.0], [100.0]]])
    mask = np.array([[1, 1], [1, 0]])
    np.testing.assert_array_equal(mean_pool(tokens, mask), [[3.0], [6.0]])


def test_embed_query_is_unit_norm_384(provider):
    vector = provider.embed_query("Does vitamin D reduce fracture risk?")
    assert len(vector) == 384
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)


def test_long_text_truncates_not_errors(provider):
    assert len(provider.embed_query("word " * 5000)) == 384


def test_truncation_is_at_256_tokens_not_the_tokenizer_files_128(provider):
    # A word inside the window must change the vector, and one past it must not. 200 words
    # sit past the 128 stored in tokenizer.json, so that limit would fail the first check.
    inside = "word " * 200
    assert provider.embed_query(inside + "cat") != provider.embed_query(inside + "dog")
    past = "word " * 300
    assert provider.embed_query(past + "cat") == provider.embed_query(past + "dog")


def test_non_ascii_query_embeds(provider):
    assert len(provider.embed_query("Ünïcödé 中文 query")) == 384


def test_empty_inputs(provider):
    assert provider.embed_query("") == []
    assert provider.embed_documents([]) == []


def test_documents_keep_input_order_and_match_single_encoding(provider):
    # Batches are sorted by length internally, so order must be restored.
    texts = [
        "a much longer sentence about protein folding in cells",
        "short",
        "mid text",
    ]
    batch = provider.embed_documents(texts)
    for text, vector in zip(texts, batch):
        np.testing.assert_allclose(vector, provider.embed_query(text), atol=1e-5)


def test_model_version_tags_runtime_variant_and_revision():
    int8 = OnnxMiniLMProvider(MODEL_DIR, variant="int8")
    assert int8.model_name == "all-MiniLM-L6-v2"
    assert int8.model_version == f"onnx-int8@{MODEL_REVISION[:12]}"
    assert int8.dimension == 384


def test_int8_variant_loads_and_embeds():
    vector = OnnxMiniLMProvider(MODEL_DIR, variant="int8").embed_query("protein")
    assert len(vector) == 384


def test_missing_model_names_the_download_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts.download_onnx_model"):
        OnnxMiniLMProvider(tmp_path).embed_query("x")


@pytest.fixture
def fresh_factory(monkeypatch):
    get_provider.cache_clear()
    yield monkeypatch
    get_provider.cache_clear()


def test_factory_respects_backend_setting(fresh_factory):
    fresh_factory.setattr(settings, "EMBEDDING_BACKEND", "torch")
    assert isinstance(get_provider(), LocalSentenceTransformerProvider)
    assert get_provider().model_version == "1.0"  # existing rows must stay valid

    get_provider.cache_clear()
    fresh_factory.setattr(settings, "EMBEDDING_BACKEND", "onnx")
    fresh_factory.setattr(settings, "EMBEDDING_ONNX_VARIANT", "int8")
    provider = get_provider()
    assert isinstance(provider, OnnxMiniLMProvider)
    assert provider.model_version.startswith("onnx-int8@")


def test_factory_defaults_to_onnx_fp32():
    fields = type(settings).model_fields
    assert fields["EMBEDDING_BACKEND"].default == "onnx"
    assert fields["EMBEDDING_ONNX_VARIANT"].default == "fp32"


def test_download_script_pins_a_hash_for_every_variant():
    assert set(MODEL_FILES.values()) <= set(download_onnx_model.FILES)
    assert {"tokenizer.json", "config.json"} <= set(download_onnx_model.FILES)
    assert all(len(sha) == 64 for sha in download_onnx_model.FILES.values())
    assert download_onnx_model.MODEL_REVISION == MODEL_REVISION


def _fake_download(content: bytes):
    def fake(*, local_dir, filename, **_):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    return fake


def test_ensure_model_rejects_a_wrong_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(download_onnx_model, "hf_hub_download", _fake_download(b"bad"))
    with pytest.raises(download_onnx_model.ChecksumMismatch):
        download_onnx_model.ensure_model(tmp_path)
    # A corrupt file must not be left behind for the provider to load.
    assert not any(p.is_file() for p in tmp_path.rglob("*"))


def test_ensure_model_skips_files_already_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download_onnx_model, "FILES", {"f.bin": download_onnx_model.sha256(b"ok")}
    )
    (tmp_path / "f.bin").write_bytes(b"ok")

    def must_not_download(**_):
        raise AssertionError("verified file was downloaded again")

    monkeypatch.setattr(download_onnx_model, "hf_hub_download", must_not_download)
    download_onnx_model.ensure_model(tmp_path)
