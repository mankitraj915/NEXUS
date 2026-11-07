"""Parity checks between the original and the lean serving runtime (gates G1 and later)."""

import gc
import random
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypedDict

import numpy as np

from app.domains.embeddings.chunker import chunk_text
from app.domains.search.analyzer import _stemmer, tokenize_text

MAX_EXAMPLES = 50


class AnalyzerParityResult(TypedDict):
    texts: int
    tokens: int
    mismatched_texts: int
    # (token, NLTK stem, serving stem)
    examples: list[tuple[str, str, str]]


def _new_stemmer() -> Callable[[str], str]:
    # The stemmer the analyzer really uses, so the gate tests what is served.
    return _stemmer.stem


def _old_stemmer() -> Callable[[str], str]:
    # Local import: NLTK is only in the `ml` group and must stay out of serving.
    from nltk.stem.snowball import SnowballStemmer

    return SnowballStemmer("english").stem


def analyzer_parity(texts: Iterable[str]) -> AnalyzerParityResult:
    """Compare real NLTK with the serving stemmer over the analyzer's tokens.

    Stopwords are kept because analyze_text filters before stemming and features.py calls
    it with remove_stopwords=False: this stream covers both ways the analyzer is used.
    """
    old_stem, new_stem = _old_stemmer(), _new_stemmer()
    stems: dict[str, tuple[str, str]] = {}  # stemmers are pure; most tokens repeat
    examples: list[tuple[str, str, str]] = []
    n_texts = n_tokens = mismatched = 0

    for text in texts:
        n_texts += 1
        differs = False
        for token in tokenize_text(text, remove_stopwords=False):
            n_tokens += 1
            pair = stems.get(token)
            if pair is None:
                pair = stems[token] = (old_stem(token), new_stem(token))
                if pair[0] != pair[1] and len(examples) < MAX_EXAMPLES:
                    examples.append((token, pair[0], pair[1]))
            differs = differs or pair[0] != pair[1]
        mismatched += differs

    return {
        "texts": n_texts,
        "tokens": n_tokens,
        "mismatched_texts": mismatched,
        "examples": examples,
    }


class CosineSummary(TypedDict):
    mean: float
    min: float
    p1: float


# Gate G2, fp32 only (spec section 5). int8 is reported separately.
G2_MIN_MEAN = 0.9999
G2_MIN_MIN = 0.999


def cosine_summary(a: np.ndarray, b: np.ndarray) -> CosineSummary:
    """Row-wise cosine between two (N, D) arrays."""
    a_unit = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_unit = b / np.linalg.norm(b, axis=1, keepdims=True)
    cosines = (a_unit * b_unit).sum(axis=1)
    return {
        "mean": float(cosines.mean()),
        "min": float(cosines.min()),
        "p1": float(np.percentile(cosines, 1)),
    }


def fp32_passes_g2(chunks: CosineSummary, queries: CosineSummary) -> bool:
    return all(
        summary["mean"] >= G2_MIN_MEAN and summary["min"] >= G2_MIN_MIN
        for summary in (chunks, queries)
    )


def sample_chunks(
    documents: Iterable[str], sample: int, seed: int
) -> tuple[int, list[str]]:
    """Split with the production chunker, then draw a seeded sample of the chunks."""
    chunks = [chunk for document in documents for chunk in chunk_text(document)]
    return len(chunks), random.Random(seed).sample(chunks, sample)


def _encode_timed(
    provider: Any, chunks: list[str], queries: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Encode as production does: documents in batches, each query on its own."""
    provider.embed_query("warm-up")  # the model load is not part of the timing

    start = time.perf_counter()
    chunk_vectors = np.array(provider.embed_documents(chunks))
    chunk_seconds = time.perf_counter() - start

    start = time.perf_counter()
    query_vectors = np.array([provider.embed_query(q) for q in queries])
    query_seconds = time.perf_counter() - start

    return (
        chunk_vectors,
        query_vectors,
        {
            "chunks": chunk_seconds / len(chunks) * 100 * 1000,
            "queries": query_seconds / len(queries) * 100 * 1000,
        },
    )


def embedding_parity(
    chunks: list[str],
    queries: list[str],
    baseline: Any,
    candidates: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare each candidate's vectors with the baseline's, text by text."""
    base_chunks, base_queries, base_ms = _encode_timed(baseline, chunks, queries)
    result: dict[str, Any] = {"candidates": {}, "ms_per_100": {"baseline": base_ms}}
    for name, provider in candidates.items():
        cand_chunks, cand_queries, ms = _encode_timed(provider, chunks, queries)
        result["candidates"][name] = {
            "chunks": cosine_summary(base_chunks, cand_chunks),
            "queries": cosine_summary(base_queries, cand_queries),
        }
        result["ms_per_100"][name] = ms
        gc.collect()
    return result
