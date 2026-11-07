"""Parity checks between the original and the lean serving runtime (gates G1 and later)."""

from collections.abc import Callable, Iterable
from typing import TypedDict

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
