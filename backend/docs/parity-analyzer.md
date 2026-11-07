# Analyzer parity (gate G1): SCIFACT

Compares real `nltk` SnowballStemmer("english") with the stemmer the analyzer serves
(`app/domains/search/_vendor/nltk_snowball`) over the analyzer's tokens, stopwords
kept (a superset of the stopword-filtered stream).

- **Result**: PASS
- **Texts**: 6292 (5183 corpus title+abstract, 1109 queries)
- **Tokens**: 1143237
- **mismatched_texts**: 0
- **Measured**: 2026-11-07 13:18 UTC at commit `347e10b`
- **Hardware**: Intel(R) Core(TM) i5-10200H CPU @ 2.40GHz, 8 logical CPUs, Windows-11-10.0.26300-SP0
- **Versions**: Python 3.14.2, nltk 3.10.3

## Reproduce

`uv sync --group ml`, then `uv run python -m scripts.benchmark parity-analyzer --dataset scifact`.

## Rejected candidates

See [parity-analyzer-rejected-candidates.md](parity-analyzer-rejected-candidates.md).
