from app.domains.search.analyzer import analyze_text


def test_analyze_basic():
    tokens = analyze_text("PostgreSQL 17 supports AI/ML workloads.")
    assert tokens == ["postgresql", "17", "support", "ai", "ml", "workload"]


def test_analyze_stopwords():
    tokens = analyze_text("The quick brown fox jumps over the lazy dog.")
    assert tokens == ["quick", "brown", "fox", "jump", "lazi", "dog"]


def test_analyze_no_stopwords_removal():
    tokens = analyze_text("to be or not to be", remove_stopwords=False)
    assert tokens == ["to", "be", "or", "not", "to", "be"]


def test_analyze_unicode_and_hyphens():
    tokens = analyze_text("AI-native state-of-the-art naïve")
    assert tokens == ["ai-nat", "state-of-the-art", "naïv"]


def test_vendored_stemmer_matches_nltk_on_fixture():
    from nltk.stem.snowball import SnowballStemmer

    from app.domains.search._vendor.nltk_snowball.snowball import EnglishStemmer

    words = [
        "generously", "dying", "news", "skies", "cancer", "cancerous", "proteins",
        "running", "lazy", "cells", "expression", "mutations", "receptors", "inhibits",
        "antibodies", "studies", "tumour", "mice", "analysis", "analyses", "bias",
        "early", "only", "sky", "howe", "atlas", "cosmos", "innings", "succeed",
        "exceeding",
    ]  # fmt: skip
    assert len(words) == 30
    new = EnglishStemmer()
    old = SnowballStemmer("english")
    differing = [w for w in words if new.stem(w) != old.stem(w)]
    assert differing == []
