from scripts.benchmark import parity


def test_analyzer_parity_counts_identical_streams():
    result = parity.analyzer_parity(["The running mice", "Proteins and cancer"])
    assert result == {
        "texts": 2,
        "tokens": 6,
        "mismatched_texts": 0,
        "examples": [],
    }


def test_analyzer_parity_reports_each_differing_stem_once(monkeypatch):
    real = parity._new_stemmer()
    monkeypatch.setattr(
        parity,
        "_new_stemmer",
        lambda: lambda t: "DIFFERENT" if t == "running" else real(t),
    )
    result = parity.analyzer_parity(["running mice", "mice", "running running"])
    assert result["texts"] == 3
    assert result["mismatched_texts"] == 2
    assert result["examples"] == [("running", "run", "DIFFERENT")]


def test_analyzer_parity_caps_examples_at_50(monkeypatch):
    monkeypatch.setattr(parity, "_new_stemmer", lambda: lambda t: t + "!")
    text = " ".join(f"word{i}" for i in range(60))
    result = parity.analyzer_parity([text])
    assert result["mismatched_texts"] == 1
    assert len(result["examples"]) == 50


def test_gate_compares_real_nltk_with_the_serving_stemmer():
    # If both sides were one implementation the gate would pass without proving anything.
    old_module = parity._old_stemmer().__module__
    new_module = parity._new_stemmer().__module__
    assert old_module == "nltk.stem.snowball"
    assert new_module == "app.domains.search._vendor.nltk_snowball.snowball"
