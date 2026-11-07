import subprocess
import sys

import pytest

# NLTK's stemmer drags in SciPy and scikit-learn; Task 2 of the lean-core plan removes it.
NEEDS_STEMMER_SWAP = pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="loaded through NLTK"
)


@pytest.mark.parametrize(
    "module",
    [
        "torch",
        "sentence_transformers",
        pytest.param("scipy", marks=NEEDS_STEMMER_SWAP),
        pytest.param("sklearn", marks=NEEDS_STEMMER_SWAP),
        pytest.param("nltk", marks=NEEDS_STEMMER_SWAP),
    ],
)
def test_app_import_loads_no_heavy_modules(module):
    code = f"import app.main, sys; print({module!r} in sys.modules)"
    # A fresh process: this test session may already have imported the module.
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.fail(f"import failed: {result.stderr}", pytrace=False)
    assert result.stdout.strip() == "False"
