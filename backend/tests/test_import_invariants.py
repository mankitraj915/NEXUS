import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    ["torch", "sentence_transformers", "scipy", "sklearn", "nltk"],
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
