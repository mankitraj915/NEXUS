import platform
from pathlib import Path

import pytest

from scripts.benchmark.provenance import python_version_line

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_line_names_the_running_interpreter():
    assert python_version_line() == f"- **Python**: {platform.python_version()}"


@pytest.mark.parametrize(
    "writer",
    [
        "benchmark/__main__.py",
        "benchmark/llm_eval.py",
        "router_training/evaluate.py",
        "router_training/train.py",
    ],
)
def test_every_report_writer_records_the_python_version(writer):
    assert "python_version_line()" in (SCRIPTS / writer).read_text(encoding="utf-8")
