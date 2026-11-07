"""Facts every benchmark report must state about the run that produced it."""

import platform


def python_version_line() -> str:
    return f"- **Python**: {platform.python_version()}"
