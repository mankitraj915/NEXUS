import importlib.util


class ExperimentalModeUnavailable(Exception):
    """An experimental search mode needs the research (`ml`) dependency group."""

    def __init__(self, mode: str, missing: str):
        super().__init__(f"Mode '{mode}' needs {missing}")
        self.mode = mode
        self.missing = missing


def require_ml(mode: str) -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        raise ExperimentalModeUnavailable(mode, "sentence_transformers")
