import abc
from collections.abc import Generator
from typing import TypedDict


class CorpusItem(TypedDict):
    id: str
    title: str
    text: str
    metadata: dict


class QueryItem(TypedDict):
    id: str
    text: str


class DatasetProvider(abc.ABC):
    @abc.abstractmethod
    def get_name(self) -> str:
        pass

    @abc.abstractmethod
    def get_version(self) -> str:
        pass

    @abc.abstractmethod
    def prepare(self):
        """Download or prepare dataset locally"""

    @abc.abstractmethod
    def iter_corpus(self) -> Generator[CorpusItem, None, None]:
        pass

    @abc.abstractmethod
    def iter_queries(self, split: str = "test") -> Generator[QueryItem, None, None]:
        pass

    @abc.abstractmethod
    def get_qrels(self, split: str = "test") -> dict[str, dict[str, int]]:
        """Returns {query_id: {doc_id: score}}"""
