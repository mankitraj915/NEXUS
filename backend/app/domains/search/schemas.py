from pydantic import BaseModel
from typing import List

class SearchResultItem(BaseModel):
    document_id: str
    url: str
    title: str
    score: float
    snippet: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
