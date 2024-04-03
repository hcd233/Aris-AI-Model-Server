from typing import Any, Dict, List

from pydantic import BaseModel


class ListRerankerResponse(BaseModel):
    rerankers: List[Dict[str, Any]]


class RerankerRequest(BaseModel):
    query: str
    documents: List[str]
    normalize: bool = True
    model: str


class RerankerResponse(BaseModel):
    data: List[Dict[str, Any]]
    model: str
