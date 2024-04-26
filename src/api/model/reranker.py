from typing import Any, Dict, List

from pydantic import BaseModel


class Document(BaseModel):
    content: str
    metadata: Dict[str, Any]


class RerankModelCard(BaseModel):
    model: str
    max_length: int


class ListRerankerResponse(BaseModel):
    rerankers: List[RerankModelCard]


class RerankerRequest(BaseModel):
    query: str
    documents: List[Document]
    model: str


class RerankObject(BaseModel):
    doc: Document
    score: float
    rank: int


class RerankerResponse(BaseModel):
    data: List[RerankObject]
    model: str
