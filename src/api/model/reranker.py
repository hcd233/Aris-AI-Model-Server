from typing import List

from pydantic import BaseModel


class RerankModelCard(BaseModel):
    model: str
    max_length: int


class ListRerankerResponse(BaseModel):
    rerankers: List[RerankModelCard]


class RerankerRequest(BaseModel):
    query: str
    documents: List[str]
    model: str
    normalize: bool = True


class RerankObject(BaseModel):
    doc: str
    score: float
    rank: int


class RerankerResponse(BaseModel):
    data: List[RerankObject]
    model: str
