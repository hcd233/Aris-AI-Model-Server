from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RerankModelCard(BaseModel):
    model: str
    max_length: int


class ListRerankerResponse(BaseModel):
    rerankers: List[RerankModelCard]


# Cohere协议的请求格式
class RerankerRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None


# Cohere协议的响应格式
class RerankerResult(BaseModel):
    index: int
    relevance_score: float


class ApiVersion(BaseModel):
    version: str


class BilledUnits(BaseModel):
    search_units: int


class Meta(BaseModel):
    api_version: ApiVersion
    billed_units: BilledUnits


class RerankerResponse(BaseModel):
    results: List[RerankerResult]
    id: str
    meta: Meta
