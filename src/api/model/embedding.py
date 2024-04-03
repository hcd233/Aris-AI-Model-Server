from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel


class ListEmbeddingResponse(BaseModel):
    embeddings: List[Dict[str, Any]]


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[List[int]]]
    model: str


class EmbeddingResponse(BaseModel):
    data: List[Dict[str, Any]]
    model: str
    object: str
    usage: Dict[Literal["prompt_tokens", "total_tokens"], int]
