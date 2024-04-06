from typing import List, Literal, Union

from pydantic import BaseModel


class EmbeddingModelCard(BaseModel):
    model: str
    max_length: int


class ListEmbeddingResponse(BaseModel):
    embeddings: List[EmbeddingModelCard]


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[List[int]]]
    model: str


class EmbeddingObject(BaseModel):
    embedding: List[float]
    index: int
    object: Literal["embedding"]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: List[EmbeddingObject]
    model: str
    object: Literal["list"]
    usage: EmbeddingUsage
