from abc import abstractmethod
from typing import Any, Literal

from pydantic import BaseModel
from typing_extensions import TypedDict


class BaseEngine(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def invoke(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `invoke` method")

    @abstractmethod
    async def stream(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `stream` method")


class RerankerResult(TypedDict):
    index: int
    relevent_score: float


class EmbeddingResult(TypedDict):
    embedding: list[float]
    index: int
    object: Literal["embedding"]


class LLMResult(TypedDict):
    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]
