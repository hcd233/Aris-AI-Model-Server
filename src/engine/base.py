from abc import abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseEngine(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def invoke(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `invoke` method")

    @abstractmethod
    def stream(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `stream` method")
