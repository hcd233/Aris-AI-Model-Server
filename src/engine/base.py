from abc import abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseEngine(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def invoke(self, *args, **kwargs) -> Any:
        pass
