from pydantic import BaseModel


class RootResponse(BaseModel):
    code: int
    status: str
    message: str
