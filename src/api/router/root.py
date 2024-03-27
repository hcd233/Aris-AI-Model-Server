from fastapi import APIRouter

from src.api.model.root import RootResponse

root_router = APIRouter(prefix="/root", tags=["root"])


@root_router.get("/", tags=["root"])
async def root() -> RootResponse:
    return RootResponse(
        code=0,
        status="success",
        message="Welcome to Fibona Embedding API!",
    )
