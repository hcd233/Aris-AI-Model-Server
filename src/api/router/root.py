from fastapi import APIRouter

from src.api.model.root import RootResponse

root_router = APIRouter(tags=["root"])


@root_router.get("/", tags=["root"])
async def root() -> RootResponse:
    return RootResponse(
        code=0,
        status="success",
        message="Welcome to Fibona Model API Server!",
    )
