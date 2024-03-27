from fastapi import APIRouter

from .embedding import embedding_router

v1_router = APIRouter(prefix="/v1", tags=["v1"])
v1_router.include_router(embedding_router)

__all__ = ["v1_router"]
