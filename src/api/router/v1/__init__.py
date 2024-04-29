from fastapi import APIRouter

from .embedding import embedding_router
from .reranker import reranker_router
from .chat_cmpl import chat_completion_router

v1_router = APIRouter(prefix="/v1", tags=["v1"])
v1_router.include_router(embedding_router)
v1_router.include_router(reranker_router)
v1_router.include_router(chat_completion_router)

__all__ = ["v1_router"]
