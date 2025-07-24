from fastapi import APIRouter, Depends

from src.api.auth.bearer import auth_secret_key
from src.api.model.chat_cmpl import ModelCard, ModelList
from src.config.gbl import (EMBEDDING_ENGINE_MAPPING, LLM_ENGINE_MAPPING,
                            RERANKER_ENGINE_MAPPING)

model_router = APIRouter()


@model_router.get("/models", response_model=ModelList, dependencies=[Depends(auth_secret_key)])
async def list_models():
    model_cards = [ModelCard(id=alias) for alias in [*LLM_ENGINE_MAPPING.keys(), *EMBEDDING_ENGINE_MAPPING.keys(), *RERANKER_ENGINE_MAPPING.keys()]]
    return ModelList(data=model_cards)

