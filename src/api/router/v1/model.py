from fastapi import APIRouter, Depends

from src.api.auth.bearer import auth_secret_key
from src.api.model.chat_cmpl import ModelCard, ModelList
from src.config.gbl import MODEL_CONTROLLER

llm_engine_mapping = MODEL_CONTROLLER.get_llm_engines()
embedding_engine_mapping = MODEL_CONTROLLER.get_embedding_engines()
reranker_engine_mapping = MODEL_CONTROLLER.get_reranker_engines()

model_router = APIRouter()


@model_router.get("/models", response_model=ModelList, dependencies=[Depends(auth_secret_key)])
async def list_models():
    model_cards = [ModelCard(id=alias) for alias in (llm_engine_mapping.keys() + embedding_engine_mapping.keys() + reranker_engine_mapping.keys())]
    return ModelList(data=model_cards)

