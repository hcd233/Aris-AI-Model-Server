from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth.bearer import auth_secret_key
from src.api.model.embedding import (EmbeddingModelCard, EmbeddingObject,
                                     EmbeddingRequest, EmbeddingResponse,
                                     EmbeddingUsage, ListEmbeddingResponse)
from src.config.gbl import EMBEDDING_ENGINE_MAPPING, TOKENIZER
from src.logger import logger

embedding_router = APIRouter()

@embedding_router.get("/embeddings", response_model=ListEmbeddingResponse, dependencies=[Depends(auth_secret_key)])
async def list_embeddings() -> ListEmbeddingResponse:
    return ListEmbeddingResponse(
        embeddings=[
            EmbeddingModelCard(
                model=engine.alias,
                max_length=engine.max_seq_len,
            )
            for engine in EMBEDDING_ENGINE_MAPPING.values()
        ]
    )


@embedding_router.post("/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(auth_secret_key)])
async def embed(request: EmbeddingRequest) -> EmbeddingResponse:
    logger.info(f"use model: {request.model}")
    try:
        engine = EMBEDDING_ENGINE_MAPPING[request.model]
    except KeyError:
        logger.error(f"[Embedding] Invalid model name: {request.model}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid model name: {request.model}",
        )

    if isinstance(request.input, str):
        request.input = [request.input]

    if isinstance(request.input[0], List):
        num_tokens_in_batch = [len(i) for i in request.input]
        tokens = sum(num_tokens_in_batch)
        request.input = TOKENIZER.decode_batch(request.input)
        logger.info(
            f"[Token Count] Tiktoken Decode Num: {len(request.input)} Avg Token: {round(tokens/len(request.input), 3)} Preview: {request.input[0][:20]}"
        )
    else:
        num_tokens_in_batch = [len(i) for i in TOKENIZER.encode_batch(request.input)]
        tokens = sum(num_tokens_in_batch)
        logger.info(
            f"[Token Count] Token Num: {len(request.input)} Avg Token: {round(tokens/len(request.input), 3)} Preview: {request.input[0][:20]}"
        )

    try:
        results = engine.invoke(request.input)
    except Exception as e:
        logger.error(f"[Embedding] Model encode error {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model encode error: {e}",
        )

    return EmbeddingResponse(
        data=[EmbeddingObject(**result) for result in results],
        model=request.model,
        object="list",
        usage=EmbeddingUsage(prompt_tokens=0, total_tokens=tokens),
    )
