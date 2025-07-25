import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth.bearer import auth_secret_key
from src.api.model.reranker import (ApiVersion, BilledUnits,
                                    ListRerankerResponse, RerankMeta,
                                    RerankModelCard, RerankRequest,
                                    RerankResponse, RerankResult)
from src.config.gbl import RERANKER_ENGINE_MAPPING
from src.logger import logger

reranker_router = APIRouter()


@reranker_router.get("/rerankers", response_model=ListRerankerResponse, dependencies=[Depends(auth_secret_key)])
async def list_rerankers() -> ListRerankerResponse:
    return ListRerankerResponse(
        rerankers=[
            RerankModelCard(
                model=engine.alias,
                max_length=engine.max_seq_len,
            )
            for engine in RERANKER_ENGINE_MAPPING.values()
        ]
    )


@reranker_router.post("/rerank", response_model=RerankResponse, dependencies=[Depends(auth_secret_key)])
async def rerank(request: RerankRequest) -> RerankResponse:
    logger.info(f"[Reranker] use model: {request.model}")
    if not request.query or not request.documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request, query and documents are required")

    try:
        engine = RERANKER_ENGINE_MAPPING[request.model]
    except KeyError:
        logger.error(f"[Reranker] Invalid model name: {request.model}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid model name: {request.model}",
        )

    results = await engine.invoke(request.query, request.documents)
    logger.debug(f"[Cohere Rerank] result: {results}")

    sorted_results = sorted(enumerate(results), key=lambda x: x[1]["relevent_score"], reverse=True)

    if request.top_n is not None:
        sorted_results = sorted_results[:request.top_n]

    cohere_results = [
        RerankResult(
            index=original_index,
            relevance_score=round(result["relevent_score"], 6)
        )
        for original_index, result in sorted_results
    ]

    return RerankResponse(
        results=cohere_results,
        id=uuid.uuid4().hex,
        meta=RerankMeta(
            api_version=ApiVersion(version="1"),
            billed_units=BilledUnits(search_units=1),
        ),
    )
