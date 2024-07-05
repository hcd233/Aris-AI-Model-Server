from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth.bearer import auth_secret_key
from src.api.model.reranker import ListRerankerResponse, RerankerRequest, RerankerResponse, RerankModelCard, RerankObject
from src.config.gbl import RERANKER_MAPPING
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
            for engine in RERANKER_MAPPING.values()
        ]
    )


@reranker_router.post("/rerankers", response_model=RerankerResponse, dependencies=[Depends(auth_secret_key)])
async def rerank(request: RerankerRequest) -> RerankerResponse:
    logger.info(f"use model: {request.model}")
    if not request.query or not request.documents:
        return RerankerResponse(data=[], model=request.model)

    try:
        engine = RERANKER_MAPPING[request.model]
    except KeyError:
        logger.error(f"[Embedding] Invalid model name: {request.model}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid model name: {request.model}",
        )
    score_rank_pairs = engine.invoke(request.query, [doc.content for doc in request.documents])

    logger.debug(f"[Rerank] result: {score_rank_pairs}")

    return RerankerResponse(
        data=sorted(
            [
                RerankObject(
                    doc=doc,
                    score=round(score, 6),
                    rank=rank,
                )
                for doc, (score, rank) in zip(request.documents, score_rank_pairs)
            ],
            key=lambda x: x.rank,
        ),
        model=request.model,
    )
