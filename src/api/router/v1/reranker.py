from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth.bearer import auth_secret_key
from src.api.model.reranker import ListRerankerResponse, RerankerRequest, RerankerResponse, RerankModelCard, RerankObject
from src.config.gbl import MODEL_CONTROLLER
from src.logger import logger

reranker_router = APIRouter()

reranker_engine_mapping = MODEL_CONTROLLER.get_reranker_engines()


@reranker_router.get("/rerankers", response_model=ListRerankerResponse, dependencies=[Depends(auth_secret_key)])
async def list_rerankers() -> ListRerankerResponse:
    return ListRerankerResponse(
        rerankers=[
            RerankModelCard(
                model=engine.alias,
                max_length=engine.max_seq_len,
            )
            for engine in reranker_engine_mapping.values()
        ]
    )


@reranker_router.post("/rerankers", response_model=RerankerResponse, dependencies=[Depends(auth_secret_key)])
async def rerank(request: RerankerRequest) -> RerankerResponse:
    logger.info(f"use model: {request.model}")
    if not request.query or not request.documents:
        return RerankerResponse(data=[], model=request.model)

    try:
        engine = reranker_engine_mapping[request.model]
    except KeyError:
        logger.error(f"[Embedding] Invalid model name: {request.model}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid model name: {request.model}",
        )
    results = engine.invoke(request.query, [doc.content for doc in request.documents])

    logger.debug(f"[Rerank] result: {results}")

    return RerankerResponse(
        data=sorted(
            [
                RerankObject(
                    doc=doc,
                    score=round(res["score"], 6),
                    rank=res["rank"],
                )
                for doc, res in zip(request.documents, results)
            ],
            key=lambda x: x.rank,
        ),
        model=request.model,
    )
