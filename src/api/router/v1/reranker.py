from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from sentence_transformers import CrossEncoder
from torch.nn.functional import softmax
from tqdm import tqdm

from src.api.auth.bearer import auth_secret_key
from src.api.model.reranker import ListRerankerResponse, RerankerRequest, RerankerResponse, RerankModelCard, RerankObject
from src.config.arg import ARGUMENTS
from src.config.env import BATCH_SIZE, DEVICE
from src.logger import logger

reranker_router = APIRouter()


def _load_reranker_models(reranks: List[str], rerank_max_lengths: List[int]) -> Dict[str, Dict[str, CrossEncoder | int]]:
    name_reranker_map = {}

    if len(reranks) != len(rerank_max_lengths):
        logger.error("[Load Rerank Models] Model and sequence length number mismatch")
        exit(-1)

    for model, max_length in tqdm(zip(reranks, rerank_max_lengths), desc="Load Rerank Models"):
        _model = Path(model)
        if not _model.exists():
            logger.error(f"[Load Rerank Models] Model not found: {model}")
            exit(-1)

        logger.debug(f"[Load Reranker Models] loading model: `{_model.name}` with max sequence length: {max_length} from path: {model}")
        name_reranker_map[_model.name] = {}
        name_reranker_map[_model.name]["model"] = CrossEncoder(model, max_length=max_length, device=DEVICE)
        name_reranker_map[_model.name]["max_length"] = max_length

    return name_reranker_map


NAME_RERANKER_MAP = _load_reranker_models(ARGUMENTS["rerank"], ARGUMENTS["rerank_seq_len"])


@reranker_router.get("/rerankers", dependencies=[Depends(auth_secret_key)])
async def list_rerankers() -> ListRerankerResponse:
    return ListRerankerResponse(
        rerankers=[
            RerankModelCard(
                model=m,
                max_length=NAME_RERANKER_MAP[m]["max_length"],
            )
            for m in NAME_RERANKER_MAP.keys()
        ]
    )


@reranker_router.post("/rerankers", response_model=RerankerResponse, dependencies=[Depends(auth_secret_key)])
async def rerank(request: RerankerRequest) -> RerankerResponse:
    pairs = []
    if not request.query or not request.documents:
        return RerankerResponse(data=[], model=request.model)

    for doc in request.documents:
        pairs.append([request.query, doc.content])

    try:
        model = NAME_RERANKER_MAP[request.model]["model"]
    except KeyError:
        logger.error(f"[Embedding] Invalid model name: {request.model}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid model name: {request.model}",
        )
    scores = model.predict(
        pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        activation_fct=lambda x: x,  # NOTE sentence_transformers CrossEncoder will use sigmoid to normalize the score. Disable it here.
        convert_to_tensor=True,
        convert_to_numpy=False,
    )

    if request.normalize:
        scores = softmax(scores, dim=0)

    scores = scores.numpy()
    scores, ranks = scores.tolist(), (-scores).argsort().argsort().tolist()

    logger.debug(f"[Rerank] scores: {scores}, ranks: {ranks}")

    return RerankerResponse(
        data=sorted(
            [
                RerankObject(
                    doc=doc,
                    score=round(score, 6),
                    rank=rank,
                )
                for doc, score, rank in zip(request.documents, scores, ranks)
            ],
            key=lambda x: x.rank,
        ),
        model=request.model,
    )
