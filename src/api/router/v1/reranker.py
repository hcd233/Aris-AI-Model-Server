from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from fastapi import APIRouter, Depends
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.api.auth.bearer import auth_secret_key
from src.api.model.reranker import ListRerankerResponse, RerankerRequest, RerankerResponse
from src.config.arg import ARGUMENTS
from src.logger import logger

reranker_router = APIRouter()


def _load_reranker_models(
    reranks: List[str], rerank_seq_lens: List[int]
) -> Dict[str, Dict[str, AutoModelForSequenceClassification | AutoTokenizer | int]]:
    name_reranker_map = {}

    if len(reranks) != len(rerank_seq_lens):
        logger.error("[Load Rerank Models] Model and sequence length number mismatch")
        exit(-1)

    for model, seq_len in tqdm(zip(reranks, rerank_seq_lens), desc="Load Rerank Models"):
        _model = Path(model)
        if not _model.exists():
            logger.error(f"[Load Rerank Models] Model not found: {model}")
            exit(-1)

        logger.debug(f"[Load Reranker Models] loading model: `{_model.name}` with sequence length: {seq_len} from path: {model}")
        name_reranker_map[_model.name] = {}
        name_reranker_map[_model.name]["model"] = AutoModelForSequenceClassification.from_pretrained(model)
        name_reranker_map[_model.name]["tokenizer"] = AutoTokenizer.from_pretrained(model)
        name_reranker_map[_model.name]["max_length"] = seq_len

    return name_reranker_map


NAME_RERANKER_MAP = _load_reranker_models(ARGUMENTS["rerank"], ARGUMENTS["rerank_seq_len"])


@reranker_router.get("/rerankers", dependencies=[Depends(auth_secret_key)])
async def list_rerankers() -> ListRerankerResponse:
    return ListRerankerResponse(rerankers=[{"name": m, "max_length": NAME_RERANKER_MAP[m]["max_length"]} for m in NAME_RERANKER_MAP.keys()])


@reranker_router.post("/rerankers", response_model=RerankerResponse, dependencies=[Depends(auth_secret_key)])
async def rerank(request: RerankerRequest) -> RerankerResponse:
    pairs = []
    if not request.query or not request.documents:
        return RerankerResponse(data=[], model=request.model)

    for doc in request.documents:
        pairs.append([request.query, doc])

    with torch.no_grad():
        model = NAME_RERANKER_MAP[request.model]["model"]
        tokenizer = NAME_RERANKER_MAP[request.model]["tokenizer"]
        max_length = NAME_RERANKER_MAP[request.model]["max_length"]

        inputs = tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        scores = (
            model(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
        )
        if request.normalize:
            scores = F.softmax(scores, dim=0)

    scores = scores.numpy()
    scores, ranks = scores.tolist(), (-scores).argsort().argsort().tolist()
    
    logger.debug(f"[Rerank] scores: {scores}, ranks: {ranks}")

    data = [{"doc": doc, "score": round(score, 6), "rank": rank} for doc, score, rank in zip(request.documents, scores, ranks)]
    data.sort(key=lambda x: x["rank"])

    return RerankerResponse(data=data, model=request.model)
