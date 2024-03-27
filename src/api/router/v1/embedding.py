from pathlib import Path
from typing import Dict, List

import numpy as np
from cachetools import LRUCache
from fastapi import APIRouter, Depends, HTTPException, status
from text2vec import SentenceModel
from tiktoken import Encoding, get_encoding
from tqdm import tqdm

from src.api.auth.bearer import auth_secret_key
from src.api.model.embedding import EmbeddingRequest, EmbeddingResponse, ListEmbeddingResponse
from src.config.arg import ARGUMENTS
from src.config.env import DEVICE
from src.logger import logger

embedding_router = APIRouter()


def _load_embedding_models(models: List[str], seq_lens: List[int]) -> Dict[str, SentenceModel]:
    name_embedding_map = {}

    if len(models) != len(seq_lens):
        logger.error("[Load Embedding Models] Model and sequence length number mismatch")
        exit(-1)

    for model, sql_len in tqdm(zip(models, seq_lens), desc="Load Embedding Models"):
        _model = Path(model)
        if not _model.exists():
            logger.error(f"[Load Embedding Models] Model not found: {model}")
            exit(-1)

        logger.debug(f"[Load Embedding Models] loading model: `{_model.name}` with sequence length: {sql_len} from path: {model}")
        name_embedding_map[_model.name] = SentenceModel(model, max_seq_length=sql_len, device=DEVICE)

    return name_embedding_map


NAME_EMBEDDING_MAP = _load_embedding_models(ARGUMENTS["model"], ARGUMENTS["seq_len"])


def _load_tiktoken_tokenizer() -> Encoding:
    tokenizer = get_encoding("cl100k_base")

    seq = "Hello, world!"
    if tokenizer.decode(tokenizer.encode(seq)) != seq:
        logger.error("[Load Tiktoken Tokenizer] Tokenizer not working properly")
        exit(-1)

    return tokenizer


TOKENIZER = _load_tiktoken_tokenizer()

LRU_CACHE = LRUCache(maxsize=1000)


@embedding_router.get("/models", dependencies=[Depends(auth_secret_key)])
async def list_models() -> ListEmbeddingResponse:
    return ListEmbeddingResponse(models=[m for m in NAME_EMBEDDING_MAP.keys()])


@embedding_router.post("/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(auth_secret_key)])
async def get_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    logger.info(f"use model: {request.model}")
    try:
        model = NAME_EMBEDDING_MAP[request.model]
    except KeyError:
        logger.error(f"[Embedding] Invalid model name: {request.model}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Invalid model name: {request.model}",
        )

    if isinstance(request.input, str):
        request.input = [request.input]

    if isinstance(request.input[0], List):
        num_tokens_in_batch = [len(i) for i in request.input]
        tokens = sum(num_tokens_in_batch)
        request.input = TOKENIZER.decode_batch(request.input)
        logger.info(f"[Tokenizer] Tiktoken Decode Num: {len(request.input)} Avg Token: {tokens/len(request.input)} Preview: {request.input[0][:20]}")
    else:
        num_tokens_in_batch = [len(i) for i in TOKENIZER.encode_batch(request.input)]
        tokens = sum(num_tokens_in_batch)

    not_cached_ids = []
    not_cached_queries = []
    cached_ids = []
    embeddings = []
    if ARGUMENTS["use_cache"]:
        for i in range(len(request.input)):
            if LRUCache.get(f"{request.model}{request.input[i]}") is None:
                not_cached_ids.append(i)
                not_cached_queries.append(request.input[i])
            else:
                cached_ids.append(i)
            embeddings.append(None)
    else:
        not_cached_queries = request.input
    try:
        no_cached_embeds = model.encode(
            not_cached_queries,
            batch_size=1,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()
    except Exception as e:
        logger.error(f"[Embedding] Model encode error {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model encode error: {e}",
        )

    if ARGUMENTS["use_cache"]:
        logger.info(f"[Cache] Hit: {len(cached_ids)}, Miss: {len(not_cached_ids)}, Cache size: {len(LRU_CACHE)}")
        for i in range(len(not_cached_ids)):
            LRU_CACHE[f"{request.model}{request.input[not_cached_ids[i]]}"] = no_cached_embeds[i]
            embeddings[not_cached_ids[i]] = no_cached_embeds[i]

        for i in range(len(cached_ids)):
            embeddings[cached_ids[i]] = LRU_CACHE[f"{request.model}{request.input[cached_ids[i]]}"]
    else:
        embeddings = no_cached_embeds

    for i in range(len(embeddings)):
        embedding = embeddings[i]
        average = np.average([embedding], axis=0, weights=[num_tokens_in_batch[i]])
        embeddings[i] = (average / np.linalg.norm(average)).tolist()

    return EmbeddingResponse(
        data=[
            {
                "embedding": embeddings[i],
                "index": i,
                "object": "embedding",
            }
            for i in range(len(embeddings))
        ],
        model=request.model,
        object="list",
        usage={"prompt_tokens": tokens, "total_tokens": tokens},
    )
