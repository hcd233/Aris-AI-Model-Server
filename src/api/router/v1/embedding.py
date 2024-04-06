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


def _load_embedding_models(embeds: List[str], embed_seq_lens: List[int]) -> Dict[str, SentenceModel]:
    name_embedding_map = {}

    if len(embeds) != len(embed_seq_lens):
        logger.error("[Load Embedding Models] Model and sequence length number mismatch")
        exit(-1)

    for embed, sql_len in tqdm(zip(embeds, embed_seq_lens), desc="Load Embedding Models"):
        _embed = Path(embed)
        if not _embed.exists():
            logger.error(f"[Load Embedding Models] Model not found: {_embed}")
            exit(-1)

        logger.debug(f"[Load Embedding Models] loading model: `{_embed.name}` with sequence length: {sql_len} from path: {embed}")
        name_embedding_map[_embed.name] = {}
        name_embedding_map[_embed.name]["model"] = SentenceModel(embed, max_seq_length=sql_len, device=DEVICE)
        name_embedding_map[_embed.name]["max_length"] = sql_len

    return name_embedding_map


NAME_EMBEDDING_MAP = _load_embedding_models(ARGUMENTS["embed"], ARGUMENTS["embed_seq_len"])


def _load_tiktoken_tokenizer() -> Encoding:
    tokenizer = get_encoding("cl100k_base")

    seq = "Hello, world!"
    if tokenizer.decode(tokenizer.encode(seq)) != seq:
        logger.error("[Load Tiktoken Tokenizer] Tokenizer not working properly")
        exit(-1)

    return tokenizer


TOKENIZER = _load_tiktoken_tokenizer()

LRU_CACHE = LRUCache(maxsize=1000)


@embedding_router.get("/embeddings", dependencies=[Depends(auth_secret_key)])
async def list_embeddings() -> ListEmbeddingResponse:
    return ListEmbeddingResponse(embeddings=[{"name": m, "max_length": NAME_EMBEDDING_MAP[m]["max_length"]} for m in NAME_EMBEDDING_MAP.keys()])


@embedding_router.post("/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(auth_secret_key)])
async def embed(request: EmbeddingRequest) -> EmbeddingResponse:
    logger.info(f"use model: {request.model}")
    try:
        model = NAME_EMBEDDING_MAP[request.model]["model"]
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
            f"[Token Count] Tiktoken Decode Num: {len(request.input)} Avg Token: {tokens/len(request.input)} Preview: {request.input[0][:20]}"
        )
    else:
        num_tokens_in_batch = [len(i) for i in TOKENIZER.encode_batch(request.input)]
        tokens = sum(num_tokens_in_batch)
        logger.info(f"[Token Count] Token Num: {len(request.input)} Avg Token: {tokens/len(request.input)} Preview: {request.input[0][:20]}")

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
