import asyncio
from typing import List

from cachetools import LRUCache
from sentence_transformers import SentenceTransformer

from src.config.arg import EmbeddingConfig
from src.config.env import DEVICE
from src.logger import logger

from ..base import BaseEngine, EmbeddingResult


class SentenceTransformerEmbeddingEngine(BaseEngine, EmbeddingConfig):
    model: SentenceTransformer
    cache: LRUCache | None

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "SentenceTransformerEmbeddingEngine":
        model = SentenceTransformer(config.path, device=DEVICE)
        model.max_seq_length = config.max_seq_len
        cache = LRUCache(maxsize=1000) if config.enable_cache else None

        logger.success(f"[SentenceTransformerEmbeddingEngine] load model from {config.path}")
        return cls(model=model, cache=cache, **config.model_dump())

    def _invoke(self, sentences: List[str]) -> List[EmbeddingResult]:
        not_cached_ids = []
        not_cached_queries = []
        cached_ids = []
        embeddings = []

        if self.cache:
            for i in range(len(sentences)):
                if LRUCache.get(f"{self.alias}{sentences[i]}") is None:
                    not_cached_ids.append(i)
                    not_cached_queries.append(sentences[i])
                else:
                    cached_ids.append(i)
                embeddings.append(None)
        else:
            not_cached_queries = sentences

        no_cached_embeds: List[List[float]] = self.model.encode(
            not_cached_queries,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()

        if self.cache:
            logger.info(f"[EmbeddingEngine] cache hit: {len(cached_ids)}, miss: {len(not_cached_ids)}, cache size: {len(self.cache)}")
            for i in range(len(not_cached_ids)):
                self.cache[f"{self.alias}{sentences[not_cached_ids[i]]}"] = no_cached_embeds[i]
                embeddings[not_cached_ids[i]] = no_cached_embeds[i]

            for i in range(len(cached_ids)):
                embeddings[cached_ids[i]] = self.cache[f"{self.alias}{sentences[cached_ids[i]]}"]
        else:
            embeddings = no_cached_embeds

        return [EmbeddingResult(embedding=embedding, index=i, object="embedding") for i, embedding in enumerate(embeddings)]

    async def invoke(self, sentences: List[str]) -> List[EmbeddingResult]:
        return await asyncio.to_thread(self.invoke, sentences)

    async def stream(self, sentences: List[str]) -> List[EmbeddingResult]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `stream` method")
