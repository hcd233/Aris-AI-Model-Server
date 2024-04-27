from typing import List

from cachetools import LRUCache
from sentence_transformers import SentenceTransformer

from src.config.arg import EmbeddingConfig
from src.config.env import DEVICE
from src.logger import logger

from .base import BaseEngine


class EmbeddingEngine(BaseEngine, EmbeddingConfig):
    model: SentenceTransformer
    cache: LRUCache | None

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "EmbeddingEngine":
        model = SentenceTransformer(config.path, device=DEVICE)
        model.max_seq_length = config.max_seq_len
        cache = LRUCache(maxsize=1000) if config.enable_cache else None

        logger.success(f"[EmbeddingEngine] load model from {config.path}")
        return cls(model=model, cache=cache, **config.model_dump())

    def invoke(self, sentences: List[str]) -> List[List[int]]:
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

        no_cached_embeds = self.model.encode(
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

        return embeddings
