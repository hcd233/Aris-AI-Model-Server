import asyncio
from typing import List

import torch
from sentence_transformers import CrossEncoder

from src.config.env import DEVICE
from src.config.model import SentenceTransformerRerankerConfig
from src.logger import logger

from ..base import BaseEngine, RerankerResult


class SentenceTransformerRerankerEngine(BaseEngine, SentenceTransformerRerankerConfig):
    model: CrossEncoder

    @classmethod
    def from_config(cls, config: SentenceTransformerRerankerConfig) -> "SentenceTransformerRerankerEngine":
        model = CrossEncoder(config.path, max_length=config.max_seq_len, device=DEVICE)

        logger.success(f"[SentenceTransformerRerankerEngine] load model from {config.path}")
        return cls(model=model, **config.model_dump())

    def _invoke(self, query: str, documents: List[str]) -> List[RerankerResult]:
        with torch.inference_mode():  # 使
            scores = self.model.predict(
                [(query, doc) for doc in documents],
                batch_size=self.batch_size,
                show_progress_bar=True,
                activation_fct=None,  # NOTE sentence_transformers CrossEncoder will use sigmoid to normalize the score
                convert_to_tensor=True,
                convert_to_numpy=False,
            )
            scores = scores.to("cpu").numpy()
            scores, indexes = scores.tolist(), (-scores).argsort().argsort().tolist()

        return [RerankerResult(index=index, relevent_score=score) for score, index in zip(scores, indexes)]

    async def stream(self, query: str, documents: List[str]) -> List[RerankerResult]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `stream` method")

    async def invoke(self, query: str, documents: List[str]) -> List[RerankerResult]:
        return await asyncio.to_thread(self._invoke, query, documents)
