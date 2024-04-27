from typing import List, Tuple

from sentence_transformers import CrossEncoder

from src.config.arg import RerankerConfig
from src.config.env import DEVICE
from src.logger import logger

from .base import BaseEngine


class RerankerEngine(BaseEngine, RerankerConfig):
    model: CrossEncoder

    @classmethod
    def from_config(cls, config: RerankerConfig) -> "RerankerEngine":
        model = CrossEncoder(config.path, max_length=config.max_seq_len, device=DEVICE)

        logger.success(f"[RerankerEngine] load model from {config.path}")
        return cls(model=model, **config.model_dump())

    def invoke(self, query: str, documents: List[str]) -> List[Tuple[float, float]]:
        scores = self.model.predict(
            [(query, doc) for doc in documents],
            batch_size=self.batch_size,
            show_progress_bar=True,
            activation_fct=None,  # NOTE sentence_transformers CrossEncoder will use sigmoid to normalize the score
            convert_to_tensor=True,
            convert_to_numpy=False,
        )
        scores = scores.to("cpu").numpy()
        scores, ranks = scores.tolist(), (-scores).argsort().argsort().tolist()

        return [(score, rank) for score, rank in zip(scores, ranks)]
