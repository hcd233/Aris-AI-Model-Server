from typing import Dict, List, Tuple

from sentence_transformers import CrossEncoder

from src.config.arg import MODEL_CONFIG, RerankerConfig
from src.config.env import DEVICE

from .base import BaseEngine


class RerankerEngine(BaseEngine, RerankerConfig):
    model: CrossEncoder

    @classmethod
    def from_config(cls, config: RerankerConfig) -> "RerankerEngine":
        model = CrossEncoder(config.path, max_length=config.max_seq_len, device=DEVICE)

        return cls(model=model, **config.model_dump())

    def invoke(self, query: str, documents: List[str]) -> List[Tuple[float, float]]:
        scores = self.model.predict(
            [(query, doc) for doc in documents],
            batch_size=self.batch_size,
            show_progress_bar=True,
            activation_fct=lambda x: x,  # NOTE sentence_transformers CrossEncoder will use sigmoid to normalize the score. Disable it here.
            convert_to_tensor=True,
            convert_to_numpy=False,
        )
        scores = scores.to("cpu").numpy()
        scores, ranks = scores.tolist(), (-scores).argsort().argsort().tolist()

        return [(score, rank) for score, rank in zip(scores, ranks)]


RERANKER_MAPPING: Dict[str, RerankerEngine] = {config.alias: RerankerEngine.from_config(config) for config in MODEL_CONFIG.reranker_configs.values()}
