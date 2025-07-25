from typing import List

from infinity_emb import AsyncEmbeddingEngine, EngineArgs
from infinity_emb.primitives import RerankReturnType

from src.config.model import InfinityRerankerConfig
from src.logger import logger

from ..base import BaseEngine, RerankerResult


class InfinityRerankerEngine(BaseEngine, InfinityRerankerConfig):
    """使用 infinity_emb 包实现的重排序引擎"""

    engine: AsyncEmbeddingEngine

    @classmethod
    def from_config(cls, config: InfinityRerankerConfig) -> "InfinityRerankerEngine":
        engine_args = EngineArgs(
            model_name_or_path=config.path,
            engine="torch",
            batch_size=config.batch_size,
            dtype=config.dtype,
            compile=True,
            bettertransformer=True,
        )

        # 创建 AsyncEmbeddingEngine
        engine = AsyncEmbeddingEngine.from_args(engine_args)

        logger.success(f"[InfinityRerankerEngine] load model from {config.path}")
        return cls(engine=engine, **config.model_dump())

    async def invoke(self, query: str, documents: List[str]) -> List[RerankerResult]:
        async with self.engine:
            results : List[RerankReturnType] = []
            results, _ = await self.engine.rerank(query=query, docs=documents)

            return [RerankerResult(
                    index=res.index,
                    relevent_score=res.relevance_score
            ) for res in results]

    async def stream(self, query: str, documents: List[str]) -> List[RerankerResult]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `stream` method")
