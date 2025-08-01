from os import PathLike
from typing import TYPE_CHECKING, Dict, Union

import yaml
from pydantic import BaseModel

from src.config.model import (EmbeddingConfig, InfinityRerankerConfig,
                              LLMConfig, MLXConfig, RerankerConfig,
                              SentenceTransformerRerankerConfig, VLLMConfig)
from src.logger import logger

if TYPE_CHECKING:
    from src.engine.embedding.sentence_transformer import \
        SentenceTransformerEmbeddingEngine
    from src.engine.llm.mlx import MLXEngine
    from src.engine.llm.vllm import VLLMEngine
    from src.engine.reranker.infinity import InfinityRerankerEngine
    from src.engine.reranker.sentence_transformer import \
        SentenceTransformerRerankerEngine


class ModelController(BaseModel):
    llm_configs: Dict[str, LLMConfig]
    embedding_configs: Dict[str, EmbeddingConfig]
    reranker_configs: Dict[str, RerankerConfig]

    @classmethod
    def from_yaml(cls, path: PathLike) -> "ModelController":
        with open(path, "r") as fp:
            config = yaml.safe_load(fp)

        llm_configs, embedding_configs, reranker_configs = {}, {}, {}

        for alias, kwargs in config.get("llm", {}).items():
            if alias in llm_configs:
                logger.error(f"Duplicate LLM alias: {alias}")
                exit(1)
            if kwargs.get("backend") == "vllm":
                llm_configs[alias] = VLLMConfig(alias=alias, **kwargs)
            elif kwargs.get("backend") == "mlx":
                llm_configs[alias] = MLXConfig(alias=alias, **kwargs)
            else:
                raise NotImplementedError(f"Unsupported LLM backend: {kwargs.get('backend')}")

        for alias, kwargs in config.get("embedding", {}).items():
            if alias in embedding_configs:
                logger.error(f"Duplicate embedding alias: {alias}")
                exit(1)
            embedding_configs[alias] = EmbeddingConfig(alias=alias, **kwargs)

        for alias, kwargs in config.get("reranker", {}).items():
            if alias in reranker_configs:
                logger.error(f"Duplicate reranker alias: {alias}")
                exit(1)
            if kwargs.get("backend") == "infinity":
                reranker_configs[alias] = InfinityRerankerConfig(alias=alias, **kwargs)
            elif kwargs.get("backend") == "sentence_transformer":
                reranker_configs[alias] = SentenceTransformerRerankerConfig(alias=alias, **kwargs)
            else:
                raise NotImplementedError(f"Unsupported reranker backend: {kwargs.get('backend')}")

        return cls(
            llm_configs=llm_configs,
            embedding_configs=embedding_configs,
            reranker_configs=reranker_configs,
        )

    def get_reranker_engines(self) -> Dict[str, "SentenceTransformerRerankerEngine"]:
        return {**self._get_sentence_transformer_reranker_engines(), **self._get_infinity_reranker_engines()}

    def get_embedding_engines(self) -> Dict[str, "SentenceTransformerEmbeddingEngine"]:
        configs = self.embedding_configs.values()
        if not configs:
            return {}
        try:
            from src.engine.embedding.sentence_transformer import \
                SentenceTransformerEmbeddingEngine
        except ImportError:
            logger.error("[ModelController] EmbeddingEngine import failed, run `pip install sentence-transformers` or `poetry install -E embedding`")
            exit(1)
        return {config.alias: SentenceTransformerEmbeddingEngine.from_config(config) for config in configs}

    def get_llm_engines(self) -> Dict[str, Union["VLLMEngine", "MLXEngine"]]:
        return {**self._get_vllm_engines(), **self._get_mlx_engines()}

    def _get_vllm_engines(self) -> Dict[str, "VLLMEngine"]:
        configs = [config for config in self.llm_configs.values() if isinstance(config, VLLMConfig)]
        if not configs:
            return {}
        try:
            from src.engine.llm.vllm import VLLMEngine
        except ImportError:
            logger.error("[ModelController] VLLMEngine import failed, run `pip install vllm==0.4.1` or `poetry install -E vllm`")
            exit(1)
        return {config.alias: VLLMEngine.from_config(config) for config in configs}

    def _get_mlx_engines(self) -> Dict[str, "MLXEngine"]:
        configs = [config for config in self.llm_configs.values() if isinstance(config, MLXConfig)]
        if not configs:
            return {}
        try:
            from src.engine.llm.mlx import MLXEngine
        except ImportError:
            logger.error("[ModelController] MLXEngine import failed, run `pip install mlx` or `poetry install -E mlx`")
            exit(1)
        return {config.alias: MLXEngine.from_config(config) for config in configs}

    def _get_sentence_transformer_reranker_engines(self) -> Dict[str, "SentenceTransformerRerankerEngine"]:
        configs = [config for config in self.reranker_configs.values() if isinstance(config, SentenceTransformerRerankerConfig)]
        if not configs:
            return {}
        try:
            from src.engine.reranker.sentence_transformer import \
                SentenceTransformerRerankerEngine
        except ImportError:
            logger.error("[ModelController] SentenceTransformerRerankerEngine import failed, run `pip install sentence-transformers` or `poetry install -E reranker`")
            exit(1)
        return {config.alias: SentenceTransformerRerankerEngine.from_config(config) for config in configs}

    def _get_infinity_reranker_engines(self) -> Dict[str, "InfinityRerankerEngine"]:
        configs = [config for config in self.reranker_configs.values() if isinstance(config, InfinityRerankerConfig)]
        if not configs:
            return {}
        try:
            from src.engine.reranker.infinity import InfinityRerankerEngine
        except ImportError:
            logger.error("[ModelController] InfinityRerankerEngine import failed, run `pip install infinity-emb` or `poetry install -E infinity`")
            exit(1)
        return {config.alias: InfinityRerankerEngine.from_config(config) for config in configs}
