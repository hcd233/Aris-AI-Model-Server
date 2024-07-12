from os import PathLike
from typing import TYPE_CHECKING, Dict

import yaml
from pydantic import BaseModel

from src.config.model import EmbeddingConfig, LLMConfig, RerankerConfig, VLLMConfig
from src.logger import logger

if TYPE_CHECKING:
    from src.engine.embedding import EmbeddingEngine
    from src.engine.reranker import RerankerEngine
    from src.engine.vllm import VLLMEngine


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
            reranker_configs[alias] = RerankerConfig(alias=alias, **kwargs)

        return cls(
            llm_configs=llm_configs,
            embedding_configs=embedding_configs,
            reranker_configs=reranker_configs,
        )

    def get_reranker_engines(self) -> Dict[str, RerankerEngine]:
        configs = self.reranker_configs.values()
        if not configs:
            return {}
        try:
            from src.engine.reranker import RerankerEngine
        except ImportError:
            logger.error("[ModelController] RerankerEngine import failed")
            exit(1)
        return {config.alias: RerankerEngine.from_config(config) for config in configs}

    def get_embedding_engines(self) -> Dict[str, EmbeddingEngine]:
        configs = self.embedding_configs.values()
        if not configs:
            return {}
        try:
            from src.engine.embedding import EmbeddingEngine
        except ImportError:
            logger.error("[ModelController] EmbeddingEngine import failed")
            exit(1)
        return {config.alias: EmbeddingEngine.from_config(config) for config in configs}

    def get_llm_engines(self) -> Dict[str, VLLMEngine]:
        return self._get_vllm_engines()

    def _get_vllm_engines(self) -> Dict[str, VLLMEngine]:
        configs = [config for config in self.llm_configs.values() if isinstance(config, VLLMConfig)]
        try:
            from src.engine.vllm import VLLMEngine
        except ImportError:
            logger.error("[ModelController] VLLMEngine import failed")
            exit(1)
        return {config.alias: VLLMEngine.from_config(config) for config in configs}
