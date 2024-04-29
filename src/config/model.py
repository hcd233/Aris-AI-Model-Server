from os import PathLike
from typing import Dict, Literal

import yaml
from pydantic import BaseModel

from src.logger import logger


class LLMConfig(BaseModel):
    alias: str
    path: str
    template: str
    max_seq_len: int


class VLLMConfig(LLMConfig):
    backend: Literal["vllm"] = "vllm"
    dtype: Literal["auto", "float16", "bfloat16"] = "auto"
    tensor_parallel_size: int
    gpu_memory_utilization: float


class EmbeddingConfig(BaseModel):
    alias: str
    path: str
    enable_cache: bool
    batch_size: int
    max_seq_len: int


class RerankerConfig(BaseModel):
    alias: str
    path: str
    batch_size: int
    max_seq_len: int


class ModelConfig(BaseModel):
    llm_configs: Dict[str, LLMConfig]
    embedding_configs: Dict[str, EmbeddingConfig]
    reranker_configs: Dict[str, RerankerConfig]

    @classmethod
    def from_yaml(cls, path: PathLike) -> "ModelConfig":
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
