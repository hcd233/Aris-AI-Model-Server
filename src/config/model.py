from typing import Literal

from pydantic import BaseModel


class BaseConfig(BaseModel):
    alias: str
    path: str


class LLMConfig(BaseConfig):
    template: str
    max_seq_len: int
    backend: str


class VLLMConfig(LLMConfig):
    backend: Literal["vllm"] = "vllm"
    dtype: Literal["auto", "float16", "bfloat16"] = "auto"
    tensor_parallel_size: int
    gpu_memory_utilization: float


class MLXConfig(LLMConfig):
    backend: Literal["mlx"] = "mlx"


class EmbeddingConfig(BaseConfig):
    enable_cache: bool
    batch_size: int
    max_seq_len: int


class RerankerConfig(BaseConfig):
    batch_size: int
    max_seq_len: int
