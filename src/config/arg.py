from argparse import ArgumentParser
from os import PathLike
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel

from src.logger import logger


def parse_args() -> Dict[str, Any]:
    parser = ArgumentParser()
    parser.add_argument("--config_path", "-cp", type=str, help="Path to model config file, in yaml format", required=True)

    parser.add_argument("--use_cache", "-uc", type=bool, help="Use cache", default=False)

    args = vars(parser.parse_args())

    path = Path(args["config_path"])
    if not path.exists():
        logger.error(f"Config file {args['config_path']} not found")
        exit(1)

    if path.suffix not in [".yaml", ".yml"]:
        logger.error(f"Config file {args['config_path']} must be in yaml format")
        exit(1)

    return args


class LLMConfig(BaseModel):
    pass


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


ARGUMENTS = parse_args()

MODEL_CONFIG = ModelConfig.from_yaml(ARGUMENTS["config_path"])
