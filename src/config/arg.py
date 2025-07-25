from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel


def parse_args() -> Dict[str, Any]:
    parser = ArgumentParser()
    parser.add_argument("--config_path", "-cp", type=str, help="Path to model config file, in yaml format", required=True)
    args = vars(parser.parse_args())

    path = Path(args["config_path"])
    if not path.exists():
        raise FileNotFoundError(f"Config file {args['config_path']} not found")

    if path.suffix not in [".yaml", ".yml"]:
        raise ValueError(f"Config file {args['config_path']} must be in yaml format")

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
    pass


