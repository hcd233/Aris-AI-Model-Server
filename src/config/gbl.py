from typing import Dict

from tiktoken import get_encoding

from src.engine.embedding import EmbeddingEngine
from src.engine.reranker import RerankerEngine

from .arg import parse_args
from .model import ModelConfig

INFO_LOG = "model-api-server-info.log"
SUCCESS_LOG = "model-api-server-success.log"
ERROR_LOG = "model-api-server-error.log"

LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>[{function}]</cyan>: <level>{message}</level>"

TOKENIZER = get_encoding("cl100k_base")

ARGUMENTS = parse_args()

MODEL_CONFIG = ModelConfig.from_yaml(ARGUMENTS["config_path"])

RERANKER_MAPPING: Dict[str, RerankerEngine] = {config.alias: RerankerEngine.from_config(config) for config in MODEL_CONFIG.reranker_configs.values()}

EMBEDDING_MAPPING: Dict[str, EmbeddingEngine] = {
    config.alias: EmbeddingEngine.from_config(config) for config in MODEL_CONFIG.embedding_configs.values()
}
