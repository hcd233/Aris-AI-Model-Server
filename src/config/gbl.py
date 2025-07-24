from tiktoken import get_encoding

from src.controller import ModelController

from .arg import parse_args

TOKENIZER = get_encoding("cl100k_base")

ARGUMENTS = parse_args()

MODEL_CONTROLLER = ModelController.from_yaml(ARGUMENTS["config_path"])

LLM_ENGINE_MAPPING = MODEL_CONTROLLER.get_llm_engines()

EMBEDDING_ENGINE_MAPPING = MODEL_CONTROLLER.get_embedding_engines()

RERANKER_ENGINE_MAPPING = MODEL_CONTROLLER.get_reranker_engines()
