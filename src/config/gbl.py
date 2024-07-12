from tiktoken import get_encoding

from src.controller import ModelController

from .arg import parse_args

TOKENIZER = get_encoding("cl100k_base")

ARGUMENTS = parse_args()

MODEL_CONTROLLER = ModelController.from_yaml(ARGUMENTS["config_path"])
