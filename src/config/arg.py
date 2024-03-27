from argparse import ArgumentParser
from typing import Any, Dict

from src.logger import logger


def parse_args() -> Dict[str, Any]:
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="The model path to load", action="append")
    parser.add_argument("--seq_len", "-sl", type=int, help="The max sequence length", action="append")
    parser.add_argument("--use_cache", "-uc", type=bool, help="Use cache", default=False)

    args = vars(parser.parse_args())

    if args["model"] is None:
        logger.error("[Check Arg] Model path is required, specify with -m or --model")

    if args["seq_len"] is None:
        logger.error("[Check Arg] Sequence length is required, specify with -sl or --seq_len")

    return args


ARGUMENTS = parse_args()
