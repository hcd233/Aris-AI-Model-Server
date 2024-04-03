from argparse import ArgumentParser
from typing import Any, Dict

from src.logger import logger


def parse_args() -> Dict[str, Any]:
    parser = ArgumentParser()
    parser.add_argument("--embed", "-e", type=str, help="The embedding path to load", action="append")
    parser.add_argument("--embed_seq_len", "-esl", type=int, help="The embedding max sequence length", action="append")
    parser.add_argument("--rerank", "-r", type=str, help="The reranker path to load", action="append")
    parser.add_argument("--rerank_seq_len", "-rsl", type=int, help="The reranker max sequence length", action="append")

    parser.add_argument("--use_cache", "-uc", type=bool, help="Use cache", default=False)

    args = vars(parser.parse_args())

    if not args["embed"]:
        logger.warning("[Check Arg] Embedding path is required, specify with -e or --embed")

    if not args["embed_seq_len"]:
        logger.warning("[Check Arg] Embedding Sequence length is required, specify with -esl or --embed_seq_len")

    if not args["rerank"]:
        logger.warning("[Check Arg] Reranker path is required, specify with -r or --rerank")

    if not args["rerank_seq_len"]:
        logger.warning("[Check Arg] Reranker Sequence length is required, specify with -rsl or --rerank_seq_len")

    return args


ARGUMENTS = parse_args()
