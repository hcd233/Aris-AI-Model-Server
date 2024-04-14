from pathlib import Path
from sys import stdout

from loguru import logger as _logger
from loguru._logger import Logger

from src.config.env import LOGGER_LEVEL, LOGGER_ROOT
from src.config.gbl import ERROR_LOG, INFO_LOG, LOG_FORMAT, SUCCESS_LOG


def init_logger() -> Logger:
    log_root = Path(LOGGER_ROOT)

    if not log_root.exists():
        log_root.mkdir(parents=True)
    if not log_root.is_dir():
        raise ValueError("LOG_ROOT is not a directory")

    _logger.remove()  # remove origin handler
    _logger.add(stdout, colorize=True, enqueue=True, level=LOGGER_LEVEL, format=LOG_FORMAT)
    _logger.add(log_root.joinpath(INFO_LOG), encoding="utf-8", rotation="100MB", enqueue=True, level="INFO", format=LOG_FORMAT)
    _logger.add(log_root.joinpath(SUCCESS_LOG), encoding="utf-8", rotation="100MB", enqueue=True, level="SUCCESS", format=LOG_FORMAT)
    _logger.add(log_root.joinpath(ERROR_LOG), encoding="utf-8", rotation="100MB", enqueue=True, level="ERROR", format=LOG_FORMAT)

    _logger.info("Init logger successfully")

    return _logger


logger = init_logger()
