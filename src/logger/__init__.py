from pathlib import Path
from sys import stdout

from loguru import logger as _logger
from loguru._logger import Logger

from src.config.env import LOGGER_LEVEL, LOGGER_PREFIX, LOGGER_ROOT


def init_logger() -> Logger:
    log_root = Path(LOGGER_ROOT)

    info = f"{LOGGER_PREFIX}-info.log"
    success = f"{LOGGER_PREFIX}-success.log"
    error = f"{LOGGER_PREFIX}-error.log"

    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>[{function}]</cyan>: <level>{message}</level>"

    if not log_root.exists():
        log_root.mkdir(parents=True)
    if not log_root.is_dir():
        raise ValueError("LOG_ROOT is not a directory")

    _logger.remove()  # remove origin handler
    _logger.add(stdout, colorize=True, enqueue=True, level=LOGGER_LEVEL, format=log_format)
    _logger.add(log_root.joinpath(info), encoding="utf-8", rotation="100MB", enqueue=True, level="INFO", format=log_format)
    _logger.add(log_root.joinpath(success), encoding="utf-8", rotation="100MB", enqueue=True, level="SUCCESS", format=log_format)
    _logger.add(log_root.joinpath(error), encoding="utf-8", rotation="100MB", enqueue=True, level="ERROR", format=log_format)

    _logger.info("Init logger successfully")

    return _logger


logger = init_logger()
