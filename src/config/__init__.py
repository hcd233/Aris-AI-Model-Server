from src.logger import logger

from .env import DEVICE, SECRET_KEY

if not SECRET_KEY:
    logger.error("[Check Env] `SECRET_KEY` is not set")
    exit(-1)

if not DEVICE:
    logger.warning("[Check Env] `DEVICE` is not set, try to use `cuda` as default")
    DEVICE = "cuda"
