import os

SECRET_KEY = os.environ.get("SECRET_KEY")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
DEVICE = os.environ.get("DEVICE")

PORT = int(os.environ.get("PORT", "8080"))

LOGGER_LEVEL = os.environ.get("LOGGER_LEVEL", "INFO")
LOGGER_ROOT = os.environ.get("LOGGER_ROOT", "./log")
