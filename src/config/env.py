import os

SECRET_KEY = os.environ.get("SECRET_KEY")
DEVICE = os.environ.get("DEVICE")

PORT = int(os.environ.get("PORT", "8080"))

LOGGER_PREFIX = os.environ.get("LOGGER_PREFIX")
LOGGER_LEVEL = os.environ.get("LOGGER_LEVEL", "INFO")
LOGGER_ROOT = os.environ.get("LOGGER_ROOT", "./log")
