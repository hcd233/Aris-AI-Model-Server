from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.logger import logger


class LoggerMiddleWare(BaseHTTPMiddleware):
    @logger.catch
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = datetime.now()
        response = await call_next(request)
        latency = (datetime.now() - start_time).total_seconds() * 1000
        latency = f"{round(latency, 3)}ms"

        log = f"{request.method} {response.status_code} {request.client.host} -> {request.url.path} {latency}"
        match response.status_code:
            case 200:
                logger_func = logger.success
            case 500:
                logger_func = logger.error
            case _:
                logger_func = logger.warning

        logger_func(log)
        return response
