import uvicorn
from fastapi import FastAPI

from src.api.router import root_router, v1_router
from src.config.env import PORT
from src.middleware.logger import LoggerMiddleWare


def main():
    app = FastAPI(
        title="Fibona Model API Server",
        version="0.1.0",
    )

    app.include_router(root_router)
    app.include_router(v1_router)

    app.add_middleware(LoggerMiddleWare)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="critical",  # ban uvicorn logger
    )


if __name__ == "__main__":
    main()
