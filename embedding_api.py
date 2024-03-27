import uvicorn
from fastapi import FastAPI

from src.config.arg import ARGUMENTS
from src.api.router import root_router, v1_router


def main():
    app = FastAPI(
        title="Fibona Embedding API",
        version="0.1.0",
    )

    app.include_router(root_router)
    app.include_router(v1_router)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=ARGUMENTS["port"],
    )


if __name__ == "__main__":
    main()
