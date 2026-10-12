from fastapi import FastAPI

from quipquill_cloud import __version__
from quipquill_cloud.api import health


def create_app() -> FastAPI:
    app = FastAPI(title="QuipQuill Cloud", version=__version__)
    app.include_router(health.router)
    return app


app = create_app()
