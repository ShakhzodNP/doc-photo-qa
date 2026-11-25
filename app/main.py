import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router as main_router
from app.document import warm_up_models
from app.logger import logger
from app.middlewares import handle_exceptions_middleware
from settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.perf_counter()
    warm_up_models()
    logger.info("Models loaded and warmed up in %.0f ms", (time.perf_counter() - start) * 1000)
    yield


app = FastAPI(
    debug=settings.DEV_MODE,
    title='Doc-photo-qa',
    description='docs photos validation service',
    lifespan=lifespan,
)

app.middleware("http")(handle_exceptions_middleware)

app.include_router(
    main_router
)
