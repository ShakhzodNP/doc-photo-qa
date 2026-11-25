from fastapi import FastAPI

from app.api import router as main_router
from app.middlewares import handle_exceptions_middleware
from settings import settings


app = FastAPI(
    debug=settings.DEV_MODE,
    title='Doc-photo-qa',
    description='docs photos validation service',
)

app.middleware("http")(handle_exceptions_middleware)

app.include_router(
    main_router
)