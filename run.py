import uvicorn

from settings import settings


def run():
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload_excludes="logs/",
        reload=settings.DEV_MODE or None,
        use_colors=True,
    )


if __name__ == "__main__":
    run()
