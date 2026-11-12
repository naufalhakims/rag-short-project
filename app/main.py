"""FastAPI application: API routes plus the single-page web UI."""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import router as api_router
from app.config import PROJECT_ROOT, Settings, get_settings
from app.logging_config import logger, setup_logging
from app.services import Services, build_services, warm_up

STATIC_DIR = PROJECT_ROOT / "static"


class RevalidatedStaticFiles(StaticFiles):
    """Static files the browser must revalidate, so UI updates show up on a plain reload."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Build the app. Tests pass ready-made `services`; normally they are built at startup."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_path)
        svc = services or build_services(settings)
        app.state.services = svc
        if services is None:
            # Model loading can take a while (and downloads on first run); do it in the
            # background so the UI is reachable immediately.
            threading.Thread(target=warm_up, args=(svc,), daemon=True).start()
        else:
            svc.ready.set()
        status = await svc.llm.describe()
        if status["model"]:
            logger.info("LLM: %s / %s", status["provider"], status["model"])
        else:
            logger.warning("No LLM provider is configured — add an API key to .env.")
        logger.info("Pocket Librarian %s is up.", __version__)
        yield
        await svc.llm.aclose()

    app = FastAPI(
        title="Pocket Librarian",
        description="Ask questions about your own documents. Answers cite their sources.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse({"detail": "Internal server error."}, status_code=500)

    app.include_router(api_router)
    app.mount("/static", RevalidatedStaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
