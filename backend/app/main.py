"""
VidSense AI — FastAPI application entrypoint.

Phase 1 scope: just enough to prove the server runs, config loads, and
storage directories exist. Video upload/transcription endpoints are
added in the next steps of Phase 1 (see docs/architecture.md).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.search import router as search_router
from app.api.video import router as video_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: make sure every storage subfolder exists before any
    # request tries to write into it (upload, audio extraction, etc.)
    settings.ensure_storage_dirs()
    logger.info("Storage directories ready at %s", settings.storage_dir)
    yield
    # Shutdown: nothing to clean up yet.


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

# Read-only demo mode. Registered before the CORS middleware so that CORS wraps it and the
# browser can read the 403 message. Search and chat are POSTs but only read, so they stay open.
READ_ONLY_OPEN_PREFIXES = ("/api/search", "/api/chat")


@app.middleware("http")
async def read_only_guard(request: Request, call_next):
    if (
        settings.read_only
        and request.method not in ("GET", "HEAD", "OPTIONS")
        and not request.url.path.startswith(READ_ONLY_OPEN_PREFIXES)
    ):
        return JSONResponse(
            status_code=403,
            content={"detail": "This deployment is read-only: uploads, processing and deletes are disabled."},
        )
    return await call_next(request)


# CORS: the frontend (Vite dev server, Phase 8) runs on a different
# origin/port than this API. Wide open for local development only --
# this is not a public-facing deployment, so there's no untrusted
# third-party origin to worry about restricting against yet.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router)
app.include_router(search_router)
app.include_router(chat_router)


@app.get("/health", tags=["system"])
def health_check() -> dict:
    """
    Liveness/readiness probe.

    Returns basic app metadata so we can confirm, from the response
    alone, that the correct config was loaded (e.g. which Whisper
    model size this deployment is configured for).
    """
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "whisper_model_size": settings.whisper_model_size,
        "whisper_device": settings.whisper_device,
        "read_only": settings.read_only,
    }
