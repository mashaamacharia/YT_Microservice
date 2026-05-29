"""
main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application entry point for the YouTube Pipeline LLM Service.

Responsibilities:
  - Creates and configures the FastAPI app instance
  - Registers all routers
  - Runs startup validation (checks .env, logs provider status)
  - Exposes /health and /status endpoints for n8n and Docker healthcheck
  - Configures structured logging

Start locally:
  uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

Via Docker:
  docker compose up -d --build
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers import research, script, analytics, keywords
from app.services.provider_router import get_available_providers


# ─── Logging setup ────────────────────────────────────────────────────────────
# Configure before anything else so all module loggers
# inherit this format from the moment they're imported.

def setup_logging(log_level: str = "info") -> None:
    """
    Configures structured logging for the entire service.
    Log level is read from .env LOG_LEVEL.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ─── Startup / shutdown lifecycle ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    Startup: validates config, logs provider status, warns about issues.
    Shutdown: logs clean exit.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("=" * 60)
    logger.info("  YouTube Pipeline — LLM Service starting up")
    logger.info("=" * 60)
    logger.info(f"  Environment : {settings.environment}")
    logger.info(f"  Port        : {settings.port}")
    logger.info(f"  Log level   : {settings.log_level}")
    logger.info(f"  LLM timeout : {settings.llm_timeout_seconds}s per provider")

    # ── Provider status ───────────────────────────────────────────
    logger.info("")
    logger.info("  Provider configuration:")
    provider_status = settings.get_configured_providers()
    for provider, status in provider_status.items():
        icon = "✅" if status == "configured" else "⬜"
        logger.info(f"    {icon} {provider:<12} {status}")

    # Check which providers are actually usable per task
    tasks = ["research", "script", "analytics", "keywords"]
    logger.info("")
    logger.info("  Available providers per task:")
    for task in tasks:
        available = get_available_providers(task)
        names = [p["name"] for p in available]
        if names:
            logger.info(f"    {task:<12} → {' → '.join(names)}")
        else:
            logger.warning(f"    {task:<12} → ⚠️  NO PROVIDERS AVAILABLE")

    # ── Config validation warnings ────────────────────────────────
    warnings = settings.validate()
    if warnings:
        logger.info("")
        logger.info("  Configuration warnings:")
        for warning in warnings:
            logger.warning(f"    {warning}")
    else:
        logger.info("")
        logger.info("  ✅ Configuration looks good")

    logger.info("")
    logger.info("  Endpoints registered:")
    logger.info("    POST /api/v1/research/aggregate")
    logger.info("    POST /api/v1/script/generate")
    logger.info("    POST /api/v1/analytics/interpret")
    logger.info("    POST /api/v1/keywords/broaden")
    logger.info("    GET  /health")
    logger.info("    GET  /status")
    logger.info("")
    logger.info("  Service ready ✅")
    logger.info("=" * 60)

    # Store startup time for /status endpoint
    app.state.started_at = time.time()

    yield  # Application runs here

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("YouTube Pipeline LLM Service shutting down")


# ─── App instance ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    Separated into a function for testability.
    """
    settings = get_settings()

    app = FastAPI(
        title="YouTube Pipeline — LLM Service",
        description=(
            "Centralized LLM microservice for the YouTube automation pipeline. "
            "Handles script generation, research aggregation, analytics "
            "interpretation, and keyword broadening. "
            "Supports Anthropic, OpenAI, and Gemini with automatic fallback."
        ),
        version="1.0.0",
        lifespan=lifespan,
        # Disable docs in production to avoid exposing prompt details
        docs_url="/docs" if settings.environment == "development" else None,
        redoc_url="/redoc" if settings.environment == "development" else None,
    )

    # ── CORS ──────────────────────────────────────────────────────
    # n8n runs on the same machine (WSL2), so localhost is fine.
    # Tighten this if you ever expose the service externally.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5678",   # n8n default port
            "http://127.0.0.1:5678",
            "http://localhost:8001",
            "http://127.0.0.1:8001",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────
    app.include_router(research.router)
    app.include_router(script.router)
    app.include_router(analytics.router)
    app.include_router(keywords.router)

    # ── Health endpoint ───────────────────────────────────────────
    @app.get(
        "/health",
        tags=["System"],
        summary="Health check",
        description=(
            "Used by Docker healthcheck and n8n to verify the service is running. "
            "Returns 200 if the service is up, regardless of provider status."
        )
    )
    async def health():
        return {"status": "ok"}

    # ── Status endpoint ───────────────────────────────────────────
    @app.get(
        "/status",
        tags=["System"],
        summary="Detailed service status",
        description=(
            "Returns full provider configuration status, available providers "
            "per task, uptime, and any configuration warnings. "
            "Useful for debugging from n8n or terminal."
        )
    )
    async def status():
        settings = get_settings()

        # Build per-task provider availability
        tasks = ["research", "script", "analytics", "keywords"]
        providers_per_task = {}
        for task in tasks:
            available = get_available_providers(task)
            providers_per_task[task] = {
                "available": [p["name"] for p in available],
                "count": len(available),
                "models": {
                    p["name"]: p["model"] for p in available
                }
            }

        # Uptime
        uptime_seconds = None
        if hasattr(app.state, "started_at"):
            uptime_seconds = round(time.time() - app.state.started_at)

        # Config warnings
        warnings = settings.validate()

        return {
            "status": "ok",
            "environment": settings.environment,
            "uptime_seconds": uptime_seconds,
            "provider_order": settings.provider_order,
            "providers": settings.get_configured_providers(),
            "providers_per_task": providers_per_task,
            "token_limits": {
                "research":  settings.research_max_tokens,
                "script":    settings.script_max_tokens,
                "analytics": settings.analytics_max_tokens,
                "keywords":  settings.keywords_max_tokens,
            },
            "llm_timeout_seconds": settings.llm_timeout_seconds,
            "warnings": warnings if warnings else []
        }

    # ── Global exception handler ──────────────────────────────────
    # Catches any unhandled exception and returns a clean JSON error
    # instead of a 500 HTML page that n8n cannot parse.
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}: "
            f"{type(exc).__name__}: {str(exc)}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Internal server error: {type(exc).__name__}: {str(exc)}",
                "path": str(request.url.path)
            }
        )

    return app


# ─── App singleton ────────────────────────────────────────────────────────────
# This is what uvicorn imports: `uvicorn app.main:app`

app = create_app()