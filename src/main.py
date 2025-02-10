"""FastAPI application factory and entrypoint."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.models import ErrorResponse
from src.api.routes import router
from src.config import get_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared resources on startup and tear them down on shutdown."""
    cfg = get_config()

    # In-memory job store (swap for Redis-backed store in production)
    app.state.jobs: dict = {}  # type: ignore[annotation-unchecked]

    # Redis connection
    app.state.redis = aioredis.from_url(
        cfg.redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
    )

    # Create LLM provider from config
    try:
        from src.llm.factory import get_llm_provider

        llm_provider = get_llm_provider(cfg)
        app.state.llm_provider = llm_provider
        logger.info("LLM provider created: %s", cfg.llm_provider)
    except Exception:
        logger.warning("LLM provider not available — using fallback in nodes", exc_info=True)
        llm_provider = None
        app.state.llm_provider = None

    # Compile LangGraph agent (lazy import to avoid circular deps)
    try:
        from src.agents.graph import build_document_graph  # type: ignore[import-untyped]

        app.state.graph = build_document_graph(llm_provider=llm_provider)
        logger.info("LangGraph agent compiled successfully")
    except Exception:
        logger.warning("LangGraph agent not available — graph endpoints will fail", exc_info=True)
        app.state.graph = None

    logger.info("Application started (redis=%s)", cfg.redis_url)
    yield

    # Shutdown
    await app.state.redis.aclose()
    logger.info("Application shut down")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="Document Processing Agent",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Request-ID middleware ---
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # --- Global error handler ---
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail="Internal server error").model_dump(),
        )

    # --- Router ---
    app.include_router(router)

    return app


app = create_app()

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(
        "src.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=True,
    )
