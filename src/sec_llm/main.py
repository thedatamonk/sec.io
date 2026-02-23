"""FastAPI application factory and lifespan."""
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sec_llm.api.router import api_router
from sec_llm.config import Settings

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting SEC-LLM application")
    yield
    # Shutdown
    logger.info("Shutting down SEC-LLM application")


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="SEC-LLM",
        version="0.1.0",
        description="AI-powered SEC financial analyst",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include all API routes defined in /api/router.py
    app.include_router(api_router)

    return app


app = create_app()
