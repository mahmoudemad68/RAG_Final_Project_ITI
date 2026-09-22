import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import Settings, get_settings
from app.services.generation import GenerationService
from app.services.retrieval import ArtifactConfigurationError, RetrievalService
from app.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    retrieval_service: Any | None = None,
    generation_service: Any | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(app_settings.log_level)
        app.state.settings = app_settings
        app.state.startup_error = None

        retrieval = retrieval_service
        if retrieval is None:
            try:
                retrieval = RetrievalService.load(app_settings)
            except ArtifactConfigurationError as exc:
                app.state.startup_error = str(exc)
                logger.warning("RAG artifacts are not ready: %s", exc)
            except Exception as exc:
                app.state.startup_error = (
                    f"Retrieval startup failed: {type(exc).__name__}"
                )
                logger.exception("Retrieval startup failed")
        app.state.retrieval_service = retrieval

        generation = generation_service
        if generation is None:
            try:
                generation = GenerationService(app_settings)
            except Exception as exc:
                app.state.startup_error = (
                    app.state.startup_error
                    or f"Generation startup failed: {type(exc).__name__}"
                )
                logger.exception("Generation startup failed")
        app.state.generation_service = generation

        if generation is not None:
            ready_check = getattr(generation, "check_ready", None)
            app.state.ollama_ready = (
                bool(ready_check()) if callable(ready_check) else True
            )
        else:
            app.state.ollama_ready = False

        if not app.state.ollama_ready and app.state.startup_error is None:
            app.state.startup_error = (
                f"Ollama model {app_settings.ollama_model!r} is not reachable"
            )

        yield

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.frontend_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.include_router(query_router)
    return app


app = create_app()
