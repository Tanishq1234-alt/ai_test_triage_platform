"""FastAPI application factory, middleware configuration, and router aggregation."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import analysis, health, reports, tests
from src.storage.database import Database, DatabaseError

logger = logging.getLogger("fastapi_app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup initialization and graceful shutdown."""
    logger.info("Initializing SQLite database tables and indexes...")
    try:
        db = Database()
        db.initialize_schema()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database during startup: {e}")
        raise
    yield
    logger.info("FastAPI application shutting down.")


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    application = FastAPI(
        title="AI-Powered Test Automation & Failure Triage Platform",
        description=(
            "REST API providing test telemetry parsing, ML-driven root cause failure classification, "
            "human-in-the-loop review feedback, and historical reporting."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Enable CORS for local Streamlit dashboard and browser clients
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global DatabaseError handler
    @application.exception_handler(DatabaseError)
    async def database_error_handler(request: Request, exc: DatabaseError) -> JSONResponse:
        logger.error(f"Database error on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Database operation failed. Please check server logs."},
        )

    # Register modular routers
    application.include_router(health.router)
    application.include_router(tests.router)
    application.include_router(analysis.router)
    application.include_router(reports.router)

    @application.get("/", tags=["Root"])
    def root() -> dict:
        return {
            "name": "AI-Powered Test Automation & Failure Triage Platform",
            "version": "1.0.0",
            "documentation": "/docs",
            "health_check": "/health",
        }

    return application


app = create_app()
