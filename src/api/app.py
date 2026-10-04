"""FastAPI Application Entry Point."""

import logging
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.context_routes import router as context_router
from src.api.generation_routes import router as generation_router
from src.api.memory_routes import router as memory_router
from src.api.query_routes import router as query_router
from src.api.rerank_routes import router as rerank_router
from src.api.retrieval_routes import router as retrieval_router
from src.api.routes import router as ingestion_router
from src.ingestion.registry import default_registry

load_dotenv()
logger = logging.getLogger("uvicorn.error")


def create_app() -> FastAPI:
    """Factory creating configured FastAPI application."""
    app = FastAPI(
        title="LLM Hallucination & Context Degradation Mitigation API",
        description="Backend API supporting multi-format document ingestion, hybrid retrieval, and context optimization.",
        version="0.1.0",
    )

    # Configure CORS origins safely
    origins_env = os.getenv("ALLOWED_ORIGINS", "")
    allowed_origins = [
        origin.strip()
        for origin in origins_env.split(",")
        if origin.strip()
    ] or [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    origin_regex = re.compile(r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$")

    def get_cors_headers(request: Request) -> dict[str, str]:
        origin = request.headers.get("origin")
        if origin and (origin in allowed_origins or origin_regex.match(origin)):
            return {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        return {}

    # Starlette ServerErrorMiddleware strips CORS headers on unhandled exceptions.
    # We register explicit exception handlers that always attach CORS headers.
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        headers = get_cors_headers(request)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        headers = get_cors_headers(request)
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")
        headers = get_cors_headers(request)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {str(exc)}"},
            headers=headers,
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routes
    app.include_router(ingestion_router)
    app.include_router(query_router)
    app.include_router(retrieval_router)
    app.include_router(rerank_router)
    app.include_router(context_router)
    app.include_router(generation_router)
    app.include_router(memory_router)

    @app.get("/api/health", tags=["Health"])
    def health_check():
        return {
            "status": "healthy",
            "module": "Ingestion",
            "supported_extensions": default_registry.get_supported_extensions(),
            "version": "0.1.0",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)
