"""FastAPI Application Entry Point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.query_routes import router as query_router
from src.api.rerank_routes import router as rerank_router
from src.api.retrieval_routes import router as retrieval_router
from src.api.routes import router as ingestion_router
from src.ingestion.registry import default_registry


def create_app() -> FastAPI:
    """Factory creating configured FastAPI application."""
    app = FastAPI(
        title="LLM Hallucination & Context Degradation Mitigation API",
        description="Backend API supporting multi-format document ingestion, hybrid retrieval, and context optimization.",
        version="0.1.0",
    )

    # Enable CORS for Next.js frontend communication
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routes
    app.include_router(ingestion_router)
    app.include_router(query_router)
    app.include_router(retrieval_router)
    app.include_router(rerank_router)

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
