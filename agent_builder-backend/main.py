"""
Agent Builder API - Main Application Entry Point

A FastAPI application for building AI agents through a visual drag-and-drop interface.
Supports multiple LLM providers, Composio tools, and custom workflows.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging

from config import settings
from database import MongoDB
from auth.routes import router as auth_router
from flows.routes import router as flows_router
from nodes.routes import router as nodes_router
from edges.routes import router as edges_router


# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Agent Builder API...")
    
    # Connect to MongoDB
    mongodb = MongoDB()
    await mongodb.connect()
    logger.info("Connected to MongoDB")
    
    yield
    
    logger.info("Shutting down Agent Builder API...")
    await mongodb.disconnect()
    logger.info("Disconnected from MongoDB")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Agent Builder API
    
    Build AI agents through a visual drag-and-drop interface.
    
    ### Features
    - **Visual Flow Builder**: Create agent workflows by connecting components
    - **Multiple LLM Providers**: Support for OpenAI, Anthropic, OpenRouter, and more
    - **Composio Integration**: Connect to 500+ external services
    - **Real-time Execution**: Execute agents with streaming responses
    - **Centralized Field Types**: Consistent input handling across all components
    
    ### New Field Types System
    The API now includes a centralized field types system that provides:
    - 25+ predefined input types (string, number, select, API key, etc.)
    - Consistent validation and schema generation
    - Frontend-friendly JSON schemas for dynamic form rendering
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": errors
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc) if settings.DEBUG else "Internal server error"
        }
    )


# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(flows_router, prefix="/api/v1")
app.include_router(nodes_router, prefix="/api/v1")
app.include_router(edges_router, prefix="/api/v1")


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Check API health status."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "app": settings.APP_NAME
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root - returns basic information."""
    return {
        "message": "Welcome to Agent Builder API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# API info endpoint
@app.get("/api/v1", tags=["API Info"])
async def api_info():
    """Get API information and available endpoints."""
    return {
        "version": "v1",
        "endpoints": {
            "auth": {
                "register": "POST /api/v1/auth/register",
                "login": "POST /api/v1/auth/login",
                "me": "GET /api/v1/auth/me"
            },
            "flows": {
                "list": "GET /api/v1/flows",
                "create": "POST /api/v1/flows",
                "get": "GET /api/v1/flows/{flow_id}",
                "update": "PUT /api/v1/flows/{flow_id}",
                "delete": "DELETE /api/v1/flows/{flow_id}",
                "execute": "POST /api/v1/flows/{flow_id}/execute",
                "stream": "POST /api/v1/flows/{flow_id}/execute/stream",
                "schema": "GET /api/v1/flows/{flow_id}/schema"
            },
            "components": {
                "list": "GET /api/v1/components",
                "categories": "GET /api/v1/components/categories",
                "get": "GET /api/v1/components/{component_type}",
                "schema": "GET /api/v1/components/{component_type}/schema",
                "validate": "POST /api/v1/components/{component_type}/validate",
                "field_types": "GET /api/v1/components/field-types",
                "validate_fields": "POST /api/v1/components/validate-fields"
            },
            "edges": {
                "validate": "POST /api/v1/edges/validate",
                "compatible_ports": "GET /api/v1/edges/compatible-ports/{component_type}"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )
