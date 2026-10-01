"""FastAPI application entrypoint for NeuroSurgical AI decision support."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.core.logging import setup_logging
from backend.app.api.v1.router import api_router

# Initialize structured logging
setup_logging()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Academic Research Prototype: AI-assisted brain tumor detection, localization, and surgical planning decision support API.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure Cross-Origin Resource Sharing (CORS) for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register v1 API Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root():
    """Root info endpoint providing API metadata and clinical disclaimer."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "api_docs": "/docs",
        "device": settings.DEVICE,
        "disclaimer": "Academic decision-support prototype. Not for medical diagnosis."
    }
