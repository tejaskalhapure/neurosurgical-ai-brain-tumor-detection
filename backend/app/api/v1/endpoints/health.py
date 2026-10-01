"""Health and environment diagnostics endpoint."""

from fastapi import APIRouter, Depends
from backend.app.core.config import settings
from backend.app.schemas.health import HealthResponse
from backend.app.services.inference_service import InferenceService
from backend.app.api.deps import get_inference_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def check_system_health(
    inference_srv: InferenceService = Depends(get_inference_service)
):
    """Returns runtime health status, execution device (CPU), and disclaimer."""
    status_info = inference_srv.get_status()
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        device=status_info["device"],
        model_loaded=status_info["model_initialized"],
        disclaimer="Academic decision-support prototype. Not for medical diagnosis."
    )
