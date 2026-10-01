"""Inference execution and model status endpoints."""

from fastapi import APIRouter, Depends, status
from typing import Dict, Any
from backend.app.services.inference_service import InferenceService
from backend.app.api.deps import get_inference_service

router = APIRouter()


@router.get("/status")
async def get_inference_status(
    inference_srv: InferenceService = Depends(get_inference_service)
) -> Dict[str, Any]:
    """Returns inference engine initialization status, hardware target, and weights availability."""
    return inference_srv.get_status()


@router.post("/predict/{scan_id}")
async def trigger_inference(
    scan_id: str,
    inference_srv: InferenceService = Depends(get_inference_service)
) -> Dict[str, Any]:
    """Triggers inference on a staged scan.

    Note: In Module 1, validates decoupled execution without fabricating clinical outputs.
    """
    return inference_srv.dispatch_inference(scan_id)
