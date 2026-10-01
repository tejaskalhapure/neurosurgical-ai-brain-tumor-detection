"""API Router aggregation for v1 endpoints."""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, mri, inference, reports

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(mri.router, prefix="/mri", tags=["MRI Ingestion"])
api_router.include_router(inference.router, prefix="/inference", tags=["Inference & Localization"])
api_router.include_router(reports.router, prefix="/reports", tags=["Surgical Reports"])
