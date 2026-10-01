"""Dependency injection providers for FastAPI endpoints."""

from backend.app.core.config import settings
from backend.app.services.mri_service import mri_service, MRIService
from backend.app.services.inference_service import inference_service, InferenceService
from backend.app.services.report_service import report_service, ReportService


def get_mri_service() -> MRIService:
    return mri_service


def get_inference_service() -> InferenceService:
    return inference_service


def get_report_service() -> ReportService:
    return report_service
