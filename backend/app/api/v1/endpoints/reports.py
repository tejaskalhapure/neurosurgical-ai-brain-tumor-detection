"""Surgical decision-support reporting endpoints."""

from fastapi import APIRouter, Depends
from backend.app.schemas.report import ReportSummaryResponse
from backend.app.services.report_service import ReportService
from backend.app.api.deps import get_report_service

router = APIRouter()


@router.get("/{scan_id}/summary", response_model=ReportSummaryResponse)
async def get_report_summary(
    scan_id: str,
    report_srv: ReportService = Depends(get_report_service)
):
    """Retrieves report generation metadata and download readiness."""
    return report_srv.prepare_report_metadata(scan_id)
