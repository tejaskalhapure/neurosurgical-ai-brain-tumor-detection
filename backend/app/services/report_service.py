"""Clinical decision-support report service."""

from pathlib import Path
from typing import Dict, Any
from backend.app.core.config import settings
from backend.app.schemas.report import ReportSummaryResponse
from datetime import datetime


class ReportService:
    """Manages report generation metadata and PDF rendering triggers."""

    def __init__(self):
        self.reports_dir = Path(settings.REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def prepare_report_metadata(self, scan_id: str) -> ReportSummaryResponse:
        """Constructs clinical decision-support report metadata.

        Enforces prominent decision-support disclaimer.
        """
        report_id = f"report_{scan_id}"
        timestamp = datetime.utcnow().isoformat() + "Z"

        return ReportSummaryResponse(
            report_id=report_id,
            scan_id=scan_id,
            download_url=f"{settings.API_V1_STR}/reports/{scan_id}/download",
            generated_at=timestamp,
            disclaimer="Academic decision-support summary. Not for medical diagnosis or independent surgical intervention."
        )


report_service = ReportService()
