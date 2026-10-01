"""Report generation schemas."""

from pydantic import BaseModel, Field
from typing import Optional


class ReportSummaryResponse(BaseModel):
    """Structured report metadata response."""
    report_id: str
    scan_id: str
    download_url: str
    generated_at: str
    disclaimer: str = Field(
        default="Academic decision-support summary. Not for medical diagnosis."
    )
