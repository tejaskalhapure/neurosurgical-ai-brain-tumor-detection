"""Health check response schema."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check status representation."""
    status: str = Field(default="healthy", description="API operational status")
    version: str = Field(..., description="API software version")
    device: str = Field(default="cpu", description="Execution hardware: cpu or cuda")
    model_loaded: bool = Field(default=False, description="Whether inference weights are loaded")
    disclaimer: str = Field(
        default="Academic decision-support prototype. Not for medical diagnosis.",
        description="Mandatory medical disclaimer"
    )
