"""Inference request and quantitative metric response schemas."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class BoundingBox3D(BaseModel):
    """3D bounding box coordinates enclosing detected tumor."""
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int


class VolumetricMetrics(BaseModel):
    """Surgical planning volumetric measurements."""
    estimated_volume_mm3: float = Field(..., description="Estimated tumor volume in cubic millimeters")
    estimated_volume_cm3: float = Field(..., description="Estimated tumor volume in cubic centimeters")
    voxel_count: int = Field(..., description="Total positive voxels segmented")
    max_slice_area_mm2: Optional[float] = Field(None, description="Maximum cross-sectional area on any single axial slice")
    max_slice_index: Optional[int] = Field(None, description="Index of slice with maximum cross-sectional tumor area")
    valid_voxel_metadata: bool = Field(..., description="Whether physical voxel dimensions were available for calculation")


class InferenceResponse(BaseModel):
    """Comprehensive tumor detection, localization, and segmentation output."""
    scan_id: str
    tumor_detected: bool = Field(..., description="Binary detection outcome")
    detection_confidence: float = Field(..., description="Confidence probability (0.0 to 1.0)")
    bounding_box_3d: Optional[BoundingBox3D] = Field(None, description="3D localization bounding box")
    volumetrics: Optional[VolumetricMetrics] = Field(None, description="Volumetric and area estimations")
    slices_with_tumor: List[int] = Field(default_factory=list, description="List of slice indices containing tumor mass")
    status: str = Field(default="completed")
    disclaimer: str = Field(
        default="Decision support output only. Requires radiological confirmation."
    )
