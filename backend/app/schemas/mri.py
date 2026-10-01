"""MRI upload and metadata schemas."""

from typing import List, Tuple, Optional
from pydantic import BaseModel, Field


class MRIUploadResponse(BaseModel):
    """Metadata response following successful MRI file upload."""
    scan_id: str = Field(..., description="Unique generated scan identifier")
    filename: str = Field(..., description="Original filename uploaded")
    file_size_bytes: int = Field(..., description="File size on disk")
    status: str = Field(default="uploaded", description="Current pipeline processing status")
    format_detected: str = Field(..., description="File format extension detected")


class MRIMetadataResponse(BaseModel):
    """Detailed volumetric header metadata extracted from NIfTI file."""
    scan_id: str
    dimensions: Tuple[int, int, int] = Field(..., description="3D matrix dimensions (Nx, Ny, Nz)")
    voxel_spacing_mm: Tuple[float, float, float] = Field(..., description="Physical voxel spacing (dx, dy, dz) in mm")
    total_slices: int = Field(..., description="Number of axial slices")
    valid_voxel_metadata: bool = Field(..., description="Indicates whether physical spacing is valid for volumetric estimation")
