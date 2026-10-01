"""MRI upload and metadata retrieval endpoints."""

from fastapi import APIRouter, UploadFile, File, Depends, status
from backend.app.schemas.mri import MRIUploadResponse
from backend.app.services.mri_service import MRIService
from backend.app.api.deps import get_mri_service

router = APIRouter()


@router.post("/upload", response_model=MRIUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_mri_scan(
    file: UploadFile = File(..., description="NIfTI (.nii, .nii.gz) or DICOM brain scan"),
    mri_srv: MRIService = Depends(get_mri_service)
):
    """Uploads an MRI brain scan for subsequent preprocessing and localization."""
    return await mri_srv.save_uploaded_file(file)
