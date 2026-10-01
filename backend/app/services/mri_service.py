"""MRI storage and metadata service."""

from pathlib import Path
import uuid
import aiofiles
from fastapi import UploadFile, HTTPException
from backend.app.core.config import settings
from backend.app.schemas.mri import MRIUploadResponse


class MRIService:
    """Manages raw MRI file staging, extension validation, and storage allocation."""

    def __init__(self):
        self.upload_dir = Path(settings.DATA_RAW_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def validate_file_extension(self, filename: str) -> str:
        """Validates allowable medical image extensions."""
        lower_name = filename.lower()
        if lower_name.endswith(".nii.gz"):
            return ".nii.gz"
        for ext in settings.ALLOWED_EXTENSIONS:
            if lower_name.endswith(ext):
                return ext
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )

    async def save_uploaded_file(self, upload_file: UploadFile) -> MRIUploadResponse:
        """Saves uploaded scan file into data/raw/<scan_id>/ directory safely."""
        extension = self.validate_file_extension(upload_file.filename or "scan.nii")
        scan_id = f"mri_{uuid.uuid4().hex[:12]}"
        scan_folder = self.upload_dir / scan_id
        scan_folder.mkdir(parents=True, exist_ok=True)

        target_file_path = scan_folder / f"source{extension}"
        total_bytes = 0

        async with aiofiles.open(target_file_path, "wb") as out_file:
            while content := await upload_file.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(content)
                if total_bytes > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    # Clean up and reject oversized files
                    target_file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB"
                    )
                await out_file.write(content)

        return MRIUploadResponse(
            scan_id=scan_id,
            filename=upload_file.filename or "unknown.nii",
            file_size_bytes=total_bytes,
            status="stored_ready_for_preprocess",
            format_detected=extension
        )


mri_service = MRIService()
