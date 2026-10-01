"""Inference service connecting FastAPI backend with the decoupled AI pipeline."""

from pathlib import Path
from typing import Dict, Any
from fastapi import HTTPException
from backend.app.core.config import settings
from ai.inference.pipeline import BrainTumorAIPipeline
from ai.config.ai_config import AIModelConfig


class InferenceService:
    """Orchestrates AI pipeline calls without coupling web logic into ML algorithms."""

    def __init__(self):
        ai_cfg = AIModelConfig(
            weights_path=settings.MODEL_WEIGHTS_PATH,
            device=settings.DEVICE
        )
        self.pipeline = BrainTumorAIPipeline(model_config=ai_cfg)

    def get_status(self) -> Dict[str, Any]:
        """Returns model initialization and hardware execution status."""
        return {
            "device": settings.DEVICE,
            "weights_configured": settings.MODEL_WEIGHTS_PATH,
            "weights_file_exists": Path(settings.MODEL_WEIGHTS_PATH).exists(),
            "model_initialized": self.pipeline.is_model_ready
        }

    def dispatch_inference(self, scan_id: str) -> Dict[str, Any]:
        """Dispatches an inference request to the decoupled AI pipeline.

        In Module 1: Enforces the architectural boundary, checks scan existence,
        and validates that no fabricated predictions are returned prior to model integration.
        """
        scan_dir = Path(settings.DATA_RAW_DIR) / scan_id
        if not scan_dir.exists():
            raise HTTPException(status_code=404, detail=f"Scan ID '{scan_id}' not found.")

        # Find raw source scan in the scan directory
        scan_files = list(scan_dir.glob("source*"))
        if not scan_files:
            raise HTTPException(status_code=404, detail=f"No source scan files found for scan ID '{scan_id}'.")

        # Call the decoupled AI pipeline
        result = self.pipeline.execute_pipeline(scan_files[0])
        return result


inference_service = InferenceService()
