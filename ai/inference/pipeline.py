"""Standalone AI Inference Pipeline.

This pipeline can be executed independently from CLI, batch test scripts,
or imported into FastAPI backend services.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from ai.config.ai_config import AIModelConfig, PreprocessingConfig, VolumetricConfig


class BrainTumorAIPipeline:
    """Decoupled, standalone pipeline orchestrating MRI preprocessing,

    inference, 3D localization bounding boxes, and volumetric quantification.
    """

    def __init__(
        self,
        model_config: Optional[AIModelConfig] = None,
        prep_config: Optional[PreprocessingConfig] = None,
        vol_config: Optional[VolumetricConfig] = None,
    ):
        self.model_config = model_config or AIModelConfig()
        self.prep_config = prep_config or PreprocessingConfig()
        self.vol_config = vol_config or VolumetricConfig()
        self.is_model_ready: bool = False

    def initialize_model(self) -> bool:
        """Checks availability of weights file and prepares device.

        In Module 1, verifies checkpoint configuration without fabricating inference.
        """
        weights_file = Path(self.model_config.weights_path)
        if weights_file.exists():
            self.is_model_ready = True
            return True
        self.is_model_ready = False
        return False

    def execute_pipeline(self, scan_path: Path) -> Dict[str, Any]:
        """Runs end-to-end processing pipeline on a designated MRI scan.

        In Module 1 architecture, this defines the exact pipeline contract
        expected by callers (FastAPI or researcher CLI).
        """
        scan_file = Path(scan_path)
        if not scan_file.exists():
            raise FileNotFoundError(f"MRI scan file does not exist at {scan_path}")

        # Returns pipeline stage metadata contract
        return {
            "file_name": scan_file.name,
            "pipeline_status": "architecture_ready",
            "model_initialized": self.is_model_ready,
            "device": self.model_config.device,
            "module_stage": "Module 1 - Architecture & Contract Specification"
        }
