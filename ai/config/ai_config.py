"""AI Pipeline Configuration Dataclass / Model.

Contains hardware selection, voxel spacing, spatial patch sizes,
normalization parameters, and post-processing thresholds.
"""

from typing import List, Tuple, Optional, Any
import os

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    from dataclasses import dataclass, field

    def Field(default=None, description=""):
        return default


if HAS_PYDANTIC:
    class AIModelConfig(BaseModel):
        """Configuration parameters for model architecture and inference."""
        model_name: str = Field(default="unet3d_prototype", description="Model architecture identifier")
        in_channels: int = Field(default=4, description="Multi-modal MRI channels: T1, T1ce, T2, FLAIR")
        out_channels: int = Field(default=3, description="Target tumor sub-regions: WT, TC, ET")
        spatial_size: Tuple[int, int, int] = Field(default=(128, 128, 128), description="Sliding window patch size")
        weights_path: str = Field(
            default=os.getenv("MODEL_WEIGHTS_PATH", "models/exports/model_weights.pth"),
            description="Path to serialized PyTorch model weights"
        )
        device: str = Field(
            default=os.getenv("DEVICE", "cpu"),
            description="Execution device: 'cpu' for local, 'cuda' for Colab"
        )
        sliding_window_overlap: float = Field(default=0.25, description="Overlap ratio for sliding window inferer")

    class PreprocessingConfig(BaseModel):
        """Configuration for NIfTI image preprocessing."""
        target_spacing: Tuple[float, float, float] = Field(default=(1.0, 1.0, 1.0), description="Target isotropic voxel spacing (mm)")
        normalize_intensities: bool = Field(default=True, description="Apply intensity standard deviation normalization")
        clip_percentiles: Tuple[float, float] = Field(default=(0.5, 99.5), description="Intensity percentile clipping limits")

    class VolumetricConfig(BaseModel):
        """Configuration for surgical planning post-processing and volumetric estimation."""
        probability_threshold: float = Field(default=0.5, description="Probability cutoff for binary tumor mask")
        min_tumor_voxels: int = Field(default=50, description="Minimum contiguous positive voxels to declare tumor detection")
else:
    @dataclass
    class AIModelConfig:
        model_name: str = "unet3d_prototype"
        in_channels: int = 4
        out_channels: int = 3
        spatial_size: Tuple[int, int, int] = (128, 128, 128)
        weights_path: str = os.getenv("MODEL_WEIGHTS_PATH", "models/exports/model_weights.pth")
        device: str = os.getenv("DEVICE", "cpu")
        sliding_window_overlap: float = 0.25

    @dataclass
    class PreprocessingConfig:
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        normalize_intensities: bool = True
        clip_percentiles: Tuple[float, float] = (0.5, 99.5)

    @dataclass
    class VolumetricConfig:
        probability_threshold: float = 0.5
        min_tumor_voxels: int = 50
