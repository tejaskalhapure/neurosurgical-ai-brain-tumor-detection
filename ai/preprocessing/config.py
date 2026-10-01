"""MRI Preprocessing Configuration.

Defines parameters for voxel resampling, orientation standardization,
foreground cropping, intensity normalization, channel ordering, and label mapping.
"""

from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import os
import json


DEFAULT_CHANNEL_ORDER: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair")
DEFAULT_TARGET_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)
DEFAULT_ORIENTATION: str = "RAS"
DEFAULT_CROP_SIZE: Tuple[int, int, int] = (128, 128, 128)

# BraTS standard to contiguous model labels
# 0: Background -> 0
# 1: NCR/NET (Necrotic core) -> 1
# 2: ED (Edema) -> 2
# 4: ET (Enhancing tumor) -> 3
DEFAULT_CONTIGUOUS_LABEL_MAP: Dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    4: 3
}

DEFAULT_INVERSE_LABEL_MAP: Dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    3: 4
}


@dataclass
class MRIPreprocessingConfig:
    """Configurable parameters for the MRI preprocessing pipeline."""

    target_spacing: Optional[Tuple[float, float, float]] = DEFAULT_TARGET_SPACING
    target_orientation: str = DEFAULT_ORIENTATION
    spatial_crop_size: Optional[Tuple[int, int, int]] = DEFAULT_CROP_SIZE
    crop_margin: int = 4
    intensity_normalization: str = "zscore_nonzero"
    clip_percentiles: Optional[Tuple[float, float]] = (0.5, 99.5)
    mri_interpolation: str = "linear"
    seg_interpolation: str = "nearest"
    channel_order: Tuple[str, ...] = DEFAULT_CHANNEL_ORDER
    convert_labels_to_contiguous: bool = True
    contiguous_label_map: Dict[int, int] = field(default_factory=lambda: dict(DEFAULT_CONTIGUOUS_LABEL_MAP))
    inverse_label_map: Dict[int, int] = field(default_factory=lambda: dict(DEFAULT_INVERSE_LABEL_MAP))
    enable_augmentation: bool = False
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Serializes configuration to dictionary."""
        d = asdict(self)
        d["target_spacing"] = list(self.target_spacing) if self.target_spacing else None
        d["spatial_crop_size"] = list(self.spatial_crop_size) if self.spatial_crop_size else None
        d["clip_percentiles"] = list(self.clip_percentiles) if self.clip_percentiles else None
        d["channel_order"] = list(self.channel_order)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MRIPreprocessingConfig":
        """Loads configuration from dictionary."""
        spacing = tuple(data["target_spacing"]) if data.get("target_spacing") else None
        crop = tuple(data["spatial_crop_size"]) if data.get("spatial_crop_size") else None
        clip = tuple(data["clip_percentiles"]) if data.get("clip_percentiles") else None
        channels = tuple(data.get("channel_order", DEFAULT_CHANNEL_ORDER))

        label_map = {int(k): int(v) for k, v in data.get("contiguous_label_map", DEFAULT_CONTIGUOUS_LABEL_MAP).items()}
        inv_map = {int(k): int(v) for k, v in data.get("inverse_label_map", DEFAULT_INVERSE_LABEL_MAP).items()}

        return cls(
            target_spacing=spacing,
            target_orientation=data.get("target_orientation", DEFAULT_ORIENTATION),
            spatial_crop_size=crop,
            crop_margin=int(data.get("crop_margin", 4)),
            intensity_normalization=data.get("intensity_normalization", "zscore_nonzero"),
            clip_percentiles=clip,
            mri_interpolation=data.get("mri_interpolation", "linear"),
            seg_interpolation=data.get("seg_interpolation", "nearest"),
            channel_order=channels,
            convert_labels_to_contiguous=bool(data.get("convert_labels_to_contiguous", True)),
            contiguous_label_map=label_map,
            inverse_label_map=inv_map,
            enable_augmentation=bool(data.get("enable_augmentation", False)),
            seed=int(data.get("seed", 42))
        )

    @classmethod
    def from_env(cls) -> "MRIPreprocessingConfig":
        """Loads configuration with environment variable overrides."""
        target_sp_env = os.getenv("PREPROCESS_TARGET_SPACING")
        spacing = tuple(float(x) for x in target_sp_env.split(",")) if target_sp_env else DEFAULT_TARGET_SPACING

        orientation = os.getenv("PREPROCESS_ORIENTATION", DEFAULT_ORIENTATION)
        norm_method = os.getenv("PREPROCESS_NORMALIZATION", "zscore_nonzero")

        return cls(
            target_spacing=spacing,
            target_orientation=orientation,
            intensity_normalization=norm_method
        )
