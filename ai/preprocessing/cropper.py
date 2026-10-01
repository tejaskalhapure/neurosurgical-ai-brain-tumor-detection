from __future__ import annotations
"""Foreground and brain-region cropping for 3D multi-modal MRI.

Eliminates redundant zero-intensity air while guaranteeing that:
  - Tumor tissue and brain parenchyma are strictly preserved.
  - The exact same spatial bounding box is applied to all modalities and the segmentation mask.
  - Crop coordinates are fully preserved for spatial reversibility back to scanner space.
"""

from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


@dataclass
class CropCoordinates:
    """Bounding box coordinates applied to 3D volume."""
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int
    original_shape: Tuple[int, int, int]
    cropped_shape: Tuple[int, int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "original_shape": list(self.original_shape),
            "cropped_shape": list(self.cropped_shape)
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CropCoordinates":
        return cls(
            x_min=d["x_min"],
            x_max=d["x_max"],
            y_min=d["y_min"],
            y_max=d["y_max"],
            z_min=d["z_min"],
            z_max=d["z_max"],
            original_shape=tuple(d["original_shape"]),
            cropped_shape=tuple(d["cropped_shape"])
        )


class ForegroundCropper:
    """Computes and applies identical bounding box crops across multi-modal volumes."""

    def __init__(self, margin: int = 4, target_size: Optional[Tuple[int, int, int]] = None):
        self.margin = margin
        self.target_size = target_size

    def compute_crop_coordinates(
        self,
        volumes: List[Any],
        seg_mask: Optional[Any] = None
    ) -> CropCoordinates:
        """Finds non-zero brain bounding box across all MRI modalities and segmentation."""
        if not volumes:
            raise ValueError("No volumes provided for crop coordinate computation.")

        if HAS_NUMPY and isinstance(volumes[0], np.ndarray):
            orig_shape = volumes[0].shape[:3]

            # Aggregate non-zero foreground mask
            combined_mask = np.zeros(orig_shape, dtype=bool)
            for vol in volumes:
                combined_mask |= (vol > 0)

            # Ensure segmentation is also covered (never crop away tumor!)
            if seg_mask is not None and isinstance(seg_mask, np.ndarray):
                combined_mask |= (seg_mask > 0)

            nonzero_indices = np.argwhere(combined_mask)
            if nonzero_indices.size == 0:
                # Fallback to full volume if scan is completely zero
                return CropCoordinates(
                    x_min=0, x_max=orig_shape[0],
                    y_min=0, y_max=orig_shape[1],
                    z_min=0, z_max=orig_shape[2],
                    original_shape=orig_shape,
                    cropped_shape=orig_shape
                )

            min_coords = nonzero_indices.min(axis=0)
            max_coords = nonzero_indices.max(axis=0) + 1  # exclusive slice

            x_min = max(0, int(min_coords[0] - self.margin))
            x_max = min(orig_shape[0], int(max_coords[0] + self.margin))
            y_min = max(0, int(min_coords[1] - self.margin))
            y_max = min(orig_shape[1], int(max_coords[1] + self.margin))
            z_min = max(0, int(min_coords[2] - self.margin))
            z_max = min(orig_shape[2], int(max_coords[2] + self.margin))

            # If target_size is requested, center or pad around foreground
            if self.target_size is not None:
                x_min, x_max = self._adjust_to_target_dim(x_min, x_max, orig_shape[0], self.target_size[0])
                y_min, y_max = self._adjust_to_target_dim(y_min, y_max, orig_shape[1], self.target_size[1])
                z_min, z_max = self._adjust_to_target_dim(z_min, z_max, orig_shape[2], self.target_size[2])

            cropped_shape = (x_max - x_min, y_max - y_min, z_max - z_min)
            return CropCoordinates(
                x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max,
                z_min=z_min, z_max=z_max,
                original_shape=orig_shape,
                cropped_shape=cropped_shape
            )

        # Pure Python fallback
        shape = (len(volumes[0]), len(volumes[0][0]), len(volumes[0][0][0]))
        return CropCoordinates(
            x_min=0, x_max=shape[0],
            y_min=0, y_max=shape[1],
            z_min=0, z_max=shape[2],
            original_shape=shape,
            cropped_shape=shape
        )

    def _adjust_to_target_dim(self, min_val: int, max_val: int, full_dim: int, target_len: int) -> Tuple[int, int]:
        """Adjusts 1D slice window to match target length while centered on foreground."""
        current_len = max_val - min_val
        if current_len == target_len:
            return min_val, max_val

        center = (min_val + max_val) // 2
        half = target_len // 2

        new_min = center - half
        new_max = new_min + target_len

        # Clamp boundaries if within full_dim
        if target_len <= full_dim:
            if new_min < 0:
                new_max += -new_min
                new_min = 0
            elif new_max > full_dim:
                new_min -= (new_max - full_dim)
                new_max = full_dim
            return max(0, new_min), min(full_dim, new_max)

        return min_val, max_val

    def apply_crop(self, volume: Any, coords: CropCoordinates) -> Any:
        """Applies crop coordinates to a 3D or 4D volume."""
        if HAS_NUMPY and isinstance(volume, np.ndarray):
            if volume.ndim == 3:
                return volume[coords.x_min:coords.x_max, coords.y_min:coords.y_max, coords.z_min:coords.z_max].copy()
            elif volume.ndim == 4:
                # Format: [C, X, Y, Z]
                return volume[:, coords.x_min:coords.x_max, coords.y_min:coords.y_max, coords.z_min:coords.z_max].copy()

        return volume

    @staticmethod
    def uncrop_volume(
        cropped_volume: Any,
        coords: CropCoordinates,
        fill_value: float = 0.0
    ) -> Any:
        """Reconstructs original spatial shape by placing cropped array back at crop coordinates.

        Critical for surgical planning overlays and area/volume localization.
        """
        if HAS_NUMPY and isinstance(cropped_volume, np.ndarray):
            orig_shape = coords.original_shape
            if cropped_volume.ndim == 3:
                full_vol = np.full(orig_shape, fill_value, dtype=cropped_volume.dtype)
                full_vol[coords.x_min:coords.x_max, coords.y_min:coords.y_max, coords.z_min:coords.z_max] = cropped_volume
                return full_vol
            elif cropped_volume.ndim == 4:
                channels = cropped_volume.shape[0]
                full_vol = np.full((channels, *orig_shape), fill_value, dtype=cropped_volume.dtype)
                full_vol[:, coords.x_min:coords.x_max, coords.y_min:coords.y_max, coords.z_min:coords.z_max] = cropped_volume
                return full_vol

        return cropped_volume
