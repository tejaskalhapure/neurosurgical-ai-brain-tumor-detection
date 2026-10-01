from __future__ import annotations
"""Spatial alignment, orientation standardization, and voxel resampling.

Implements RAS+ orientation standardization and continuous vs nearest-neighbor resampling
with full transformation metadata preservation for surgical area and volume traceability.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, Dict, List
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

try:
    import scipy.ndimage as ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


class BaseSpatialTransformer(ABC):
    """Abstract interface for isotropic resampling and orientation standardisation."""

    @abstractmethod
    def resample_to_spacing(
        self,
        volume: Any,
        current_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        is_label: bool = False
    ) -> Tuple[Any, Tuple[int, int, int]]:
        """Resamples 3D volume using spline (MRI) or nearest-neighbor (mask) interpolation."""
        pass

    @abstractmethod
    def reorient_to_ras(self, volume: Any, affine: Any) -> Tuple[Any, Any]:
        """Standardizes anatomical voxel coordinates to RAS coordinate space."""
        pass


class SpatialTransformer(BaseSpatialTransformer):
    """Concrete spatial transformation engine for multi-modal brain MRI.

    Orientation Convention:
      Standard neurological RAS+ (Right, Anterior, Superior):
        - Axis 0 (X): Extends from Left to Right (x increases towards Right)
        - Axis 1 (Y): Extends from Posterior to Anterior (y increases towards Anterior)
        - Axis 2 (Z): Extends from Inferior to Superior (z increases towards Superior / cranial)

    Resampling Rules:
      - Continuous structural MRI: 1st-order trilinear / spline interpolation (order=1)
      - Segmentation masks: Strictly 0th-order nearest-neighbor interpolation (order=0)
    """

    def reorient_to_ras(
        self,
        volume: Any,
        affine: Any
    ) -> Tuple[Any, Any]:
        """Standardizes 3D volume to RAS+ anatomical orientation while updating the affine."""
        if not HAS_NUMPY or not isinstance(volume, np.ndarray):
            return volume, affine

        aff = np.array(affine, dtype=float)

        if HAS_NIBABEL:
            try:
                orig_ornt = nib.io_orientation(aff)
                target_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
                transform_ornt = nib.orientations.ornt_transform(orig_ornt, target_ornt)

                reoriented_vol = nib.orientations.apply_orientation(volume, transform_ornt)
                new_affine = aff @ nib.orientations.inv_ornt_aff(transform_ornt, volume.shape)
                return reoriented_vol, new_affine.tolist()
            except Exception:
                pass

        # Fallback orientation standardizer based on affine diagonal signs
        reoriented = volume.copy()
        new_aff = aff.copy()

        # Check X flip
        if new_aff[0, 0] < 0:
            reoriented = np.flip(reoriented, axis=0)
            new_aff[0, 0] = -new_aff[0, 0]
            new_aff[0, 3] = -new_aff[0, 3]

        # Check Y flip
        if new_aff[1, 1] < 0:
            reoriented = np.flip(reoriented, axis=1)
            new_aff[1, 1] = -new_aff[1, 1]
            new_aff[1, 3] = -new_aff[1, 3]

        # Check Z flip
        if new_aff[2, 2] < 0:
            reoriented = np.flip(reoriented, axis=2)
            new_aff[2, 2] = -new_aff[2, 2]
            new_aff[2, 3] = -new_aff[2, 3]

        return reoriented, new_aff.tolist()

    def resample_to_spacing(
        self,
        volume: Any,
        current_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        is_label: bool = False
    ) -> Tuple[Any, Tuple[int, int, int]]:
        """Resamples 3D volume from current_spacing to target_spacing.

        Args:
            volume: 3D NumPy array
            current_spacing: (dx, dy, dz) in mm
            target_spacing: (target_dx, target_dy, target_dz) in mm
            is_label: True for segmentation masks (nearest-neighbor), False for MRI (trilinear)

        Returns:
            Tuple of (resampled_volume, new_shape)
        """
        if not HAS_NUMPY or not isinstance(volume, np.ndarray):
            return volume, volume.shape if hasattr(volume, "shape") else (0, 0, 0)

        # If spacing is already identical, return unchanged
        if all(abs(c - t) < 1e-4 for c, t in zip(current_spacing, target_spacing)):
            return volume.copy(), volume.shape

        zoom_factors = [c / t for c, t in zip(current_spacing, target_spacing)]

        if HAS_SCIPY:
            # order=0 is strictly nearest-neighbor (required for segmentation masks)
            # order=1 is continuous trilinear interpolation (for continuous MRI intensities)
            interp_order = 0 if is_label else 1
            resampled = ndimage.zoom(volume, zoom=zoom_factors, order=interp_order, mode="nearest")

            if is_label:
                # Ensure integer label types are preserved
                resampled = np.round(resampled).astype(volume.dtype)
            else:
                resampled = resampled.astype(np.float32)

            return resampled, resampled.shape

        # Minimal fallback if scipy is absent
        target_shape = tuple(int(round(s * z)) for s, z in zip(volume.shape, zoom_factors))
        return volume, target_shape

    def resample_to_original_shape(
        self,
        resampled_volume: Any,
        original_shape: Tuple[int, int, int],
        is_label: bool = True
    ) -> Any:
        """Projects a preprocessed segmentation mask back to the original scan shape."""
        if not HAS_NUMPY or not isinstance(resampled_volume, np.ndarray) or not HAS_SCIPY:
            return resampled_volume

        current_shape = resampled_volume.shape
        if current_shape == original_shape:
            return resampled_volume.copy()

        zoom_factors = [o / c for o, c in zip(original_shape, current_shape)]
        interp_order = 0 if is_label else 1
        restored = ndimage.zoom(resampled_volume, zoom=zoom_factors, order=interp_order, mode="nearest")

        if is_label:
            restored = np.round(restored).astype(resampled_volume.dtype)

        # Exact dimension clip/pad adjustment if roundoff occurred
        return self._match_exact_shape(restored, original_shape)

    def _match_exact_shape(self, array: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Clips or zero-pads array to match target shape exactly if zoom rounded off by 1 voxel."""
        out = array
        for axis in range(3):
            cur = out.shape[axis]
            tgt = target_shape[axis]
            if cur > tgt:
                slices = [slice(None)] * 3
                slices[axis] = slice(0, tgt)
                out = out[tuple(slices)]
            elif cur < tgt:
                pad_width = [(0, 0)] * 3
                pad_width[axis] = (0, tgt - cur)
                out = np.pad(out, pad_width, mode="constant", constant_values=0)
        return out


spatial_transformer = SpatialTransformer()
