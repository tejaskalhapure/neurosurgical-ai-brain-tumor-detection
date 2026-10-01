from __future__ import annotations
"""Intensity normalization implementation for multi-modal brain MRI.

Applies Z-score standardisation (zero mean, unit variance) strictly on non-zero
brain tissue voxels to prevent background air from distorting intensity statistics.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


class BaseIntensityNormalizer(ABC):
    """Abstract interface for MRI intensity normalization."""

    @abstractmethod
    def normalize_nonzero(
        self,
        volume: Any,
        clip_percentiles: Optional[Tuple[float, float]] = (0.5, 99.5)
    ) -> Any:
        """Applies Z-score normalization strictly on non-zero brain voxels."""
        pass

    @abstractmethod
    def clip_percentiles(
        self,
        volume: Any,
        lower_pct: float = 0.5,
        upper_pct: float = 99.5
    ) -> Any:
        """Clips extreme intensity outliers prior to neural network ingestion."""
        pass


class MRIIntensityNormalizer(BaseIntensityNormalizer):
    """Robust Z-score intensity normalizer for MRI volumes.

    Characteristics:
      - Ignores background air (intensity == 0) during mean and std calculation.
      - Safely handles zero, constant, or near-constant volumes without division by zero.
      - Per-volume normalization: zero cross-patient data leakage.
      - Preserves background as 0.0.
    """

    def normalize_nonzero(
        self,
        volume: Any,
        clip_percentiles: Optional[Tuple[float, float]] = (0.5, 99.5)
    ) -> Any:
        """Performs non-zero Z-score normalization with optional outlier clipping."""
        if HAS_NUMPY and isinstance(volume, np.ndarray):
            return self._normalize_numpy(volume, clip_percentiles)

        # Fallback for pure Python arrays
        return self._normalize_python(volume, clip_percentiles)

    def clip_percentiles(
        self,
        volume: Any,
        lower_pct: float = 0.5,
        upper_pct: float = 99.5
    ) -> Any:
        """Clips intensity values to the given percentile range computed on non-zero voxels."""
        if HAS_NUMPY and isinstance(volume, np.ndarray):
            nonzero_mask = volume > 0
            if not np.any(nonzero_mask):
                return volume.copy()
            nonzero_vals = volume[nonzero_mask]
            lower_val, upper_val = np.percentile(nonzero_vals, [lower_pct, upper_pct])
            clipped = volume.copy()
            clipped[nonzero_mask] = np.clip(nonzero_vals, lower_val, upper_val)
            return clipped

        return volume

    def _normalize_numpy(
        self,
        volume: np.ndarray,
        clip_percentiles: Optional[Tuple[float, float]]
    ) -> np.ndarray:
        """NumPy optimized non-zero Z-score normalization."""
        norm_vol = volume.astype(np.float32, copy=True)
        nonzero_mask = norm_vol > 0

        # If volume is empty / all zeros, return safely
        if not np.any(nonzero_mask):
            return norm_vol

        nonzero_vals = norm_vol[nonzero_mask]

        # Optional percentile clipping on non-zero tissue
        if clip_percentiles is not None:
            lower_pct, upper_pct = clip_percentiles
            lower_val, upper_val = np.percentile(nonzero_vals, [lower_pct, upper_pct])
            nonzero_vals = np.clip(nonzero_vals, lower_val, upper_val)
            norm_vol[nonzero_mask] = nonzero_vals

        mean = float(np.mean(nonzero_vals))
        std = float(np.std(nonzero_vals))

        # Safe division handling for constant/near-constant scans
        if std < 1e-6:
            norm_vol[nonzero_mask] = 0.0
        else:
            norm_vol[nonzero_mask] = (nonzero_vals - mean) / (std + 1e-8)

        return norm_vol

    def _normalize_python(
        self,
        volume: Any,
        clip_percentiles: Optional[Tuple[float, float]]
    ) -> Any:
        """Pure-Python non-zero Z-score fallback."""
        def flatten(nested):
            for item in nested:
                if isinstance(item, (list, tuple)):
                    yield from flatten(item)
                else:
                    yield item

        vals = [float(x) for x in flatten(volume) if float(x) > 0]
        if not vals:
            return volume

        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = math.sqrt(variance)

        if std < 1e-6:
            denom = 1.0
        else:
            denom = std + 1e-8

        def transform(nested):
            if isinstance(nested, (list, tuple)):
                return [transform(x) for x in nested]
            val = float(nested)
            if val > 0:
                return (val - mean) / denom
            return 0.0

        return transform(volume)


mri_normalizer = MRIIntensityNormalizer()
