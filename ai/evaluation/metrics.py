from __future__ import annotations
"""Mathematical definitions and evaluation formulas for medical image segmentation

and surgical planning decision-support metrics.
"""

from typing import Tuple, Dict, Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


def dice_similarity_coefficient(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Computes the Dice Similarity Coefficient (DSC):

    DSC = (2 * |X ∩ Y|) / (|X| + |Y|)
    """
    y_true_f = y_true.astype(bool).flatten()
    y_pred_f = y_pred.astype(bool).flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    cardinality = y_true_f.sum() + y_pred_f.sum()
    return float((2.0 * intersection + smooth) / (cardinality + smooth))


def intersection_over_union(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Computes the Jaccard Index / Intersection over Union (IoU):

    IoU = |X ∩ Y| / |X ∪ Y|
    """
    y_true_f = y_true.astype(bool).flatten()
    y_pred_f = y_pred.astype(bool).flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return float((intersection + smooth) / (union + smooth))


def calculate_slice_area_mm2(binary_slice_mask: np.ndarray, dx: float, dy: float) -> float:
    """Calculates 2D tumor cross-sectional surface area on a specific slice:

    Area (mm²) = Pixel_Count * dx * dy
    """
    pixel_count = int(np.count_nonzero(binary_slice_mask))
    return float(pixel_count * dx * dy)


def calculate_tumor_volume(
    binary_volume_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float]
) -> Dict[str, Any]:
    """Calculates 3D tumor volume when valid physical voxel spacing metadata is available:

    Volume (mm³) = Positive_Voxel_Count * (dx * dy * dz)
    Volume (cm³) = Volume (mm³) / 1000.0
    """
    dx, dy, dz = voxel_spacing
    if dx <= 0 or dy <= 0 or dz <= 0:
        return {
            "estimated_volume_mm3": 0.0,
            "estimated_volume_cm3": 0.0,
            "voxel_count": 0,
            "valid_voxel_metadata": False,
            "note": "Invalid or missing voxel spacing metadata"
        }

    positive_voxels = int(np.count_nonzero(binary_volume_mask))
    voxel_volume_mm3 = float(dx * dy * dz)
    total_volume_mm3 = float(positive_voxels * voxel_volume_mm3)
    total_volume_cm3 = float(total_volume_mm3 / 1000.0)

    return {
        "estimated_volume_mm3": round(total_volume_mm3, 2),
        "estimated_volume_cm3": round(total_volume_cm3, 3),
        "voxel_count": positive_voxels,
        "valid_voxel_metadata": True,
        "note": "Valid volumetric calculation based on physical voxel spacing"
    }


def calculate_bounding_box_3d(binary_volume_mask: np.ndarray) -> Optional[Dict[str, int]]:
    """Calculates the 3D bounding box coordinates enclosing the detected tumor mass.

    Returns:
        Dict with x_min, x_max, y_min, y_max, z_min, z_max or None if no tumor voxels.
    """
    coords = np.argwhere(binary_volume_mask)
    if coords.size == 0:
        return None

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    return {
        "x_min": int(x_min),
        "x_max": int(x_max),
        "y_min": int(y_min),
        "y_max": int(y_max),
        "z_min": int(z_min),
        "z_max": int(z_max),
    }
