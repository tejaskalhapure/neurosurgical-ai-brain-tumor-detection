from __future__ import annotations
"""BraTS Segmentation Label Processing & Contiguous Re-mapping.

Handles conversion between official non-contiguous BraTS labels (0, 1, 2, 4)
and contiguous neural network class indices (0, 1, 2, 3) with exact bidirectional invertibility.
"""

from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


# Official BraTS Label Semantics
# Label 0: Background
# Label 1: Necrotic Tumor Core (NCR) & Non-Enhancing Tumor (NET)
# Label 2: Peritumoral Edematous Tissue (ED)
# Label 4: GD-Enhancing Tumor (ET)
BRATS_OFFICIAL_LABELS: Dict[int, str] = {
    0: "Background",
    1: "NCR/NET (Necrotic Core)",
    2: "ED (Peritumoral Edema)",
    4: "ET (Enhancing Tumor)"
}

CONTIGUOUS_FORWARD_MAP: Dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    4: 3
}

CONTIGUOUS_INVERSE_MAP: Dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    3: 4
}


class BraTSLabelHandler:
    """Manages ground-truth label conversions and clinical sub-region derivations."""

    def __init__(
        self,
        forward_map: Optional[Dict[int, int]] = None,
        inverse_map: Optional[Dict[int, int]] = None
    ):
        self.forward_map = forward_map or dict(CONTIGUOUS_FORWARD_MAP)
        self.inverse_map = inverse_map or dict(CONTIGUOUS_INVERSE_MAP)

    def to_contiguous(self, seg_mask: Any) -> Any:
        """Converts official BraTS labels {0, 1, 2, 4} to contiguous model classes {0, 1, 2, 3}."""
        if not HAS_NUMPY or not isinstance(seg_mask, np.ndarray):
            return seg_mask

        contiguous = np.zeros_like(seg_mask, dtype=np.int64)
        for orig_label, cont_label in self.forward_map.items():
            contiguous[seg_mask == orig_label] = cont_label

        return contiguous

    def to_original_brats(self, contiguous_mask: Any) -> Any:
        """Converts model output class indices {0, 1, 2, 3} back to official BraTS labels {0, 1, 2, 4}."""
        if not HAS_NUMPY or not isinstance(contiguous_mask, np.ndarray):
            return contiguous_mask

        original = np.zeros_like(contiguous_mask, dtype=np.int64)
        for cont_label, orig_label in self.inverse_map.items():
            original[contiguous_mask == cont_label] = orig_label

        return original

    def extract_subregions(
        self,
        seg_mask: Any,
        is_contiguous: bool = False
    ) -> Dict[str, Any]:
        """Extracts binary masks for the three standard surgical evaluation regions:

          - WT (Whole Tumor): Complete pathological mass
          - TC (Tumor Core): Necrotic core + active enhancing rim
          - ET (Enhancing Tumor): Active hyper-vascular rim
        """
        if not HAS_NUMPY or not isinstance(seg_mask, np.ndarray):
            return {}

        if is_contiguous:
            # Classes: 1=NCR, 2=ED, 3=ET
            wt_mask = (seg_mask == 1) | (seg_mask == 2) | (seg_mask == 3)
            tc_mask = (seg_mask == 1) | (seg_mask == 3)
            et_mask = (seg_mask == 3)
        else:
            # Official BraTS classes: 1=NCR, 2=ED, 4=ET
            wt_mask = (seg_mask == 1) | (seg_mask == 2) | (seg_mask == 4)
            tc_mask = (seg_mask == 1) | (seg_mask == 4)
            et_mask = (seg_mask == 4)

        return {
            "WT": wt_mask,
            "TC": tc_mask,
            "ET": et_mask
        }

    def verify_invertibility(self, seg_mask: np.ndarray) -> bool:
        """Tests that to_original_brats(to_contiguous(mask)) strictly equals mask."""
        cont = self.to_contiguous(seg_mask)
        restored = self.to_original_brats(cont)
        return bool(np.array_equal(seg_mask, restored))


label_handler = BraTSLabelHandler()
