from __future__ import annotations
"""Preprocessing visualization utility.

Generates 5-panel comparative visualization:
  1. Original MRI slice
  2. Preprocessed MRI slice
  3. Original segmentation mask
  4. Preprocessed segmentation mask
  5. Preprocessed overlay (MRI + segmentation)
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ai.preprocessing.pipeline import BraTSPreprocessedCase
from ai.preprocessing.nifti_handler import nifti_handler


class PreprocessingVisualizer:
    """Visualizes before/after effects of orientation, resampling, cropping, and normalization."""

    @staticmethod
    def render_5panel_comparison(
        raw_mri_path: Path,
        raw_seg_path: Optional[Path],
        preprocessed_case: BraTSPreprocessedCase,
        output_image_path: Path,
        modality_channel_idx: int = 3,  # default to FLAIR (index 3)
        slice_ratio: float = 0.5  # middle slice
    ) -> Path:
        """Renders 5-panel comparative inspection figure to disk."""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            raise NotImplementedError("Visualization requires 'matplotlib' and 'numpy'.")

        # 1. Load Raw MRI and Raw Seg
        raw_mri = nifti_handler.load_volume_array(Path(raw_mri_path))
        raw_slice_idx = int(raw_mri.shape[2] * slice_ratio)
        raw_mri_slice = raw_mri[:, :, raw_slice_idx]

        raw_seg_slice = None
        if raw_seg_path and Path(raw_seg_path).is_file():
            raw_seg = nifti_handler.load_volume_array(Path(raw_seg_path))
            raw_seg_slice = raw_seg[:, :, raw_slice_idx]

        # 2. Extract Preprocessed MRI & Seg
        prep_img = preprocessed_case.image
        if hasattr(prep_img, "cpu"):
            prep_img = prep_img.cpu().numpy()

        prep_slice_idx = int(prep_img.shape[3] * slice_ratio)
        prep_mri_slice = prep_img[modality_channel_idx, :, :, prep_slice_idx]

        prep_seg_slice = None
        if preprocessed_case.label is not None:
            prep_lbl = preprocessed_case.label
            if hasattr(prep_lbl, "cpu"):
                prep_lbl = prep_lbl.cpu().numpy()
            prep_seg_slice = prep_lbl[0, :, :, prep_slice_idx]

        # 3. Create 5-panel layout
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

        # Panel 1: Original MRI
        axes[0].imshow(raw_mri_slice.T, cmap="gray", origin="lower")
        axes[0].set_title(f"1. Original MRI\nShape: {raw_mri.shape}", fontsize=10)
        axes[0].axis("off")

        # Panel 2: Preprocessed MRI (Normalized & Cropped)
        axes[1].imshow(prep_mri_slice.T, cmap="gray", origin="lower")
        channel_name = preprocessed_case.metadata.get("channel_order", ["T1", "T1ce", "T2", "FLAIR"])[modality_channel_idx].upper()
        axes[1].set_title(f"2. Preprocessed MRI ({channel_name})\nShape: {prep_mri_slice.shape}", fontsize=10)
        axes[1].axis("off")

        # Colormap for segmentation (0=black, 1=red, 2=green, 3=yellow/4=yellow)
        cmap_seg = mcolors.ListedColormap(["black", "crimson", "mediumseagreen", "gold"])
        bounds = [0, 0.5, 1.5, 2.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap_seg.N)

        # Panel 3: Original Segmentation
        if raw_seg_slice is not None:
            axes[2].imshow(raw_seg_slice.T, cmap=cmap_seg, norm=norm, origin="lower")
            axes[2].set_title("3. Original Mask\nLabels: {0, 1, 2, 4}", fontsize=10)
        else:
            axes[2].text(0.5, 0.5, "No Seg Label", ha="center", va="center", color="gray")
            axes[2].set_title("3. Original Mask", fontsize=10)
        axes[2].axis("off")

        # Panel 4: Preprocessed Segmentation
        if prep_seg_slice is not None:
            axes[3].imshow(prep_seg_slice.T, cmap=cmap_seg, norm=norm, origin="lower")
            axes[3].set_title("4. Preprocessed Mask\nContiguous: {0, 1, 2, 3}", fontsize=10)
        else:
            axes[3].text(0.5, 0.5, "No Seg Label", ha="center", va="center", color="gray")
            axes[3].set_title("4. Preprocessed Mask", fontsize=10)
        axes[3].axis("off")

        # Panel 5: Preprocessed Overlay
        axes[4].imshow(prep_mri_slice.T, cmap="gray", origin="lower")
        if prep_seg_slice is not None:
            masked = np.ma.masked_where(prep_seg_slice.T == 0, prep_seg_slice.T)
            axes[4].imshow(masked, cmap=cmap_seg, norm=norm, alpha=0.5, origin="lower")
            axes[4].set_title("5. Preprocessed Overlay\nAligned Alignment Check", fontsize=10)
        else:
            axes[4].set_title("5. Preprocessed Overlay\n(No Label)", fontsize=10)
        axes[4].axis("off")

        case_id = preprocessed_case.case_id
        fig.suptitle(f"Preprocessing Traceability Verification - Case: {case_id}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        out_path = Path(output_image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return out_path
