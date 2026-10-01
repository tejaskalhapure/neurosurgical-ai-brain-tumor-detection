"""BraTS Dataset Exploration and Slice Visualizer.

Provides multi-planar slice inspection (Axial, Coronal, Sagittal),
ground-truth segmentation mask visualization, and alpha-blended overlays.
Strictly relies on authentic dataset files without synthetic medical images.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from ai.preprocessing.nifti_handler import NiftiHandler, nifti_handler
from ai.config.dataset_config import LABEL_MAPPINGS

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class BraTSVisualizer:
    """Explores and visualizes authentic BraTS cases and segmentation overlays."""

    def __init__(self, n_handler: Optional[NiftiHandler] = None):
        self.nifti_handler = n_handler or nifti_handler

    def extract_slice_data(
        self,
        mri_path: Path,
        seg_path: Optional[Path] = None,
        plane: str = "axial",
        slice_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """Extracts a 2D slice from an authentic 3D NIfTI file on disk.

        Planes:
          - 'axial': slice along Z dimension (axis=2)
          - 'coronal': slice along Y dimension (axis=1)
          - 'sagittal': slice along X dimension (axis=0)
        """
        mri_file = Path(mri_path)
        if not mri_file.is_file():
            raise FileNotFoundError(f"MRI file does not exist: {mri_file}")

        meta = self.nifti_handler.read_header_metadata(mri_file)
        nx, ny, nz = meta["dimensions"]

        # Determine axis and slice index
        plane_lower = plane.lower()
        if plane_lower == "axial":
            max_slices = nz
            axis = 2
        elif plane_lower == "coronal":
            max_slices = ny
            axis = 1
        elif plane_lower == "sagittal":
            max_slices = nx
            axis = 0
        else:
            raise ValueError(f"Unknown anatomical plane '{plane}'. Allowed: axial, coronal, sagittal")

        idx = slice_index if slice_index is not None else (max_slices // 2)
        if idx < 0 or idx >= max_slices:
            raise IndexError(f"Slice index {idx} out of range [0, {max_slices - 1}] for plane '{plane}'")

        # Load volume
        mri_volume = self.nifti_handler.load_volume_array(mri_file)

        if axis == 0:
            mri_slice = mri_volume[idx, :, :]
        elif axis == 1:
            mri_slice = mri_volume[:, idx, :]
        else:
            mri_slice = mri_volume[:, :, idx]

        seg_slice = None
        has_tumor_in_slice = False
        tumor_labels_present: List[int] = []

        if seg_path and Path(seg_path).is_file():
            seg_volume = self.nifti_handler.load_volume_array(Path(seg_path))
            if axis == 0:
                seg_slice = seg_volume[idx, :, :]
            elif axis == 1:
                seg_slice = seg_volume[:, idx, :]
            else:
                seg_slice = seg_volume[:, :, idx]

            if HAS_NUMPY:
                unique_labels = np.unique(seg_slice)
                tumor_labels_present = [int(lbl) for lbl in unique_labels if lbl > 0]
                has_tumor_in_slice = len(tumor_labels_present) > 0

        return {
            "mri_filename": mri_file.name,
            "plane": plane_lower,
            "slice_index": idx,
            "total_slices_in_plane": max_slices,
            "mri_slice": mri_slice,
            "seg_slice": seg_slice,
            "has_tumor_in_slice": has_tumor_in_slice,
            "tumor_labels_present": tumor_labels_present,
            "voxel_spacing": meta["voxel_spacing"]
        }

    def render_and_save_comparison(
        self,
        mri_path: Path,
        seg_path: Optional[Path],
        output_image_path: Path,
        plane: str = "axial",
        slice_index: Optional[int] = None,
        title: Optional[str] = None
    ) -> Path:
        """Renders MRI slice, segmentation mask, and overlay into a 3-panel figure."""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            raise NotImplementedError("Rendering images to disk requires 'matplotlib' and 'numpy'.")

        data = self.extract_slice_data(mri_path, seg_path, plane=plane, slice_index=slice_index)
        mri_slice = data["mri_slice"]
        seg_slice = data["seg_slice"]
        idx = data["slice_index"]
        plane_str = data["plane"].capitalize()

        fig, axes = plt.subplots(1, 3 if seg_slice is not None else 1, figsize=(15 if seg_slice is not None else 6, 5))
        if seg_slice is None:
            axes = [axes]

        # 1. Base MRI Slice
        axes[0].imshow(mri_slice.T, cmap="gray", origin="lower")
        axes[0].set_title(f"MRI Slice ({plane_str} #{idx})")
        axes[0].axis("off")

        if seg_slice is not None:
            # 2. Discrete Segmentation Mask
            cmap_seg = mcolors.ListedColormap(["black", "red", "green", "yellow"])
            bounds = [0, 0.5, 1.5, 3.0, 5.0]
            norm = mcolors.BoundaryNorm(bounds, cmap_seg.N)

            axes[1].imshow(seg_slice.T, cmap=cmap_seg, norm=norm, origin="lower")
            axes[1].set_title("Ground-Truth Mask (NCR/ED/ET)")
            axes[1].axis("off")

            # 3. Semi-transparent Overlay
            axes[2].imshow(mri_slice.T, cmap="gray", origin="lower")
            masked_seg = np.ma.masked_where(seg_slice.T == 0, seg_slice.T)
            axes[2].imshow(masked_seg, cmap=cmap_seg, norm=norm, alpha=0.45, origin="lower")
            axes[2].set_title("MRI + Segmentation Overlay")
            axes[2].axis("off")

        fig_title = title or f"BraTS Slice Exploration: {Path(mri_path).name}"
        fig.suptitle(fig_title, fontsize=12, fontweight="bold")
        plt.tight_layout()

        out_path = Path(output_image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return out_path
