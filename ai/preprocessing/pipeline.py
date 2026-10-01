from __future__ import annotations
"""BraTS Master MRI Preprocessing Pipeline.

Orchestrates NIfTI loading, RAS+ orientation standardization, isotropic resampling,
unified foreground cropping, non-zero intensity normalization, channel stacking [4, D, H, W],
and comprehensive spatial traceability back to scanner coordinates.
"""

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
import json

from ai.preprocessing.config import MRIPreprocessingConfig, DEFAULT_CHANNEL_ORDER
from ai.preprocessing.normalizer import MRIIntensityNormalizer, mri_normalizer
from ai.preprocessing.cropper import ForegroundCropper, CropCoordinates
from ai.preprocessing.registration import SpatialTransformer, spatial_transformer
from ai.preprocessing.label_handler import BraTSLabelHandler, label_handler
from ai.preprocessing.nifti_handler import NiftiHandler, nifti_handler

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class BraTSPreprocessedCase:
    """Preprocessed multi-modal volume, segmentation mask, and traceability metadata."""
    case_id: str
    image: Any  # Shape: [4, D, H, W]
    label: Optional[Any]  # Shape: [1, D, H, W] or None
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "has_label": self.label is not None,
            "image_shape": list(self.image.shape) if hasattr(self.image, "shape") else None,
            "label_shape": list(self.label.shape) if (self.label is not None and hasattr(self.label, "shape")) else None,
            "metadata": self.metadata
        }


class BraTSPreprocessingPipeline:
    """End-to-end preprocessing pipeline from raw NIfTI files to model-ready tensors."""

    def __init__(
        self,
        config: Optional[MRIPreprocessingConfig] = None,
        n_handler: Optional[NiftiHandler] = None,
        normalizer: Optional[MRIIntensityNormalizer] = None,
        transformer: Optional[SpatialTransformer] = None,
        cropper: Optional[ForegroundCropper] = None,
        lbl_handler: Optional[BraTSLabelHandler] = None
    ):
        self.config = config or MRIPreprocessingConfig()
        self.nifti_handler = n_handler or nifti_handler
        self.normalizer = normalizer or mri_normalizer
        self.transformer = transformer or spatial_transformer
        self.cropper = cropper or ForegroundCropper(
            margin=self.config.crop_margin,
            target_size=self.config.spatial_crop_size
        )
        self.label_handler = lbl_handler or label_handler

    def preprocess_case(
        self,
        modality_paths: Dict[str, Path],
        case_id: str,
        seg_path: Optional[Path] = None,
        split: str = "train",
        return_tensors: bool = True
    ) -> BraTSPreprocessedCase:
        """Executes the full preprocessing pipeline on a single patient case."""
        # 1. Validate required channels are provided
        for mod in self.config.channel_order:
            if mod not in modality_paths or not Path(modality_paths[mod]).is_file():
                raise FileNotFoundError(f"Case '{case_id}' missing required sequence '{mod}'")

        # 2. Load Raw NIfTI Volumes and Read Header Metadata
        raw_volumes: Dict[str, Any] = {}
        primary_mod = self.config.channel_order[0]
        meta = self.nifti_handler.read_header_metadata(Path(modality_paths[primary_mod]))

        orig_shape = meta["dimensions"]
        orig_spacing = meta["voxel_spacing"]
        orig_affine = meta["affine"]
        orig_orientation = meta.get("orientation", "UNKNOWN")

        for mod in self.config.channel_order:
            raw_volumes[mod] = self.nifti_handler.load_volume_array(Path(modality_paths[mod]))

        raw_seg = None
        if seg_path and Path(seg_path).is_file():
            raw_seg = self.nifti_handler.load_volume_array(Path(seg_path))

        # 3. Orientation Standardization (Standardize to RAS+)
        oriented_volumes: Dict[str, Any] = {}
        current_affine = orig_affine

        for mod in self.config.channel_order:
            ornt_vol, current_affine = self.transformer.reorient_to_ras(raw_volumes[mod], current_affine)
            oriented_volumes[mod] = ornt_vol

        oriented_seg = None
        if raw_seg is not None:
            oriented_seg, _ = self.transformer.reorient_to_ras(raw_seg, orig_affine)

        # 4. Spatial Resampling (to config.target_spacing, e.g. 1.0mm isotropic)
        resampled_volumes: Dict[str, Any] = {}
        resampled_shape = orig_shape
        current_spacing = orig_spacing

        if self.config.target_spacing is not None:
            target_sp = self.config.target_spacing
            for mod in self.config.channel_order:
                res_vol, resampled_shape = self.transformer.resample_to_spacing(
                    oriented_volumes[mod],
                    current_spacing=orig_spacing,
                    target_spacing=target_sp,
                    is_label=False  # Continuous trilinear interpolation
                )
                resampled_volumes[mod] = res_vol

            current_spacing = target_sp

            resampled_seg = None
            if oriented_seg is not None:
                # CRITICAL: Segmentation masks strictly use nearest-neighbor interpolation
                resampled_seg, _ = self.transformer.resample_to_spacing(
                    oriented_seg,
                    current_spacing=orig_spacing,
                    target_spacing=target_sp,
                    is_label=True  # Strictly Nearest Neighbor
                )
        else:
            resampled_volumes = oriented_volumes
            resampled_seg = oriented_seg

        # 5. Foreground Brain Cropping (Zero tumor tissue loss)
        vol_list = [resampled_volumes[m] for m in self.config.channel_order]
        crop_coords = self.cropper.compute_crop_coordinates(vol_list, seg_mask=resampled_seg)

        cropped_volumes: Dict[str, Any] = {}
        for mod in self.config.channel_order:
            cropped_volumes[mod] = self.cropper.apply_crop(resampled_volumes[mod], crop_coords)

        cropped_seg = None
        if resampled_seg is not None:
            cropped_seg = self.cropper.apply_crop(resampled_seg, crop_coords)

        # 6. Intensity Normalization (Z-score strictly on non-zero brain tissue)
        normalized_volumes: Dict[str, Any] = {}
        for mod in self.config.channel_order:
            normalized_volumes[mod] = self.normalizer.normalize_nonzero(
                cropped_volumes[mod],
                clip_percentiles=self.config.clip_percentiles
            )

        # 7. Channel Organization [C, D, H, W] where C=4 in order [T1, T1ce, T2, FLAIR]
        ordered_list = [normalized_volumes[mod] for mod in self.config.channel_order]

        if HAS_NUMPY:
            image_array = np.stack(ordered_list, axis=0).astype(np.float32)

            # 8. Segmentation Label Processing
            if cropped_seg is not None:
                if self.config.convert_labels_to_contiguous:
                    label_processed = self.label_handler.to_contiguous(cropped_seg)
                else:
                    label_processed = cropped_seg.copy()
                label_array = np.expand_dims(label_processed, axis=0).astype(np.int64)
            else:
                label_array = None

            # Optional PyTorch Tensor conversion
            if return_tensors and HAS_TORCH:
                final_image = torch.from_numpy(image_array)
                final_label = torch.from_numpy(label_array) if label_array is not None else None
            else:
                final_image = image_array
                final_label = label_array
        else:
            final_image = ordered_list
            final_label = cropped_seg

        # 9. Spatial Traceability Metadata Compilation
        traceability_metadata = {
            "case_id": case_id,
            "split": split,
            "channel_order": list(self.config.channel_order),
            "original_shape": list(orig_shape),
            "original_spacing": list(orig_spacing),
            "original_affine": orig_affine,
            "original_orientation": orig_orientation,
            "target_orientation": self.config.target_orientation,
            "target_spacing": list(self.config.target_spacing) if self.config.target_spacing else None,
            "resampled_shape": list(resampled_shape),
            "crop_coordinates": crop_coords.to_dict(),
            "preprocessed_shape": list(final_image.shape if hasattr(final_image, "shape") else ()),
            "convert_labels_to_contiguous": self.config.convert_labels_to_contiguous,
            "contiguous_label_map": self.config.contiguous_label_map,
            "inverse_label_map": self.config.inverse_label_map,
            "intensity_normalization": self.config.intensity_normalization,
            "clip_percentiles": list(self.config.clip_percentiles) if self.config.clip_percentiles else None,
            "mri_interpolation": self.config.mri_interpolation,
            "seg_interpolation": self.config.seg_interpolation
        }

        return BraTSPreprocessedCase(
            case_id=case_id,
            image=final_image,
            label=final_label,
            metadata=traceability_metadata
        )

    def map_prediction_to_original_space(
        self,
        predicted_mask: Any,
        metadata: Dict[str, Any]
    ) -> Any:
        """Projects a preprocessed segmentation mask back to original patient NIfTI space.

        Reverses:
          1. Contiguous class mapping (3 -> 4)
          2. Foreground bounding box crop (uncrop to resampled shape)
          3. Spatial resampling (nearest-neighbor back to original shape)
        """
        if not HAS_NUMPY:
            return predicted_mask

        # Remove channel dimension if [1, D, H, W]
        if hasattr(predicted_mask, "ndim") and predicted_mask.ndim == 4:
            mask = predicted_mask[0]
        else:
            mask = predicted_mask

        if HAS_TORCH and isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # Step 1: Revert contiguous mapping if applied
        if metadata.get("convert_labels_to_contiguous", True):
            mask = self.label_handler.to_original_brats(mask)

        # Step 2: Uncrop to resampled shape
        crop_dict = metadata.get("crop_coordinates")
        if crop_dict:
            coords = CropCoordinates.from_dict(crop_dict)
            mask = self.cropper.uncrop_volume(mask, coords, fill_value=0)

        # Step 3: Resample back to original resolution
        orig_shape = tuple(metadata["original_shape"])
        if mask.shape != orig_shape:
            mask = self.transformer.resample_to_original_shape(mask, orig_shape, is_label=True)

        return mask
