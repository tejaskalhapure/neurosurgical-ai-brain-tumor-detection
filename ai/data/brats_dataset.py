"""BraTS PyTorch Dataset Implementation.

Provides multi-modal 3D MRI volume loading, ground-truth segmentation masks,
MONAI dictionary transform compatibility, and strict preservation of physical spatial metadata.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from ai.config.dataset_config import DatasetConfig, DEFAULT_MODALITIES
from ai.preprocessing.nifti_handler import NiftiHandler, nifti_handler

# Optional PyTorch & NumPy imports with graceful standard-library compatibility
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class TorchDataset:  # type: ignore
        """Minimal fallback for environments before PyTorch installation."""
        pass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class BraTSDataset(TorchDataset):
    """Multi-modal 3D BraTS PyTorch Dataset.

    Features:
      - Stacks multi-modal MRI sequences into a multi-channel tensor [C, D, H, W]
      - Preserves 4x4 affine matrix, voxel spacing (dx, dy, dz), and original shape
      - Compatible with MONAI dictionary-based transforms (e.g. MapTransform)
      - Does NOT apply irreversible downsampling or lossy conversions
    """

    def __init__(
        self,
        case_records: List[Any],
        modalities: Tuple[str, ...] = DEFAULT_MODALITIES,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        load_data: bool = True,
        require_segmentation: bool = True,
        n_handler: Optional[NiftiHandler] = None
    ):
        self.case_records = case_records
        self.modalities = tuple(modalities)
        self.transform = transform
        self.load_data = load_data
        self.require_segmentation = require_segmentation
        self.nifti_handler = n_handler or nifti_handler

    def __len__(self) -> int:
        return len(self.case_records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.case_records[idx]

        # Extract record attributes whether record is a CaseIndexRecord or dict
        if isinstance(record, dict):
            case_id = record["case_id"]
            mod_paths = {
                mod: Path(record[f"{mod}_path"])
                for mod in self.modalities
                if record.get(f"{mod}_path")
            }
            seg_path = Path(record["seg_path"]) if record.get("seg_path") else None
        else:
            case_id = record.case_id
            mod_paths = {}
            for mod in self.modalities:
                p = getattr(record, f"{mod}_path", None)
                if p:
                    mod_paths[mod] = Path(p)
            seg_path = Path(record.seg_path) if getattr(record, "seg_path", None) else None

        # 1. Read spatial metadata from the first available modality
        sample_mod_path = next(iter(mod_paths.values()))
        meta = self.nifti_handler.read_header_metadata(sample_mod_path)

        item: Dict[str, Any] = {
            "case_id": case_id,
            "affine": meta["affine"],
            "voxel_spacing": meta["voxel_spacing"],
            "original_shape": meta["dimensions"],
            "orientation": meta.get("orientation", "UNKNOWN"),
            "modality_paths": {k: str(v) for k, v in mod_paths.items()},
            "segmentation_path": str(seg_path) if seg_path else None
        }

        # 2. If load_data is disabled (e.g. metadata-only indexing), return early
        if not self.load_data:
            if self.transform is not None:
                item = self.transform(item)
            return item

        # 3. Load MRI volumes
        mod_arrays = []
        for mod in self.modalities:
            if mod not in mod_paths:
                raise KeyError(f"Case '{case_id}' missing expected modality '{mod}'")
            arr = self.nifti_handler.load_volume_array(mod_paths[mod])
            mod_arrays.append(arr)

        if HAS_NUMPY:
            # Stack modalities along channel dimension: [C, D, H, W]
            image_np = np.stack(mod_arrays, axis=0).astype(np.float32)

            if seg_path and seg_path.is_file():
                label_arr = self.nifti_handler.load_volume_array(seg_path)
                # Expand label to [1, D, H, W]
                label_np = np.expand_dims(label_arr, axis=0).astype(np.int64)
            else:
                label_np = None

            if HAS_TORCH:
                item["image"] = torch.from_numpy(image_np)
                item["label"] = torch.from_numpy(label_np) if label_np is not None else None
                item["affine"] = torch.tensor(meta["affine"], dtype=torch.float32)
            else:
                item["image"] = image_np
                item["label"] = label_np
        else:
            # Fallback if numpy not installed
            item["image"] = mod_arrays
            item["label"] = None

        # 4. Apply MONAI / custom dictionary transforms if provided
        if self.transform is not None:
            item = self.transform(item)

        return item
