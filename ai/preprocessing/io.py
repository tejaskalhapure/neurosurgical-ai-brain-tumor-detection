from __future__ import annotations
"""I/O utilities for saving, loading, and caching preprocessed BraTS cases and metadata."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
from ai.preprocessing.pipeline import BraTSPreprocessedCase, BraTSPreprocessingPipeline
from ai.preprocessing.config import MRIPreprocessingConfig

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


def save_preprocessed_case(
    case: BraTSPreprocessedCase,
    output_dir: Path
) -> Tuple[Path, Path]:
    """Saves preprocessed 4D image array and metadata to disk."""
    out = Path(output_dir) / case.case_id
    out.mkdir(parents=True, exist_ok=True)

    meta_file = out / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(case.metadata, f, indent=2)

    data_file = out / "tensors.npz"
    if HAS_NUMPY:
        img_arr = case.image.cpu().numpy() if (HAS_TORCH and isinstance(case.image, torch.Tensor)) else case.image
        save_dict = {"image": img_arr}
        if case.label is not None:
            lbl_arr = case.label.cpu().numpy() if (HAS_TORCH and isinstance(case.label, torch.Tensor)) else case.label
            save_dict["label"] = lbl_arr
        np.savez_compressed(data_file, **save_dict)

    return data_file, meta_file


def load_preprocessed_case(
    case_dir: Path,
    as_tensors: bool = True
) -> BraTSPreprocessedCase:
    """Loads a saved preprocessed case from disk."""
    folder = Path(case_dir)
    meta_file = folder / "metadata.json"
    data_file = folder / "tensors.npz"

    if not meta_file.is_file() or not data_file.is_file():
        raise FileNotFoundError(f"Missing preprocessed artifacts in {case_dir}")

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    loaded = np.load(data_file)
    image = loaded["image"]
    label = loaded["label"] if "label" in loaded else None

    if as_tensors and HAS_TORCH:
        image = torch.from_numpy(image)
        label = torch.from_numpy(label) if label is not None else None

    return BraTSPreprocessedCase(
        case_id=metadata["case_id"],
        image=image,
        label=label,
        metadata=metadata
    )


def preprocess_dataset(
    case_records: List[Any],
    output_dir: Path,
    config: Optional[MRIPreprocessingConfig] = None,
    split: str = "train"
) -> List[Dict[str, Any]]:
    """Runs batch preprocessing across a cohort and saves metadata manifests."""
    pipeline = BraTSPreprocessingPipeline(config=config)
    results = []

    for record in case_records:
        case_id = record.case_id if hasattr(record, "case_id") else record["case_id"]
        mod_paths = {}
        for mod in (config or MRIPreprocessingConfig()).channel_order:
            val = getattr(record, f"{mod}_path", None) if hasattr(record, f"{mod}_path") else record.get(f"{mod}_path")
            if val:
                mod_paths[mod] = Path(val)

        seg_val = getattr(record, "seg_path", None) if hasattr(record, "seg_path") else record.get("seg_path")
        seg_path = Path(seg_val) if seg_val else None

        preprocessed = pipeline.preprocess_case(
            modality_paths=mod_paths,
            case_id=case_id,
            seg_path=seg_path,
            split=split
        )
        save_preprocessed_case(preprocessed, output_dir)
        results.append(preprocessed.to_dict())

    return results
