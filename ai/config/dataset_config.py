"""BraTS Dataset Configuration.

Defines environment-driven, non-hardcoded dataset paths, modality conventions,
segmentation label schemas, and reproducible patient split parameters.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import os
import json
from ai.config.paths import ROOT_DIR, DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR


# Standard BraTS multi-modal MRI sequences and ground-truth sub-regions
DEFAULT_MODALITIES: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair")
DEFAULT_LABEL_NAME: str = "seg"

# BraTS Tumor Sub-Region Label Mappings (BraTS 2020 / 2021 / 2023)
# Label 0: Background
# Label 1: NCR / NET (Necrotic core and non-enhancing tumor)
# Label 2: ED (Peritumoral edema)
# Label 4: ET (GD-enhancing tumor)
LABEL_MAPPINGS: Dict[int, str] = {
    0: "Background",
    1: "Necrotic Core (NCR)",
    2: "Peritumoral Edema (ED)",
    4: "Enhancing Tumor (ET)"
}

# Derived clinical evaluation regions for surgical planning
TARGET_REGIONS: Dict[str, List[int]] = {
    "WT": [1, 2, 4],  # Whole Tumor: all pathological tissue
    "TC": [1, 4],     # Tumor Core: necrotic core + enhancing tumor
    "ET": [4]         # Enhancing Tumor: active hyper-intense rim
}


class DatasetConfig:
    """Configurable parameters for BraTS dataset discovery, validation,

    indexing, and patient-level splitting.
    """

    def __init__(
        self,
        data_root: Optional[Path] = None,
        train_dir: Optional[Path] = None,
        val_dir: Optional[Path] = None,
        test_dir: Optional[Path] = None,
        dataset_name: str = "BraTS2021",
        modalities: Tuple[str, ...] = DEFAULT_MODALITIES,
        label_modality: str = DEFAULT_LABEL_NAME,
        allowed_extensions: Tuple[str, ...] = (".nii.gz", ".nii"),
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        index_output_dir: Optional[Path] = None,
        require_segmentation: bool = True
    ):
        # Resolve data_root from parameter, environment, or default project directory
        raw_env_root = os.getenv("BRATS_DATA_ROOT")
        if data_root is not None:
            self.data_root = Path(data_root)
        elif raw_env_root:
            self.data_root = Path(raw_env_root)
        else:
            self.data_root = RAW_DATA_DIR

        self.train_dir = Path(train_dir) if train_dir else None
        self.val_dir = Path(val_dir) if val_dir else None
        self.test_dir = Path(test_dir) if test_dir else None

        self.dataset_name = os.getenv("BRATS_DATASET_NAME", dataset_name)
        self.modalities = tuple(modalities)
        self.label_modality = label_modality
        self.allowed_extensions = tuple(allowed_extensions)

        # Splitting ratios (must sum to 1.0 within floating tolerance)
        self.train_ratio = float(os.getenv("BRATS_TRAIN_RATIO", train_ratio))
        self.val_ratio = float(os.getenv("BRATS_VAL_RATIO", val_ratio))
        self.test_ratio = float(os.getenv("BRATS_TEST_RATIO", test_ratio))
        self._validate_ratios()

        self.seed = int(os.getenv("BRATS_SEED", seed))

        # Output index storage
        if index_output_dir is not None:
            self.index_output_dir = Path(index_output_dir)
        else:
            self.index_output_dir = INTERIM_DATA_DIR

        self.require_segmentation = require_segmentation

    def _validate_ratios(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Split ratios must sum to 1.0. Given train={self.train_ratio}, "
                f"val={self.val_ratio}, test={self.test_ratio} (sum={total:.3f})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes dataset configuration to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "data_root": str(self.data_root),
            "train_dir": str(self.train_dir) if self.train_dir else None,
            "val_dir": str(self.val_dir) if self.val_dir else None,
            "test_dir": str(self.test_dir) if self.test_dir else None,
            "modalities": list(self.modalities),
            "label_modality": self.label_modality,
            "allowed_extensions": list(self.allowed_extensions),
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "index_output_dir": str(self.index_output_dir),
            "require_segmentation": self.require_segmentation,
            "label_mappings": LABEL_MAPPINGS,
            "target_regions": TARGET_REGIONS
        }

    @classmethod
    def from_json(cls, json_path: Path) -> "DatasetConfig":
        """Loads configuration from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            data_root=Path(data["data_root"]) if data.get("data_root") else None,
            train_dir=Path(data["train_dir"]) if data.get("train_dir") else None,
            val_dir=Path(data["val_dir"]) if data.get("val_dir") else None,
            test_dir=Path(data["test_dir"]) if data.get("test_dir") else None,
            dataset_name=data.get("dataset_name", "BraTS2021"),
            modalities=tuple(data.get("modalities", DEFAULT_MODALITIES)),
            label_modality=data.get("label_modality", DEFAULT_LABEL_NAME),
            allowed_extensions=tuple(data.get("allowed_extensions", (".nii.gz", ".nii"))),
            train_ratio=data.get("train_ratio", 0.70),
            val_ratio=data.get("val_ratio", 0.15),
            test_ratio=data.get("test_ratio", 0.15),
            seed=data.get("seed", 42),
            index_output_dir=Path(data["index_output_dir"]) if data.get("index_output_dir") else None,
            require_segmentation=data.get("require_segmentation", True)
        )
