"""BraTS Dataset Scanner.

Scans dataset directories to identify patient/case folders and multi-modal NIfTI files
without hardcoded paths or assumptions about a single BraTS release.
Supports BraTS 2021, BraTS 2020, BraTS 2023, and custom directory layouts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import re
from dataclasses import dataclass, field
from ai.config.dataset_config import DatasetConfig, DEFAULT_MODALITIES, DEFAULT_LABEL_NAME


@dataclass
class ScannedCase:
    """Representation of an identified BraTS patient case on disk."""
    case_id: str
    case_dir: Optional[Path]
    modality_paths: Dict[str, Path] = field(default_factory=dict)
    segmentation_path: Optional[Path] = None
    all_files: List[Path] = field(default_factory=list)
    has_segmentation: bool = False
    is_complete: bool = False
    missing_modalities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_dir": str(self.case_dir) if self.case_dir else None,
            "modality_paths": {k: str(v) for k, v in self.modality_paths.items()},
            "segmentation_path": str(self.segmentation_path) if self.segmentation_path else None,
            "has_segmentation": self.has_segmentation,
            "is_complete": self.is_complete,
            "missing_modalities": self.missing_modalities
        }


class BraTSDatasetScanner:
    """Scans and indexes raw patient directories and multi-modal MRI scans."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.expected_modalities = set(self.config.modalities)
        self.label_name = self.config.label_modality
        self.allowed_extensions = self.config.allowed_extensions

    def scan(self, data_root: Optional[Path] = None) -> List[ScannedCase]:
        """Scans the dataset directory and returns a list of ScannedCase objects.

        Handles both directory-per-case structures (standard BraTS)
        and flat directory structures with case-prefixed files.
        """
        root = Path(data_root or self.config.data_root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

        cases_map: Dict[str, ScannedCase] = {}

        # 1. First search for case subdirectories (Standard BraTS release convention)
        subdirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]

        if subdirs:
            for subdir in sorted(subdirs):
                case_id = subdir.name
                files = self._collect_nifti_files(subdir)
                if files:
                    case = self._parse_case_files(case_id=case_id, case_dir=subdir, files=files)
                    cases_map[case_id] = case

        # 2. Also search root directly in case files are stored in a flat directory
        flat_files = self._collect_nifti_files(root)
        if flat_files:
            # Group flat files by inferred case_id prefix
            grouped_flat = self._group_flat_files(flat_files)
            for case_id, files in grouped_flat.items():
                if case_id not in cases_map:
                    case = self._parse_case_files(case_id=case_id, case_dir=root, files=files)
                    cases_map[case_id] = case

        # Sort by case_id for determinism
        return [cases_map[cid] for cid in sorted(cases_map.keys())]

    def _collect_nifti_files(self, directory: Path) -> List[Path]:
        """Finds all NIfTI files (.nii, .nii.gz) in a single directory."""
        files = []
        for ext in self.allowed_extensions:
            # Non-recursive inside case directory
            files.extend(directory.glob(f"*{ext}"))
        # Remove duplicates if .nii.gz and .nii matched overlapping names
        seen = set()
        unique_files = []
        for f in files:
            if f.name not in seen:
                seen.add(f.name)
                unique_files.append(f)
        return sorted(unique_files)

    def _parse_case_files(self, case_id: str, case_dir: Path, files: List[Path]) -> ScannedCase:
        """Categorizes files into modalities ('t1', 't1ce', 't2', 'flair') and segmentation."""
        modality_paths: Dict[str, Path] = {}
        segmentation_path: Optional[Path] = None

        for file_path in files:
            name_lower = file_path.name.lower()

            # Check if it's the segmentation mask
            # e.g., *_seg.nii, *_seg.nii.gz, *_mask.nii
            if self._is_segmentation_file(name_lower):
                segmentation_path = file_path
                continue

            # Check each modality
            matched_mod = self._identify_modality(name_lower)
            if matched_mod:
                modality_paths[matched_mod] = file_path

        missing = sorted(list(self.expected_modalities - set(modality_paths.keys())))
        has_seg = (segmentation_path is not None)
        is_complete = (len(missing) == 0) and (not self.config.require_segmentation or has_seg)

        return ScannedCase(
            case_id=case_id,
            case_dir=case_dir,
            modality_paths=modality_paths,
            segmentation_path=segmentation_path,
            all_files=files,
            has_segmentation=has_seg,
            is_complete=is_complete,
            missing_modalities=missing
        )

    def _identify_modality(self, filename: str) -> Optional[str]:
        """Identifies standard BraTS sequence modalities from filename suffix.

        Handles:
          - 't1ce' or 't1-ce' or 't1gd' -> 't1ce'
          - 'flair' -> 'flair'
          - 't2' -> 't2'
          - 't1' -> 't1' (ensures t1ce is not misclassified as t1)
        """
        # Remove extension
        stem = filename
        for ext in self.allowed_extensions:
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
                break

        # Priority check: t1ce must precede t1
        if stem.endswith("t1ce") or stem.endswith("t1-ce") or stem.endswith("t1_ce") or stem.endswith("t1gd"):
            return "t1ce"
        if stem.endswith("flair"):
            return "flair"
        if stem.endswith("t2"):
            return "t2"
        if stem.endswith("t1"):
            return "t1"

        # Regex fallback for embedded tokens, e.g. BraTS2021_00000_t1ce_raw
        if re.search(r"(?:_|-|^)t1ce(?:_|-|$)", stem) or re.search(r"(?:_|-|^)t1gd(?:_|-|$)", stem):
            return "t1ce"
        if re.search(r"(?:_|-|^)flair(?:_|-|$)", stem):
            return "flair"
        if re.search(r"(?:_|-|^)t2(?:_|-|$)", stem):
            return "t2"
        if re.search(r"(?:_|-|^)t1(?:_|-|$)", stem):
            return "t1"

        return None

    def _is_segmentation_file(self, filename: str) -> bool:
        """Determines if a filename represents ground-truth segmentation."""
        stem = filename
        for ext in self.allowed_extensions:
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
                break

        if stem.endswith("seg") or stem.endswith("mask") or stem.endswith("gts"):
            return True
        if re.search(r"(?:_|-|^)seg(?:_|-|$)", stem):
            return True
        return False

    def _group_flat_files(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Groups flat NIfTI files by patient case prefix."""
        groups: Dict[str, List[Path]] = {}
        for f in files:
            case_id = self._infer_case_id(f.name)
            if case_id:
                groups.setdefault(case_id, []).append(f)
        return groups

    def _infer_case_id(self, filename: str) -> Optional[str]:
        """Infers the case ID prefix from standard BraTS filename patterns.

        Examples:
          - BraTS2021_00000_flair.nii.gz -> BraTS2021_00000
          - BraTS20_Training_001_t1.nii -> BraTS20_Training_001
          - BraTS-GLI-00001-000_t1ce.nii.gz -> BraTS-GLI-00001-000
        """
        for token in ["_flair", "_t1ce", "_t1", "_t2", "_seg", "-flair", "-t1ce", "-t1", "-t2", "-seg"]:
            idx = filename.find(token)
            if idx > 0:
                return filename[:idx]
        return None
