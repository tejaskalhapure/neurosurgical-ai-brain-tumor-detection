"""BraTS Dataset Validator.

Performs strict, comprehensive clinical and computational validation across
multi-modal MRI scans and segmentation masks without silently discarding errors.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from ai.config.dataset_config import DatasetConfig, DEFAULT_MODALITIES
from ai.data.dataset_scanner import ScannedCase, BraTSDatasetScanner
from ai.preprocessing.nifti_handler import NiftiHandler, CorruptedNiftiError


@dataclass
class CaseValidationResult:
    """Validation outcome and extracted metadata for an individual patient case."""
    case_id: str
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dimensions: Optional[Tuple[int, int, int]] = None
    voxel_spacing: Optional[Tuple[float, float, float]] = None
    orientation: Optional[str] = None
    affine: Optional[List[List[float]]] = None
    modality_shapes: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    has_segmentation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "dimensions": list(self.dimensions) if self.dimensions else None,
            "voxel_spacing": list(self.voxel_spacing) if self.voxel_spacing else None,
            "orientation": self.orientation,
            "has_segmentation": self.has_segmentation,
            "modality_shapes": {k: list(v) for k, v in self.modality_shapes.items()}
        }


@dataclass
class DatasetValidationSummary:
    """Aggregated validation statistics across the entire cohort."""
    total_cases_scanned: int
    valid_cases_count: int
    invalid_cases_count: int
    validation_passed: bool
    case_results: List[CaseValidationResult] = field(default_factory=list)
    invalid_cases: List[CaseValidationResult] = field(default_factory=list)
    common_errors: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases_scanned": self.total_cases_scanned,
            "valid_cases_count": self.valid_cases_count,
            "invalid_cases_count": self.invalid_cases_count,
            "validation_passed": self.validation_passed,
            "invalid_cases": [c.to_dict() for c in self.invalid_cases],
            "common_errors": self.common_errors
        }


class BraTSDatasetValidator:
    """Validates multi-modal MRI NIfTI volumes, header metadata, and segmentation masks."""

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        nifti_handler: Optional[NiftiHandler] = None
    ):
        self.config = config or DatasetConfig()
        self.nifti_handler = nifti_handler or NiftiHandler()

    def validate_case(self, case: ScannedCase) -> CaseValidationResult:
        """Runs full validation checks on a single patient case."""
        errors: List[str] = []
        warnings: List[str] = []
        modality_shapes: Dict[str, Tuple[int, int, int]] = {}
        modality_spacings: Dict[str, Tuple[float, float, float]] = {}

        primary_dim: Optional[Tuple[int, int, int]] = None
        primary_spacing: Optional[Tuple[float, float, float]] = None
        primary_affine: Optional[List[List[float]]] = None
        primary_orientation: Optional[str] = None

        # 1. Verify required modalities are present
        for mod in self.config.modalities:
            if mod not in case.modality_paths:
                errors.append(f"Missing required modality: '{mod}'")

        # 2. Verify segmentation mask if required
        if self.config.require_segmentation and not case.segmentation_path:
            errors.append("Missing required segmentation mask ('seg')")

        # 3. Open each available modality and inspect header metadata
        for mod, path in case.modality_paths.items():
            if not path.is_file():
                errors.append(f"Modality file not found on disk: {path}")
                continue

            try:
                meta = self.nifti_handler.read_header_metadata(path)
                dims = meta["dimensions"]
                spacing = meta["voxel_spacing"]

                modality_shapes[mod] = dims
                modality_spacings[mod] = spacing

                if primary_dim is None:
                    primary_dim = dims
                    primary_spacing = spacing
                    primary_affine = meta["affine"]
                    primary_orientation = meta["orientation"]
                else:
                    # Check shape consistency across modalities for the same patient
                    if dims != primary_dim:
                        errors.append(
                            f"Dimension mismatch in modality '{mod}': shape {dims} "
                            f"does not match primary shape {primary_dim}"
                        )
                    # Check spacing consistency
                    if spacing != primary_spacing:
                        warnings.append(
                            f"Voxel spacing discrepancy in '{mod}': {spacing} vs {primary_spacing}"
                        )
            except CorruptedNiftiError as e:
                errors.append(f"Corrupted or invalid NIfTI file in '{mod}': {e}")
            except Exception as e:
                errors.append(f"Failed to inspect modality '{mod}' ({path.name}): {e}")

        # 4. Open and validate segmentation mask if present
        if case.segmentation_path and case.segmentation_path.is_file():
            try:
                seg_meta = self.nifti_handler.read_header_metadata(case.segmentation_path)
                seg_dims = seg_meta["dimensions"]

                if primary_dim is not None and seg_dims != primary_dim:
                    errors.append(
                        f"Segmentation dimension mismatch: mask shape {seg_dims} "
                        f"does not correspond to MRI volume shape {primary_dim}"
                    )
            except CorruptedNiftiError as e:
                errors.append(f"Corrupted segmentation file ({case.segmentation_path.name}): {e}")
            except Exception as e:
                errors.append(f"Failed to inspect segmentation file: {e}")

        is_valid = (len(errors) == 0)

        return CaseValidationResult(
            case_id=case.case_id,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            dimensions=primary_dim,
            voxel_spacing=primary_spacing,
            orientation=primary_orientation,
            affine=primary_affine,
            modality_shapes=modality_shapes,
            has_segmentation=(case.segmentation_path is not None)
        )

    def validate_dataset(self, cases: List[ScannedCase]) -> DatasetValidationSummary:
        """Validates all scanned cases across the dataset and checks for duplicate IDs."""
        case_results: List[CaseValidationResult] = []
        invalid_cases: List[CaseValidationResult] = []
        error_frequency: Dict[str, int] = {}
        seen_case_ids: Set[str] = set()

        for case in cases:
            res = self.validate_case(case)

            # Check for duplicate case identifiers
            if case.case_id in seen_case_ids:
                res.is_valid = False
                res.errors.append(f"Duplicate case ID detected: '{case.case_id}'")
            else:
                seen_case_ids.add(case.case_id)

            if not res.is_valid:
                invalid_cases.append(res)
                for err in res.errors:
                    # Strip specific filenames to aggregate common error categories
                    category = err.split(":")[0] if ":" in err else err
                    error_frequency[category] = error_frequency.get(category, 0) + 1

            case_results.append(res)

        valid_count = sum(1 for r in case_results if r.is_valid)
        invalid_count = len(invalid_cases)
        all_passed = (invalid_count == 0 and len(case_results) > 0)

        return DatasetValidationSummary(
            total_cases_scanned=len(cases),
            valid_cases_count=valid_count,
            invalid_cases_count=invalid_count,
            validation_passed=all_passed,
            case_results=case_results,
            invalid_cases=invalid_cases,
            common_errors=error_frequency
        )
