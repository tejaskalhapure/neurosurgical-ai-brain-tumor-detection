"""BraTS Dataset Indexer.

Constructs structured index representations (CSV, JSON, Pandas DataFrame)
mapping patient IDs to modality filepaths, validity statuses, and physical spatial metadata.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import csv
import json
from dataclasses import dataclass, asdict
from ai.config.dataset_config import DatasetConfig, DEFAULT_MODALITIES
from ai.data.dataset_scanner import ScannedCase, BraTSDatasetScanner
from ai.data.validator import BraTSDatasetValidator, CaseValidationResult, DatasetValidationSummary

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class CaseIndexRecord:
    """Flat representation of an indexed patient case for CSV/DataFrame serialization."""
    case_id: str
    t1_path: Optional[str]
    t1ce_path: Optional[str]
    t2_path: Optional[str]
    flair_path: Optional[str]
    seg_path: Optional[str]
    modalities_available: str
    num_modalities: int
    has_all_modalities: bool
    has_segmentation: bool
    is_valid: bool
    validation_errors: str
    dim_x: Optional[int]
    dim_y: Optional[int]
    dim_z: Optional[int]
    spacing_x: Optional[float]
    spacing_y: Optional[float]
    spacing_z: Optional[float]
    orientation: Optional[str]
    affine_summary: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BraTSDatasetIndexer:
    """Indexes BraTS cohorts and exports structured tabular indexes."""

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        scanner: Optional[BraTSDatasetScanner] = None,
        validator: Optional[BraTSDatasetValidator] = None
    ):
        self.config = config or DatasetConfig()
        self.scanner = scanner or BraTSDatasetScanner(self.config)
        self.validator = validator or BraTSDatasetValidator(self.config)

    def build_index(self, data_root: Optional[Path] = None) -> Tuple[List[CaseIndexRecord], DatasetValidationSummary]:
        """Scans, validates, and builds an index for all cases in data_root."""
        scanned_cases = self.scanner.scan(data_root)
        validation_summary = self.validator.validate_dataset(scanned_cases)

        validation_map: Dict[str, CaseValidationResult] = {
            r.case_id: r for r in validation_summary.case_results
        }

        records: List[CaseIndexRecord] = []
        for case in scanned_cases:
            val_res = validation_map.get(case.case_id)
            is_valid = val_res.is_valid if val_res else False
            errors = "; ".join(val_res.errors) if val_res and val_res.errors else ""

            dims = val_res.dimensions if val_res else None
            spacing = val_res.voxel_spacing if val_res else None
            orientation = val_res.orientation if val_res else None
            affine = val_res.affine if val_res else None

            affine_summary = json.dumps(affine) if affine else None

            available_mods = sorted(list(case.modality_paths.keys()))

            record = CaseIndexRecord(
                case_id=case.case_id,
                t1_path=str(case.modality_paths.get("t1")) if "t1" in case.modality_paths else None,
                t1ce_path=str(case.modality_paths.get("t1ce")) if "t1ce" in case.modality_paths else None,
                t2_path=str(case.modality_paths.get("t2")) if "t2" in case.modality_paths else None,
                flair_path=str(case.modality_paths.get("flair")) if "flair" in case.modality_paths else None,
                seg_path=str(case.segmentation_path) if case.segmentation_path else None,
                modalities_available=",".join(available_mods),
                num_modalities=len(available_mods),
                has_all_modalities=(len(available_mods) == len(self.config.modalities)),
                has_segmentation=case.has_segmentation,
                is_valid=is_valid,
                validation_errors=errors,
                dim_x=dims[0] if dims else None,
                dim_y=dims[1] if dims else None,
                dim_z=dims[2] if dims else None,
                spacing_x=spacing[0] if spacing else None,
                spacing_y=spacing[1] if spacing else None,
                spacing_z=spacing[2] if spacing else None,
                orientation=orientation,
                affine_summary=affine_summary
            )
            records.append(record)

        return records, validation_summary

    def export_csv(self, records: List[CaseIndexRecord], output_path: Path) -> Path:
        """Exports index records to a standard CSV file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if not records:
            # Write header only
            fieldnames = list(CaseIndexRecord.__annotations__.keys())
            with open(out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return out

        fieldnames = list(records[0].to_dict().keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r.to_dict())

        return out

    def export_json(self, records: List[CaseIndexRecord], output_path: Path) -> Path:
        """Exports index records to JSON format."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in records]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return out

    def to_dataframe(self, records: List[CaseIndexRecord]) -> Any:
        """Returns a Pandas DataFrame if pandas is installed, otherwise a list of dicts."""
        dicts = [r.to_dict() for r in records]
        if HAS_PANDAS:
            return pd.DataFrame(dicts)
        return dicts
