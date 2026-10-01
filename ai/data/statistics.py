"""BraTS Dataset Statistics Analyzer.

Computes comprehensive diagnostic metrics across the cohort:
  - Valid vs invalid case counts
  - Modality availability percentages
  - Volume dimensional distributions
  - Physical voxel spacing summary (mean, min, max)
  - Ground-truth label availability
"""

from typing import Dict, List, Optional, Tuple, Any
import json
from collections import Counter
from ai.data.dataset_indexer import CaseIndexRecord
from ai.data.validator import DatasetValidationSummary


class BraTSDatasetStatistics:
    """Calculates summary statistics across indexed BraTS cohorts."""

    @staticmethod
    def compute_summary(
        records: List[CaseIndexRecord],
        validation_summary: Optional[DatasetValidationSummary] = None
    ) -> Dict[str, Any]:
        """Calculates cohort-level statistics from indexed records."""
        total = len(records)
        if total == 0:
            return {
                "total_cases": 0,
                "valid_cases": 0,
                "invalid_cases": 0,
                "valid_percentage": 0.0,
                "modality_availability": {},
                "dimensions": {},
                "voxel_spacing": {},
                "orientations": {},
                "segmentation_availability": 0.0
            }

        valid_count = sum(1 for r in records if r.is_valid)
        invalid_count = total - valid_count

        # 1. Modality Availability
        mod_counts: Counter = Counter()
        seg_count = 0
        for r in records:
            if r.has_segmentation:
                seg_count += 1
            if r.modalities_available:
                for mod in r.modalities_available.split(","):
                    if mod.strip():
                        mod_counts[mod.strip()] += 1

        modality_percentages = {
            mod: round((cnt / total) * 100.0, 2)
            for mod, cnt in mod_counts.items()
        }

        # 2. Dimensional Distributions (Valid Cases)
        dim_counter: Counter = Counter()
        spacings_x, spacings_y, spacings_z = [], [], []
        orientations: Counter = Counter()

        for r in records:
            if r.dim_x is not None and r.dim_y is not None and r.dim_z is not None:
                dim_str = f"{r.dim_x} x {r.dim_y} x {r.dim_z}"
                dim_counter[dim_str] += 1

            if r.spacing_x is not None and r.spacing_y is not None and r.spacing_z is not None:
                spacings_x.append(r.spacing_x)
                spacings_y.append(r.spacing_y)
                spacings_z.append(r.spacing_z)

            if r.orientation:
                orientations[r.orientation] += 1

        # 3. Voxel Spacing Stats
        def stats(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"mean": 0.0, "min": 0.0, "max": 0.0}
            mean = sum(vals) / len(vals)
            return {
                "mean": round(mean, 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4)
            }

        spacing_summary = {
            "dx_mm": stats(spacings_x),
            "dy_mm": stats(spacings_y),
            "dz_mm": stats(spacings_z)
        }

        summary = {
            "total_cases": total,
            "valid_cases": valid_count,
            "invalid_cases": invalid_count,
            "valid_percentage": round((valid_count / total) * 100.0, 2),
            "segmentation_available_count": seg_count,
            "segmentation_percentage": round((seg_count / total) * 100.0, 2),
            "modality_availability": modality_percentages,
            "dimension_distributions": dict(dim_counter.most_common(5)),
            "voxel_spacing_summary": spacing_summary,
            "orientation_distributions": dict(orientations.most_common(5))
        }

        if validation_summary:
            summary["common_validation_errors"] = validation_summary.common_errors

        return summary

    @staticmethod
    def format_text_report(stats: Dict[str, Any]) -> str:
        """Renders an ASCII text report suitable for CLI terminal output."""
        lines = [
            "=" * 70,
            "              BraTS DATASET INGESTION & VALIDATION REPORT",
            "=" * 70,
            f" Total Cases Scanned        : {stats['total_cases']}",
            f" Valid Cases                : {stats['valid_cases']} ({stats.get('valid_percentage', 0.0)}%)",
            f" Invalid Cases              : {stats['invalid_cases']}",
            f" Cases with Segmentation    : {stats['segmentation_available_count']} ({stats.get('segmentation_percentage', 0.0)}%)",
            "-" * 70,
            " Modality Availability:"
        ]

        for mod, pct in stats.get("modality_availability", {}).items():
            lines.append(f"   - {mod:<10}: {pct:>6.2f}%")

        lines.extend([
            "-" * 70,
            " Spatial Dimensions Observed:"
        ])
        for dim, count in stats.get("dimension_distributions", {}).items():
            lines.append(f"   - {dim:<20}: {count} cases")

        sp = stats.get("voxel_spacing_summary", {})
        lines.extend([
            "-" * 70,
            " Voxel Spacing (dx, dy, dz):",
            f"   - dx (mm) : mean={sp.get('dx_mm', {}).get('mean', 0.0)}, min={sp.get('dx_mm', {}).get('min', 0.0)}, max={sp.get('dx_mm', {}).get('max', 0.0)}",
            f"   - dy (mm) : mean={sp.get('dy_mm', {}).get('mean', 0.0)}, min={sp.get('dy_mm', {}).get('min', 0.0)}, max={sp.get('dy_mm', {}).get('max', 0.0)}",
            f"   - dz (mm) : mean={sp.get('dz_mm', {}).get('mean', 0.0)}, min={sp.get('dz_mm', {}).get('min', 0.0)}, max={sp.get('dz_mm', {}).get('max', 0.0)}",
            "-" * 70,
            " Orientations:"
        ])
        for ori, cnt in stats.get("orientation_distributions", {}).items():
            lines.append(f"   - {ori}: {cnt} cases")

        if stats.get("common_validation_errors"):
            lines.extend([
                "-" * 70,
                " Validation Issues Encountered:"
            ])
            for err, cnt in stats["common_validation_errors"].items():
                lines.append(f"   [!] {err}: {cnt} occurrence(s)")

        lines.append("=" * 70)
        return "\n".join(lines)
