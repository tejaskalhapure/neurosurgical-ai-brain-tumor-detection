"""BraTS Dataset Pipeline CLI.

Command-line interface to scan, validate, index, and split BraTS cohorts.

Usage:
    python -m ai.data.cli --data-root /path/to/BraTS2021 --output-dir data/interim
"""

import argparse
import sys
from pathlib import Path
import json

from ai.config.dataset_config import DatasetConfig
from ai.data.dataset_scanner import BraTSDatasetScanner
from ai.data.validator import BraTSDatasetValidator
from ai.data.dataset_indexer import BraTSDatasetIndexer
from ai.data.split import PatientLevelSplitter
from ai.data.statistics import BraTSDatasetStatistics


def main():
    parser = argparse.ArgumentParser(
        description="BraTS Dataset Ingestion, Validation, Indexing, and Splitting Pipeline"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to BraTS dataset root directory (or defaults to BRATS_DATA_ROOT / data/raw)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/interim",
        help="Output directory for generated indexes and manifests"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="BraTS2021",
        help="BraTS dataset label/version (default: BraTS2021)"
    )
    parser.add_argument(
        "--require-seg",
        action="store_true",
        default=True,
        help="Enforce ground-truth segmentation existence (default: True)"
    )
    parser.add_argument(
        "--no-require-seg",
        dest="require_seg",
        action="store_false",
        help="Allow scans without ground truth masks (e.g. hidden test splits)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient-level split (default: 42)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of cases for training (default: 0.70)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of cases for validation (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of cases for testing (default: 0.15)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  BraTS DATASET INGESTION, VALIDATION & INDEXING PIPELINE")
    print("=" * 70)

    # 1. Initialize configuration
    cfg = DatasetConfig(
        data_root=Path(args.data_root) if args.data_root else None,
        dataset_name=args.dataset_name,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        require_segmentation=args.require_seg,
        index_output_dir=Path(args.output_dir)
    )

    print(f"[*] Dataset Root       : {cfg.data_root}")
    print(f"[*] Output Directory   : {cfg.index_output_dir}")
    print(f"[*] Expected Modalities: {cfg.modalities}")
    print(f"[*] Require Seg Label  : {cfg.require_segmentation}")
    print(f"[*] Split Ratios       : {cfg.train_ratio} / {cfg.val_ratio} / {cfg.test_ratio} (Seed: {cfg.seed})")
    print("-" * 70)

    if not cfg.data_root.exists():
        print(f"[!] Error: Data root '{cfg.data_root}' does not exist on disk.")
        print("    Please set --data-root to your actual BraTS dataset directory,")
        print("    or set the environment variable: BRATS_DATA_ROOT=/path/to/dataset")
        sys.exit(1)

    # 2. Build index and validate
    print("[*] Scanning and validating dataset cases...")
    indexer = BraTSDatasetIndexer(config=cfg)
    records, validation_summary = indexer.build_index()

    # 3. Export structured index files
    out_dir = Path(cfg.index_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "brats_index.csv"
    json_path = out_dir / "brats_index.json"
    indexer.export_csv(records, csv_path)
    indexer.export_json(records, json_path)
    print(f"[+] Structured index exported to : {csv_path}")
    print(f"[+] Detailed JSON index exported to: {json_path}")

    # 4. Perform patient-level split on valid cases
    valid_case_ids = [r.case_id for r in records if r.is_valid]
    print(f"[*] Performing patient-level splitting on {len(valid_case_ids)} valid cases...")
    splitter = PatientLevelSplitter(config=cfg)
    split = splitter.split_cases(valid_case_ids)
    manifest_path = out_dir / "split_manifest.json"
    splitter.save_split_manifest(split, manifest_path)
    print(f"[+] Patient split manifest saved: {manifest_path}")
    print(f"    - Train : {len(split.train_cases)} cases")
    print(f"    - Val   : {len(split.val_cases)} cases")
    print(f"    - Test  : {len(split.test_cases)} cases")

    # 5. Compute & display cohort statistics
    stats = BraTSDatasetStatistics.compute_summary(records, validation_summary)
    stats_path = out_dir / "dataset_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    report_text = BraTSDatasetStatistics.format_text_report(stats)
    print("\n" + report_text)
    print(f"\n[+] Full pipeline completed successfully. Artifacts stored in: {out_dir}\n")


if __name__ == "__main__":
    main()
