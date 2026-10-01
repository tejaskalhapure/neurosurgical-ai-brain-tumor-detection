"""Module 2 Standalone Test Runner.

Executes all Module 2 test suites using Python's built-in unittest framework
to guarantee execution even prior to external test runner installations.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ai.config.dataset_config import DatasetConfig
from ai.data.dataset_scanner import BraTSDatasetScanner
from ai.data.validator import BraTSDatasetValidator
from ai.data.dataset_indexer import BraTSDatasetIndexer
from ai.data.split import PatientLevelSplitter
from ai.data.brats_dataset import BraTSDataset
from ai.data.statistics import BraTSDatasetStatistics
from tests.utils import write_test_nifti


class TestBraTSPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="brats_test_runner_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discovery_and_metadata(self):
        """Tests discovery of case directories and reading valid NIfTI header metadata."""
        case_dir = self.temp_dir / "BraTS2021_00001"
        case_dir.mkdir(parents=True)
        for mod in ["flair", "t1", "t1ce", "t2", "seg"]:
            write_test_nifti(
                case_dir / f"BraTS2021_00001_{mod}.nii.gz",
                dimensions=(64, 64, 32),
                voxel_spacing=(1.0, 1.0, 1.0)
            )

        cfg = DatasetConfig(data_root=self.temp_dir)
        scanner = BraTSDatasetScanner(cfg)
        cases = scanner.scan()
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "BraTS2021_00001")
        self.assertTrue(cases[0].is_complete)

    def test_invalid_file_handling(self):
        """Tests detection of shape mismatches and corrupted files."""
        case_dir = self.temp_dir / "BraTS2021_00002"
        case_dir.mkdir(parents=True)
        # flair shape
        write_test_nifti(case_dir / "BraTS2021_00002_flair.nii.gz", dimensions=(64, 64, 32))
        # t1 shape mismatch
        write_test_nifti(case_dir / "BraTS2021_00002_t1.nii.gz", dimensions=(64, 64, 48))
        write_test_nifti(case_dir / "BraTS2021_00002_t1ce.nii.gz", dimensions=(64, 64, 32))
        write_test_nifti(case_dir / "BraTS2021_00002_t2.nii.gz", dimensions=(64, 64, 32))
        write_test_nifti(case_dir / "BraTS2021_00002_seg.nii.gz", dimensions=(64, 64, 32))

        cfg = DatasetConfig(data_root=self.temp_dir)
        indexer = BraTSDatasetIndexer(cfg)
        records, summary = indexer.build_index()
        self.assertEqual(summary.valid_cases_count, 0)
        self.assertEqual(summary.invalid_cases_count, 1)
        self.assertFalse(records[0].is_valid)

    def test_case_indexing(self):
        """Tests structured indexing and CSV export."""
        case_dir = self.temp_dir / "BraTS2021_00003"
        case_dir.mkdir(parents=True)
        for mod in ["flair", "t1", "t1ce", "t2", "seg"]:
            write_test_nifti(case_dir / f"BraTS2021_00003_{mod}.nii.gz", dimensions=(64, 64, 32))

        out_dir = self.temp_dir / "out"
        cfg = DatasetConfig(data_root=self.temp_dir, index_output_dir=out_dir)
        indexer = BraTSDatasetIndexer(cfg)
        records, summary = indexer.build_index()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].dim_x, 64)
        self.assertEqual(records[0].dim_z, 32)

        csv_path = out_dir / "index.csv"
        indexer.export_csv(records, csv_path)
        self.assertTrue(csv_path.exists())

    def test_patient_level_splitting_no_leakage(self):
        """Tests patient-level splitting without patient data leakage."""
        case_ids = [f"BraTS2021_{i:05d}" for i in range(1, 101)]
        splitter = PatientLevelSplitter()
        split = splitter.split_cases(case_ids, seed=42, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

        train_set = set(split.train_cases)
        val_set = set(split.val_cases)
        test_set = set(split.test_cases)

        self.assertTrue(train_set.isdisjoint(val_set))
        self.assertTrue(train_set.isdisjoint(test_set))
        self.assertTrue(val_set.isdisjoint(test_set))
        self.assertEqual(len(train_set) + len(val_set) + len(test_set), 100)

    def test_deterministic_splitting(self):
        """Tests deterministic splitting with constant seed."""
        case_ids = [f"BraTS2021_{i:05d}" for i in range(1, 31)]
        splitter = PatientLevelSplitter()
        s1 = splitter.split_cases(case_ids, seed=123)
        s2 = splitter.split_cases(case_ids, seed=123)
        s3 = splitter.split_cases(case_ids, seed=456)

        self.assertEqual(s1.train_cases, s2.train_cases)
        self.assertNotEqual(s1.train_cases, s3.train_cases)

    def test_dataset_getitem_and_metadata_preservation(self):
        """Tests BraTSDataset item retrieval and preservation of spatial metadata."""
        case_dir = self.temp_dir / "BraTS2021_00004"
        case_dir.mkdir(parents=True)
        for mod in ["flair", "t1", "t1ce", "t2", "seg"]:
            write_test_nifti(
                case_dir / f"BraTS2021_00004_{mod}.nii.gz",
                dimensions=(64, 64, 32),
                voxel_spacing=(1.0, 1.0, 1.0)
            )

        cfg = DatasetConfig(data_root=self.temp_dir)
        indexer = BraTSDatasetIndexer(cfg)
        records, _ = indexer.build_index()

        dataset = BraTSDataset(case_records=records, load_data=False)
        self.assertEqual(len(dataset), 1)

        item = dataset[0]
        self.assertEqual(item["case_id"], "BraTS2021_00004")
        self.assertEqual(item["original_shape"], (64, 64, 32))
        self.assertEqual(item["voxel_spacing"], (1.0, 1.0, 1.0))
        self.assertIn("t1", item["modality_paths"])
        self.assertIn("flair", item["modality_paths"])
        self.assertIsNotNone(item["segmentation_path"])

    def test_dataset_statistics(self):
        """Tests dataset statistics reporting."""
        case_dir = self.temp_dir / "BraTS2021_00005"
        case_dir.mkdir(parents=True)
        for mod in ["flair", "t1", "t1ce", "t2", "seg"]:
            write_test_nifti(case_dir / f"BraTS2021_00005_{mod}.nii.gz", dimensions=(64, 64, 32))

        cfg = DatasetConfig(data_root=self.temp_dir)
        indexer = BraTSDatasetIndexer(cfg)
        records, summary = indexer.build_index()

        stats = BraTSDatasetStatistics.compute_summary(records, summary)
        self.assertEqual(stats["total_cases"], 1)
        self.assertEqual(stats["valid_cases"], 1)
        self.assertEqual(stats["modality_availability"]["flair"], 100.0)
        report = BraTSDatasetStatistics.format_text_report(stats)
        self.assertIn("BraTS DATASET INGESTION", report)


if __name__ == "__main__":
    unittest.main()
