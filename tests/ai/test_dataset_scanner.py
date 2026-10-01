"""Unit tests for BraTS dataset discovery and case scanner."""

import unittest
from pathlib import Path
import tempfile
import shutil

from ai.config.dataset_config import DatasetConfig
from ai.data.dataset_scanner import BraTSDatasetScanner
from tests.utils import write_test_nifti


class TestDatasetScanner(unittest.TestCase):

    def setUp(self):
        self.temp_dataset_dir = Path(tempfile.mkdtemp(prefix="brats_test_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dataset_dir, ignore_errors=True)

    def test_scanner_directory_per_case_discovery(self):
        """Verifies scanner discovers cases in standard directory-per-case layouts."""
        case_dir = self.temp_dataset_dir / "BraTS2021_00100"
        case_dir.mkdir(parents=True)

        for mod in ["flair", "t1", "t1ce", "t2", "seg"]:
            write_test_nifti(case_dir / f"BraTS2021_00100_{mod}.nii.gz")

        cfg = DatasetConfig(data_root=self.temp_dataset_dir)
        scanner = BraTSDatasetScanner(cfg)
        cases = scanner.scan()

        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.case_id, "BraTS2021_00100")
        self.assertIn("flair", case.modality_paths)
        self.assertIn("t1", case.modality_paths)
        self.assertIn("t1ce", case.modality_paths)
        self.assertIn("t2", case.modality_paths)
        self.assertTrue(case.has_segmentation)
        self.assertTrue(case.is_complete)
        self.assertEqual(len(case.missing_modalities), 0)

    def test_scanner_missing_modality_detection(self):
        """Verifies scanner identifies cases with missing modalities."""
        case_dir = self.temp_dataset_dir / "BraTS2021_00101"
        case_dir.mkdir(parents=True)

        # Write only flair and t1 (missing t1ce and t2)
        write_test_nifti(case_dir / "BraTS2021_00101_flair.nii.gz")
        write_test_nifti(case_dir / "BraTS2021_00101_t1.nii.gz")

        cfg = DatasetConfig(data_root=self.temp_dataset_dir)
        scanner = BraTSDatasetScanner(cfg)
        cases = scanner.scan()

        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertIn("t1ce", case.missing_modalities)
        self.assertIn("t2", case.missing_modalities)
        self.assertFalse(case.is_complete)

    def test_scanner_flat_directory_discovery(self):
        """Verifies scanner identifies cases stored in a flat directory."""
        for mod in ["flair", "t1", "t1ce", "t2", "seg"]:
            write_test_nifti(self.temp_dataset_dir / f"BraTS20_Training_042_{mod}.nii")

        cfg = DatasetConfig(data_root=self.temp_dataset_dir)
        scanner = BraTSDatasetScanner(cfg)
        cases = scanner.scan()

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "BraTS20_Training_042")
        self.assertTrue(cases[0].has_segmentation)


if __name__ == "__main__":
    unittest.main()
