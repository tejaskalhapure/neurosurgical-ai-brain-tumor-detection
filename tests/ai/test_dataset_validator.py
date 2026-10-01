"""Unit tests for BraTS dataset validation and error detection."""

import unittest
from pathlib import Path
import tempfile
import shutil

from ai.config.dataset_config import DatasetConfig
from ai.data.dataset_scanner import BraTSDatasetScanner
from ai.data.validator import BraTSDatasetValidator
from tests.utils import write_test_nifti


class TestDatasetValidator(unittest.TestCase):

    def setUp(self):
        self.temp_dataset_dir = Path(tempfile.mkdtemp(prefix="brats_val_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dataset_dir, ignore_errors=True)

    def test_validator_valid_case_passes(self):
        """Verifies that a well-formed case with all modalities and matching dimensions passes."""
        case_dir = self.temp_dataset_dir / "BraTS2021_00001"
        case_dir.mkdir(parents=True)

        for mod in ["flair", "t1", "t1ce", "t2", "seg"]:
            write_test_nifti(
                case_dir / f"BraTS2021_00001_{mod}.nii.gz",
                dimensions=(64, 64, 32),
                voxel_spacing=(1.0, 1.0, 1.0)
            )

        cfg = DatasetConfig(data_root=self.temp_dataset_dir)
        scanner = BraTSDatasetScanner(cfg)
        validator = BraTSDatasetValidator(cfg)

        cases = scanner.scan()
        summary = validator.validate_dataset(cases)

        self.assertTrue(summary.validation_passed)
        self.assertEqual(summary.valid_cases_count, 1)
        self.assertEqual(summary.invalid_cases_count, 0)

    def test_validator_dimension_mismatch_fails(self):
        """Verifies that cross-modality dimension mismatch is flagged as an invalid case."""
        case_dir = self.temp_dataset_dir / "BraTS2021_00002"
        case_dir.mkdir(parents=True)

        # flair has (64, 64, 32)
        write_test_nifti(case_dir / "BraTS2021_00002_flair.nii.gz", dimensions=(64, 64, 32))
        # t1 has mismatched shape (64, 64, 48)
        write_test_nifti(case_dir / "BraTS2021_00002_t1.nii.gz", dimensions=(64, 64, 48))
        write_test_nifti(case_dir / "BraTS2021_00002_t1ce.nii.gz", dimensions=(64, 64, 32))
        write_test_nifti(case_dir / "BraTS2021_00002_t2.nii.gz", dimensions=(64, 64, 32))
        write_test_nifti(case_dir / "BraTS2021_00002_seg.nii.gz", dimensions=(64, 64, 32))

        cfg = DatasetConfig(data_root=self.temp_dataset_dir)
        scanner = BraTSDatasetScanner(cfg)
        validator = BraTSDatasetValidator(cfg)

        cases = scanner.scan()
        summary = validator.validate_dataset(cases)

        self.assertEqual(summary.valid_cases_count, 0)
        self.assertEqual(summary.invalid_cases_count, 1)
        invalid_case = summary.invalid_cases[0]
        self.assertTrue(any("Dimension mismatch" in err for err in invalid_case.errors))

    def test_validator_segmentation_mismatch_fails(self):
        """Verifies that segmentation shape mismatch with image volume is flagged."""
        case_dir = self.temp_dataset_dir / "BraTS2021_00003"
        case_dir.mkdir(parents=True)

        for mod in ["flair", "t1", "t1ce", "t2"]:
            write_test_nifti(case_dir / f"BraTS2021_00003_{mod}.nii.gz", dimensions=(64, 64, 32))

        # seg has different dimensions
        write_test_nifti(case_dir / "BraTS2021_00003_seg.nii.gz", dimensions=(32, 32, 16))

        cfg = DatasetConfig(data_root=self.temp_dataset_dir)
        scanner = BraTSDatasetScanner(cfg)
        validator = BraTSDatasetValidator(cfg)

        cases = scanner.scan()
        summary = validator.validate_dataset(cases)

        self.assertEqual(summary.invalid_cases_count, 1)
        self.assertTrue(any("Segmentation dimension mismatch" in err for err in summary.invalid_cases[0].errors))

    def test_validator_corrupted_file_fails(self):
        """Verifies that truncated or unparseable NIfTI files are reported clearly."""
        case_dir = self.temp_dataset_dir / "BraTS2021_00004"
        case_dir.mkdir(parents=True)

        with open(case_dir / "BraTS2021_00004_flair.nii.gz", "wb") as f:
            f.write(b"NOT_A_VALID_GZIP_OR_NIFTI_FILE")

        for mod in ["t1", "t1ce", "t2", "seg"]:
            write_test_nifti(case_dir / f"BraTS2021_00004_{mod}.nii.gz")

        cfg = DatasetConfig(data_root=self.temp_dataset_dir)
        scanner = BraTSDatasetScanner(cfg)
        validator = BraTSDatasetValidator(cfg)

        cases = scanner.scan()
        summary = validator.validate_dataset(cases)

        self.assertEqual(summary.invalid_cases_count, 1)
        self.assertTrue(any("Corrupted" in err or "Failed" in err for err in summary.invalid_cases[0].errors))


if __name__ == "__main__":
    unittest.main()
