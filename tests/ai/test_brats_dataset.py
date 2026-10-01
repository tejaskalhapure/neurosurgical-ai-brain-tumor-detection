"""Unit tests for BraTSDataset and spatial metadata preservation."""

import unittest
from pathlib import Path
import tempfile
import shutil

from ai.config.dataset_config import DatasetConfig
from ai.data.dataset_indexer import BraTSDatasetIndexer
from ai.data.brats_dataset import BraTSDataset
from tests.utils import write_test_nifti


class TestBraTSDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dataset_dir = Path(tempfile.mkdtemp(prefix="brats_ds_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dataset_dir, ignore_errors=True)

    def test_brats_dataset_metadata_preservation(self):
        """Verifies that BraTSDataset preserves affine, voxel spacing, and original shape."""
        case_dir = self.temp_dataset_dir / "BraTS2021_00010"
        case_dir.mkdir(parents=True)

        expected_shape = (64, 64, 32)
        expected_spacing = (1.0, 1.0, 1.0)

        for mod in ["flair", "t1", "t1ce", "t2", "seg"]:
            write_test_nifti(
                case_dir / f"BraTS2021_00010_{mod}.nii.gz",
                dimensions=expected_shape,
                voxel_spacing=expected_spacing
            )

        cfg = DatasetConfig(data_root=self.temp_dataset_dir)
        indexer = BraTSDatasetIndexer(cfg)
        records, _ = indexer.build_index()

        self.assertEqual(len(records), 1)

        dataset = BraTSDataset(case_records=records, load_data=False)
        self.assertEqual(len(dataset), 1)

        item = dataset[0]
        self.assertEqual(item["case_id"], "BraTS2021_00010")
        self.assertEqual(item["original_shape"], expected_shape)
        self.assertEqual(item["voxel_spacing"], expected_spacing)
        self.assertIsNotNone(item["affine"])
        self.assertIn("t1", item["modality_paths"])
        self.assertIn("t1ce", item["modality_paths"])
        self.assertIn("t2", item["modality_paths"])
        self.assertIn("flair", item["modality_paths"])
        self.assertIsNotNone(item["segmentation_path"])

    def test_brats_dataset_custom_transform(self):
        """Verifies that custom dictionary transforms can intercept and process items."""
        case_dir = self.temp_dataset_dir / "BraTS2021_00020"
        case_dir.mkdir(parents=True)

        for mod in ["flair", "t1", "t1ce", "t2", "seg"]:
            write_test_nifti(case_dir / f"BraTS2021_00020_{mod}.nii.gz")

        cfg = DatasetConfig(data_root=self.temp_dataset_dir)
        indexer = BraTSDatasetIndexer(cfg)
        records, _ = indexer.build_index()

        def mock_transform(data):
            data["transformed_flag"] = True
            return data

        dataset = BraTSDataset(case_records=records, transform=mock_transform, load_data=False)
        item = dataset[0]
        self.assertTrue(item.get("transformed_flag"))


if __name__ == "__main__":
    unittest.main()
