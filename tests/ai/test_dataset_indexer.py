"""Unit tests for BraTS dataset index generation and serialization."""

import unittest
from pathlib import Path
import tempfile
import shutil
import csv
import json

from ai.config.dataset_config import DatasetConfig
from ai.data.dataset_indexer import BraTSDatasetIndexer
from tests.utils import write_test_nifti


class TestDatasetIndexer(unittest.TestCase):

    def setUp(self):
        self.temp_dataset_dir = Path(tempfile.mkdtemp(prefix="brats_idx_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dataset_dir, ignore_errors=True)

    def test_indexer_creates_valid_csv_and_json(self):
        """Verifies indexer generates complete records and exports CSV and JSON correctly."""
        for i in range(1, 4):
            case_dir = self.temp_dataset_dir / f"BraTS2021_{i:05d}"
            case_dir.mkdir(parents=True)
            for mod in ["flair", "t1", "t1ce", "t2", "seg"]:
                write_test_nifti(
                    case_dir / f"BraTS2021_{i:05d}_{mod}.nii.gz",
                    dimensions=(64, 64, 32),
                    voxel_spacing=(1.0, 1.0, 1.0)
                )

        out_dir = self.temp_dataset_dir / "index_out"
        cfg = DatasetConfig(data_root=self.temp_dataset_dir, index_output_dir=out_dir)
        indexer = BraTSDatasetIndexer(cfg)

        records, summary = indexer.build_index()

        self.assertEqual(len(records), 3)
        self.assertTrue(all(r.is_valid for r in records))
        self.assertEqual(records[0].dim_x, 64)
        self.assertEqual(records[0].dim_z, 32)
        self.assertEqual(records[0].spacing_x, 1.0)

        # Test CSV Export
        csv_file = out_dir / "index.csv"
        indexer.export_csv(records, csv_file)
        self.assertTrue(csv_file.exists())

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["case_id"], "BraTS2021_00001")
            self.assertEqual(rows[0]["dim_x"], "64")

        # Test JSON Export
        json_file = out_dir / "index.json"
        indexer.export_json(records, json_file)
        self.assertTrue(json_file.exists())

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0]["case_id"], "BraTS2021_00001")


if __name__ == "__main__":
    unittest.main()
