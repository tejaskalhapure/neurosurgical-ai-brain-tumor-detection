"""Unit tests for patient-level dataset splitting and leakage prevention.

Supports both pytest and standard unittest execution.
"""

import unittest
from ai.data.split import PatientLevelSplitter
from ai.config.dataset_config import DatasetConfig


class TestPatientSplitting(unittest.TestCase):

    def test_patient_splitting_no_leakage(self):
        """Verifies that patient cases are mutually exclusive across train, val, and test splits."""
        case_ids = [f"BraTS2021_{i:05d}" for i in range(1, 101)]

        splitter = PatientLevelSplitter()
        split = splitter.split_cases(case_ids, seed=42, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

        train_set = set(split.train_cases)
        val_set = set(split.val_cases)
        test_set = set(split.test_cases)

        # 1. Zero overlap (Strict prevention of patient data leakage)
        self.assertTrue(train_set.isdisjoint(val_set))
        self.assertTrue(train_set.isdisjoint(test_set))
        self.assertTrue(val_set.isdisjoint(test_set))

        # 2. Total count preservation
        self.assertEqual(len(train_set) + len(val_set) + len(test_set), 100)
        self.assertEqual(len(train_set), 70)
        self.assertEqual(len(val_set), 15)
        self.assertEqual(len(test_set), 15)

    def test_patient_splitting_deterministic_reproducibility(self):
        """Verifies that the exact same seed produces identical case assignments."""
        case_ids = [f"BraTS2021_{i:05d}" for i in range(1, 51)]

        splitter = PatientLevelSplitter()
        split1 = splitter.split_cases(case_ids, seed=42)
        split2 = splitter.split_cases(case_ids, seed=42)
        split3 = splitter.split_cases(case_ids, seed=999)

        self.assertEqual(split1.train_cases, split2.train_cases)
        self.assertEqual(split1.val_cases, split2.val_cases)
        self.assertEqual(split1.test_cases, split2.test_cases)

        # Different seed produces different assignments
        self.assertNotEqual(split1.train_cases, split3.train_cases)


if __name__ == "__main__":
    unittest.main()
