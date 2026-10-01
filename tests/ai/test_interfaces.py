"""Unit tests for AI evaluation metrics and mathematical formulas."""

import unittest

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ai.evaluation.metrics import (
    dice_similarity_coefficient,
    intersection_over_union,
    calculate_slice_area_mm2,
    calculate_tumor_volume,
    calculate_bounding_box_3d
)


class TestMetrics(unittest.TestCase):

    def test_dice_similarity_perfect_overlap(self):
        """Identical masks must yield a Dice score of 1.0."""
        if not HAS_NUMPY:
            self.skipTest("numpy required for metric arrays")
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 2:6] = True
        dice = dice_similarity_coefficient(mask, mask)
        self.assertAlmostEqual(dice, 1.0, places=3)

    def test_dice_similarity_zero_overlap(self):
        """Completely disjoint masks must yield a Dice score close to 0.0."""
        if not HAS_NUMPY:
            self.skipTest("numpy required for metric arrays")
        mask1 = np.zeros((10, 10), dtype=bool)
        mask2 = np.zeros((10, 10), dtype=bool)
        mask1[0:2, 0:2] = True
        mask2[8:10, 8:10] = True
        dice = dice_similarity_coefficient(mask1, mask2)
        self.assertAlmostEqual(dice, 0.0, places=3)

    def test_calculate_tumor_volume(self):
        """Volume calculation must equal count * (dx * dy * dz) mm3."""
        if not HAS_NUMPY:
            self.skipTest("numpy required for metric arrays")
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:4, 2:4, 2:4] = True  # 2 x 2 x 2 = 8 voxels
        result = calculate_tumor_volume(mask, (1.0, 1.0, 1.0))
        self.assertEqual(result["voxel_count"], 8)
        self.assertEqual(result["estimated_volume_mm3"], 8.0)
        self.assertEqual(result["estimated_volume_cm3"], 0.008)
        self.assertTrue(result["valid_voxel_metadata"])

    def test_calculate_bounding_box_3d(self):
        """3D bounding box must match the extremal coordinates."""
        if not HAS_NUMPY:
            self.skipTest("numpy required for metric arrays")
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[5:11, 4:13, 6:16] = True
        bbox = calculate_bounding_box_3d(mask)
        self.assertIsNotNone(bbox)
        self.assertEqual(bbox["z_min"], 5)
        self.assertEqual(bbox["z_max"], 10)
        self.assertEqual(bbox["y_min"], 4)
        self.assertEqual(bbox["y_max"], 12)
        self.assertEqual(bbox["x_min"], 6)
        self.assertEqual(bbox["x_max"], 15)


if __name__ == "__main__":
    unittest.main()
