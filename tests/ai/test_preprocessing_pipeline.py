"""Unit tests for the Module 3 MRI Preprocessing Pipeline.

Tests orientation, resampling, nearest-neighbor segmentation interpolation,
nonzero intensity normalization, constant image safety, channel ordering,
label preservation, and spatial metadata traceability.
"""

import unittest
from pathlib import Path
import tempfile
import shutil

from ai.preprocessing.config import MRIPreprocessingConfig
from ai.preprocessing.normalizer import MRIIntensityNormalizer
from ai.preprocessing.cropper import ForegroundCropper, CropCoordinates
from ai.preprocessing.registration import SpatialTransformer
from ai.preprocessing.label_handler import BraTSLabelHandler
from ai.preprocessing.pipeline import BraTSPreprocessingPipeline
from tests.utils import write_test_nifti

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TestPreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="prep_test_"))
        self.normalizer = MRIIntensityNormalizer()
        self.cropper = ForegroundCropper(margin=2)
        self.transformer = SpatialTransformer()
        self.label_handler = BraTSLabelHandler()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_intensity_normalization_nonzero(self):
        """Verifies Z-score normalization on non-zero brain voxels while preserving zero background."""
        if not HAS_NUMPY:
            self.skipTest("NumPy required for intensity normalization test")

        vol = np.zeros((10, 10, 10), dtype=np.float32)
        # Create non-zero foreground region with known values
        vol[2:8, 2:8, 2:8] = np.arange(10, 10 + 6*6*6).reshape(6, 6, 6).astype(np.float32)

        norm_vol = self.normalizer.normalize_nonzero(vol, clip_percentiles=None)

        # Background must remain strictly 0
        self.assertEqual(norm_vol[0, 0, 0], 0.0)
        self.assertEqual(norm_vol[9, 9, 9], 0.0)

        # Non-zero region must have mean ~ 0 and std ~ 1
        nonzero_vals = norm_vol[norm_vol != 0]
        self.assertAlmostEqual(float(np.mean(nonzero_vals)), 0.0, places=4)
        self.assertAlmostEqual(float(np.std(nonzero_vals)), 1.0, places=4)

    def test_zero_and_constant_image_handling(self):
        """Verifies safe handling of empty or constant scans without division by zero."""
        if not HAS_NUMPY:
            self.skipTest("NumPy required")

        # 1. All-zero volume
        zero_vol = np.zeros((8, 8, 8), dtype=np.float32)
        norm_zero = self.normalizer.normalize_nonzero(zero_vol)
        self.assertTrue(np.all(norm_zero == 0.0))
        self.assertFalse(np.any(np.isnan(norm_zero)))

        # 2. Constant volume (std == 0)
        const_vol = np.full((8, 8, 8), 100.0, dtype=np.float32)
        norm_const = self.normalizer.normalize_nonzero(const_vol)
        self.assertFalse(np.any(np.isnan(norm_const)))
        self.assertFalse(np.any(np.isinf(norm_const)))

    def test_resampling_nearest_neighbor_for_segmentation(self):
        """Verifies that segmentation masks STRICTLY use nearest-neighbor interpolation.

        Discrete labels must NEVER be converted to floating-point intermediate values.
        """
        if not HAS_NUMPY:
            self.skipTest("NumPy required")

        # Create a synthetic mask with discrete labels: 0, 1, 2, 4
        mask = np.zeros((16, 16, 16), dtype=np.int64)
        mask[2:6, 2:6, 2:6] = 1   # NCR
        mask[6:10, 6:10, 6:10] = 2 # ED
        mask[10:14, 10:14, 10:14] = 4 # ET

        # Resample from 2.0mm to 1.0mm (upsampling factor 2)
        resampled_mask, new_shape = self.transformer.resample_to_spacing(
            mask,
            current_spacing=(2.0, 2.0, 2.0),
            target_spacing=(1.0, 1.0, 1.0),
            is_label=True
        )

        unique_labels = set(np.unique(resampled_mask))
        # Unique labels must strictly be a subset of {0, 1, 2, 4}
        # Absolutely NO intermediate fractional numbers like 1.5, 2.7, 3.1
        self.assertTrue(unique_labels.issubset({0, 1, 2, 4}))
        self.assertIn(1, unique_labels)
        self.assertIn(2, unique_labels)
        self.assertIn(4, unique_labels)

    def test_foreground_cropper_and_coordinate_tracking(self):
        """Verifies that unified foreground crop is applied and reversible."""
        if not HAS_NUMPY:
            self.skipTest("NumPy required")

        mri = np.zeros((30, 30, 30), dtype=np.float32)
        seg = np.zeros((30, 30, 30), dtype=np.int64)

        # Place brain / tumor mass inside [5:25, 6:24, 7:23]
        mri[5:25, 6:24, 7:23] = 100.0
        seg[10:15, 10:15, 10:15] = 4  # Enhancing tumor

        cropper = ForegroundCropper(margin=2)
        coords = cropper.compute_crop_coordinates([mri], seg_mask=seg)

        # Confirm crop encompasses foreground
        self.assertLessEqual(coords.x_min, 5)
        self.assertGreaterEqual(coords.x_max, 25)
        self.assertLessEqual(coords.y_min, 6)
        self.assertGreaterEqual(coords.y_max, 24)

        cropped_mri = cropper.apply_crop(mri, coords)
        cropped_seg = cropper.apply_crop(seg, coords)

        # Check shapes match crop coordinates
        self.assertEqual(cropped_mri.shape, coords.cropped_shape)
        self.assertEqual(cropped_seg.shape, coords.cropped_shape)

        # Verify uncrop reversibility
        uncropped_seg = ForegroundCropper.uncrop_volume(cropped_seg, coords)
        self.assertEqual(uncropped_seg.shape, (30, 30, 30))
        self.assertTrue(np.array_equal(uncropped_seg, seg))

    def test_label_handler_bidirectional_invertibility(self):
        """Verifies that contiguous mapping (0->0, 1->1, 2->2, 4->3) is exactly invertible."""
        if not HAS_NUMPY:
            self.skipTest("NumPy required")

        mask = np.array([0, 1, 2, 4, 1, 4, 0, 2], dtype=np.int64)
        cont = self.label_handler.to_contiguous(mask)

        # 4 must map to 3
        expected_cont = np.array([0, 1, 2, 3, 1, 3, 0, 2], dtype=np.int64)
        self.assertTrue(np.array_equal(cont, expected_cont))

        # Restored must equal original
        restored = self.label_handler.to_original_brats(cont)
        self.assertTrue(np.array_equal(mask, restored))
        self.assertTrue(self.label_handler.verify_invertibility(mask))

    def test_full_pipeline_channel_ordering_and_metadata(self):
        """Verifies full pipeline execution, [4, D, H, W] channel ordering, and metadata preservation."""
        if not HAS_NUMPY:
            self.skipTest("NumPy required for full array tensor processing")

        case_dir = self.temp_dir / "BraTS2021_00099"
        case_dir.mkdir(parents=True)

        shape = (48, 48, 24)
        spacing = (1.0, 1.0, 1.0)
        mod_paths = {}

        for mod in ["t1", "t1ce", "t2", "flair"]:
            p = case_dir / f"BraTS2021_00099_{mod}.nii.gz"
            write_test_nifti(p, dimensions=shape, voxel_spacing=spacing)
            mod_paths[mod] = p

        seg_p = case_dir / "BraTS2021_00099_seg.nii.gz"
        write_test_nifti(seg_p, dimensions=shape, voxel_spacing=spacing)

        cfg = MRIPreprocessingConfig(
            target_spacing=None,  # test without resampling to verify dimensions
            crop_margin=2
        )
        pipeline = BraTSPreprocessingPipeline(config=cfg)
        result = pipeline.preprocess_case(
            modality_paths=mod_paths,
            case_id="BraTS2021_00099",
            seg_path=seg_p,
            return_tensors=False
        )

        # Channel dimension must be 4: [4, D, H, W]
        self.assertEqual(result.image.shape[0], 4)
        if result.label is not None:
            self.assertEqual(result.label.shape[0], 1)

        # Verify metadata completeness
        meta = result.metadata
        self.assertEqual(meta["case_id"], "BraTS2021_00099")
        self.assertEqual(meta["channel_order"], ["t1", "t1ce", "t2", "flair"])
        self.assertEqual(meta["original_shape"], [48, 48, 24])
        self.assertEqual(meta["original_spacing"], [1.0, 1.0, 1.0])
        self.assertIn("crop_coordinates", meta)
        self.assertEqual(meta["contiguous_label_map"][4], 3)
        self.assertEqual(meta["inverse_label_map"][3], 4)

    def test_deterministic_preprocessing_consistency(self):
        """Verifies that validation/test preprocessing is 100% deterministic."""
        if not HAS_NUMPY:
            self.skipTest("NumPy required for deterministic volume processing")

        case_dir = self.temp_dir / "BraTS2021_00088"
        case_dir.mkdir(parents=True)

        mod_paths = {}
        for mod in ["t1", "t1ce", "t2", "flair"]:
            p = case_dir / f"BraTS2021_00088_{mod}.nii.gz"
            write_test_nifti(p, dimensions=(32, 32, 16))
            mod_paths[mod] = p

        cfg = MRIPreprocessingConfig()
        pipeline = BraTSPreprocessingPipeline(config=cfg)

        res1 = pipeline.preprocess_case(mod_paths, "BraTS2021_00088", split="val", return_tensors=False)
        res2 = pipeline.preprocess_case(mod_paths, "BraTS2021_00088", split="val", return_tensors=False)

        if HAS_NUMPY:
            self.assertTrue(np.array_equal(res1.image, res2.image))


    def test_config_serialization_and_defaults(self):
        """Verifies default parameters and round-trip dictionary serialization for MRIPreprocessingConfig."""
        cfg = MRIPreprocessingConfig()
        self.assertEqual(cfg.target_spacing, (1.0, 1.0, 1.0))
        self.assertEqual(cfg.target_orientation, "RAS")
        self.assertEqual(cfg.channel_order, ("t1", "t1ce", "t2", "flair"))
        self.assertTrue(cfg.convert_labels_to_contiguous)
        self.assertEqual(cfg.contiguous_label_map[4], 3)
        self.assertEqual(cfg.inverse_label_map[3], 4)

        d = cfg.to_dict()
        restored = MRIPreprocessingConfig.from_dict(d)
        self.assertEqual(restored.target_spacing, cfg.target_spacing)
        self.assertEqual(restored.target_orientation, cfg.target_orientation)
        self.assertEqual(restored.channel_order, cfg.channel_order)

    def test_crop_coordinates_serialization(self):
        """Verifies serialization and deserialization of bounding box CropCoordinates."""
        coords = CropCoordinates(
            x_min=10, x_max=50,
            y_min=15, y_max=55,
            z_min=5, z_max=25,
            original_shape=(64, 64, 32),
            cropped_shape=(40, 40, 20)
        )
        d = coords.to_dict()
        restored = CropCoordinates.from_dict(d)
        self.assertEqual(restored.x_min, 10)
        self.assertEqual(restored.x_max, 50)
        self.assertEqual(restored.cropped_shape, (40, 40, 20))
        self.assertEqual(restored.original_shape, (64, 64, 32))

    def test_pure_python_intensity_normalizer(self):
        """Verifies pure-Python intensity normalization on nested lists without NumPy."""
        nested = [[[0.0, 10.0], [20.0, 30.0]]]
        norm = self.normalizer._normalize_python(nested, clip_percentiles=None)
        # Background remains 0.0
        self.assertEqual(norm[0][0][0], 0.0)
        # Nonzero values have been standardized
        self.assertNotEqual(norm[0][0][1], 10.0)


if __name__ == "__main__":
    unittest.main()
