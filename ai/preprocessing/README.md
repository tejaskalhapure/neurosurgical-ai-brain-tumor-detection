# AI Preprocessing Package: BraTS MRI Pipeline

## Overview
This package implements the medical image preprocessing pipeline for the **AI Based Brain Tumor Detection and Localization for Surgical Planning** project. It transforms validated multi-modal BraTS scans into model-ready tensors while guaranteeing complete spatial reversibility to original scanner space.

## Modules Summary
- **`config.py`**: `MRIPreprocessingConfig` managing spacing, orientation, cropping, normalization, and label re-mapping.
- **`normalizer.py`**: `MRIIntensityNormalizer` implementing non-zero tissue Z-score normalization with percentile clipping.
- **`cropper.py`**: `ForegroundCropper` computing aggregate non-zero brain bounding boxes and uncropping volumes back to raw coordinates.
- **`registration.py`**: `SpatialTransformer` standardizing orientation to RAS+ and performing continuous (MRI) vs nearest-neighbor (mask) resampling.
- **`label_handler.py`**: `BraTSLabelHandler` managing bidirectional conversion between official BraTS labels {0, 1, 2, 4} and contiguous classes {0, 1, 2, 3}.
- **`pipeline.py`**: `BraTSPreprocessingPipeline` orchestrating the 8-stage transformation and original-space projection.
- **`io.py`**: Case serialization, compression, and cohort batch preprocessing utilities.
- **`visualizer.py`**: `PreprocessingVisualizer` generating 5-panel comparative inspection slices.

## Key Rules
1. **Zero Data Leakage:** Intensity statistics are computed strictly per-volume.
2. **Nearest-Neighbor Masks:** Segmentation labels are strictly interpolated via 0th-order nearest neighbor.
3. **No GPU Requirement:** Vectorized for local CPU execution on Intel Iris Xe laptops.
4. **Physical Traceability:** Affine, voxel spacing, and crop coordinates are preserved at every stage.
