# Module 3: MRI Preprocessing Pipeline Specification

> **ACADEMIC RESEARCH PROTOTYPE DISCLAIMER:**  
> This preprocessing pipeline is an integral component of an academic engineering capstone prototype for neurosurgical decision support. It does not provide medical diagnoses, surgical margins, or standalone clinical conclusions. All outputs require validation by certified clinical specialists.

---

## 1. Preprocessing Pipeline Flow & Architecture

```
[Raw Validated BraTS Case]
   (T1, T1ce, T2, FLAIR, Seg, Affine, Spacing, Orientation)
              │
              ▼
   [Stage 1: NIfTI Ingestion & Header Verification]
   • Validates 3D matrix integrity & positive voxel spacing
              │
              ▼
   [Stage 2: Orientation Standardization to RAS+]
   • Standardizes voxel axes: Right (X), Anterior (Y), Superior (Z)
   • Simultaneously updates 4x4 coordinate affine matrix
              │
              ▼
   [Stage 3: Spatial Resampling]
   • Structural MRI: Continuous trilinear / spline interpolation (order=1)
   • Segmentation Mask: Strictly 0th-order Nearest-Neighbor (order=0)
   • Default: 1.0mm³ isotropic spacing (fully configurable)
              │
              ▼
   [Stage 4: Unified Foreground Brain Cropping]
   • Identifies aggregate non-zero bounding box across all 4 sequences + mask
   • Guarantees zero tumor tissue or parenchyma is cropped away
   • Records [x_min:x_max, y_min:y_max, z_min:z_max] crop coordinates
              │
              ▼
   [Stage 5: Non-Zero Z-Score Intensity Normalization]
   • Computed strictly on non-zero tissue voxels (per-volume, zero leakage)
   • Percentile outlier clipping (0.5%, 99.5%)
   • Safe handling of constant/empty volumes (std < 1e-6)
   • Background air remains strictly 0.0
              │
              ▼
   [Stage 6: Deterministic Channel Organization]
   • Stacks sequences into 4D tensor [C, D, H, W]
   • Enforces deterministic channel order: [T1, T1ce, T2, FLAIR]
              │
              ▼
   [Stage 7: Segmentation Contiguous Re-Mapping]
   • Maps non-contiguous BraTS classes {0, 1, 2, 4} -> contiguous {0, 1, 2, 3}
   • Bidirectional invertibility preserved in metadata {0:0, 1:1, 2:2, 3:4}
              │
              ▼
   [Stage 8: Spatial Traceability Metadata Compilation]
   • Encapsulates original & resampled affine, spacing, shape, and crop coordinates
   • Ready for model ingestion and surgical space reconstruction
```

---

## 2. Exact Transform Execution Order

1. **`LoadNiftiVolumes`**: Reads the 4 structural sequences and segmentation mask. Extracts original shape, spacing, affine, and orientation.
2. **`StandardizeOrientation`**: Applies coordinate orientation transforms to align patient axes with standard neurological **RAS+** coordinates.
3. **`SpatialResampling`**: Resamples grid resolution to `target_spacing` (default: $(1.0, 1.0, 1.0)$ mm).
4. **`ForegroundCropping`**: Computes non-zero brain bounding box with safety padding margin; applies identical spatial crop to all modalities and mask.
5. **`NormalizeNonZeroIntensities`**: Z-score standardizes each MRI modality independently on non-zero brain voxels.
6. **`StackChannels`**: Packs volumes into shape $[4, D, H, W]$ in fixed sequence `[t1, t1ce, t2, flair]`.
7. **`MapLabelsContiguous`**: Re-indexes label 4 (enhancing tumor) to contiguous index 3; expands mask to $[1, D, H, W]$.
8. **`CompileTraceabilityMetadata`**: Records all spatial transformations for reversibility.

---

## 3. Preprocessing Configuration Schema

```python
from ai.preprocessing.config import MRIPreprocessingConfig

config = MRIPreprocessingConfig(
    target_spacing=(1.0, 1.0, 1.0),      # Default 1.0mm isotropic (or None to disable)
    target_orientation="RAS",             # Standard neurological RAS+
    spatial_crop_size=(128, 128, 128),    # Bounding box or target dimensions
    crop_margin=4,                        # Voxel padding around non-zero brain tissue
    intensity_normalization="zscore_nonzero",
    clip_percentiles=(0.5, 99.5),         # Percentile clipping limits
    mri_interpolation="linear",           # Continuous trilinear interpolation
    seg_interpolation="nearest",          # Strictly Nearest-Neighbor
    channel_order=("t1", "t1ce", "t2", "flair"),
    convert_labels_to_contiguous=True,    # 0->0, 1->1, 2->2, 4->3
    enable_augmentation=False,            # Disabled for val/test/inference
    seed=42
)
```

---

## 4. Interpolation Decisions: Continuous vs Nearest-Neighbor

| Data Type | Interpolation Scheme | Order | Rationale |
| :--- | :--- | :--- | :--- |
| **Structural MRI (`t1`, `t1ce`, `t2`, `flair`)** | Trilinear / Spline | `order=1` | MRI intensities represent smoothly varying physical tissue relaxation times. Linear interpolation produces continuous, artifact-free anatomical gradients. |
| **Segmentation Mask (`seg`)** | **Strictly Nearest-Neighbor** | `order=0` | **Mandatory:** Segmentation labels are categorical discrete integers ($0, 1, 2, 4$). Linear interpolation would generate synthetic, non-existent fractional labels (e.g. $1.7$ or $3.2$), destroying anatomical ground truth. |

---

## 5. Intensity Normalization & Data Leakage Prevention

### Non-Zero Z-Score Standardisation:
$$\mu = \frac{1}{|V_{>0}|} \sum_{v \in V_{>0}} X(v), \quad \sigma = \sqrt{\frac{1}{|V_{>0}|} \sum_{v \in V_{>0}} (X(v) - \mu)^2}$$
$$X_{\text{norm}}(v) = \begin{cases} \frac{X(v) - \mu}{\sigma + 10^{-8}} & \text{if } X(v) > 0 \\ 0.0 & \text{if } X(v) = 0 \end{cases}$$

### Zero Cross-Patient Leakage:
- Normalization is performed **strictly per-volume**.
- No population-wide global statistics or test-set metrics are ever calculated or leaked into training.
- Background air ($X=0$) is excluded from mean and variance calculation, preventing scanner-dependent air volume from distorting brain tissue statistics.

---

## 6. Segmentation Label Mapping & Clinical Sub-Regions

### Bidirectional Contiguous Mapping:
- **Forward Mapping (Input to Model):**
  - Label 0 (Background) $\rightarrow$ Class 0
  - Label 1 (Necrotic Core / NCR) $\rightarrow$ Class 1
  - Label 2 (Peritumoral Edema / ED) $\rightarrow$ Class 2
  - Label 4 (Enhancing Tumor / ET) $\rightarrow$ Class 3
- **Inverse Mapping (Model Output to BraTS Space):**
  - Class 0 $\rightarrow$ Label 0
  - Class 1 $\rightarrow$ Label 1
  - Class 2 $\rightarrow$ Label 2
  - Class 3 $\rightarrow$ Label 4

### Derived Clinical Sub-Regions for Surgical Planning:
- **WT (Whole Tumor):** Classes $1 + 2 + 3$ (Official labels $1 + 2 + 4$) — Total neoplastic burden.
- **TC (Tumor Core):** Classes $1 + 3$ (Official labels $1 + 4$) — Necrotic cavity plus active border.
- **ET (Enhancing Tumor):** Class 3 (Official label $4$) — Hyper-vascular active rim targeted for maximal safe resection.

---

## 7. Original Space Projection & Reversibility

To calculate authentic physical measurements (maximum 2D slice area in $\text{mm}^2$ and 3D tumor volume in $\text{cm}^3$) and display overlays on the patient's original MRI:

```python
from ai.preprocessing.pipeline import BraTSPreprocessingPipeline

pipeline = BraTSPreprocessingPipeline(config=config)

# Map model prediction back to raw patient scanner space:
original_space_mask = pipeline.map_prediction_to_original_space(
    predicted_mask=model_output_tensor,
    metadata=case.metadata
)

# original_space_mask matches case.metadata["original_shape"] with 100% precision
```

---

## 8. Memory Management on Intel Iris Xe CPU
- **CPU Optimization:** Runs purely on CPU with vectorized NumPy/SciPy routines. Zero NVIDIA CUDA requirements.
- **Bounding Box Cropping:** Crops out empty background air immediately, reducing volume size from $240 \times 240 \times 155$ to approximately $128 \times 128 \times 128$, reducing RAM consumption by over 60%.
- **Precision:** Uses single-precision `float32` for image tensors and `int64` for label masks.
