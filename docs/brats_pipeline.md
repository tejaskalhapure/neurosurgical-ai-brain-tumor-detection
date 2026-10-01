# Module 2: BraTS Ingestion, Validation, Indexing & DataLoader Pipeline

## 1. Objectives & Scope
Module 2 establishes the end-to-end data ingestion foundation for the **AI Based Brain Tumor Detection and Localization for Surgical Planning** project. It guarantees:
1. Flexible directory discovery across different BraTS releases (2020, 2021, 2023).
2. Rigorous multi-modal header and dimension validation before training.
3. Structured indexing with preservation of physical spatial metadata (affine matrix, voxel spacing).
4. Patient-level data partitioning with zero cross-slice leakage.
5. PyTorch Dataset (`BraTSDataset`) integration prepared for MONAI dictionary transforms in Module 3.

---

## 2. Supported BraTS Releases & Modalities
The pipeline is designed to support:
- **BraTS 2021 Adult Glioma**: Multi-modal structural scans acquired with standardized protocols across multiple clinical institutions.
- **BraTS 2020 / 2019 / 2018**: Compatible with legacy naming patterns.

### Modality Specifications
| Modality Key | Full Sequence Name | Clinical Utility |
| :--- | :--- | :--- |
| `t1` | Native T1-weighted | Anatomical structural reference, grey-white matter boundary |
| `t1ce` | Contrast-enhanced T1-weighted (Gd) | Delineates hyper-vascular, active neoplastic tumor margins |
| `t2` | T2-weighted structural | Depicts fluid and edema boundaries |
| `flair` | T2-Fluid Attenuated Inversion Recovery | Suppresses pure cerebrospinal fluid (CSF) to isolate peritumoral edema |

### Segmentation Target Labels
| Label ID | Tissue Component | Target Region |
| :--- | :--- | :--- |
| **0** | Normal tissue / Non-tumor brain | Background |
| **1** | Necrotic tumor core (NCR) & non-enhancing tumor (NET) | WT & TC |
| **2** | Peritumoral edematous tissue (ED) | WT |
| **4** | GD-enhancing tumor (ET) | WT, TC & ET |

---

## 3. Preservation of Spatial Metadata
Physical measurements critical for surgical planning (e.g. maximum cross-sectional area in $mm^2$ and approximate 3D tumor volume in $cm^3$) require authentic physical voxel metadata.
The pipeline extracts and preserves:
- **`affine`**: 4x4 coordinate transformation matrix relating voxel coordinates $(i, j, k)$ to scanner space $(x, y, z)$ in millimeters.
- **`voxel_spacing`**: $(dx, dy, dz)$ voxel dimensions in millimeters.
- **`original_shape`**: $(Nx, Ny, Nz)$ matrix dimensions.
- **`orientation`**: Standardized anatomical orientation codes (e.g. `RAS`).

These fields are carried inside every record in `brats_index.csv`, `brats_index.json`, and the item dictionary produced by `BraTSDataset.__getitem__()`.

---

## 4. Patient-Level Splitting & Leakage Prevention
In medical imaging, splitting slices from the same 3D patient volume into both training and validation sets causes severe **data leakage** (the model memorizes patient-specific anatomical features rather than learning generalizable tumor morphology).

`PatientLevelSplitter` enforces:
1. Splitting exclusively on unique patient IDs (`case_id`).
2. Disjoint sets:
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
3. Exact reproducibility via configurable integer seed (default: `42`).
4. Persistence of split manifests into `data/interim/split_manifest.json`.

---

## 5. Execution Commands
To run the automated validation and indexing pipeline on an authentic dataset:
```bash
python -m ai.data.cli --data-root /path/to/BraTS2021 --output-dir data/interim
```
To run the test suite:
```bash
python -m unittest tests/run_module2_tests.py -v
```
