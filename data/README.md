# BraTS Dataset Ingestion, Directory Structure & Validation Guide

## 1. Directory Layout
The data pipeline is designed to support the **RSNA-ASNR-MICCAI BraTS (Brain Tumor Segmentation)** benchmark cohorts (BraTS 2021, BraTS 2020, BraTS 2023 Adult Glioma) as well as custom multi-modal MRI cohorts.

```
data/
├── raw/                         # Default root for uncompressed/downloaded patient cases
│   ├── BraTS2021_00000/
│   │   ├── BraTS2021_00000_flair.nii.gz
│   │   ├── BraTS2021_00000_t1.nii.gz
│   │   ├── BraTS2021_00000_t1ce.nii.gz
│   │   ├── BraTS2021_00000_t2.nii.gz
│   │   └── BraTS2021_00000_seg.nii.gz
│   ├── BraTS2021_00002/
│   │   └── ...
│   └── .gitkeep
├── interim/                     # Output directory for index files & patient split manifests
│   ├── brats_index.csv          # Structured tabular index (case IDs, paths, shapes, voxels)
│   ├── brats_index.json         # Detailed JSON index with 4x4 affine matrices
│   ├── split_manifest.json      # Patient-level train / val / test case ID lists
│   ├── dataset_statistics.json  # Cohort-level diagnostic stats and error frequencies
│   └── .gitkeep
└── processed/                   # Reserved for Module 3 cached/preprocessed tensors
    └── .gitkeep
```

---

## 2. Supported MRI Modalities & Segmentation Labels

### 4 Multi-Modal MRI Sequences
1. **`flair`**: T2-weighted Fluid Attenuated Inversion Recovery (highlights peritumoral edema)
2. **`t1`**: Native T1-weighted anatomical sequence
3. **`t1ce`**: T1-weighted post-contrast / Gadolinium enhanced (highlights active tumor margins)
4. **`t2`**: T2-weighted structural sequence

### Ground-Truth Tumor Sub-Regions (`seg`)
- **Label 0**: Background (normal brain parenchyma / non-tumor tissue)
- **Label 1**: Necrotic Tumor Core (NCR) and non-enhancing tumor (NET)
- **Label 2**: Peritumoral Edematous Tissue (ED)
- **Label 4**: GD-Enhancing Tumor (ET) (hyper-intense active neoplastic border)

---

## 3. How to Configure `DATA_ROOT`
You can configure the location of your BraTS dataset without modifying any code:

### Option A: Via Environment Variable (Recommended for Local Dev & Colab)
In your `.env` or system terminal:
```bash
# Windows PowerShell
$env:BRATS_DATA_ROOT="D:\Datasets\BraTS2021_Training_Data"

# Linux / macOS / Google Colab
export BRATS_DATA_ROOT="/content/drive/MyDrive/BraTS2021"
```

### Option B: Via Command-Line Flag
```bash
python -m ai.data.cli --data-root "D:\Datasets\BraTS2021_Training_Data" --output-dir data/interim
```

---

## 4. How to Validate the Dataset & Generate the Index
Run the dataset CLI scanner:
```bash
# With active Python virtual environment (.venv)
python -m ai.data.cli --data-root data/raw --output-dir data/interim
```

### Validation Checks Performed
- **Modality completeness**: Confirms all 4 sequences (`flair`, `t1`, `t1ce`, `t2`) exist.
- **Label presence**: Confirms `seg` exists when required.
- **Integrity**: Verifies file headers are readable and files are not truncated or corrupt.
- **Dimensional consistency**: Ensures all 4 modalities and the segmentation mask share the exact same spatial dimensions (e.g. `240 x 240 x 155`).
- **Physical voxel spacing**: Extracts $(dx, dy, dz)$ in millimeters.
- **Case uniqueness**: Detects and reports duplicate case IDs.

---

## 5. Patient-Level Splitting & Data Leakage Prevention
- Splitting is performed strictly at the **patient / case level**.
- Individual 2D slices from the same 3D volume are **never** partitioned across train, validation, or test sets.
- By default, uses `seed=42` with 70% Train, 15% Validation, 15% Test.
- Generates `data/interim/split_manifest.json` ensuring 100% reproducible training experiments.

---

## 6. How to Run Dataset Exploration
Open the exploration notebook or execute slice visualization:
```bash
python -c "
from ai.data.visualizer import BraTSVisualizer
from pathlib import Path
v = BraTSVisualizer()
# Extracts axial middle slice data from an authentic scan
data = v.extract_slice_data(Path('data/raw/BraTS2021_00000/BraTS2021_00000_flair.nii.gz'))
print('Slice extracted successfully. Dimensions:', data['mri_slice'].shape)
"
```
