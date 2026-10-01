# End-to-End System Data Flow

```
[Clinical User / Researcher]
          │
          │ 1. Upload MRI (NIfTI .nii/.nii.gz or DICOM series)
          ▼
[Frontend: React + Vite]
          │
          │ 2. POST /api/v1/mri/upload (multipart/form-data)
          ▼
[Backend: FastAPI (mri_service)]
          │
          │ 3. Store raw scan in data/raw/<scan_id>/ & generate session ticket
          │ 4. Read volume header metadata (voxel spacing dx, dy, dz, dimensions, affine matrix)
          ▼
[AI Pipeline: Preprocessing Engine (ai/preprocessing)]
          │
          │ 5. Intensity Normalization (Z-score / Min-Max / Nyúl histogram matching)
          │ 6. Resampling to isotropic 1.0mm³ spacing (if required by model)
          │ 7. Skull stripping / Brain extraction / ROI clipping
          │ 8. Generate preprocessed tensor shape (e.g., [C, D, H, W])
          ▼
[AI Pipeline: Inference Engine (ai/inference)]
          │
          │ 9. Load PyTorch / MONAI model onto CPU device (`torch.device('cpu')`)
          │ 10. Sliding window inference / full-volume forward pass (torch.no_grad())
          │ 11. Softmax/Sigmoid probability thresholding (e.g., threshold > 0.5)
          ▼
[AI Pipeline: Post-processing & Volumetric Quantification (ai/inference)]
          │
          │ 12. Connected component labeling & 3D Bounding Box localization (Xmin..Xmax, Ymin..Ymax, Zmin..Zmax)
          │ 13. Tumor presence detection (Binary classification confidence)
          │ 14. 2D Maximum Axial Slice Area estimation (mm² = voxel_count * dx * dy)
          │ 15. 3D Tumor Volume calculation (mm³ / cm³ = total_positive_voxels * (dx * dy * dz))
          ▼
[Backend: Result Aggregation & Reporting (services/report_service)]
          │
          │ 16. Compile JSON response payload + Generate 2D slice previews (PNG/Base64 with overlays)
          │ 17. Render downloadable Clinical Decision-Support PDF Summary
          ▼
[Frontend: Visualization & Review]
          │
          │ 18. Multi-planar reconstruction viewer (Axial, Coronal, Sagittal)
          │ 19. Interactive mask toggle, opacity slider, bounding box highlights
          │ 20. Volume and surgical trajectory planning metric display
```

---

## Data Transformation States

| Stage | Input Representation | Output Representation | Primary Tool/Library |
| :--- | :--- | :--- | :--- |
| **Ingestion** | `.nii` / `.nii.gz` binary stream | Validated file on disk + header dict | FastAPI, `shutil`, `tempfile` |
| **Parsing** | File on disk | 3D/4D NumPy array + affine matrix + voxel spacing `(dx, dy, dz)` | `nibabel`, `SimpleITK` |
| **Preprocessing** | Raw NumPy array | Normalized, cropped PyTorch Tensor `[1, C, D, H, W]` | `monai.transforms`, `numpy` |
| **Inference** | Preprocessed Tensor | Logits / Segmentation Probability Map `[1, Classes, D, H, W]` | `torch` (CPU eval mode) |
| **Quantification** | Binary Mask `[D, H, W]` + Voxel spacing | Bounding Box `[x1, y1, z1, x2, y2, z2]`, Volume `cm³`, Area `mm²` | `scipy.ndimage`, `numpy` |
| **Presentation** | Slices + Mask overlays + Metrics | Interactive HTML5 Canvas viewer + JSON report | React, Tailwind, Lucide |
