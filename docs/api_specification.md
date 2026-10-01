# Backend REST API Specification

**Base URL:** `/api/v1`  
**Standard Header:** `Accept: application/json`

---

## 1. System Health
### `GET /health`
Returns the status of the backend API, environment configuration, and device availability (CPU).

#### Response (200 OK)
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cpu",
  "model_loaded": false,
  "disclaimer": "Academic decision-support prototype. Not for medical diagnosis."
}
```

---

## 2. MRI Ingestion & Metadata Extraction
### `POST /mri/upload`
Uploads a multi-modal or single 3D MRI file (`.nii`, `.nii.gz`).

#### Request
- Form Data: `file: UploadFile` (Binary stream)
- Optional Query/Form: `sequence_type` (`t1`, `t1ce`, `t2`, `flair`, or `multimodal`)

#### Response (201 Created)
```json
{
  "scan_id": "mri_9b2e87c0_42a1",
  "filename": "patient_042_flair.nii.gz",
  "dimensions": [240, 240, 155],
  "voxel_spacing_mm": [1.0, 1.0, 1.0],
  "total_slices": 155,
  "status": "stored_ready_for_preprocess"
}
```

---

## 3. Preprocessing
### `POST /mri/{scan_id}/preprocess`
Executes brain skull-stripping/cropping, affine normalization, and min-max/Z-score normalization.

#### Response (200 OK)
```json
{
  "scan_id": "mri_9b2e87c0_42a1",
  "status": "preprocessed",
  "normalized_shape": [1, 128, 128, 128],
  "voxel_volume_mm3": 1.0
}
```

---

## 4. Tumor Detection & Segmentation Inference
### `POST /inference/predict/{scan_id}`
Triggers inference through the decoupled AI pipeline.

#### Response (200 OK)
```json
{
  "scan_id": "mri_9b2e87c0_42a1",
  "tumor_detected": true,
  "detection_confidence": 0.942,
  "bounding_box_3d": {
    "x_min": 64,
    "x_max": 112,
    "y_min": 80,
    "y_max": 128,
    "z_min": 45,
    "z_max": 78
  },
  "quantitative_metrics": {
    "max_slice_area_mm2": 842.5,
    "estimated_volume_cm3": 14.82,
    "voxel_count": 14820,
    "valid_voxel_metadata": true
  },
  "slices_with_tumor": [45, 46, 47, 48, 77, 78],
  "status": "completed"
}
```

---

## 5. Decision-Support Report
### `GET /reports/{scan_id}/download`
Generates and downloads a structured PDF clinical summary report including slice captures, tumor coordinates, volume measurements, and explicit medical disclaimer notices.
