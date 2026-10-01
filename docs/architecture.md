# System Architecture Specification

## Project Overview
**Title:** AI Based Brain Tumor Detection and Localization for Surgical Planning  
**Level:** Final-Year B.E. Computer Engineering Capstone Project  
**Category:** Medical Image Processing & Clinical Decision Support Prototype  

---

## 1. Architectural Principles
1. **Strict Decoupling:** The core AI pipeline is decoupled from web frameworks (FastAPI/React). It can execute as a standalone Python CLI or library for research experiments and batch evaluations.
2. **Layered Separation of Concerns:**
   - **Presentation Layer (Frontend):** React + TypeScript + Tailwind CSS for MRI slice navigation, overlay opacity control, bounding boxes, size/volume display, and PDF report triggers.
   - **API & Orchestration Layer (Backend):** FastAPI + Pydantic for validation, asynchronous processing, job status tracking, file storage handling, and dispatching to the AI inference service.
   - **Inference & Processing Layer (AI Core):** PyTorch + MONAI + NiBabel + SimpleITK for NIfTI I/O, intensity normalization, affine registration/resampling, inference execution, and post-processing (connected component analysis, localization bounding cubes, voxel-based volumetric quantification).
   - **Training & Experimentation Layer:** Google Colab GPU notebooks using the BraTS dataset, outputting versioned model weights (`.pt`/`.pth`) to be exported for local CPU inference.
3. **Environment Adaptability:** Local execution relies entirely on standard x86 CPU instructions (Intel Iris Xe without CUDA requirement). Heavy GPU training is delegated to cloud environments (Google Colab).
4. **Configuration-Driven:** Zero hardcoded paths; all parameters (slice thresholds, model checkpoint paths, voxel spacing tolerances, upload limits) are managed via `.env` and Pydantic Settings.

---

## 2. Directory Architecture & Layer Responsibilities

```
root/
├── frontend/        # React + TypeScript + Vite + Tailwind CSS Web Client
├── backend/         # FastAPI REST service & workflow orchestration
├── ai/              # Pure AI/ML pipeline (PyTorch, MONAI, preprocessing, metrics)
├── data/            # Local data directories (raw, processed, interim - gitignored)
├── models/          # Checkpoint storage and exported weights (gitignored)
├── notebooks/       # Jupyter notebooks for exploratory data analysis & Colab GPU training
├── reports/         # PDF/Markdown clinical decision-support report templates & outputs
├── tests/           # Unit, integration, and interface verification test suite
└── docs/            # Architecture, API, Colab guides, and medical disclaimers
```

---

## 3. Communication Contracts
- **Frontend to Backend:** HTTP REST via JSON payloads and `multipart/form-data` for MRI uploads. Endpoints follow versioned REST standards (`/api/v1/...`).
- **Backend to AI Core:** Direct Python module invocation via typed Service interfaces (`InferenceService` -> `AIPipeline.predict()`). No network overhead between backend and AI engine on local deployment.
- **Model Storage:** Versioned model weights loaded from `models/exports/` configured via `MODEL_WEIGHTS_PATH` in backend environment.
