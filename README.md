# NeuroSurgical-AI: Brain Tumor Detection and Localization for Surgical Planning

> **ACADEMIC RESEARCH PROTOTYPE NOTICE:**  
> This software is developed as a Bachelor of Engineering (B.E.) Computer Engineering final-year capstone project. It is strictly a **clinical decision-support prototype** (CADe/CADx) and is **NOT** a certified medical diagnostic device. It must **never** be used as a standalone or primary diagnostic tool without licensed radiological and neurosurgical verification.

---

## 1. Project Overview & Objectives
Brain tumor surgery requires millimetric precision to maximize pathological tissue resection while sparing critical eloquent brain parenchyma. This project establishes an AI-assisted computational framework for:
1. **Brain MRI Multi-Modal Ingestion:** Processing 3D volumetric scans (T1, T1ce, T2, FLAIR) in NIfTI format (`.nii`, `.nii.gz`).
2. **Standardized Preprocessing:** Intensity normalization (Z-score), skull-stripping/cropping, and isotropic voxel resampling.
3. **Deep Learning Tumor Detection & Localization:** Identification of neoplastic tissue presence and spatial 3D bounding box coordinates.
4. **Sub-Region Segmentation:** Delineation of tumor core, enhancing tumor, and peritumoral edema.
5. **Quantitative Surgical Volumetrics:** Rigorous estimation of maximum 2D cross-sectional area ($mm^2$) and 3D tumor volume ($cm^3$ / $mm^3$) when valid physical voxel dimensions ($dx, dy, dz$) are present in scan headers.
6. **Pre-Surgical Decision-Support Reporting:** Multi-planar reconstruction (Axial, Coronal, Sagittal) previews and downloadable clinical planning summaries.

---

## 2. System Architecture & Folder Structure

```
.
├── backend/                  # FastAPI Application & API Orchestration
│   ├── app/
│   │   ├── api/v1/           # Versioned API routes (health, mri, inference, reports)
│   │   ├── core/             # Pydantic Settings, environment configs, logging
│   │   ├── schemas/          # Pydantic request/response validation schemas
│   │   └── services/         # Orchestration (MRI storage, decoupled AI caller, reports)
│   ├── requirements.txt      # FastAPI, Uvicorn, Pydantic dependencies
│   └── .env.example          # Backend environment template
│
├── ai/                       # Standalone AI/ML Pipeline (Framework-agnostic)
│   ├── config/               # Hardware (CPU/CUDA), patch size, and threshold configs
│   ├── data/                 # Dataset reader & PyTorch transform interfaces
│   ├── models/               # Base segmentation contracts & model registry
│   ├── preprocessing/        # NIfTI header inspection, normalizer, and resampling
│   ├── inference/            # Standalone predictor & execution pipeline
│   ├── evaluation/           # Dice coefficient, IoU, Area & Volume formulas
│   ├── requirements-local.txt# PyTorch CPU dependencies for local laptop
│   └── requirements-colab.txt# PyTorch CUDA dependencies for Google Colab GPU
│
├── frontend/                 # React + TypeScript + Vite + Tailwind CSS Client
│   ├── src/
│   │   ├── components/       # UI widgets, upload dropzone, viewer canvases
│   │   ├── services/         # Typed API clients for FastAPI communication
│   │   └── types/            # TypeScript interfaces matching backend schemas
│   ├── package.json
│   └── .env.example
│
├── data/                     # Data directories (Protected by .gitignore)
│   ├── raw/                  # Original patient NIfTI scans
│   ├── interim/              # Resampled & aligned intermediate arrays
│   ├── processed/            # Normalized tensors ready for batch testing
│   └── README.md
│
├── models/                   # Model weight storage
│   ├── checkpoints/          # Training iteration checkpoints (Google Colab)
│   ├── exports/              # Serialized CPU-mapped weights for local inference
│   └── README.md
│
├── notebooks/                # Research & cloud execution notebooks
│   ├── 01_brats_data_exploration.ipynb
│   ├── 02_colab_training_pipeline.ipynb
│   └── README.md
│
├── reports/                  # Pre-surgical summaries
│   ├── templates/            # Markdown & PDF planning summary templates
│   ├── generated/            # Auto-generated clinical review outputs (gitignored)
│   └── README.md
│
├── tests/                    # Verification suite
│   ├── backend/              # Health & schema unit tests
│   ├── ai/                   # Metrics math & interface validation tests
│   └── conftest.py
│
└── docs/                     # Comprehensive engineering & regulatory docs
    ├── architecture.md       # Architectural principles & subsystem contracts
    ├── data_flow.md          # Step-by-step transformation flow diagram
    ├── hardware_and_environment.md # CPU vs Colab GPU separation strategy
    ├── colab_training_guide.md     # Colab execution & weight export workflow
    ├── api_specification.md  # Detailed REST endpoint documentation
    └── medical_disclaimer.md # Formal medical prototype disclaimer
```

---

## 3. Hardware & Environment Separation

| Feature | Local Laptop (Windows) | Cloud Training (Google Colab) |
| :--- | :--- | :--- |
| **Processor** | Intel Core CPU (Intel Iris Xe graphics) | NVIDIA Tesla T4 / V100 / A100 GPU |
| **CUDA Support** | **None** (CPU-only execution) | **Yes** (CUDA 12.x / cuDNN) |
| **Tasks** | Backend API, React Web UI, Testing, Single-scan CPU inference | Multi-epoch 3D U-Net training on BraTS dataset |
| **PyTorch Wheel** | `pip install -r ai/requirements-local.txt` (CPU wheel) | `pip install -r ai/requirements-colab.txt` (CUDA wheel) |
| **Model Weights** | Loads exported weights with `map_location='cpu'` | Exports trained weights as `.pth` |

---

## 4. Local Development Quickstart

### Prerequisites
- Python 3.10+ installed
- Node.js 18+ & npm installed
- VS Code with Python and Prettier extensions

### Step 1: Clone and Set Up Python Virtual Environment
```bash
# In Windows PowerShell / Command Prompt
git clone <YOUR_REPOSITORY_URL>
cd <PROJECT_ROOT>

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.venv\Scripts\activate.bat
```

### Step 2: Install Backend & AI CPU Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install backend dependencies
pip install -r backend/requirements.txt

# Install AI CPU-only dependencies
pip install -r ai/requirements-local.txt
```

### Step 3: Run the FastAPI Backend
```bash
# Copy backend environment template
cd backend
cp .env.example .env

# Start FastAPI server on port 8000
uvicorn app.main:app --reload --port 8000
```
- Interactive Swagger API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`

### Step 4: Run the React Frontend
```bash
# In a separate terminal
cd frontend
cp .env.example .env
npm install
npm run dev
```
- Access the web interface at `http://localhost:3000` (or `http://localhost:5173`)

### Step 5: Execute Test Suite
```bash
# From project root with .venv activated
pytest tests/ -v
```

---

## 5. Development Roadmap
- **Module 1 (Current):** System Architecture, Directory Blueprint, Interface Contracts, and Environment Configuration.
- **Module 2:** NIfTI MRI Data Ingestion, Voxel Spacing Extraction, and MONAI Preprocessing Pipelines.
- **Module 3:** Deep Learning Architecture (3D U-Net / MONAI) & Google Colab Training Notebooks.
- **Module 4:** Local CPU Model Inference, Sliding Window Predictor, and Connected-Component Localization.
- **Module 5:** Quantitative Volumetrics (Area $mm^2$, Volume $cm^3$) & 3D Bounding Box Logic.
- **Module 6:** Interactive Multi-Planar React Visualization & Pre-Surgical PDF Report Generation.
- **Module 7:** Comprehensive Model Evaluation (Dice, IoU, Hausdorff95) on BraTS Benchmark Cohorts.
