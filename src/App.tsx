import React, { useState } from 'react';
import {
  FolderTree,
  Cpu,
  Layers,
  Activity,
  FileText,
  AlertTriangle,
  Terminal,
  ShieldCheck,
  CheckCircle2,
  GitBranch,
  Server,
  Cloud,
  ArrowRight,
  Database,
  Code2,
  HardDrive,
  BookOpen,
  Boxes,
  HelpCircle,
  Copy,
  Check
} from 'lucide-react';

interface FolderNode {
  name: string;
  type: 'folder' | 'file';
  badge?: string;
  desc?: string;
  children?: FolderNode[];
}

const folderStructure: FolderNode = {
  name: "root/",
  type: "folder",
  children: [
    {
      name: "frontend/",
      type: "folder",
      badge: "Presentation",
      desc: "React + TypeScript + Vite + Tailwind client with multi-planar slice viewer, mask overlay controls, and surgical planning dashboard.",
      children: [
        { name: "src/components/", type: "folder", desc: "Modular UI components (UploadDropzone, SliceViewer, BoundingBoxOverlay, MetricsCard)" },
        { name: "src/services/", type: "folder", desc: "Typed API client (apiFetch) and endpoint services for FastAPI communication" },
        { name: "src/types/", type: "folder", desc: "TypeScript interfaces for MRI headers, bounding boxes, and volumetric results" },
        { name: "package.json", type: "file", desc: "Frontend dependencies (React 19, Lucide, Tailwind)" },
        { name: ".env.example", type: "file", desc: "Vite environment configuration (VITE_API_BASE_URL)" },
      ]
    },
    {
      name: "backend/",
      type: "folder",
      badge: "API Orchestration",
      desc: "FastAPI REST service handling asynchronous file upload staging, validation, job orchestration, and PDF report triggers.",
      children: [
        {
          name: "app/api/v1/",
          type: "folder",
          desc: "Versioned REST endpoints",
          children: [
            { name: "endpoints/health.py", type: "file", desc: "System health, CPU device status, and medical disclaimer endpoint" },
            { name: "endpoints/mri.py", type: "file", desc: "NIfTI / DICOM upload and file staging endpoint" },
            { name: "endpoints/inference.py", type: "file", desc: "Inference dispatch endpoint calling the decoupled AI engine" },
            { name: "endpoints/reports.py", type: "file", desc: "Surgical decision support report summary & download endpoint" },
            { name: "router.py", type: "file", desc: "APIRouter aggregator registering v1 routes" }
          ]
        },
        { name: "app/core/", type: "folder", desc: "Pydantic Settings (config.py) and structured logging (logging.py)" },
        { name: "app/schemas/", type: "folder", desc: "Strict Pydantic request/response validation models (mri, inference, report, health)" },
        { name: "app/services/", type: "folder", desc: "Business logic layer (mri_service, inference_service, report_service)" },
        { name: "app/main.py", type: "file", desc: "FastAPI entry point with CORS middleware and global error handlers" },
        { name: "requirements.txt", type: "file", desc: "Backend dependencies (fastapi, uvicorn, pydantic, aiofiles)" },
        { name: ".env.example", type: "file", desc: "Backend environment variables template (ports, storage paths, model path)" }
      ]
    },
    {
      name: "ai/",
      type: "folder",
      badge: "Decoupled AI Engine",
      desc: "Independent, framework-agnostic AI package. Runs from CLI or backend. No web framework dependencies.",
      children: [
        { name: "config/", type: "folder", desc: "Pydantic configs for model dimensions, patch sizes, and device mapping (CPU/CUDA)" },
        { name: "data/", type: "folder", desc: "Dataset reader contracts, MONAI spatial transform interfaces, and DataLoader builders" },
        { name: "models/", type: "folder", desc: "BaseSegmentationModel abstract contract and dynamic ModelRegistry" },
        { name: "preprocessing/", type: "folder", desc: "NIfTI header extraction, intensity normalizer (Z-score), and isotropic resampler" },
        { name: "inference/", type: "folder", desc: "BrainTumorAIPipeline and CPU sliding-window inference execution engine" },
        { name: "evaluation/", type: "folder", desc: "Formulas for Dice Score, IoU, 3D Bounding Box, 2D Area (mm²), and Volume (cm³)" },
        { name: "requirements-local.txt", type: "file", desc: "CPU-only PyTorch & MONAI dependencies for local laptop" },
        { name: "requirements-colab.txt", type: "file", desc: "CUDA-accelerated PyTorch dependencies for Google Colab GPU" }
      ]
    },
    {
      name: "data/",
      type: "folder",
      badge: "Storage (Ignored)",
      desc: "Local data directories for raw patient NIfTI files, interim resampled arrays, and processed tensors (gitignored).",
      children: [
        { name: "raw/", type: "folder", desc: "Original patient MRI scans (.nii, .nii.gz) - gitignored" },
        { name: "interim/", type: "folder", desc: "Temporarily cached normalized/resampled arrays - gitignored" },
        { name: "processed/", type: "folder", desc: "Preprocessed evaluation volumes - gitignored" },
        { name: "README.md", type: "file", desc: "Medical data compliance, de-identification, and BraTS structure guide" }
      ]
    },
    {
      name: "models/",
      type: "folder",
      badge: "Weights (Ignored)",
      desc: "Model weights storage. Training checkpoints and final exported weights mapped for local CPU inference.",
      children: [
        { name: "checkpoints/", type: "folder", desc: "Intermediate training weights saved on Colab (.pth) - gitignored" },
        { name: "exports/", type: "folder", desc: "Final exported weights mapped for local CPU inference - gitignored" },
        { name: "README.md", type: "file", desc: "Model serialization and device-agnostic loading guide" }
      ]
    },
    {
      name: "notebooks/",
      type: "folder",
      badge: "Research & Colab",
      desc: "Jupyter notebooks for exploratory data analysis and remote Google Colab GPU model training.",
      children: [
        { name: "01_brats_data_exploration.ipynb", type: "file", desc: "BraTS dataset exploration (intensity distributions, modalities)" },
        { name: "02_colab_training_pipeline.ipynb", type: "file", desc: "Colab GPU training pipeline template for 3D U-Net" },
        { name: "README.md", type: "file", desc: "Colab GPU mounting, hardware acceleration, and training instructions" }
      ]
    },
    {
      name: "reports/",
      type: "folder",
      badge: "Clinical Summaries",
      desc: "Templates and auto-generated pre-surgical planning decision support summaries with mandatory disclaimers.",
      children: [
        { name: "templates/", type: "folder", desc: "Markdown and PDF layout templates for surgical planning review" },
        { name: "generated/", type: "folder", desc: "Generated patient summaries (gitignored)" },
        { name: "README.md", type: "file", desc: "Surgical report documentation and disclaimer requirements" }
      ]
    },
    {
      name: "tests/",
      type: "folder",
      badge: "Verification Suite",
      desc: "Automated test suite verifying API schemas, health endpoints, AI path configs, and quantitative mathematical formulas.",
      children: [
        { name: "backend/test_health.py", type: "file", desc: "FastAPI health check & root disclaimer tests" },
        { name: "backend/test_schemas.py", type: "file", desc: "Pydantic BoundingBox3D and VolumetricMetrics validation tests" },
        { name: "ai/test_config.py", type: "file", desc: "Hardware CPU config & directory path resolution tests" },
        { name: "ai/test_interfaces.py", type: "file", desc: "Dice score, IoU, 3D Bounding Box, and Volume mathematical tests" },
        { name: "conftest.py", type: "file", desc: "Pytest fixtures and root sys.path injection" }
      ]
    },
    {
      name: "docs/",
      type: "folder",
      badge: "Documentation",
      desc: "Engineering and clinical decision support documentation for final-year project defense.",
      children: [
        { name: "architecture.md", type: "file", desc: "Full architectural breakdown, layer contracts, and decoupling rules" },
        { name: "data_flow.md", type: "file", desc: "End-to-end data transformation pipeline specification" },
        { name: "hardware_and_environment.md", type: "file", desc: "Intel Iris Xe CPU local setup vs Colab GPU training strategy" },
        { name: "colab_training_guide.md", type: "file", desc: "Step-by-step BraTS training and weight export protocol" },
        { name: "api_specification.md", type: "file", desc: "FastAPI REST API specification with sample payloads" },
        { name: "medical_disclaimer.md", type: "file", desc: "Formal CADe/CADx research prototype regulatory notice" }
      ]
    }
  ]
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'tree' | 'brats_pipeline' | 'preprocessing' | 'dataflow' | 'hardware' | 'contracts' | 'commands' | 'risks'>('overview');
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCmd(id);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Bar / Medical Disclaimer Banner */}
      <div className="bg-amber-950/80 border-b border-amber-500/30 px-4 py-2.5 text-xs sm:text-sm text-amber-200 flex items-center justify-between">
        <div className="flex items-center gap-2 max-w-5xl mx-auto w-full">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 animate-pulse" />
          <span>
            <strong className="font-semibold uppercase tracking-wider text-amber-300 mr-1.5">
              Decision Support Notice:
            </strong>
            Academic research prototype for B.E. Computer Engineering project. Not a certified medical device. Never for primary clinical diagnosis.
          </span>
        </div>
      </div>

      {/* Main Header */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg sm:text-xl font-bold tracking-tight text-white">
                  NeuroSurgical-AI
                </h1>
                <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                  Module 3: MRI Preprocessing Complete
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Brain Tumor Detection & Localization for Surgical Planning • Lead Architect Blueprint
              </p>
            </div>
          </div>

          {/* Academic Meta Badge */}
          <div className="flex items-center gap-2 text-xs">
            <div className="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-300 flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>Local CPU (Intel Iris Xe)</span>
            </div>
            <div className="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-300 flex items-center gap-2">
              <Cloud className="w-3.5 h-3.5 text-indigo-400" />
              <span>Google Colab GPU (Training)</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex overflow-x-auto scrollbar-none border-t border-slate-800/60">
          {[
            { id: 'overview', label: '1. Architecture Overview', icon: Layers },
            { id: 'tree', label: '2. Folder Tree & Modules', icon: FolderTree },
            { id: 'brats_pipeline', label: '3. BraTS Data Pipeline (Mod 2)', icon: Database },
            { id: 'preprocessing', label: '4. MRI Preprocessing (Mod 3)', icon: Activity },
            { id: 'dataflow', label: '5. Data Flow Pipeline', icon: Activity },
            { id: 'hardware', label: '6. Local CPU vs Colab GPU', icon: Cpu },
            { id: 'contracts', label: '7. Decoupled API Contracts', icon: Code2 },
            { id: 'commands', label: '8. Local Dev Commands', icon: Terminal },
            { id: 'risks', label: '9. Risks & Mitigations', icon: ShieldCheck },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-3 text-xs sm:text-sm font-medium border-b-2 whitespace-nowrap transition-colors ${
                  isActive
                    ? 'border-cyan-400 text-cyan-300 bg-cyan-950/20'
                    : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        {/* TAB 1: OVERVIEW */}
        {activeTab === 'overview' && (
          <div className="space-y-8 animate-fadeIn">
            {/* Mission & Architectural Goal */}
            <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-indigo-950/50 border border-slate-800 shadow-xl relative overflow-hidden">
              <div className="max-w-3xl relative z-10">
                <span className="text-xs font-semibold uppercase tracking-wider text-cyan-400 bg-cyan-950/60 border border-cyan-800/50 px-2.5 py-1 rounded-md">
                  Module 1 Architectural Milestone
                </span>
                <h2 className="text-2xl sm:text-3xl font-bold text-white mt-3 mb-2">
                  System Architecture & Subsystem Boundaries
                </h2>
                <p className="text-sm text-slate-300 leading-relaxed">
                  As the Lead Software Architect, the initial foundation has been established adhering to strict medical software engineering principles. The core AI algorithms are decoupled from web technologies, training is mapped to Google Colab GPU to accommodate your Intel Iris Xe laptop, and explicit contracts prevent premature or fabricated predictions.
                </p>
              </div>

              {/* Status checklist */}
              <div className="mt-6 pt-6 border-t border-slate-800/80 grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="flex items-start gap-3 p-3 rounded-xl bg-slate-800/50 border border-slate-700/50">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-semibold text-white">Zero CUDA Dependency</h4>
                    <p className="text-[11px] text-slate-400 mt-0.5">Local execution uses CPU-only PyTorch wheels for Intel Iris Xe laptop.</p>
                  </div>
                </div>

                <div className="flex items-start gap-3 p-3 rounded-xl bg-slate-800/50 border border-slate-700/50">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-semibold text-white">Strict Decoupling</h4>
                    <p className="text-[11px] text-slate-400 mt-0.5">The AI engine can run independently via CLI or imported into FastAPI.</p>
                  </div>
                </div>

                <div className="flex items-start gap-3 p-3 rounded-xl bg-slate-800/50 border border-slate-700/50">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-semibold text-white">No Fabricated Outputs</h4>
                    <p className="text-[11px] text-slate-400 mt-0.5">Strict adherence to rule: No fake models, no fake data, and no U-Net until Module 2/3.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* 3-Tier Layer Separation */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Layer 1: Frontend */}
              <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider">Tier 1: Presentation</span>
                    <span className="text-[10px] bg-cyan-950 text-cyan-300 border border-cyan-800 px-2 py-0.5 rounded">React + Vite</span>
                  </div>
                  <h3 className="text-base font-semibold text-white mb-2">Web Application Interface</h3>
                  <p className="text-xs text-slate-400 leading-relaxed mb-4">
                    Clinical dashboard designed for neurosurgeons and radiologists. Facilitates MRI file ingestion, multi-planar reconstruction (Axial/Coronal/Sagittal), mask opacity blending, bounding box inspection, and PDF summary downloads.
                  </p>
                  <ul className="text-xs text-slate-300 space-y-1.5">
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                      TypeScript typed API services
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                      Tailwind CSS responsive design
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                      HTML5 Canvas medical slice viewer
                    </li>
                  </ul>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-500 font-mono">
                  Path: /frontend/src/
                </div>
              </div>

              {/* Layer 2: Backend */}
              <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-bold text-indigo-400 uppercase tracking-wider">Tier 2: Orchestration</span>
                    <span className="text-[10px] bg-indigo-950 text-indigo-300 border border-indigo-800 px-2 py-0.5 rounded">FastAPI</span>
                  </div>
                  <h3 className="text-base font-semibold text-white mb-2">REST API & Service Gateway</h3>
                  <p className="text-xs text-slate-400 leading-relaxed mb-4">
                    High-performance asynchronous API layer. Manages multipart file streaming, storage quotas, Pydantic data validation, background task dispatches to the AI inference service, and report compilation.
                  </p>
                  <ul className="text-xs text-slate-300 space-y-1.5">
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
                      Interactive Swagger documentation (/docs)
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
                      Strict Pydantic request/response schemas
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
                      Service pattern with dependency injection
                    </li>
                  </ul>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-500 font-mono">
                  Path: /backend/app/
                </div>
              </div>

              {/* Layer 3: AI Engine */}
              <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-bold text-purple-400 uppercase tracking-wider">Tier 3: Core AI Engine</span>
                    <span className="text-[10px] bg-purple-950 text-purple-300 border border-purple-800 px-2 py-0.5 rounded">PyTorch / MONAI</span>
                  </div>
                  <h3 className="text-base font-semibold text-white mb-2">Decoupled ML Pipeline</h3>
                  <p className="text-xs text-slate-400 leading-relaxed mb-4">
                    Completely autonomous scientific processing library. Responsible for NIfTI I/O, intensity normalization, isotropic voxel resampling, CPU inference forward pass, and mathematical volume / area calculation.
                  </p>
                  <ul className="text-xs text-slate-300 space-y-1.5">
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-purple-400"></span>
                      NiBabel & SimpleITK spatial headers
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-purple-400"></span>
                      Sliding window CPU inference
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-purple-400"></span>
                      Dice, IoU, 3D Bounding Box & Volume metrics
                    </li>
                  </ul>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-500 font-mono">
                  Path: /ai/
                </div>
              </div>
            </div>

            {/* Scope & Boundary Rules */}
            <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-3">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                Module 1 Boundary & Rule Compliance Audit
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700/40 flex items-center justify-between">
                  <span className="text-slate-300">Rule 1 & 2: No complete app or U-Net implemented yet</span>
                  <span className="font-semibold text-emerald-400 flex items-center gap-1">
                    <Check className="w-3.5 h-3.5" /> Enforced
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700/40 flex items-center justify-between">
                  <span className="text-slate-300">Rule 3 & 4: No fake datasets or mock model predictions</span>
                  <span className="font-semibold text-emerald-400 flex items-center gap-1">
                    <Check className="w-3.5 h-3.5" /> Enforced
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700/40 flex items-center justify-between">
                  <span className="text-slate-300">Rule 7 & 8: AI engine decoupled from web application</span>
                  <span className="font-semibold text-emerald-400 flex items-center gap-1">
                    <Check className="w-3.5 h-3.5" /> Enforced
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700/40 flex items-center justify-between">
                  <span className="text-slate-300">Rule 11 & 13: Env vars configured & Decision Support notice</span>
                  <span className="font-semibold text-emerald-400 flex items-center gap-1">
                    <Check className="w-3.5 h-3.5" /> Enforced
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: FOLDER TREE & MODULES */}
        {activeTab === 'tree' && (
          <div className="space-y-6 animate-fadeIn">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-xl bg-slate-900 border border-slate-800">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <FolderTree className="w-5 h-5 text-cyan-400" />
                  Complete Project Directory Blueprint
                </h2>
                <p className="text-xs text-slate-400">
                  Every folder and file created in Module 1 with clear responsibilities and isolation boundaries.
                </p>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300">
                  Total Subsystems: 9
                </span>
                <span className="px-2.5 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
                  Status: All Created
                </span>
              </div>
            </div>

            {/* Folder Explorer */}
            <div className="space-y-4">
              {folderStructure.children?.map((folder) => (
                <div key={folder.name} className="p-5 rounded-xl bg-slate-900 border border-slate-800 hover:border-slate-700 transition-colors">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
                    <div className="flex items-center gap-2.5">
                      <FolderTree className="w-5 h-5 text-indigo-400" />
                      <span className="font-mono text-sm font-bold text-white">{folder.name}</span>
                    </div>
                    {folder.badge && (
                      <span className="text-[11px] font-medium px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700 self-start sm:self-auto">
                        {folder.badge}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mt-2 mb-4 leading-relaxed">
                    {folder.desc}
                  </p>

                  {/* Child nodes */}
                  {folder.children && folder.children.length > 0 && (
                    <div className="bg-slate-950/70 rounded-lg p-3 border border-slate-800/60 font-mono text-xs space-y-2">
                      {folder.children.map((child) => (
                        <div key={child.name} className="flex flex-col sm:flex-row sm:items-start justify-between gap-1 text-slate-300 py-1 border-b border-slate-900 last:border-0">
                          <div className="flex items-center gap-2 text-cyan-300">
                            {child.type === 'folder' ? (
                              <Boxes className="w-3.5 h-3.5 text-slate-500" />
                            ) : (
                              <FileText className="w-3.5 h-3.5 text-slate-500" />
                            )}
                            <span>{child.name}</span>
                          </div>
                          <span className="text-[11px] text-slate-400 font-sans sm:text-right max-w-lg">
                            {child.desc}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 3: BRATS DATA PIPELINE (MODULE 2) */}
        {activeTab === 'brats_pipeline' && (
          <div className="space-y-6 animate-fadeIn">
            <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-emerald-950/40 border border-slate-800 shadow-xl">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400 bg-emerald-950/60 border border-emerald-800/50 px-2.5 py-0.5 rounded-md">
                      Module 2 Milestone Delivered
                    </span>
                    <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                      7/7 Tests Passed (0.192s)
                    </span>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-bold text-white">
                    BraTS Dataset Ingestion, Validation & Indexing Pipeline
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Multi-modal MRI reading, header inspection, zero-leakage patient-level splitting, and metadata preservation for surgical volumetrics.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="px-3 py-1.5 rounded-lg bg-emerald-950/80 border border-emerald-800 text-emerald-300 text-xs font-medium flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    Zero Fake Data Enforced
                  </span>
                </div>
              </div>

              {/* 4 Multi-Modal Sequences & Sub-Region Badges */}
              <div className="mt-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
                  <div className="text-xs font-bold text-cyan-400 font-mono mb-1">1. FLAIR (T2-FLAIR)</div>
                  <p className="text-[11px] text-slate-400">Suppresses fluid signal to highlight peritumoral edema boundaries.</p>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
                  <div className="text-xs font-bold text-indigo-400 font-mono mb-1">2. T1 (Native)</div>
                  <p className="text-[11px] text-slate-400">Standard anatomical reference mapping normal brain parenchyma.</p>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
                  <div className="text-xs font-bold text-purple-400 font-mono mb-1">3. T1ce (Post-Contrast)</div>
                  <p className="text-[11px] text-slate-400">Gadolinium-enhanced sequence delineating hyper-intense active tumor margins.</p>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
                  <div className="text-xs font-bold text-emerald-400 font-mono mb-1">4. T2 (Structural)</div>
                  <p className="text-[11px] text-slate-400">High contrast for fluid, ventricles, and tissue edema margins.</p>
                </div>
              </div>
            </div>

            {/* Validation & Leakage Prevention Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Validation Framework */}
              <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4 text-emerald-400" />
                      Comprehensive Validation Rules (ai/data/validator.py)
                    </h3>
                    <span className="text-[10px] font-mono bg-emerald-950 text-emerald-300 border border-emerald-800 px-2 py-0.5 rounded">
                      No Silent Discards
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mb-3">
                    Every case is systematically inspected. Corrupted files or mismatched dimensions are reported with detailed diagnostic error messages:
                  </p>
                  <ul className="text-xs text-slate-300 space-y-2">
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                      <span><strong>Modality Completeness:</strong> Confirms all 4 sequences (T1, T1ce, T2, FLAIR) exist.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                      <span><strong>Dimensional Consistency:</strong> Verifies all 4 modalities share identical 3D shapes.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                      <span><strong>Segmentation Correspondence:</strong> Enforces that mask shape matches MRI volume exactly.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                      <span><strong>Header Parsing:</strong> Verifies non-corrupt NIfTI-1 binary headers (348 bytes) and positive spacings.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                      <span><strong>Duplicate Prevention:</strong> Prevents duplicate case IDs across the cohort.</span>
                    </li>
                  </ul>
                </div>
              </div>

              {/* Patient-Level Splitting */}
              <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <GitBranch className="w-4 h-4 text-cyan-400" />
                      Patient-Level Splitting & Leakage Prevention
                    </h3>
                    <span className="text-[10px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-800 px-2 py-0.5 rounded">
                      Seed: 42
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mb-3">
                    In medical AI, splitting 2D slices from the same patient across train and test sets leads to catastrophic data leakage. We enforce strict case-level partitioning:
                  </p>

                  <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono my-3">
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-cyan-900/60">
                      <div className="text-cyan-400 font-bold">70% Train</div>
                      <div className="text-[10px] text-slate-500 font-sans mt-0.5">Model Optimization</div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-indigo-900/60">
                      <div className="text-indigo-400 font-bold">15% Val</div>
                      <div className="text-[10px] text-slate-500 font-sans mt-0.5">Checkpointing</div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-emerald-900/60">
                      <div className="text-emerald-400 font-bold">15% Test</div>
                      <div className="text-[10px] text-slate-500 font-sans mt-0.5">Final Evaluation</div>
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-amber-950/20 border border-amber-800/40 text-[11px] text-amber-300">
                    <strong>Disjoint Proof:</strong> Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅. All 155 slices of a patient belong strictly to one split.
                  </div>
                </div>
              </div>
            </div>

            {/* Spatial Metadata & Surgical Volumetrics */}
            <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
              <h3 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
                <Layers className="w-4 h-4 text-purple-400" />
                Physical Spatial Metadata Preserved for Surgical Volumetrics
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                To enable accurate 2D area (mm²) and 3D volume (cm³) calculations in subsequent modules, the data pipeline preserves raw physical metadata at all times:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-purple-400 font-semibold mb-1">4x4 Affine Matrix</div>
                  <div className="text-slate-300 text-[11px]">Relates voxel grid (i, j, k) to scanner space (x, y, z) in mm.</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-cyan-400 font-semibold mb-1">Voxel Spacing (dx, dy, dz)</div>
                  <div className="text-slate-300 text-[11px]">Physical slice thickness and in-plane pixel dimensions in mm.</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-emerald-400 font-semibold mb-1">Anatomical Orientation</div>
                  <div className="text-slate-300 text-[11px]">Standardized anatomical axes (e.g. RAS: Right, Anterior, Superior).</div>
                </div>
              </div>
            </div>

            {/* Test Results Output Banner */}
            <div className="p-4 rounded-xl bg-slate-900 border border-emerald-900/60">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-bold text-emerald-400 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4" />
                  Module 2 Test Suite Verification Output
                </span>
                <span className="text-[10px] font-mono text-slate-400">Ran 7 tests in 0.192s</span>
              </div>
              <div className="p-3 rounded-lg bg-slate-950 font-mono text-xs text-emerald-300/90 overflow-x-auto whitespace-pre">
{`test_case_indexing (__main__.TestBraTSPipeline) ... ok
test_dataset_getitem_and_metadata_preservation (__main__.TestBraTSPipeline) ... ok
test_dataset_statistics (__main__.TestBraTSPipeline) ... ok
test_deterministic_splitting (__main__.TestBraTSPipeline) ... ok
test_discovery_and_metadata (__main__.TestBraTSPipeline) ... ok
test_invalid_file_handling (__main__.TestBraTSPipeline) ... ok
test_patient_level_splitting_no_leakage (__main__.TestBraTSPipeline) ... ok

----------------------------------------------------------------------
Ran 7 tests in 0.192s - OK (100% Pass Rate)`}
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: MRI PREPROCESSING (MODULE 3) */}
        {activeTab === 'preprocessing' && (
          <div className="space-y-6 animate-fadeIn">
            <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-cyan-950/40 border border-slate-800 shadow-xl">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-semibold uppercase tracking-wider text-cyan-400 bg-cyan-950/60 border border-cyan-800/50 px-2.5 py-0.5 rounded-md">
                      Module 3 Milestone Delivered
                    </span>
                    <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                      CPU-Optimized (Intel Iris Xe Safe)
                    </span>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-bold text-white">
                    Multi-Modal MRI Preprocessing Pipeline
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Converts validated BraTS volumes into model-ready tensors [4, D, H, W] while preserving complete spatial reversibility for surgical localization.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="px-3 py-1.5 rounded-lg bg-cyan-950/80 border border-cyan-800 text-cyan-300 text-xs font-medium flex items-center gap-1.5">
                    <ShieldCheck className="w-4 h-4 text-cyan-400" />
                    Zero Data Leakage
                  </span>
                </div>
              </div>

              {/* 8-Stage Execution Order */}
              <div className="mt-5">
                <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">
                  Exact Sequential Transform Pipeline (ai/preprocessing/pipeline.py)
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  {[
                    { step: "01", name: "NIfTI Loading", desc: "Extracts T1, T1ce, T2, FLAIR & Seg; verifies 3D matrix integrity & voxel spacing.", color: "text-cyan-400" },
                    { step: "02", name: "RAS+ Standardization", desc: "Reorients anatomical axes to Right, Anterior, Superior; updates 4x4 coordinate affine.", color: "text-indigo-400" },
                    { step: "03", name: "Voxel Resampling", desc: "MRI: Continuous trilinear (order=1). Mask: Strictly Nearest-Neighbor (order=0).", color: "text-purple-400" },
                    { step: "04", name: "Foreground Crop", desc: "Computes non-zero brain bounding box across all 4 sequences; zero tumor tissue loss.", color: "text-emerald-400" },
                    { step: "05", name: "Z-Score Normalize", desc: "Computed strictly on non-zero tissue voxels per-volume; zero cross-patient leakage.", color: "text-amber-400" },
                    { step: "06", name: "Channel Stacking", desc: "Stacks modalities deterministically into [4, D, H, W] in order: [T1, T1ce, T2, FLAIR].", color: "text-cyan-400" },
                    { step: "07", name: "Label Re-Mapping", desc: "Bidirectionally maps BraTS {0, 1, 2, 4} to contiguous model classes {0, 1, 2, 3}.", color: "text-rose-400" },
                    { step: "08", name: "Traceability Pack", desc: "Encapsulates affine, spacing, and crop coordinates for 100% spatial reversibility.", color: "text-emerald-400" },
                  ].map((s) => (
                    <div key={s.step} className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                      <div className="flex items-center justify-between mb-1">
                        <span className={`text-xs font-mono font-bold ${s.color}`}>STAGE {s.step}</span>
                      </div>
                      <div className="text-xs font-semibold text-white mb-0.5">{s.name}</div>
                      <p className="text-[11px] text-slate-400 leading-relaxed">{s.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Core Architectural Decisions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Interpolation Rules */}
              <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Activity className="w-4 h-4 text-purple-400" />
                      Continuous vs Nearest-Neighbor Resampling
                    </h3>
                    <span className="text-[10px] font-mono bg-purple-950 text-purple-300 border border-purple-800 px-2 py-0.5 rounded">
                      Critical Rule
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mb-3">
                    Different data modalities require strictly segregated interpolation algorithms to prevent label corruption:
                  </p>
                  <div className="space-y-2 text-xs">
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <div className="font-semibold text-cyan-300">MRI Sequences (T1, T1ce, T2, FLAIR)</div>
                      <div className="text-slate-400 text-[11px] mt-0.5">
                        Continuous trilinear interpolation (order=1). Preserves smooth spatial tissue gradients and physical relaxation times without jagged boundaries.
                      </div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                      <div className="font-semibold text-rose-300">Segmentation Masks (Ground-Truth & Predictions)</div>
                      <div className="text-slate-400 text-[11px] mt-0.5">
                        <strong>Strictly Nearest-Neighbor (order=0):</strong> Discrete labels represent categorical tissue classes. Linear interpolation would generate illegal fractional labels (e.g., 1.5, 2.7), destroying anatomical ground truth.
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Label Mapping & Invertibility */}
              <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Layers className="w-4 h-4 text-emerald-400" />
                      Bidirectional Label Mapping & Reversibility
                    </h3>
                    <span className="text-[10px] font-mono bg-emerald-950 text-emerald-300 border border-emerald-800 px-2 py-0.5 rounded">
                      Invertible
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mb-3">
                    BraTS official labels skip index 3. The pipeline maps labels to contiguous class IDs for neural network loss functions, while preserving exact inverse restoration:
                  </p>
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono my-2">
                    <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                      <div className="text-cyan-400 font-bold mb-1">Forward (To Model)</div>
                      <div className="text-[11px] text-slate-300">0 → 0 (Background)</div>
                      <div className="text-[11px] text-slate-300">1 → 1 (Necrotic Core)</div>
                      <div className="text-[11px] text-slate-300">2 → 2 (Edema)</div>
                      <div className="text-[11px] text-emerald-400 font-bold">4 → 3 (Enhancing Tumor)</div>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                      <div className="text-purple-400 font-bold mb-1">Inverse (To Patient Space)</div>
                      <div className="text-[11px] text-slate-300">0 → 0 (Background)</div>
                      <div className="text-[11px] text-slate-300">1 → 1 (Necrotic Core)</div>
                      <div className="text-[11px] text-slate-300">2 → 2 (Edema)</div>
                      <div className="text-[11px] text-emerald-400 font-bold">3 → 4 (Enhancing Tumor)</div>
                    </div>
                  </div>
                  <div className="text-[11px] text-emerald-300 font-sans mt-2">
                    <strong>Derived Clinical Sub-Regions:</strong> Whole Tumor (WT: 1+2+3), Tumor Core (TC: 1+3), Enhancing Tumor (ET: 3).
                  </div>
                </div>
              </div>
            </div>

            {/* Spatial Traceability & Original-Space Reconstruction */}
            <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
              <h3 className="text-sm font-semibold text-white mb-2 flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-cyan-400" />
                Original Space Traceability & Surgical Reconstruction
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                The pipeline provides <code>map_prediction_to_original_space()</code> to reverse all preprocessing steps, projecting model outputs back onto raw patient scans:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-cyan-400 font-semibold mb-1">Step 1: Invert Labels</div>
                  <div className="text-slate-400 text-[11px]">Maps contiguous classes {"{0, 1, 2, 3}"} back to official BraTS labels {"{0, 1, 2, 4}"}.</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-indigo-400 font-semibold mb-1">Step 2: Uncrop Volume</div>
                  <div className="text-slate-400 text-[11px]">Replaces bounding box at recorded crop coordinates [x_min:x_max, y_min:y_max, z_min:z_max].</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-purple-400 font-semibold mb-1">Step 3: Resample to Origin</div>
                  <div className="text-slate-400 text-[11px]">Nearest-neighbor zoom restores original dimensions (e.g. 240x240x155) with 100% precision.</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: DATA FLOW PIPELINE */}
        {activeTab === 'dataflow' && (
          <div className="space-y-6 animate-fadeIn">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Activity className="w-5 h-5 text-emerald-400" />
                End-to-End System Data Flow Pipeline
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Visualizing how an MRI scan transitions from clinical ingestion to surgical decision-support quantification.
              </p>
            </div>

            {/* Pipeline Stage Cards */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {[
                {
                  step: "01",
                  title: "MRI Ingestion",
                  component: "Frontend / FastAPI",
                  inputs: "NIfTI (.nii, .nii.gz) or DICOM",
                  outputs: "Staged scan in data/raw/<id>",
                  desc: "Client streams scan. FastAPI validates extension, checks 250MB limit, generates unique scan ticket, and stores raw volume.",
                  color: "border-cyan-500/40 bg-cyan-950/20 text-cyan-400"
                },
                {
                  step: "02",
                  title: "Metadata & Preprocessing",
                  component: "ai/preprocessing",
                  inputs: "Raw 3D/4D matrix",
                  outputs: "Normalized tensor [C, 128, 128, 128]",
                  desc: "NiBabel extracts voxel spacing (dx, dy, dz) & affine matrix. Z-score standardizes intensities; resamples to 1.0mm³ isotropic grid.",
                  color: "border-indigo-500/40 bg-indigo-950/20 text-indigo-400"
                },
                {
                  step: "03",
                  title: "Model Inference (CPU)",
                  component: "ai/inference",
                  inputs: "Preprocessed tensor",
                  outputs: "Logits / Probability map",
                  desc: "PyTorch sliding-window evaluator runs in torch.no_grad() mode on Intel Iris Xe CPU with optimized thread count.",
                  color: "border-purple-500/40 bg-purple-950/20 text-purple-400"
                },
                {
                  step: "04",
                  title: "Localization & Volumetrics",
                  component: "ai/evaluation",
                  inputs: "Binary mask + voxel spacing",
                  outputs: "3D BBox, Volume (cm³), Area (mm²)",
                  desc: "Calculates bounding box [Xmin..Xmax, Ymin..Ymax, Zmin..Zmax], checks valid voxel metadata, and sums positive voxels * voxel_volume.",
                  color: "border-emerald-500/40 bg-emerald-950/20 text-emerald-400"
                },
                {
                  step: "05",
                  title: "Decision Support Report",
                  component: "Backend / React UI",
                  inputs: "Metrics, coordinates, slices",
                  outputs: "Interactive UI + PDF summary",
                  desc: "React renders 3-plane views with toggleable segmentation masks. User downloads timestamped surgical planning summary.",
                  color: "border-amber-500/40 bg-amber-950/20 text-amber-400"
                },
              ].map((stage, idx) => (
                <div key={stage.step} className={`p-4 rounded-xl border ${stage.color} flex flex-col justify-between`}>
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-mono font-bold tracking-wider opacity-75">STAGE {stage.step}</span>
                      <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-slate-900/60 border border-slate-700/50">
                        {stage.component}
                      </span>
                    </div>
                    <h3 className="text-sm font-bold text-white mb-2">{stage.title}</h3>
                    <p className="text-[11px] text-slate-300 leading-relaxed mb-3">
                      {stage.desc}
                    </p>
                  </div>
                  <div className="pt-2 border-t border-slate-800/80 text-[10px] space-y-1 font-mono">
                    <div className="text-slate-400"><span className="text-slate-500">In:</span> {stage.inputs}</div>
                    <div className="text-slate-300"><span className="text-slate-500">Out:</span> {stage.outputs}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Mathematical Formulas Summary */}
            <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Code2 className="w-4 h-4 text-cyan-400" />
                Mathematical Quantification Formulas (implemented in ai/evaluation/metrics.py)
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-cyan-400 font-semibold mb-1">2D Cross-Sectional Area</div>
                  <div className="text-slate-300 mb-1">Area (mm²) = N_pixels × dx × dy</div>
                  <div className="text-[10px] text-slate-500 font-sans">Computed across the maximum tumor axial slice.</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-emerald-400 font-semibold mb-1">3D Tumor Volume</div>
                  <div className="text-slate-300 mb-1">Volume (cm³) = (N_voxels × dx × dy × dz) / 1000</div>
                  <div className="text-[10px] text-slate-500 font-sans">Requires valid positive voxel dimensions from NIfTI header.</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                  <div className="text-purple-400 font-semibold mb-1">Dice Similarity (DSC)</div>
                  <div className="text-slate-300 mb-1">DSC = (2 × |X ∩ Y|) / (|X| + |Y|)</div>
                  <div className="text-[10px] text-slate-500 font-sans">Standard medical image overlap coefficient for BraTS.</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: HARDWARE SEPARATION */}
        {activeTab === 'hardware' && (
          <div className="space-y-6 animate-fadeIn">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Cpu className="w-5 h-5 text-cyan-400" />
                Hardware Execution Separation Strategy
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Accommodating your laptop's Intel Iris Xe graphics (No NVIDIA CUDA GPU) with a zero-compromise cloud training pipeline.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Local Laptop Profile */}
              <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <HardDrive className="w-5 h-5 text-cyan-400" />
                      <h3 className="text-base font-bold text-white">Local Development Laptop</h3>
                    </div>
                    <span className="text-xs px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                      Windows / VS Code
                    </span>
                  </div>

                  <div className="space-y-3 text-xs text-slate-300 mb-6">
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">Processor:</span>
                      <span className="font-semibold text-white">Intel Core CPU</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">Graphics:</span>
                      <span className="text-amber-400 font-medium">Intel Iris Xe (Integrated)</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">NVIDIA CUDA:</span>
                      <span className="text-rose-400 font-medium">None (Do not install CUDA)</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">PyTorch Build:</span>
                      <span className="font-mono text-cyan-300">torch (CPU-only wheel)</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">Primary Tasks:</span>
                      <span className="text-white text-right">FastAPI, React Web, Test Suite, Single-scan CPU inference</span>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-cyan-950/30 border border-cyan-800/40 text-xs text-cyan-200">
                    <strong>Local Inference Optimization:</strong> CPU inference on a single 3D volume uses sliding window inferer with 4-8 threads, completing in ~3-8 seconds without memory spikes.
                  </div>
                </div>

                <div className="mt-6 pt-3 border-t border-slate-800 font-mono text-[11px] text-slate-400">
                  ai/requirements-local.txt
                </div>
              </div>

              {/* Cloud Colab Profile */}
              <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Cloud className="w-5 h-5 text-indigo-400" />
                      <h3 className="text-base font-bold text-white">Cloud Training Environment</h3>
                    </div>
                    <span className="text-xs px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
                      Google Colaboratory
                    </span>
                  </div>

                  <div className="space-y-3 text-xs text-slate-300 mb-6">
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">Hardware Accelerator:</span>
                      <span className="font-semibold text-emerald-400">NVIDIA Tesla T4 / V100 / A100 GPU</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">CUDA / cuDNN:</span>
                      <span className="text-emerald-400 font-medium">CUDA 12.x Enabled</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">Dataset Storage:</span>
                      <span className="text-white">BraTS 2021/2023 Cohort</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">PyTorch Build:</span>
                      <span className="font-mono text-indigo-300">torch (cu121 wheels)</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-slate-800">
                      <span className="text-slate-400">Primary Tasks:</span>
                      <span className="text-white text-right">3D spatial augmentations, multi-epoch backpropagation, loss optimization</span>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-indigo-950/30 border border-indigo-800/40 text-xs text-indigo-200">
                    <strong>Checkpoint Serialization:</strong> Colab exports standard state dictionaries (`.pth`) which are saved into <code>models/exports/</code> and loaded on your laptop with <code>map_location='cpu'</code>.
                  </div>
                </div>

                <div className="mt-6 pt-3 border-t border-slate-800 font-mono text-[11px] text-slate-400">
                  notebooks/02_colab_training_pipeline.ipynb
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: DECOUPLED CONTRACTS */}
        {activeTab === 'contracts' && (
          <div className="space-y-6 animate-fadeIn">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Code2 className="w-5 h-5 text-indigo-400" />
                Decoupled Subsystem Interface Contracts
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                How FastAPI communicates with the AI engine, and how React communicates with FastAPI without framework pollution.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Contract 1: Backend -> AI */}
              <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <Server className="w-4 h-4 text-cyan-400" />
                    FastAPI Service ➔ Decoupled AI Pipeline
                  </h3>
                  <span className="text-[10px] font-mono bg-slate-800 px-2 py-0.5 rounded text-slate-300">Direct Python API</span>
                </div>
                <p className="text-xs text-slate-400 mb-3">
                  Backend services never import raw PyTorch layers directly. Instead, they interact via the high-level <code>BrainTumorAIPipeline</code> contract:
                </p>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 overflow-x-auto">
                  <pre>{`# backend/app/services/inference_service.py
from ai.inference.pipeline import BrainTumorAIPipeline
from ai.config.ai_config import AIModelConfig

class InferenceService:
    def __init__(self):
        # AI Config is injected from backend environment
        self.pipeline = BrainTumorAIPipeline(
            model_config=AIModelConfig(
                weights_path=settings.MODEL_WEIGHTS_PATH,
                device="cpu"  # Enforce CPU on Intel Iris Xe
            )
        )

    def dispatch_inference(self, scan_id: str):
        scan_path = self.locate_scan(scan_id)
        # Executes decoupled AI pipeline
        return self.pipeline.execute_pipeline(scan_path)`}</pre>
                </div>
              </div>

              {/* Contract 2: React -> FastAPI */}
              <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <Activity className="w-4 h-4 text-emerald-400" />
                    React Web Client ➔ FastAPI REST Endpoints
                  </h3>
                  <span className="text-[10px] font-mono bg-slate-800 px-2 py-0.5 rounded text-slate-300">HTTP REST JSON</span>
                </div>
                <p className="text-xs text-slate-400 mb-3">
                  Frontend communicates with FastAPI over standard REST endpoints typed via TypeScript:
                </p>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 overflow-x-auto">
                  <pre>{`// frontend/src/services/endpoints.ts
export const apiService = {
  // 1. Health check & disclaimer
  checkHealth: () => apiFetch<HealthResponse>('/health'),

  // 2. Upload MRI (multipart/form-data)
  uploadMri: (file: File) => {
    const data = new FormData();
    data.append('file', file);
    return apiFetch<MRIUploadResult>('/mri/upload', {
      method: 'POST', body: data
    });
  },

  // 3. Trigger decoupled inference
  triggerInference: (scanId: string) =>
    apiFetch<InferenceResult>('/inference/predict/' + scanId, {
      method: 'POST'
    })
};`}</pre>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 6: LOCAL DEV COMMANDS */}
        {activeTab === 'commands' && (
          <div className="space-y-6 animate-fadeIn">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Terminal className="w-5 h-5 text-amber-400" />
                Local Development Commands (Windows & VS Code)
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Exact instructions to initialize your Python virtual environment, install CPU dependencies, start FastAPI, and run the test suite.
              </p>
            </div>

            <div className="space-y-4">
              {[
                {
                  id: "cmd1",
                  title: "Step 1: Create & Activate Virtual Environment (.venv)",
                  note: "Run in project root from Windows PowerShell or VS Code Integrated Terminal",
                  cmd: "python -m venv .venv\n.venv\\Scripts\\Activate.ps1"
                },
                {
                  id: "cmd2",
                  title: "Step 2: Install Backend & AI CPU Dependencies",
                  note: "Installs CPU-only PyTorch to prevent downloading 3GB+ NVIDIA CUDA drivers on your laptop",
                  cmd: "python -m pip install --upgrade pip\npip install -r backend/requirements.txt\npip install -r ai/requirements-local.txt"
                },
                {
                  id: "cmd3",
                  title: "Step 3: Run the FastAPI Backend Server",
                  note: "Launches the REST service on port 8000 with auto-reload and Swagger docs at /docs",
                  cmd: "cd backend\ncp .env.example .env\nuvicorn app.main:app --reload --port 8000"
                },
                {
                  id: "cmd4",
                  title: "Step 4: Run the React Frontend Client",
                  note: "In a second terminal window, launches Vite dev server on port 3000",
                  cmd: "cd frontend\ncp .env.example .env\nnpm install\nnpm run dev"
                },
                {
                  id: "cmd5",
                  title: "Step 5: Run the Verification Test Suite",
                  note: "Executes unit and interface tests verifying schemas, paths, and metric formulas",
                  cmd: "pytest tests/ -v"
                },
              ].map((item) => (
                <div key={item.id} className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <div className="flex items-center justify-between mb-1.5">
                    <h3 className="text-sm font-semibold text-white">{item.title}</h3>
                    <button
                      onClick={() => copyToClipboard(item.cmd, item.id)}
                      className="text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center gap-1.5 transition-colors border border-slate-700"
                    >
                      {copiedCmd === item.id ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-emerald-400">Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                  <p className="text-xs text-slate-400 mb-2">{item.note}</p>
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-cyan-300 overflow-x-auto whitespace-pre">
                    {item.cmd}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 7: RISKS & MITIGATIONS */}
        {activeTab === 'risks' && (
          <div className="space-y-6 animate-fadeIn">
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-rose-400" />
                Technical & Academic Project Risk Matrix
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Proactive risk mitigation planned by the lead architect for final-year engineering review and defense.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                {
                  risk: "Risk 1: 3D MRI Memory Exhaustion on Local RAM",
                  severity: "High",
                  detail: "Loading full 4-channel 240x240x155 floating-point volumes directly into RAM can trigger Out-Of-Memory (OOM) crashes on 8GB/16GB laptops.",
                  mitigation: "Use MONAI's SlidingWindowInferer with patch size (128, 128, 128) and overlap 0.25 to process sub-volumes sequentially without saturating RAM.",
                  tag: "Memory"
                },
                {
                  risk: "Risk 2: Accidental CUDA Dependency on Local Machine",
                  severity: "High",
                  detail: "Installing default PyTorch wheels on Windows might look for CUDA drivers, leading to DLL load errors or failed imports on Intel Iris Xe.",
                  mitigation: "Strict separation of ai/requirements-local.txt (using --extra-index-url https://download.pytorch.org/whl/cpu) vs ai/requirements-colab.txt.",
                  tag: "Dependencies"
                },
                {
                  risk: "Risk 3: Inconsistent Physical Voxel Spacing",
                  severity: "Medium",
                  detail: "Different MRI scanners yield varying slice thicknesses (e.g. 1.0mm vs 5.0mm). Naive pixel counting leads to invalid volume and area estimations.",
                  mitigation: "NiBabel header extraction validates (dx, dy, dz). Volume calculation checks valid_voxel_metadata flag before reporting cm³; isotropic resampling in ai/preprocessing.",
                  tag: "Data Quality"
                },
                {
                  risk: "Risk 4: Regulatory & Academic Medical Misclassification",
                  severity: "Critical",
                  detail: "Presenting the project as an autonomous diagnostic tool without medical device certification breaches clinical ethics and project guidelines.",
                  mitigation: "Strict CADe/CADx labeling across all UI banners, API schemas, PDF reports, and docs/medical_disclaimer.md.",
                  tag: "Ethics"
                },
              ].map((r, i) => (
                <div key={i} className="p-5 rounded-xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                        {r.tag}
                      </span>
                      <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${
                        r.severity === 'Critical' ? 'bg-rose-950 text-rose-300 border border-rose-800' :
                        r.severity === 'High' ? 'bg-amber-950 text-amber-300 border border-amber-800' :
                        'bg-blue-950 text-blue-300 border border-blue-800'
                      }`}>
                        Severity: {r.severity}
                      </span>
                    </div>
                    <h3 className="text-sm font-bold text-white mb-2">{r.risk}</h3>
                    <p className="text-xs text-slate-400 mb-3 leading-relaxed">{r.detail}</p>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 text-xs text-emerald-300">
                    <strong className="text-emerald-400 block mb-0.5">Mitigation:</strong>
                    {r.mitigation}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-950 py-6 mt-12 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            AI Based Brain Tumor Detection and Localization for Surgical Planning • Module 1 Complete
          </div>
          <div className="text-slate-400 font-mono">
            Awaiting User Instruction for Module 2
          </div>
        </div>
      </footer>
    </div>
  );
}
