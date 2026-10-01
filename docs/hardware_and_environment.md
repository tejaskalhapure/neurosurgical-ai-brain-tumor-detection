# Hardware & Environment Strategy

## 1. Local Development Profile
- **Host Operating System:** Microsoft Windows 10/11
- **Processor:** Modern x86-64 Intel Core (with integrated Intel Iris Xe Graphics)
- **Dedicated GPU:** **None** (No NVIDIA CUDA hardware available locally)
- **Local Runtime Target:** Python 3.10+ Virtual Environment (`.venv`), Node.js 18+ for React frontend
- **Execution Role:**
  - Fast local development and debugging
  - Backend API orchestration (FastAPI)
  - Frontend UI hot reloading and medical viewer rendering (React/Vite)
  - **CPU-based Inference Only** using pre-trained TorchScript or state dictionary checkpoints
  - Fast execution on small batches or individual patient volumes using `torch.device('cpu')` with `torch.set_num_threads()` optimization.

---

## 2. Cloud Training Profile (Google Colab)
- **Platform:** Google Colaboratory
- **Hardware Acceleration:** NVIDIA GPU runtime (T4, V100, or A100 based on allocation)
- **CUDA Toolkit:** CUDA 12.x / cuDNN enabled automatically
- **Dataset:** BraTS (Brain Tumor Segmentation Challenge - multi-modal MRI scans: T1, T1ce, T2, FLAIR)
- **Training Role:**
  - Data loading and heavy 3D spatial data augmentations
  - Deep 3D convolutional network backbones (e.g. 3D U-Net / SegResNet / UNETR via MONAI)
  - Multi-epoch loss optimization (Dice-Cross-Entropy Loss, AdamW optimizer)
  - Checkpoint export: PyTorch state dictionary (`.pth`) and optimized TorchScript / ONNX weights.

---

## 3. Safe Transfer Architecture: Colab -> Local Laptop
To deploy trained models from Google Colab to the local CPU laptop:
1. **Device-Agnostic Saving:**
   ```python
   # On Google Colab (GPU)
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'metrics': {'val_dice': best_dice},
       'config': model_config
   }, 'models/checkpoints/best_model_weights.pth')
   ```
2. **CPU-Mapped Loading Locally:**
   ```python
   # On Local Laptop (CPU)
   checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   ```
3. **Local Memory Considerations:**
   - 3D MRI volumes (e.g., 240 x 240 x 155 voxels) can consume significant RAM if loaded naively.
   - Inference uses sliding window inference with MONAI's `SlidingWindowInferer` or sub-sampled grid processing to operate smoothly within 8GB-16GB of local system RAM without thrashing memory.
