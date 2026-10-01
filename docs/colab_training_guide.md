# Google Colab GPU Training Workflow

## Overview
Because local laptops equipped with Intel Iris Xe graphics lack dedicated CUDA acceleration, all compute-heavy 3D convolutional network training is conducted on Google Colab with NVIDIA GPUs (T4 / V100 / A100).

---

## Step-by-Step Training Protocol

### Phase 1: Environment Setup in Colab
1. Open Google Colab and set Runtime to **GPU** (`Runtime > Change runtime type > T4 GPU`).
2. Verify GPU status:
   ```bash
   !nvidia-smi
   ```
3. Install required deep learning and medical imaging dependencies:
   ```bash
   !pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   !pip install monai nibabel SimpleITK scikit-learn
   ```

### Phase 2: Dataset Acquisition (BraTS)
1. Mount Google Drive or download the official BraTS dataset into Colab local scratch space (`/content/dataset/`).
2. Decompress patient directories containing the 4 standard MRI sequences:
   - `*_t1.nii.gz` (Native T1-weighted)
   - `*_t1ce.nii.gz` (Post-contrast T1-weighted)
   - `*_t2.nii.gz` (T2-weighted)
   - `*_flair.nii.gz` (T2 Fluid Attenuated Inversion Recovery)
   - `*_seg.nii.gz` (Ground truth segmentation annotations: GD-enhancing tumor, peritumoral edema, necrotic core)

### Phase 3: Model Training via Decoupled Module
1. Clone the project repository or import `ai/`:
   ```bash
   !git clone <YOUR_REPO_URL>
   %cd <YOUR_REPO>/notebooks
   ```
2. Run `notebooks/02_colab_training_pipeline.ipynb` (to be implemented in future modules).
3. Log validation metrics:
   - Mean Dice Similarity Coefficient (DSC)
   - 95% Hausdorff Distance ($HD_{95}$)
   - Cross-Entropy Loss & Focal Tversky Loss.

### Phase 4: Model Export & Local Deployment
1. Serialize the best model checkpoint to standard PyTorch format:
   ```python
   torch.save({
       'epoch': best_epoch,
       'model_state_dict': model.state_dict(),
       'input_shape': [4, 128, 128, 128],
       'classes': ['edema', 'non_enhancing', 'enhancing']
   }, 'best_brain_tumor_model.pth')
   ```
2. Download `best_brain_tumor_model.pth` and place it inside your local project directory:
   ```
   models/exports/best_brain_tumor_model.pth
   ```
3. Update `backend/.env`:
   ```env
   MODEL_WEIGHTS_PATH=../models/exports/best_brain_tumor_model.pth
   DEVICE=cpu
   ```
