# Model Weights & Artifacts Directory

## Subdirectories
- `models/checkpoints/`: Epoch-by-epoch training checkpoints saved during Google Colab training runs (e.g. `checkpoint_epoch_40.pth`).
- `models/exports/`: Final exported, optimized model weights specifically packaged for local CPU inference (e.g. `brats_segmentation_cpu_v1.pth` or TorchScript `.pt` file).

## Deployment Flow
1. Train model on Google Colab with GPU acceleration.
2. Select checkpoint with highest validation Dice Similarity Coefficient ($DSC$).
3. Export state dict or traced model using device-agnostic saving.
4. Place the file in `models/exports/` and specify the path in `backend/.env` under `MODEL_WEIGHTS_PATH`.
