"""AI Preprocessing Package for Brain MRI Volumes."""

from ai.preprocessing.config import (
    MRIPreprocessingConfig,
    DEFAULT_CHANNEL_ORDER,
    DEFAULT_TARGET_SPACING,
    DEFAULT_ORIENTATION,
    DEFAULT_CROP_SIZE,
    DEFAULT_CONTIGUOUS_LABEL_MAP,
    DEFAULT_INVERSE_LABEL_MAP
)
from ai.preprocessing.nifti_handler import NiftiHandler, nifti_handler, CorruptedNiftiError
from ai.preprocessing.normalizer import BaseIntensityNormalizer, MRIIntensityNormalizer, mri_normalizer
from ai.preprocessing.registration import BaseSpatialTransformer, SpatialTransformer, spatial_transformer
from ai.preprocessing.cropper import ForegroundCropper, CropCoordinates
from ai.preprocessing.label_handler import BraTSLabelHandler, label_handler, BRATS_OFFICIAL_LABELS
from ai.preprocessing.pipeline import BraTSPreprocessingPipeline, BraTSPreprocessedCase
from ai.preprocessing.io import save_preprocessed_case, load_preprocessed_case, preprocess_dataset
from ai.preprocessing.visualizer import PreprocessingVisualizer

__all__ = [
    "MRIPreprocessingConfig",
    "DEFAULT_CHANNEL_ORDER",
    "DEFAULT_TARGET_SPACING",
    "DEFAULT_ORIENTATION",
    "DEFAULT_CROP_SIZE",
    "DEFAULT_CONTIGUOUS_LABEL_MAP",
    "DEFAULT_INVERSE_LABEL_MAP",
    "NiftiHandler",
    "nifti_handler",
    "CorruptedNiftiError",
    "BaseIntensityNormalizer",
    "MRIIntensityNormalizer",
    "mri_normalizer",
    "BaseSpatialTransformer",
    "SpatialTransformer",
    "spatial_transformer",
    "ForegroundCropper",
    "CropCoordinates",
    "BraTSLabelHandler",
    "label_handler",
    "BRATS_OFFICIAL_LABELS",
    "BraTSPreprocessingPipeline",
    "BraTSPreprocessedCase",
    "save_preprocessed_case",
    "load_preprocessed_case",
    "preprocess_dataset",
    "PreprocessingVisualizer"
]
