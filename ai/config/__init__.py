"""AI Configuration Package."""

from ai.config.ai_config import AIModelConfig, PreprocessingConfig, VolumetricConfig
from ai.config.paths import ROOT_DIR, DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR
from ai.config.dataset_config import (
    DatasetConfig,
    DEFAULT_MODALITIES,
    DEFAULT_LABEL_NAME,
    LABEL_MAPPINGS,
    TARGET_REGIONS
)

__all__ = [
    "AIModelConfig",
    "PreprocessingConfig",
    "VolumetricConfig",
    "DatasetConfig",
    "DEFAULT_MODALITIES",
    "DEFAULT_LABEL_NAME",
    "LABEL_MAPPINGS",
    "TARGET_REGIONS",
    "ROOT_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
]
