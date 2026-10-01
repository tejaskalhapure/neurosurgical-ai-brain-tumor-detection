"""FastAPI Settings configuration using Pydantic Settings."""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application runtime settings loaded from environment or defaults."""
    PROJECT_NAME: str = "NeuroSurgical AI - Decision Support Prototype"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "*"
    ]

    # Model and hardware settings
    DEVICE: str = Field(default="cpu", description="Compute device: cpu on local Iris Xe, cuda on Colab")
    MODEL_WEIGHTS_PATH: str = Field(
        default="../models/exports/model_weights.pth",
        description="Path to serialized PyTorch model checkpoint"
    )

    # Storage paths
    DATA_RAW_DIR: str = "../data/raw"
    DATA_PROCESSED_DIR: str = "../data/processed"
    REPORTS_DIR: str = "../reports/generated"

    MAX_UPLOAD_SIZE_MB: int = 250
    ALLOWED_EXTENSIONS: List[str] = [".nii", ".nii.gz", ".dcm"]

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"


settings = Settings()
