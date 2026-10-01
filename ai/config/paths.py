"""Path resolution utilities for the AI package.

Ensures zero hardcoding and portable execution between local CPU environments
and remote Google Colab GPU runtimes.
"""

from pathlib import Path
import os

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Core subdirectories
AI_DIR = ROOT_DIR / "ai"
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT_DIR / "models")))
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTS_DIR = MODELS_DIR / "exports"

REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(ROOT_DIR / "reports")))
GENERATED_REPORTS_DIR = REPORTS_DIR / "generated"
REPORT_TEMPLATES_DIR = REPORTS_DIR / "templates"
