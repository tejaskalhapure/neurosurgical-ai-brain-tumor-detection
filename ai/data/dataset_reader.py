"""Dataset reader interfaces for medical NIfTI imaging.

Defines the contract for parsing BraTS-structured cohorts and patient directories.
Implementation of actual dataset loaders will be completed in Module 2.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pathlib import Path


class BaseDatasetReader(ABC):
    """Abstract Base Class for reading medical imaging datasets."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

    @abstractmethod
    def scan_dataset(self) -> List[Dict[str, Path]]:
        """Scans the directory and indexes patient scan modality paths.

        Returns:
            List of dictionaries containing paths to 't1', 't1ce', 't2', 'flair', 'seg'.
        """
        pass

    @abstractmethod
    def get_patient_modalities(self, patient_id: str) -> Dict[str, Path]:
        """Retrieves individual file paths for a specified patient identifier."""
        pass
