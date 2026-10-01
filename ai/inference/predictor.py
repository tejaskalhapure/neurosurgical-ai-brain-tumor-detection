"""Predictor abstraction for brain tumor inference.

Encapsulates model weight loading on CPU, sliding window evaluation,
and post-processing for bounding box extraction and volumetric quantification.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class BaseTumorPredictor(ABC):
    """Abstract interface for tumor predictor engine."""

    @abstractmethod
    def load_model(self, weights_path: Path, device: str = "cpu") -> None:
        """Loads serialized model weights onto the target device."""
        pass

    @abstractmethod
    def predict_tensor(self, preprocessed_tensor: Any) -> Dict[str, Any]:
        """Performs forward inference on a preprocessed tensor.

        Returns:
            Dictionary containing:
                - probability_map: 3D probability array
                - binary_mask: Boolean/Integer segmentation mask
                - tumor_detected: bool
                - confidence_score: float
        """
        pass
