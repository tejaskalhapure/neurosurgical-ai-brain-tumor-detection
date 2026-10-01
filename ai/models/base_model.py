"""Base class specification for brain tumor segmentation models.

Defines the contract for model weight loading, forward pass execution,
and device placement without implementing specific architectures (such as U-Net)
prior to subsequent modules.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path


class BaseSegmentationModel(ABC):
    """Abstract interface for 3D Brain Tumor Deep Learning Architectures."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3, device: str = "cpu"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self._model: Optional[Any] = None

    @abstractmethod
    def build_network(self) -> Any:
        """Instantiates the underlying PyTorch / MONAI neural network module."""
        pass

    @abstractmethod
    def load_weights(self, weights_path: Path) -> None:
        """Loads serialized model weights safely mapping to the configured device."""
        pass

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Executes forward pass inference."""
        pass
