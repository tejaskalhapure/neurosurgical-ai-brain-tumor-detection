"""Transform interface contracts and MONAI-compatible pipeline preparation.

Prepares the dataset pipeline so that MONAI transforms can be added in Module 3
without applying irreversible downsampling or lossy conversions prematurely.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable

try:
    import monai.transforms as mt
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


class BaseTransformPipeline(ABC):
    """Abstract interface for medical image transformation pipelines."""

    @abstractmethod
    def get_train_transforms(self) -> Any:
        """Returns the composition of data augmentation transforms for training."""
        pass

    @abstractmethod
    def get_val_transforms(self) -> Any:
        """Returns the deterministic transforms for validation / testing."""
        pass

    @abstractmethod
    def get_inference_transforms(self) -> Any:
        """Returns minimal deterministic preprocessing transforms for inference."""
        pass


class BraTSTransformPipeline(BaseTransformPipeline):
    """Configurable MONAI dictionary transform provider for BraTS multi-modal volumes.

    In Module 2, provides identity and structure-preserving dictionary passes
    while preparing interfaces for heavy spatial/intensity augmentations in Module 3.
    """

    def __init__(self, keys: tuple = ("image", "label"), allow_missing_keys: bool = True):
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def get_train_transforms(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Returns training transforms. If MONAI is available, configures typed dict pass."""
        if HAS_MONAI:
            return mt.Compose([
                mt.EnsureTyped(keys=["image"], allow_missing_keys=self.allow_missing_keys)
            ])
        return lambda data: data

    def get_val_transforms(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Returns validation transforms (deterministic)."""
        if HAS_MONAI:
            return mt.Compose([
                mt.EnsureTyped(keys=["image"], allow_missing_keys=self.allow_missing_keys)
            ])
        return lambda data: data

    def get_inference_transforms(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Returns inference transforms."""
        if HAS_MONAI:
            return mt.Compose([
                mt.EnsureTyped(keys=["image"], allow_missing_keys=self.allow_missing_keys)
            ])
        return lambda data: data
