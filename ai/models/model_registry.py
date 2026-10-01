"""Model Registry for registering and retrieving neural network architectures."""

from typing import Dict, Type
from ai.models.base_model import BaseSegmentationModel

_MODEL_REGISTRY: Dict[str, Type[BaseSegmentationModel]] = {}


def register_model(name: str):
    """Decorator to register a segmentation model architecture."""
    def decorator(cls: Type[BaseSegmentationModel]):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str) -> Type[BaseSegmentationModel]:
    """Retrieves a registered model class by key."""
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Registered architectures: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]
