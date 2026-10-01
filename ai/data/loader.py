"""DataLoader builder interfaces for training and validation splits."""

from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseDataLoaderBuilder(ABC):
    """Abstract interface for constructing PyTorch DataLoaders."""

    @abstractmethod
    def build_dataloaders(
        self,
        batch_size: int,
        val_split: float,
        num_workers: int
    ) -> Tuple[Any, Any]:
        """Builds train and validation dataloaders without executing on fake data."""
        pass
