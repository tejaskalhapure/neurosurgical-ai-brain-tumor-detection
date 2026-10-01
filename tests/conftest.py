"""Pytest configuration and common fixtures."""

import pytest
import os
import sys
from pathlib import Path

# Add project root to sys.path so tests can resolve backend and ai imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def mock_voxel_spacing():
    """Returns standard 1.0mm isotropic voxel spacing."""
    return (1.0, 1.0, 1.0)
