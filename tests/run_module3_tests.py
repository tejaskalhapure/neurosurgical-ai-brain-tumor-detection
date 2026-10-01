"""Module 3 Preprocessing Test Runner.

Executes all Module 3 unit tests verifying normalization, resampling,
nearest-neighbor mask interpolation, foreground cropping, channel ordering,
label mapping, and metadata preservation.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tests.ai.test_preprocessing_pipeline import TestPreprocessingPipeline

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessingPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
