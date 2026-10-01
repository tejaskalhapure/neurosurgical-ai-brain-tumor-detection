"""Unit tests for AI configuration and path resolution."""

import unittest
from ai.config.ai_config import AIModelConfig, PreprocessingConfig, VolumetricConfig
from ai.config.paths import ROOT_DIR, DATA_DIR, MODELS_DIR


class TestAIConfig(unittest.TestCase):

    def test_ai_config_defaults(self):
        """Verifies default device is CPU for local laptop architecture."""
        cfg = AIModelConfig()
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.in_channels, 4)
        self.assertEqual(cfg.out_channels, 3)

    def test_paths_integrity(self):
        """Verifies repository paths resolve correctly."""
        self.assertTrue(ROOT_DIR.exists())
        self.assertTrue((ROOT_DIR / "ai").exists())
        self.assertTrue((ROOT_DIR / "backend").exists())


if __name__ == "__main__":
    unittest.main()
