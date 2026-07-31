from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


class MetricsPackageTests(unittest.TestCase):
    def test_registry_and_harmora_smoke(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "harmora_metrics"))
        from metrics import MetricConfig, compute_all_metrics

        generator = torch.Generator().manual_seed(123)
        hidden = torch.randn(2, 48, 12, generator=generator, dtype=torch.float64)
        cfg = MetricConfig(harmora_K_l=5, harmora_K_max=12)
        result = compute_all_metrics(
            hidden_states=hidden,
            metrics=["harmora", "participation_ratio", "anisotropy"],
            config=cfg,
            skip_errors=False,
        )
        self.assertIn("harmora", result)
        self.assertIn("score", result["harmora"])
        scores = np.asarray(result["harmora"]["score"], dtype=float)
        self.assertEqual(scores.shape[0], 2)
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
