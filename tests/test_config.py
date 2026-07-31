from __future__ import annotations

import unittest
from pathlib import Path

from harmora_downstream.config import load_config


EXPECTED_MODELS = [
    "minilm_l6",
    "mpnet_base",
    "e5_base_v2",
    "e5_large_v2",
    "bge_base_en_v15",
    "bge_large_en_v15",
    "snowflake_arctic_embed_m",
]

EXPECTED_TASKS = [
    "Banking77Classification.v2",
    "EmotionClassification.v2",
    "HUMEEmotionClassification",
    "ArXivHierarchicalClusteringP2P",
    "ArXivHierarchicalClusteringS2S",
    "BiorxivClusteringP2P.v2",
    "LegalBenchPC",
    "STS15",
    "STS16",
    "STSBenchmark",
    "SICK-R",
]


class ConfigTests(unittest.TestCase):
    def test_exact_11_tasks_7_models_4_seeds(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "paper.yaml")
        self.assertEqual([m["alias"] for m in cfg["models"]], EXPECTED_MODELS)
        self.assertEqual(cfg["mteb"]["include_task_names"], EXPECTED_TASKS)
        self.assertEqual(cfg["seeds"], [11, 22, 33, 44])

    def test_metrics_and_correlations_are_enabled_but_generalization_is_absent(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "paper.yaml")
        self.assertIn("metrics", cfg)
        self.assertIn("metric_extraction", cfg)
        self.assertIn("correlation", cfg)
        self.assertNotIn("generalization", cfg)
        self.assertNotIn("generalization_gap", cfg)
        self.assertEqual(cfg["correlation"]["target_name"], "downstream_primary_score")
        self.assertEqual(len(cfg["metrics"]["names"]), 9)


if __name__ == "__main__":
    unittest.main()

