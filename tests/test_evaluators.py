from __future__ import annotations

import unittest

import numpy as np

from harmora_downstream.evaluators import (
    classification_profile,
    clustering_profile,
    pair_classification_profile,
    sts_profile,
)


class EvaluatorTests(unittest.TestCase):
    def test_classification_profile(self):
        rng = np.random.default_rng(0)
        labels = [0] * 20 + [1] * 20
        base = np.concatenate([rng.normal(-2, 0.2, (20, 3)), rng.normal(2, 0.2, (20, 3))])
        embeddings = np.stack([base, base * 2], axis=0).astype(np.float32)
        train = list(range(0, 15)) + list(range(20, 35))
        test = list(range(15, 20)) + list(range(35, 40))
        scores = classification_profile(embeddings, labels, train, test, seed=11)
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(score > 0.9 for score in scores))

    def test_clustering_profile(self):
        rng = np.random.default_rng(1)
        labels = [0] * 20 + [1] * 20
        base = np.concatenate([rng.normal(-3, 0.1, (20, 2)), rng.normal(3, 0.1, (20, 2))])
        embeddings = np.stack([base, base], axis=0).astype(np.float32)
        scores = clustering_profile(embeddings, labels, seed=11)
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(score > 0.9 for score in scores))

    def test_pair_classification_profile(self):
        a = np.array([[[1, 0], [1, 0], [0, 1], [0, 1]]], dtype=np.float32)
        b = np.array([[[1, 0], [-1, 0], [0, 1], [0, -1]]], dtype=np.float32)
        labels = [1, 0, 1, 0]
        scores = pair_classification_profile(a, b, labels)
        self.assertGreater(scores[0], 0.9)

    def test_sts_profile(self):
        a = np.array([[[1, 0], [1, 0], [1, 0], [1, 0]]], dtype=np.float32)
        b = np.array([[[1, 0], [0.8, 0.2], [0.2, 0.8], [-1, 0]]], dtype=np.float32)
        gold = [4.0, 3.0, 2.0, 1.0]
        scores = sts_profile(a, b, gold)
        self.assertGreater(scores[0], 0.9)


if __name__ == "__main__":
    unittest.main()
