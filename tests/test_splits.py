from __future__ import annotations

import unittest

from harmora_downstream.splits import deterministic_stratified_split


class SplitTests(unittest.TestCase):
    def test_reproducible_and_disjoint(self):
        labels = [0] * 20 + [1] * 20 + [2] * 20
        train_a, test_a, _ = deterministic_stratified_split(labels, 0.30, 11)
        train_b, test_b, _ = deterministic_stratified_split(labels, 0.30, 11)
        self.assertEqual(train_a.tolist(), train_b.tolist())
        self.assertEqual(test_a.tolist(), test_b.tolist())
        self.assertFalse(set(train_a.tolist()) & set(test_a.tolist()))
        self.assertEqual(set(train_a.tolist()) | set(test_a.tolist()), set(range(len(labels))))

    def test_different_seeds_change_split(self):
        labels = [0] * 20 + [1] * 20
        train_a, test_a, _ = deterministic_stratified_split(labels, 0.30, 11)
        train_b, test_b, _ = deterministic_stratified_split(labels, 0.30, 22)
        self.assertNotEqual(test_a.tolist(), test_b.tolist())

    def test_singleton_stays_in_train(self):
        labels = [0] * 10 + [1]
        train, test, meta = deterministic_stratified_split(labels, 0.30, 11)
        self.assertIn(10, train.tolist())
        self.assertNotIn(10, test.tolist())
        self.assertEqual(meta["n_singleton_classes"], 1)


if __name__ == "__main__":
    unittest.main()
