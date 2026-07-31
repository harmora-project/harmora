from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from harmora_downstream.aggregation import summarize_profiles


class AggregationTests(unittest.TestCase):
    def test_mean_std_variance(self):
        rows = []
        values = [0.1, 0.2, 0.3, 0.4]
        for seed, score in zip([11, 22, 33, 44], values):
            rows.append({
                "model_alias": "m",
                "hf_name": "hf",
                "task": "t",
                "task_type": "Classification",
                "probe_family": "classification",
                "primary_metric": "accuracy",
                "primary_evaluator": "logistic_regression",
                "layer": 0,
                "primary_score": score,
                "seed": seed,
                "sampling_fingerprint": "sf",
                "sample_hash": "s",
                "embedding_hash": "e",
                "encoder_fingerprint": "ef",
                "evaluation_fingerprint": "vf",
                "seed_affects_evaluation": True,
            })
        out = summarize_profiles(pd.DataFrame(rows))
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out.iloc[0]["score_mean"], np.mean(values))
        self.assertAlmostEqual(out.iloc[0]["score_variance"], np.var(values, ddof=1))
        self.assertAlmostEqual(out.iloc[0]["score_std"], np.std(values, ddof=1))
        self.assertEqual(out.iloc[0]["n_seeds"], 4)


if __name__ == "__main__":
    unittest.main()
