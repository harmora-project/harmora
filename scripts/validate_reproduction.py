#!/usr/bin/env python3
"""Validate the frozen paper results and exact artifact inventory."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def get(path: Path, key: str, value: str) -> dict[str, str]:
    for row in rows(path):
        if row.get(key) == value:
            return row
    raise AssertionError(f"Missing {key}={value} in {path}")


def close(actual: float, expected: float, tol: float = 5e-4) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"Expected {expected}, got {actual}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/paper")
    args = parser.parse_args()
    artifact = Path(args.artifact_dir)
    if not artifact.is_absolute():
        artifact = ROOT / artifact

    data = artifact / "data"
    if not data.exists():
        raise SystemExit("Run `python scripts/reproduce_paper.py` first.")

    within_rows = rows(data / "within_model/figure1_seed_level_points.csv")
    assert len({r["task"] for r in within_rows}) == 11
    assert len({r["model"] for r in within_rows}) == 7
    assert sorted({int(r["seed"]) for r in within_rows}) == [11, 22, 33, 44]
    assert len({r["metric"] for r in within_rows}) == 9

    within = get(data / "within_model/table1_within_model_summary.csv", "metric", "Harmora")
    close(float(within["mean_spearman"]), 0.432387, 1e-6)
    assert int(within["rank"]) == 1

    spectral_gap = get(data / "within_model/table1_within_model_summary.csv", "metric", "Spectral gap")
    close(float(spectral_gap["mean_spearman"]), 0.270963, 1e-6)

    cross = get(data / "cross_model/cross_model_selection_overall_summary.csv", "metric", "Harmora")
    close(float(cross["selected_percentile_mean"]), 0.845779, 1e-6)
    close(float(cross["ndcg_at_1_mean"]), 0.843242, 1e-6)
    close(float(cross["ndcg_at_3_mean"]), 0.795474, 1e-6)
    close(float(cross["top5_overlap_mean"]), 0.372727, 1e-6)

    runtime = rows(data / "runtime/real_overall_runtime_summary.csv")[0]
    close(float(runtime["exhaustive_evaluator_s"]), 762.298868, 1e-5)
    close(float(runtime["hybrid_total_s"]), 166.995998, 1e-5)
    close(float(runtime["hybrid_speedup"]), 4.564773, 1e-6)

    precision = [
        r for r in rows(data / "precision/method_summary.csv")
        if r["fraction"] == "0.25" and r["method"] == "adaptive_precision"
    ][0]
    close(float(precision["mean_ndcg_at_k"]), 0.830090, 1e-6)
    close(float(precision["mean_utility_spearman"]), 0.491196, 1e-6)

    official = rows(data / "precision/official_validation.csv")[0]
    assert int(official["n_candidates"]) == 1166
    assert float(official["max_abs_error"]) < 1e-6

    required = [
        artifact / "figures/main/figure_1_task_dependent_selection.png",
        artifact / "figures/main/figure_1_task_dependent_selection.pdf",
        artifact / "figures/main/figure_2_within_model_layer_ranking.pdf",
        artifact / "figures/main/figure_3_harmonic_diagnostics.pdf",
        artifact / "tables/main/table_1_cross_model_selection.tex",
        artifact / "tables/main/table_2_bandwidth_sensitivity.tex",
        artifact / "RESULTS.json",
        artifact / "MANIFEST_SHA256.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AssertionError("Missing artifacts:\n" + "\n".join(missing))

    print("Validation passed: paper results, task/model/seed counts, and required artifacts match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
