#!/usr/bin/env python3
"""Generate only the analyses and artifacts reported in the Harmora paper."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ANALYSES = ("within", "cross", "spectral", "precision", "runtime")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def load_settings(config_path: Path) -> dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML mapping: {config_path}")
    return cfg


def values(items: list[Any]) -> list[str]:
    return [str(item) for item in items]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the five analysis groups used in the Harmora paper."
    )
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--only", nargs="*", choices=ANALYSES, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-real-runtime", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_settings(config_path)
    settings = cfg.get("paper_analysis", {})
    generated_root = ROOT / settings.get("output_dir", "artifacts/generated")
    core_output = ROOT / cfg["paths"]["output_dir"]
    selected = tuple(args.only) if args.only else ANALYSES

    if "within" in selected:
        run([sys.executable, "experiments/make_within_model_results.py"])

    if "cross" in selected:
        run([sys.executable, "experiments/make_cross_model_results.py"])

    if "spectral" in selected:
        spectral = settings.get("bandwidth", {})
        run([
            sys.executable,
            "experiments/make_spectral_results.py",
            "--project-root", str(ROOT),
            "--output-dir", str(generated_root / "spectral"),
            "--fixed-k", str(spectral.get("fixed_k", 10)),
            "--bandwidth-k-max", str(spectral.get("curve_max_k", 47)),
            "--spectral-k-max", str(spectral.get("spectral_max_k", 80)),
            "--support-k-max", str(spectral.get("support_max_k", 80)),
            "--require-all-tasks",
        ])

    if "precision" in selected:
        precision = settings.get("adaptive_precision", {})
        run([
            sys.executable,
            "experiments/run_adaptive_precision_ablation.py",
            "--representation-dir", str(core_output / "embedding_cache"),
            "--utility-csv", str(core_output / "correlations/downstream_targets_long.csv"),
            "--metrics-package-dir", str(ROOT / "harmora_metrics"),
            "--official-metrics-dir", str(core_output / "metrics"),
            "--target-source", "downstream",
            "--target-name", "downstream_primary_score_mean",
            "--expected-task-count", str(cfg["expected_task_count"]),
            "--expected-candidates-per-task", "106",
            "--expected-model-count", str(cfg["expected_model_count"]),
            "--harmonic-k", str(cfg["metrics"]["config"]["harmora_K_l"]),
            "--sigma0-l2", str(cfg["metrics"]["config"]["harmora_sigma_l2"]),
            "--fractions", *values(precision.get("fractions", [0.25, 0.5, 0.75, 1.0])),
            "--max-samples", str(precision.get("max_samples", 192)),
            "--min-samples", str(precision.get("min_samples", 12)),
            "--uncertainty-views", str(precision.get("calibration_views", 5)),
            "--evaluation-views", str(precision.get("evaluation_views", 5)),
            "--subsample-seeds", *values(precision.get("subsample_seeds", [11, 29, 47, 71, 101])),
            "--top-k", str(precision.get("top_k", 5)),
            "--precision-shuffles", str(precision.get("precision_shuffles", 50)),
            "--shrinkage-strength", str(precision.get("shrinkage_strength", 5.0)),
            "--task-bootstrap", str(precision.get("task_bootstrap", 10000)),
            "--validation-tolerance", str(precision.get("validation_tolerance", 1e-6)),
            "--device", args.device,
            "--bank-cache-dir", str(ROOT / "cache/precision_ablation"),
            "--out-dir", str(generated_root / "adaptive_precision"),
        ])

    if "runtime" in selected:
        runtime = settings.get("runtime", {})
        command = [
            sys.executable,
            "experiments/run_runtime_analysis.py",
            "--config", str(config_path),
            "--output-dir", str(generated_root / "runtime"),
            "--expected-candidates", str(runtime.get("expected_candidates", 106)),
            "--shortlist-size", str(runtime.get("shortlist_size", 5)),
            "--K", str(cfg["metrics"]["config"]["harmora_K_l"]),
            "--sigma2", str(cfg["metrics"]["config"]["harmora_sigma_l2"]),
            "--eig-tolerance", str(runtime.get("eig_tolerance", 1e-8)),
            "--eig-maxiter", str(runtime.get("eig_maxiter", 5000)),
            "--real-warmups", str(runtime.get("real_warmups", 1)),
            "--real-repeats", str(runtime.get("real_repeats", 3)),
            "--scaling-warmups", str(runtime.get("scaling_warmups", 1)),
            "--scaling-repeats", str(runtime.get("scaling_repeats", 5)),
            "--validation-cases", str(runtime.get("validation_cases", 12)),
            "--validation-max-N", str(runtime.get("validation_max_samples", 300)),
        ]
        if args.skip_real_runtime:
            command.append("--skip-real")
        run(command)

    print(f"\nPaper analyses completed under {generated_root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
