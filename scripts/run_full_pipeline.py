#!/usr/bin/env python3
"""Run the locked 11-task, 7-model Harmora experiment without legacy extras."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STAGES = ("samples", "downstream", "metrics", "aggregate", "correlate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the experiment stages required by the Harmora paper."
    )
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--from-stage", choices=STAGES, default="samples")
    parser.add_argument("--to-stage", choices=STAGES, default="correlate")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Heavy libraries are imported only after argument parsing so that
    # `--help` works before the full environment is installed.
    from harmora_downstream.aggregation import aggregate_all
    from harmora_downstream.config import load_config, resolve_path
    from harmora_downstream.correlation_pipeline import run_all_correlations
    from harmora_downstream.metrics_pipeline import extract_metrics
    from harmora_downstream.pipeline import (
        build_all_task_caches,
        prepare_cfg,
        run_downstream,
    )
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = prepare_cfg(load_config(config_path))

    start = STAGES.index(args.from_stage)
    end = STAGES.index(args.to_stage)
    if start > end:
        raise SystemExit("--from-stage must not come after --to-stage")

    selected = STAGES[start : end + 1]
    print("Stages:", ", ".join(selected))

    if "samples" in selected:
        payloads = build_all_task_caches(
            cfg,
            requested_tasks=args.tasks,
            overwrite=args.overwrite,
        )
        print(f"Prepared {len(payloads)} task sample caches.")

    if "downstream" in selected:
        manifest = run_downstream(
            cfg,
            model_aliases=args.models,
            requested_tasks=args.tasks,
            seed_override=args.seeds,
            overwrite_results=args.overwrite,
            overwrite_embeddings=args.overwrite,
            overwrite_splits=args.overwrite,
            show_progress=not args.no_progress,
        )
        print(f"Completed {len(manifest['runs'])} downstream runs.")

    if "metrics" in selected:
        paths = extract_metrics(
            cfg,
            model_aliases=args.models,
            requested_tasks=args.tasks,
            overwrite_metrics=args.overwrite,
            overwrite_embeddings=args.overwrite,
            overwrite_augmentations=args.overwrite,
            show_progress=not args.no_progress,
        )
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))

    if "aggregate" in selected:
        out = resolve_path(cfg, "output_dir")
        agg_cfg = cfg.get("aggregation", {})
        paths = aggregate_all(
            out,
            confidence_level=float(agg_cfg.get("confidence_level", 0.95)),
            save_pdf=bool(agg_cfg.get("save_pdf_figures", True)),
            save_summary_figures=bool(agg_cfg.get("save_summary_figures", True)),
        )
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))

    if "correlate" in selected:
        paths = run_all_correlations(cfg)
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))

    print("\nCore experiment completed.")
    print("Next: python scripts/run_paper_analyses.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
