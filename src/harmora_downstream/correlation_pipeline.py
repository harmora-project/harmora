from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import ensure_output_dirs, resolve_path
from .io_utils import safe_name, save_json
from .statistics_utils import (
    all_correlations,
    ndcg_at_k,
    pairwise_preference_accuracy,
    regret_at_1,
    sample_summary,
    topk_overlap,
)


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return path


def downstream_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        _required(output_dir / "csv" / "downstream_profiles_long.csv"),
        _required(output_dir / "csv" / "downstream_profiles_summary.csv"),
    )


def metric_path(output_dir: Path) -> Path:
    return _required(output_dir / "metric_csv" / "metrics_layerwise_long.csv")


def export_downstream_targets(output_dir: Path) -> Dict[str, Path]:
    long_path, summary_path = downstream_paths(output_dir)
    long_df = pd.read_csv(long_path)
    summary_df = pd.read_csv(summary_path)

    target_dir = output_dir / "downstream_targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_rows = []
    for (model, task), group in summary_df.groupby(["model_alias", "task"]):
        group = group.sort_values("layer")
        seed_group = long_df[(long_df["model_alias"] == model) & (long_df["task"] == task)]
        per_seed = {
            str(int(seed)): seed_data.sort_values("layer")["primary_score"].astype(float).tolist()
            for seed, seed_data in seed_group.groupby("seed")
        }
        payload = {
            "model_alias": model,
            "task": task,
            "task_type": str(group["task_type"].iloc[0]),
            "probe_family": str(group["probe_family"].iloc[0]),
            "primary_metric": str(group["primary_metric"].iloc[0]),
            "primary_evaluator": str(group["primary_evaluator"].iloc[0]),
            "sample_hash": str(group["sample_hash"].iloc[0]),
            "embedding_hash": str(group["embedding_hash"].iloc[0]),
            "layer_indices": group["layer"].astype(int).tolist(),
            "downstream_mean": group["score_mean"].astype(float).tolist(),
            "downstream_variance": group["score_variance"].astype(float).tolist(),
            "downstream_std": group["score_std"].astype(float).tolist(),
            "downstream_ci_low": group["ci_low"].astype(float).tolist(),
            "downstream_ci_high": group["ci_high"].astype(float).tolist(),
            "downstream_per_seed": per_seed,
            "n_seeds": int(group["n_seeds"].iloc[0]),
            "target_source": "downstream",
            "target_name": "downstream_primary_score",
            "multi_probe": False,
            "generalization": False,
        }
        path = target_dir / safe_name(model) / f"{safe_name(task)}.json"
        save_json(path, payload)
        for row in group.itertuples():
            target_rows.append({
                "model_alias": model,
                "task": task,
                "task_type": row.task_type,
                "probe_family": row.probe_family,
                "primary_metric": row.primary_metric,
                "layer": int(row.layer),
                "target_source": "downstream",
                "target_name": "downstream_primary_score_mean",
                "target_value": float(row.score_mean),
                "target_variance": float(row.score_variance),
                "target_std": float(row.score_std),
                "n_seeds": int(row.n_seeds),
                "sample_hash": row.sample_hash,
                "embedding_hash": row.embedding_hash,
            })
    target_csv = output_dir / "correlations" / "downstream_targets_long.csv"
    pd.DataFrame(target_rows).to_csv(target_csv, index=False)
    return {"target_csv": target_csv, "target_dir": target_dir}


def _correlate_group(metric_group: pd.DataFrame, target_group: pd.DataFrame, min_layers: int) -> Dict[str, float]:
    merged = metric_group[["layer", "oriented_value"]].merge(
        target_group[["layer", "target_value"]], on="layer", how="inner"
    ).sort_values("layer")
    if len(merged) < int(min_layers):
        return {"pearson": np.nan, "spearman": np.nan, "kendall": np.nan, "distance": np.nan, "n_layers": len(merged)}
    return all_correlations(merged["oriented_value"], merged["target_value"])


def layerwise_correlations(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Compute seed-specific correlations and seed-aware task summaries.

    Primary task-level estimates are formed by averaging correlations across
    seeds within each model and then averaging models within each task. The
    correlation against the seed-mean downstream curve is retained as a
    secondary descriptive output.
    """
    ensure_output_dirs(cfg)
    output_dir = resolve_path(cfg, "output_dir")
    long_path, summary_path = downstream_paths(output_dir)
    metrics_path = metric_path(output_dir)
    downstream_long = pd.read_csv(long_path)
    downstream_summary = pd.read_csv(summary_path)
    metrics = pd.read_csv(metrics_path)
    min_layers = int(cfg.get("correlation", {}).get("min_layers", 3))
    confidence = float(cfg.get("correlation", {}).get("confidence_level", 0.95))

    per_seed_rows: list[dict[str, Any]] = []
    for (model, task, seed), target in downstream_long.groupby(
        ["model_alias", "task", "seed"]
    ):
        target = target.rename(columns={"primary_score": "target_value"})
        metric_sub = metrics[
            (metrics["model_alias"] == model) & (metrics["task"] == task)
        ]
        for feature, metric_group in metric_sub.groupby("feature"):
            stats = _correlate_group(
                metric_group,
                target,
                min_layers=min_layers,
            )
            per_seed_rows.append({
                "model_alias": model,
                "task": task,
                "task_type": str(target["task_type"].iloc[0]),
                "probe_family": str(target["probe_family"].iloc[0]),
                "primary_metric": str(target["primary_metric"].iloc[0]),
                "seed": int(seed),
                "feature": feature,
                "metric": str(metric_group["metric"].iloc[0]),
                "field": str(metric_group["field"].iloc[0]),
                "direction": int(metric_group["direction"].iloc[0]),
                "sample_hash": str(target["sample_hash"].iloc[0]),
                "embedding_hash": str(target["embedding_hash"].iloc[0]),
                "target_source": "downstream_per_seed",
                **stats,
            })
    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_csv = (
        output_dir
        / "correlations"
        / "per_seed"
        / "metric_correlations_per_seed.csv"
    )
    per_seed_df.to_csv(per_seed_csv, index=False)

    summary_rows = []
    group_cols = [
        "model_alias",
        "task",
        "task_type",
        "probe_family",
        "primary_metric",
        "feature",
        "metric",
        "field",
        "direction",
    ]
    for keys, group in per_seed_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["seeds"] = ",".join(
            str(x) for x in sorted(group["seed"].unique())
        )
        row["n_seeds_configured"] = int(group["seed"].nunique())
        for method in ["pearson", "spearman", "kendall", "distance"]:
            summary = sample_summary(
                group[method],
                confidence_level=confidence,
            )
            for key, value in summary.items():
                row[f"{method}_{key}"] = value
            finite = group[method].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            row[f"{method}_positive_fraction"] = (
                float(np.mean(finite > 0)) if len(finite) else np.nan
            )
            row[f"{method}_sign_consistency"] = (
                float(max(np.mean(finite >= 0), np.mean(finite <= 0)))
                if len(finite)
                else np.nan
            )
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = (
        output_dir
        / "correlations"
        / "summary"
        / "metric_correlations_seed_summary.csv"
    )
    summary_df.to_csv(summary_csv, index=False)

    # Secondary descriptive estimate: correlate each metric with the downstream
    # curve obtained after averaging the four downstream seeds at every layer.
    mean_rows = []
    for (model, task), target in downstream_summary.groupby(
        ["model_alias", "task"]
    ):
        target = target.rename(columns={"score_mean": "target_value"})
        metric_sub = metrics[
            (metrics["model_alias"] == model) & (metrics["task"] == task)
        ]
        for feature, metric_group in metric_sub.groupby("feature"):
            stats = _correlate_group(
                metric_group,
                target,
                min_layers=min_layers,
            )
            mean_rows.append({
                "model_alias": model,
                "task": task,
                "task_type": str(target["task_type"].iloc[0]),
                "probe_family": str(target["probe_family"].iloc[0]),
                "primary_metric": str(target["primary_metric"].iloc[0]),
                "feature": feature,
                "metric": str(metric_group["metric"].iloc[0]),
                "field": str(metric_group["field"].iloc[0]),
                "direction": int(metric_group["direction"].iloc[0]),
                "target_source": "downstream_seed_mean_curve",
                **stats,
            })
    mean_df = pd.DataFrame(mean_rows)
    mean_csv = (
        output_dir
        / "correlations"
        / "summary"
        / "metric_correlations_mean_downstream.csv"
    )
    mean_df.to_csv(mean_csv, index=False)

    # Primary task-level estimate: seeds are repeated measurements within a
    # model-task case, and models are repeated cases within a task. Tasks remain
    # the independent inferential units.
    model_level = (
        per_seed_df.groupby(
            [
                "model_alias",
                "task",
                "task_type",
                "feature",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            pearson=("pearson", "mean"),
            spearman=("spearman", "mean"),
            kendall=("kendall", "mean"),
            distance=("distance", "mean"),
            n_seeds_total=("seed", "nunique"),
            n_seeds_valid_pearson=("pearson", "count"),
            n_seeds_valid_spearman=("spearman", "count"),
            n_seeds_valid_kendall=("kendall", "count"),
            n_seeds_valid_distance=("distance", "count"),
        )
    )
    task_level = (
        model_level.groupby(
            ["task", "task_type", "feature", "metric"],
            as_index=False,
        )
        .agg(
            pearson=("pearson", "mean"),
            spearman=("spearman", "mean"),
            kendall=("kendall", "mean"),
            distance=("distance", "mean"),
            n_models_total=("model_alias", "nunique"),
            n_models_valid_pearson=("pearson", "count"),
            n_models_valid_spearman=("spearman", "count"),
            n_models_valid_kendall=("kendall", "count"),
            n_models_valid_distance=("distance", "count"),
            mean_valid_seeds_spearman=(
                "n_seeds_valid_spearman",
                "mean",
            ),
        )
    )
    task_level["aggregation"] = (
        "mean_seed_correlations_within_model_then_mean_models"
    )
    task_level_csv = (
        output_dir
        / "correlations"
        / "summary"
        / "task_level_metric_correlations.csv"
    )
    task_level.to_csv(task_level_csv, index=False)

    # Retain the historical seed-mean-curve task aggregation as a secondary
    # sensitivity output, not as the primary inferential table.
    mean_task_level = (
        mean_df.groupby(
            ["task", "task_type", "feature", "metric"],
            as_index=False,
        )
        .agg(
            pearson=("pearson", "mean"),
            spearman=("spearman", "mean"),
            kendall=("kendall", "mean"),
            distance=("distance", "mean"),
            n_models_total=("model_alias", "nunique"),
            n_models_valid_spearman=("spearman", "count"),
        )
    )
    mean_task_level["aggregation"] = (
        "correlation_with_seed_mean_curve_then_mean_models"
    )
    mean_task_level_csv = (
        output_dir
        / "correlations"
        / "summary"
        / "task_level_metric_correlations_mean_downstream.csv"
    )
    mean_task_level.to_csv(mean_task_level_csv, index=False)

    family_rows = []
    for (task_type, feature), group in task_level.groupby(
        ["task_type", "feature"]
    ):
        row = {
            "task_type": task_type,
            "feature": feature,
            "n_tasks": int(group["task"].nunique()),
            "aggregation": (
                "mean_seed_correlations_within_model_then_mean_models"
            ),
        }
        for method in ["pearson", "spearman", "kendall", "distance"]:
            ss = sample_summary(
                group[method],
                confidence_level=confidence,
            )
            for key, value in ss.items():
                row[f"{method}_{key}"] = value
        family_rows.append(row)
    family_df = pd.DataFrame(family_rows)
    family_csv = (
        output_dir
        / "correlations"
        / "summary"
        / "task_family_metric_correlations.csv"
    )
    family_df.to_csv(family_csv, index=False)

    return {
        "per_seed": per_seed_csv,
        "seed_summary": summary_csv,
        "mean_target": mean_csv,
        "task_level": task_level_csv,
        "task_level_mean_curve": mean_task_level_csv,
        "family_summary": family_csv,
    }

def _model_logratio(group: pd.DataFrame) -> np.ndarray:
    """Architecture-relative log ratio using a model-local positive shift.

    The transform is target-free. Each model is shifted independently only
    when required to obtain positive values, then normalized by that model's
    median. This avoids allowing an extreme value from one architecture to
    determine the transform applied to every other architecture.
    """
    values = group["oriented_value"].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("model_logratio requires finite oriented values.")

    models = group["model_alias"].astype(str).to_numpy()
    transformed = np.empty_like(values, dtype=float)
    for model in np.unique(models):
        indices = np.where(models == model)[0]
        model_values = values[indices]
        lo = float(np.min(model_values))
        hi = float(np.max(model_values))
        span = max(hi - lo, float(np.max(np.abs(model_values))), 1.0)
        epsilon = 1e-12 * span
        shift = (-lo + epsilon) if lo <= 0.0 else 0.0
        positive = model_values + shift
        reference = float(np.median(positive))
        if reference <= 0.0 or np.any(positive <= 0.0):
            raise RuntimeError(
                f"model_logratio produced non-positive values for {model}."
            )
        transformed[indices] = np.log(positive / reference)
    return transformed

def _candidate_metrics(group: pd.DataFrame, score: np.ndarray, top_ks: Sequence[int]) -> Dict[str, float]:
    utility = group["target_value"].to_numpy(dtype=float)
    models = group["model_alias"].astype(str).to_numpy()
    corr = all_correlations(score, utility)
    all_pref, n_all = pairwise_preference_accuracy(score, utility, models, mode="all")
    within_pref, n_within = pairwise_preference_accuracy(score, utility, models, mode="within")
    cross_pref, n_cross = pairwise_preference_accuracy(score, utility, models, mode="cross")
    regret = regret_at_1(score, utility)
    result: Dict[str, float] = {
        **corr,
        "preference_accuracy": all_pref,
        "n_preference_pairs": n_all,
        "within_model_preference_accuracy": within_pref,
        "n_within_model_pairs": n_within,
        "cross_model_preference_accuracy": cross_pref,
        "n_cross_model_pairs": n_cross,
        **regret,
    }
    for k in top_ks:
        result[f"ndcg_at_{k}"] = ndcg_at_k(score, utility, k)
        result[f"topk_overlap_at_{k}"] = topk_overlap(score, utility, k)
    return result


def cross_model_candidate_analysis(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Evaluate model-layer candidate ranking separately for every seed.

    The primary estimand is computed for each task, evaluation seed, metric,
    and calibration. Seed-level results are then averaged within each task;
    tasks receive equal weight in the global summary. This preserves seed
    variability without treating repeated seeds as independent tasks.
    """
    output_dir = resolve_path(cfg, "output_dir")
    metrics = pd.read_csv(metric_path(output_dir))
    long_path, _ = downstream_paths(output_dir)
    downstream = pd.read_csv(long_path).rename(
        columns={"primary_score": "target_value"}
    )
    merged = metrics.merge(
        downstream[
            [
                "model_alias",
                "task",
                "task_type",
                "layer",
                "seed",
                "target_value",
            ]
        ],
        on=["model_alias", "task", "layer"],
        how="inner",
        suffixes=("_metric", "_target"),
    )
    merged["candidate_id"] = (
        merged["model_alias"].astype(str)
        + "::L"
        + merged["layer"].astype(str)
    )
    merged["target_source"] = "downstream_per_seed"
    merged["target_name"] = "downstream_primary_score"
    merged["candidate_weighting"] = "all_model_layer_candidates"
    candidates_csv = (
        output_dir
        / "analysis"
        / "cross_model"
        / "cross_model_candidates_long.csv"
    )
    merged.to_csv(candidates_csv, index=False)

    calibrations = list(
        cfg.get("correlation", {}).get(
            "calibrations",
            ["raw", "model_logratio"],
        )
    )
    allowed_calibrations = {"raw", "model_logratio"}
    unknown = sorted(set(calibrations).difference(allowed_calibrations))
    if unknown:
        raise ValueError(
            "Unknown cross-model calibration(s): " + ", ".join(unknown)
        )
    if len(set(calibrations)) != len(calibrations):
        raise ValueError("Cross-model calibrations must be unique.")

    top_ks = [
        int(x)
        for x in cfg.get("correlation", {}).get(
            "top_k_values",
            [1, 3, 5],
        )
    ]
    seed_detail_rows = []
    for (task, seed, feature), group in merged.groupby(
        ["task", "seed", "feature"]
    ):
        group = group.sort_values(
            ["model_alias", "layer"],
            kind="mergesort",
        ).copy()
        for calibration in calibrations:
            if calibration == "raw":
                score = group["oriented_value"].to_numpy(dtype=float)
            elif calibration == "model_logratio":
                score = _model_logratio(group)
            else:  # Protected by validation above.
                raise AssertionError(calibration)
            stats = _candidate_metrics(group, score, top_ks)
            seed_detail_rows.append({
                "task": task,
                "task_type": str(group["task_type_target"].iloc[0]),
                "seed": int(seed),
                "feature": feature,
                "calibration": calibration,
                "n_candidates": int(len(group)),
                "n_models": int(group["model_alias"].nunique()),
                "candidate_weighting": "all_model_layer_candidates",
                **stats,
            })
    seed_detail = pd.DataFrame(seed_detail_rows)
    seed_detail_csv = (
        output_dir
        / "analysis"
        / "cross_model"
        / "cross_model_candidate_alignment_seed_detail.csv"
    )
    seed_detail.to_csv(seed_detail_csv, index=False)

    metric_cols = [
        "pearson",
        "spearman",
        "kendall",
        "distance",
        "preference_accuracy",
        "within_model_preference_accuracy",
        "cross_model_preference_accuracy",
        "regret_at_1",
        "selected_percentile",
    ] + [f"ndcg_at_{k}" for k in top_ks] + [
        f"topk_overlap_at_{k}" for k in top_ks
    ]

    # Average repeated seeds inside each task. The task, not the seed, remains
    # the independent unit used by global summaries and statistical inference.
    task_rows = []
    for (task, task_type, feature, calibration), group in seed_detail.groupby(
        ["task", "task_type", "feature", "calibration"]
    ):
        row: Dict[str, Any] = {
            "task": task,
            "task_type": task_type,
            "feature": feature,
            "calibration": calibration,
            "n_seeds": int(group["seed"].nunique()),
            "n_candidates": int(group["n_candidates"].iloc[0]),
            "n_models": int(group["n_models"].iloc[0]),
            "candidate_weighting": "all_model_layer_candidates",
            "aggregation": "mean_across_seeds_within_task",
        }
        for col in metric_cols:
            ss = sample_summary(group[col])
            row[col] = ss["mean"]
            row[f"{col}_seed_std"] = ss["std"]
            row[f"{col}_seed_min"] = ss["min"]
            row[f"{col}_seed_max"] = ss["max"]
        task_rows.append(row)
    detail = pd.DataFrame(task_rows)
    detail_csv = (
        output_dir
        / "analysis"
        / "cross_model"
        / "cross_model_candidate_alignment_detail.csv"
    )
    detail.to_csv(detail_csv, index=False)

    summary_rows = []
    for (feature, calibration), group in detail.groupby(
        ["feature", "calibration"]
    ):
        row: Dict[str, Any] = {
            "feature": feature,
            "calibration": calibration,
            "n_tasks": int(group["task"].nunique()),
            "n_seeds": int(seed_detail["seed"].nunique()),
            "candidate_weighting": "all_model_layer_candidates",
            "aggregation": (
                "per_seed_candidate_analysis_then_equal_weight_task_mean"
            ),
        }
        for col in metric_cols:
            ss = sample_summary(group[col])
            row[f"mean_{col}"] = ss["mean"]
            row[f"std_{col}"] = ss["std"]
            row[f"variance_{col}"] = ss["variance"]
            row[f"ci_low_{col}"] = ss["ci_low"]
            row[f"ci_high_{col}"] = ss["ci_high"]
            seed_std_col = f"{col}_seed_std"
            row[f"mean_seed_std_{col}"] = float(
                group[seed_std_col].mean()
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_csv = (
        output_dir
        / "analysis"
        / "cross_model"
        / "cross_model_candidate_alignment_global_summary.csv"
    )
    summary.to_csv(summary_csv, index=False)

    family_summary = (
        detail.groupby(
            ["task_type", "feature", "calibration"],
            as_index=False,
        )
        .agg(
            n_tasks=("task", "nunique"),
            mean_spearman=("spearman", "mean"),
            std_spearman=("spearman", "std"),
            mean_seed_std_spearman=("spearman_seed_std", "mean"),
            mean_cross_model_preference=(
                "cross_model_preference_accuracy",
                "mean",
            ),
            mean_regret_at_1=("regret_at_1", "mean"),
            mean_top1_overlap=("topk_overlap_at_1", "mean"),
            mean_top3_overlap=("topk_overlap_at_3", "mean"),
        )
    )
    family_csv = (
        output_dir
        / "analysis"
        / "cross_model"
        / "cross_model_candidate_alignment_family_summary.csv"
    )
    family_summary.to_csv(family_csv, index=False)
    return {
        "candidates": candidates_csv,
        "seed_detail": seed_detail_csv,
        "detail": detail_csv,
        "summary": summary_csv,
        "family": family_csv,
    }

def run_all_correlations(cfg: Dict[str, Any]) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    paths.update(export_downstream_targets(resolve_path(cfg, "output_dir")))
    paths.update(layerwise_correlations(cfg))
    paths.update(cross_model_candidate_analysis(cfg))
    save_json(
        resolve_path(cfg, "output_dir")
        / "manifests"
        / "correlation_manifest.json",
        {
            "target": "task_matched_downstream_primary_score",
            "within_model_primary": (
                "per_seed_correlations_averaged_within_model_then_task"
            ),
            "mean_downstream_curve": "secondary_descriptive",
            "cross_model_primary": (
                "per_seed_candidate_analysis_then_equal_weight_task_mean"
            ),
            "cross_model_candidate_weighting": (
                "all_model_layer_candidates"
            ),
            "calibrations": list(
                cfg.get("correlation", {}).get(
                    "calibrations",
                    ["raw", "model_logratio"],
                )
            ),
            "multi_probe": False,
            "generalization": False,
            "outputs": {key: str(value) for key, value in paths.items()},
        },
    )
    return paths

