#!/usr/bin/env python3
"""Generate the within-model layer-ranking figure and appendix tables.

Input
-----
``outputs/full_experiment/correlations/per_seed/metric_correlations_per_seed.csv``

Outputs
-------
``artifacts/generated/within_model/``

The figure matches the paper design: all seed-level points use one circular
marker, vertical bars show the interquartile range of seed-averaged
model--task correlations, and diamonds show equal-task family means. The
summary panel reports equal-task aggregation across the eleven tasks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from matplotlib.lines import Line2D


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Leave as None for automatic discovery.
# An explicit path may be absolute or relative to PROJECT_ROOT.
INPUT_CSV: Path | None = (
    PROJECT_ROOT
    / "outputs"
    / "full_experiment"
    / "correlations"
    / "per_seed"
    / "metric_correlations_per_seed.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "generated" / "within_model"
FIGURES_DIR = OUTPUT_ROOT / "figures"
TABLES_DIR = OUTPUT_ROOT / "tables"
RUN_INFO_PATH = OUTPUT_ROOT / "run_info.json"

EXPECTED_SEEDS = (11, 22, 33, 44)
EXPORT_PNG_DPI = 600

# Metric aliases are normalized only for display.
HARMORA_NAMES = {
    "harmora",
    "official_rho_energy",
    "harmora_official",
}

# Four task families used in the paper.
FAMILY_ORDER = [
    "Core class-organized",
    "Cluster-organized",
    "Pairwise relational",
    "Similarity-dominated",
]

# Canonical task aliases. The matching function removes punctuation and case.
TASK_FAMILY_ALIASES = {
    "Core class-organized": [
        "banking77",
        "emotion",
        "humeemotion",
        "diversity1",       # retained only for compatibility with older runs
    ],
    "Cluster-organized": [
        "arxivp2p",
        "arxivs2s",
        "arxivhierarchicalclusteringp2p",
        "arxivhierarchicalclusterings2s",
        "biorxiv",
    ],
    "Pairwise relational": [
        "legalbenchpc",
    ],
    "Similarity-dominated": [
        "sts15",
        "sts16",
        "stsb",
        "sickr",
    ],
}

# # Marker shape identifies the downstream seed.
# SEED_MARKERS = {
#     11: "o",
#     22: "^",
#     33: "s",
#     44: "D",
# }

# Common input-column aliases.
COLUMN_ALIASES = {
    "metric": [
        "metric",
        "metric_name",
        "metric_key",
        "method",
        "method_name",
        "diagnostic",
        "diagnostic_name",
        "score_name",
    ],
    "task": [
        "task",
        "task_name",
        "task_key",
        "task_display",
        "dataset",
        "dataset_name",
        "benchmark",
    ],
    "model": [
        "model",
        "model_name",
        "model_alias",
        "encoder",
        "encoder_name",
        "backbone",
    ],
    "seed": [
        "seed",
        "downstream_seed",
        "probe_seed",
        "run_seed",
        "random_seed",
    ],
    "pearson": [
        "pearson",
        "pearson_r",
        "pearson_correlation",
        "mean_pearson",
    ],
    "spearman": [
        "spearman",
        "spearman_rho",
        "rho",
        "rank_correlation",
        "spearman_correlation",
        "correlation",
        "corr",
        "mean_spearman",
        "seed_spearman",
        "layer_spearman",
    ],
    "kendall": [
        "kendall",
        "kendall_tau",
        "kendall_correlation",
        "mean_kendall",
    ],
    "distance": [
        "distance",
        "distance_correlation",
        "distance_corr",
        "dcor",
        "mean_distance",
    ],
}


# =============================================================================
# SAFETY
# =============================================================================

_MARKER_FILE = ".generated_by_make_within_model_results"
_ALLOWED_GENERATED_DIRS = {"figures", "tables"}


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_paths() -> None:
    project = resolved(PROJECT_ROOT)
    script_dir = resolved(SCRIPT_DIR)

    if not project.exists():
        raise FileNotFoundError(f"Project directory does not exist:\n{project}")

    for target in (FIGURES_DIR, TABLES_DIR):
        target = resolved(target)
        if target.parent != resolved(OUTPUT_ROOT):
            raise RuntimeError("Generated folders must remain inside the configured output root.")
        if target.name not in _ALLOWED_GENERATED_DIRS:
            raise RuntimeError(
                "Only folders named 'figures' and 'tables' may be recreated."
            )


def reset_generated_directory(target: Path) -> None:
    """
    Delete only a directory that was previously created by this script.

    For a pre-existing unmarked directory, the script stops instead of deleting
    anything. Delete or rename that directory manually once if it is disposable.
    """
    target = resolved(target)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    marker = target / _MARKER_FILE

    if target.exists():
        if not marker.exists():
            raise RuntimeError(
                f"Refusing to delete an existing unmarked directory:\n{target}\n\n"
                f"Delete or rename it manually once, then rerun the script."
            )
        shutil.rmtree(target)

    target.mkdir(parents=False, exist_ok=False)
    marker.write_text(
        "Generated by experiments/make_within_model_results.py.\n",
        encoding="utf-8",
    )


def reset_generated_outputs() -> None:
    validate_paths()
    reset_generated_directory(FIGURES_DIR)
    reset_generated_directory(TABLES_DIR)


# =============================================================================
# INPUT DISCOVERY
# =============================================================================

def canonical_column(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def canonical_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def infer_columns(columns: Iterable[str]) -> dict[str, str]:
    normalized = {canonical_column(c): c for c in columns}
    result: dict[str, str] = {}

    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = canonical_column(alias)
            if key in normalized:
                result[canonical_name] = normalized[key]
                break

    # Conservative fallback for longer Spearman column names.
    if "spearman" not in result:
        candidates: list[tuple[int, int, str]] = []
        for normalized_name, original_name in normalized.items():
            excluded = any(
                token in normalized_name
                for token in (
                    "pvalue",
                    "p_value",
                    "ci_low",
                    "ci_high",
                    "rank",
                    "kendall",
                    "pearson",
                )
            )
            accepted = (
                "spearman" in normalized_name
                or normalized_name in {"mean_rho", "rho_mean"}
            )
            if accepted and not excluded:
                priority = 2 if "spearman" in normalized_name else 1
                candidates.append((priority, -len(normalized_name), original_name))

        if candidates:
            candidates.sort(reverse=True)
            result["spearman"] = candidates[0][2]

    return result


def input_candidate_score(path: Path, mapping: dict[str, str]) -> int:
    required = {"metric", "task", "model", "seed", "pearson", "spearman", "kendall", "distance"}
    if not required.issubset(mapping):
        return -1

    text = str(path).lower()
    name = path.name.lower()
    score = 20

    score += 8 if "within_model" in text or "within-model" in text else 0
    score += 7 if "seed" in name or "per_seed" in text else 0
    score += 5 if "spearman" in name else 0
    score += 4 if "correlation" in name or "corr" in name else 0
    score += 2 if "layer" in name else 0
    score += 2 if "downstream" in text else 0

    # Avoid selecting derived cross-model or paper-output files.
    score -= 15 if "cross_model" in text or "cross-model" in text else 0
    score -= 10 if "pairwise" in name else 0
    score -= 8 if "global" in name else 0
    score -= 6 if "summary" in name else 0
    score -= 20 if resolved(SCRIPT_DIR) in path.resolve().parents else 0

    return score


def discover_input_csv() -> tuple[Path, dict[str, str]]:
    if INPUT_CSV is not None:
        path = Path(INPUT_CSV)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = resolved(path)

        if not path.exists():
            raise FileNotFoundError(f"INPUT_CSV does not exist:\n{path}")

        header = pd.read_csv(path, nrows=0)
        mapping = infer_columns(header.columns)
        missing = {"metric", "task", "model", "seed", "pearson", "spearman", "kendall", "distance"} - set(mapping)
        if missing:
            raise ValueError(
                f"INPUT_CSV is missing recognizable columns: {sorted(missing)}\n"
                f"Columns found: {list(header.columns)}"
            )
        return path, mapping

    excluded_parts = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "figures",
        "tables",
        "paper",
        "node_modules",
    }

    candidates: list[tuple[int, Path, dict[str, str]]] = []
    inspected: list[tuple[Path, list[str]]] = []

    for path in resolved(PROJECT_ROOT).rglob("*.csv"):
        relative_parts = {
            part.lower()
            for part in path.relative_to(resolved(PROJECT_ROOT)).parts
        }
        if relative_parts & excluded_parts:
            continue
        if resolved(SCRIPT_DIR) in path.resolve().parents:
            continue

        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue

        columns = list(header.columns)
        inspected.append((path, columns))
        mapping = infer_columns(columns)
        score = input_candidate_score(path, mapping)
        if score >= 0:
            candidates.append((score, path, mapping))

    if not candidates:
        preview = "\n".join(
            f"  - {path}: {columns}"
            for path, columns in inspected[:25]
        )
        raise FileNotFoundError(
            "No per-seed within-model Spearman CSV was found under:\n"
            f"{resolved(PROJECT_ROOT)}\n\n"
            "The required columns are equivalent to:\n"
            "  metric, task, model, seed, Spearman correlation\n\n"
            "First readable CSV headers:\n"
            f"{preview if preview else '  (no readable CSV files found)'}\n\n"
            "Set INPUT_CSV explicitly near the top of make_results.py if needed."
        )

    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    best_score, best_path, best_mapping = candidates[0]

    print("[input] Auto-selected per-seed correlation CSV:")
    print(f"        {best_path}")
    print(f"[input] Recognized columns: {best_mapping}")
    print(f"[input] Discovery score: {best_score}")

    if len(candidates) > 1:
        print("[input] Other plausible files:")
        for score, path, mapping in candidates[1:6]:
            print(f"        score={score:2d}  {path}")
            print(f"                  columns={mapping}")

    return best_path, best_mapping


# =============================================================================
# DATA PREPARATION
# =============================================================================

def normalize_metric_name(value: object) -> str:
    text = str(value).strip()
    canonical = canonical_token(text)

    if canonical in {canonical_token(x) for x in HARMORA_NAMES}:
        return "Harmora"

    display_names = {
        "spectralgap": "Spectral gap",
        "participationratio": "Participation ratio",
        "matrixentropy": "Matrix entropy",
        "infonce": "InfoNCE",
        "infonceloss": "InfoNCE",
        "lidar": "LiDAR",
        "dime": "DiME",
        "anisotropy": "Anisotropy",
        "curvature": "Curvature",
    }
    return display_names.get(canonical, text.replace("_", " ").strip().title())


def assign_task_family(task: object) -> str | None:
    key = canonical_token(task)

    # Match longer aliases first so HUMEEmotion is not confused with Emotion.
    candidates: list[tuple[int, str, str]] = []
    for family, aliases in TASK_FAMILY_ALIASES.items():
        for alias in aliases:
            canonical_alias = canonical_token(alias)
            candidates.append((len(canonical_alias), canonical_alias, family))

    for _, alias, family in sorted(candidates, reverse=True):
        if alias and alias in key:
            return family

    return None


def exclude_unwanted_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the downstream-evaluation setting used by the paper when
    descriptor columns are present.
    """
    df = raw.copy()
    descriptor_columns = [
        c
        for c in df.columns
        if canonical_column(c)
        in {
            "experiment",
            "experiment_name",
            "evaluation",
            "evaluation_type",
            "setting",
            "target",
            "target_name",
            "analysis",
            "analysis_type",
            "variant",
            "calibration",
            "score_scale",
            "specification",
            "probe_type",
        }
    ]

    bad_pattern = re.compile(
        r"generalization|generalisation|multi[\s_-]*probe|"
        r"model[\s_-]*logratio|log[\s_-]*ratio|generalization[\s_-]*gap",
        flags=re.IGNORECASE,
    )

    for column in descriptor_columns:
        values = df[column].astype(str)
        df = df.loc[~values.str.contains(bad_pattern, na=False)].copy()

    return df


def load_seed_level_data(
    csv_path: Path,
    mapping: dict[str, str],
) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw = exclude_unwanted_rows(raw)

    rename_map = {
        original_name: canonical_name
        for canonical_name, original_name in mapping.items()
    }
    df = raw.rename(columns=rename_map)

    correlation_columns = ["pearson", "spearman", "kendall", "distance"]
    required = ["metric", "task", "model", "seed", *correlation_columns]
    df = df[required].copy()

    df["metric"] = df["metric"].map(normalize_metric_name)
    df["task"] = df["task"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

    for column in correlation_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required)
    df["seed"] = df["seed"].astype(int)
    df = df[df["seed"].isin(EXPECTED_SEEDS)].copy()

    if df.empty:
        raise ValueError(
            "No valid rows remained for downstream seeds "
            f"{list(EXPECTED_SEEDS)} in:\n{csv_path}"
        )

    aggregations = {
        column: (column, "mean")
        for column in correlation_columns
    }
    df = (
        df.groupby(
            ["metric", "task", "model", "seed"],
            as_index=False,
        )
        .agg(**aggregations)
    )

    df["family"] = df["task"].map(assign_task_family)
    unmapped = sorted(df.loc[df["family"].isna(), "task"].unique())
    if unmapped:
        raise ValueError(
            "These tasks could not be assigned to one of the four families:\n"
            + "\n".join(f"  - {task}" for task in unmapped)
            + "\n\nAdd their aliases to TASK_FAMILY_ALIASES near the top of the script."
        )

    present_seeds = set(df["seed"].unique())
    missing_seeds = set(EXPECTED_SEEDS) - present_seeds
    if missing_seeds:
        raise ValueError(
            "The selected CSV does not contain all four required downstream "
            f"seeds. Missing: {sorted(missing_seeds)}"
        )

    n_tasks = df["task"].nunique()
    n_models = df["model"].nunique()
    n_metrics = df["metric"].nunique()

    print(
        f"[input] Loaded {len(df):,} model-task-seed correlations: "
        f"{n_tasks} tasks, {n_models} models, {n_metrics} metrics, "
        f"{len(present_seeds)} seeds."
    )

    return df


def build_summaries(
    seed_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Figure 1 uses Spearman correlations.

    The appendix tables report Pearson, Spearman, Kendall, and distance
    correlation. For each correlation, models are averaged within each task
    for every seed, tasks are averaged with equal weight, and mean plus sample
    variance are then computed across the four seeds.
    """
    correlations = ["pearson", "spearman", "kendall", "distance"]

    mean_aggregations = {
        column: (column, "mean")
        for column in correlations
    }

    model_task = (
        seed_df.groupby(
            ["metric", "family", "task", "model"],
            as_index=False,
        )
        .agg(
            **mean_aggregations,
            n_seeds=("seed", "nunique"),
        )
    )

    task_metric = (
        model_task.groupby(
            ["metric", "family", "task"],
            as_index=False,
        )
        .agg(
            **mean_aggregations,
            n_models=("model", "nunique"),
        )
    )

    seed_task_metric = (
        seed_df.groupby(
            ["metric", "family", "task", "seed"],
            as_index=False,
        )
        .agg(
            **mean_aggregations,
            n_models=("model", "nunique"),
        )
    )

    seed_family_metric = (
        seed_task_metric.groupby(
            ["metric", "family", "seed"],
            as_index=False,
        )
        .agg(
            **mean_aggregations,
            n_tasks=("task", "nunique"),
        )
    )

    family_named_aggregations: dict[str, tuple] = {}
    for correlation in correlations:
        family_named_aggregations[f"{correlation}_mean"] = (
            correlation,
            "mean",
        )
        family_named_aggregations[f"{correlation}_variance"] = (
            correlation,
            lambda values: values.var(ddof=1),
        )

    family_seed_stats = (
        seed_family_metric.groupby(
            ["metric", "family"],
            as_index=False,
        )
        .agg(
            **family_named_aggregations,
            n_seeds=("seed", "nunique"),
            n_tasks=("n_tasks", "max"),
        )
    )

    family_seed_stats["family_mean"] = family_seed_stats["spearman_mean"]
    family_seed_stats["family_variance"] = (
        family_seed_stats["spearman_variance"]
    )

    model_task_quartiles = (
        model_task.groupby(["metric", "family"])["spearman"]
        .agg(
            family_q1=lambda s: s.quantile(0.25),
            family_median="median",
            family_q3=lambda s: s.quantile(0.75),
            family_min="min",
            family_max="max",
            n_model_task_cases="size",
        )
        .reset_index()
    )

    family_summary = family_seed_stats.merge(
        model_task_quartiles,
        on=["metric", "family"],
        how="left",
    )

    seed_overall_metric = (
        seed_task_metric.groupby(
            ["metric", "seed"],
            as_index=False,
        )
        .agg(
            **mean_aggregations,
            n_tasks=("task", "nunique"),
        )
    )

    overall_named_aggregations: dict[str, tuple] = {}
    for correlation in correlations:
        overall_named_aggregations[f"mean_{correlation}"] = (
            correlation,
            "mean",
        )
        overall_named_aggregations[f"variance_{correlation}"] = (
            correlation,
            lambda values: values.var(ddof=1),
        )

    overall_seed_stats = (
        seed_overall_metric.groupby("metric", as_index=False)
        .agg(
            **overall_named_aggregations,
            n_seeds=("seed", "nunique"),
        )
    )

    overall_task_stats = (
        task_metric.groupby("metric", as_index=False)
        .agg(
            median_spearman=("spearman", "median"),
            q1=("spearman", lambda s: s.quantile(0.25)),
            q3=("spearman", lambda s: s.quantile(0.75)),
            min_spearman=("spearman", "min"),
            max_spearman=("spearman", "max"),
            positive_tasks=("spearman", lambda s: int((s > 0).sum())),
            n_tasks=("task", "nunique"),
        )
    )

    overall_summary = (
        overall_seed_stats.merge(
            overall_task_stats,
            on="metric",
            how="left",
        )
        .sort_values(
            ["mean_spearman", "metric"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    overall_summary["rank"] = np.arange(1, len(overall_summary) + 1)

    return model_task, task_metric, family_summary, overall_summary


# =============================================================================
# FIGURE 1
# =============================================================================

def stable_jitter(
    task: object,
    model: object,
    seed: object,
    amplitude: float = 0.20,
) -> float:
    key = f"{task}|{model}|{seed}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    unit = integer / float(2**64 - 1)
    return (unit - 0.5) * 2.0 * amplitude


def default_metric_colors(metrics: list[str]) -> dict[str, str]:
    """
    Use Matplotlib's active default color cycle without hard-coding a palette.
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = [f"C{i}" for i in range(max(1, len(metrics)))]
    return {
        metric: cycle[index % len(cycle)]
        for index, metric in enumerate(metrics)
    }


def common_y_limits(values: pd.Series) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 0.15)
    margin = max(0.06, 0.08 * span)
    return max(-1.0, low - margin), min(1.0, high + margin)


def draw_family_panel(
    ax: plt.Axes,
    family: str,
    panel_letter: str,
    seed_df: pd.DataFrame,
    family_summary: pd.DataFrame,
    metric_order: list[str],
    metric_colors: dict[str, str],
    y_limits: tuple[float, float],
    show_ylabel: bool,
) -> None:
    family_points = seed_df[seed_df["family"] == family].copy()
    family_stats = family_summary[family_summary["family"] == family].copy()

    x_positions = {metric: index for index, metric in enumerate(metric_order)}

    for metric in metric_order:
        subset = family_points[family_points["metric"] == metric].copy()
        if subset.empty:
            continue

        color = metric_colors[metric]
        center = x_positions[metric]

        xs = [
            center + stable_jitter(row.task, row.model, row.seed)
            for row in subset.itertuples(index=False)
        ]

        ax.scatter(
            xs,
            subset["spearman"].to_numpy(),
            marker="o",
            s=13,
            alpha=0.44,
            linewidths=0,
            color=color,
            zorder=2,
        )

        stat_rows = family_stats[family_stats["metric"] == metric]
        if stat_rows.empty:
            continue
        stat = stat_rows.iloc[0]

        # Descriptive IQR across seed-averaged model-task cases.
        ax.plot(
            [center, center],
            [stat["family_q1"], stat["family_q3"]],
            linewidth=3.2,
            solid_capstyle="round",
            color=color,
            zorder=4,
        )
        cap_width = 0.10
        ax.plot(
            [center - cap_width, center + cap_width],
            [stat["family_q1"], stat["family_q1"]],
            linewidth=1.2,
            color=color,
            zorder=4,
        )
        ax.plot(
            [center - cap_width, center + cap_width],
            [stat["family_q3"], stat["family_q3"]],
            linewidth=1.2,
            color=color,
            zorder=4,
        )

        # Task-balanced family mean.
        ax.scatter(
            [center],
            [stat["family_mean"]],
            marker="D",
            s=43,
            color=color,
            edgecolors="white",
            linewidths=0.55,
            zorder=5,
        )

    ax.axhline(0.0, linewidth=0.8, alpha=0.55, zorder=1)
    ax.set_ylim(*y_limits)
    ax.set_xlim(-0.55, len(metric_order) - 0.45)
    ax.set_xticks(range(len(metric_order)))
    ax.set_xticklabels(
        metric_order,
        rotation=57,
        ha="right",
        fontsize=9.2,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Spearman correlation" if show_ylabel else "",
        fontsize=14.0,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.18, linewidth=0.65)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="y", labelsize=9.5)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    n_tasks = family_points["task"].nunique()
    n_models = family_points["model"].nunique()
    n_seeds = family_points["seed"].nunique()
    n_points = len(family_points)

    ax.text(
        0.02,
        0.975,
        f"{n_tasks} tasks · {n_models} models · {n_seeds} seeds\n"
        f"{n_points:,} metric cases",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        fontweight="bold",
        alpha=0.88,
    )

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def draw_overall_summary_panel(
    ax: plt.Axes,
    task_metric: pd.DataFrame,
    family_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    metric_order: list[str],
    metric_colors: dict[str, str],
    x_limits: tuple[float, float],
) -> None:
    """
    Horizontal summary:
      thin line = task min-max
      thick line = task Q1-Q3
      circles = family means
      diamond = task-balanced overall mean
    """
    y_positions = {
        metric: len(metric_order) - 1 - index
        for index, metric in enumerate(metric_order)
    }

    family_offsets = {
        family: offset
        for family, offset in zip(
            FAMILY_ORDER,
            np.linspace(-0.18, 0.18, len(FAMILY_ORDER)),
        )
    }

    rank_lookup = overall_summary.set_index("metric")["rank"].to_dict()

    for metric in metric_order:
        color = metric_colors[metric]
        y = y_positions[metric]
        row = overall_summary[overall_summary["metric"] == metric].iloc[0]

        ax.plot(
            [row["min_spearman"], row["max_spearman"]],
            [y, y],
            linewidth=0.9,
            alpha=0.55,
            color=color,
            zorder=2,
        )
        ax.plot(
            [row["q1"], row["q3"]],
            [y, y],
            linewidth=5.0,
            alpha=0.78,
            solid_capstyle="round",
            color=color,
            zorder=3,
        )

        family_rows = family_summary[
            family_summary["metric"] == metric
        ]
        for family in FAMILY_ORDER:
            candidate = family_rows[family_rows["family"] == family]
            if candidate.empty:
                continue
            family_mean = float(candidate.iloc[0]["family_mean"])
            ax.scatter(
                [family_mean],
                [y + family_offsets[family]],
                marker="o",
                s=18,
                alpha=0.70,
                color=color,
                edgecolors="white",
                linewidths=0.35,
                zorder=4,
            )

        ax.scatter(
            [row["mean_spearman"]],
            [y],
            marker="D",
            s=55,
            color=color,
            edgecolors="white",
            linewidths=0.65,
            zorder=5,
        )

        ax.text(
            x_limits[1] - 0.012 * (x_limits[1] - x_limits[0]),
            y,
            f"{row['mean_spearman']:.3f}",
            ha="right",
            va="center",
            fontsize=9.1,
            fontweight="bold",
        )

    ax.axvline(0.0, linewidth=0.8, alpha=0.55, zorder=1)
    ax.set_xlim(*x_limits)
    ax.set_ylim(-0.65, len(metric_order) - 0.35)
    ax.set_yticks([y_positions[m] for m in metric_order])
    ax.set_yticklabels(
        metric_order,
        fontsize=9.4,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Spearman correlation",
        fontsize=14.0,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.18, linewidth=0.65)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9.5)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")


def build_figure_1(
    seed_df: pd.DataFrame,
    task_metric: pd.DataFrame,
    family_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
) -> list[Path]:
    metric_order = overall_summary["metric"].tolist()
    metric_colors = default_metric_colors(metric_order)

    y_limits = common_y_limits(seed_df["spearman"])

    # Summary-panel range includes task values and leaves space for annotations.
    summary_min = min(
        float(task_metric["spearman"].min()),
        float(overall_summary["min_spearman"].min()),
    )
    summary_max = max(
        float(task_metric["spearman"].max()),
        float(overall_summary["max_spearman"].max()),
    )
    span = max(summary_max - summary_min, 0.20)
    summary_x_limits = (
        max(-1.0, summary_min - 0.08 * span),
        min(1.0, summary_max + 0.32 * span),
    )

    plt.rcParams.update({"font.size": 10.5})
    fig = plt.figure(figsize=(23.0, 6.6))

    # Outer grid: summary panel on the left and four family panels on the right
    outer = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.35, 4.25],
        wspace=0.165,
    )

    # Left summary panel
    summary_ax = fig.add_subplot(outer[0, 0])

    # Four task-family panels on the right
    right_grid = outer[0, 1].subgridspec(
        nrows=1,
        ncols=4,
        width_ratios=[1.00, 1.00, 0.92, 1.08],
        wspace=0.08,
    )

    family_axes = [
        fig.add_subplot(right_grid[0, index])
        for index in range(4)
    ]

    for index, (ax, family) in enumerate(zip(family_axes, FAMILY_ORDER)):
        draw_family_panel(
            ax=ax,
            family=family,
            panel_letter=chr(ord("a") + index),
            seed_df=seed_df,
            family_summary=family_summary,
            metric_order=metric_order,
            metric_colors=metric_colors,
            y_limits=y_limits,
            show_ylabel=(index == 0),
        )

        if index > 0:
            ax.tick_params(axis="y", labelleft=False)

    draw_overall_summary_panel(
        ax=summary_ax,
        task_metric=task_metric,
        family_summary=family_summary,
        overall_summary=overall_summary,
        metric_order=metric_order,
        metric_colors=metric_colors,
        x_limits=summary_x_limits,
    )

    # # Compact seed legend. Other visual elements are described in the caption.
    # legend_handles = [
    #     Line2D(
    #         [0],
    #         [0],
    #         marker=SEED_MARKERS[seed],
    #         linestyle="None",
    #         markersize=5.0,
    #         label=f"Seed {seed}",
    #     )
    #     for seed in EXPECTED_SEEDS
    # ]
    # fig.legend(
    #     handles=legend_handles,
    #     loc="upper center",
    #     bbox_to_anchor=(0.50, 0.975),
    #     ncol=4,
    #     frameon=False,
    #     prop={"size": 10.2, "weight": "bold"},
    #     handletextpad=0.40,
    #     columnspacing=1.35,
    # )


    fig.subplots_adjust(
        left=0.045,
        right=0.992,
        top=0.93,
        bottom=0.18,
    )

    base = FIGURES_DIR / "figure_2_within_model_layer_ranking"
    outputs = [
        base.with_suffix(".pdf"),
        base.with_suffix(".svg"),
        base.with_suffix(".png"),
    ]

    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(
        outputs[2],
        dpi=EXPORT_PNG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    return outputs


# =============================================================================
# TABLE EXPORTS
# =============================================================================

APPENDIX_METRIC_ORDER = [
    "Matrix entropy",
    "Participation ratio",
    "Anisotropy",
    "Curvature",
    "Spectral gap",
    "InfoNCE",
    "DiME",
    "LiDAR",
    "Harmora",
]

APPENDIX_METRIC_LABELS = {
    "Matrix entropy": r"Matrix Entropy $\uparrow$",
    "Participation ratio": r"Participation Ratio $\uparrow$",
    "Anisotropy": r"Anisotropy $\downarrow$",
    "Curvature": r"Curvature $\downarrow$",
    "Spectral gap": r"Spectral Gap $\uparrow$",
    "InfoNCE": r"InfoNCE Loss $\downarrow$",
    "DiME": r"DiME $\uparrow$",
    "LiDAR": r"LiDAR $\uparrow$",
    "Harmora": r"\oursrow Harmora (ours) $\uparrow$",
}

APPENDIX_FAMILY_COLUMNS = [
    (
        "Core class-organized",
        "Classification",
        "Banking77, Emotion, HUMEEmotion",
    ),
    (
        "Cluster-organized",
        "Clustering",
        "ArXivP2P, ArXivS2S, Biorxiv",
    ),
    (
        "Pairwise relational",
        "Pair classification",
        "LegalBenchPC",
    ),
    (
        "Similarity-dominated",
        "Semantic textual similarity",
        "STS15, STS16, STS-B, SICK-R",
    ),
]

APPENDIX_CORRELATIONS = [
    ("pearson", "Pearson"),
    ("spearman", "Spearman"),
    ("kendall", "Kendall"),
    ("distance", "Dist."),
]


def format_number(value: object) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.3f}"


def format_mean_variance(mean: object, variance: object) -> str:
    return (
        rf"\mv{{{format_number(mean)}}}"
        rf"{{{format_number(variance)}}}"
    )


def ordered_available_metrics(values: Iterable[str]) -> list[str]:
    available = set(values)
    ordered = [
        metric
        for metric in APPENDIX_METRIC_ORDER
        if metric in available
    ]
    extras = sorted(available - set(ordered))
    return ordered + extras


def write_family_appendix_table(
    path: Path,
    family_summary: pd.DataFrame,
) -> None:
    metrics = ordered_available_metrics(family_summary["metric"].unique())
    lookup = family_summary.set_index(["metric", "family"])

    rows: list[str] = []
    for metric in metrics:
        cells: list[str] = []

        for family_key, _, _ in APPENDIX_FAMILY_COLUMNS:
            if (metric, family_key) not in lookup.index:
                cells.extend([r"\mv{--}{--}"] * len(APPENDIX_CORRELATIONS))
                continue

            record = lookup.loc[(metric, family_key)]
            if isinstance(record, pd.DataFrame):
                record = record.iloc[0]

            for correlation, _ in APPENDIX_CORRELATIONS:
                cells.append(
                    format_mean_variance(
                        record[f"{correlation}_mean"],
                        record[f"{correlation}_variance"],
                    )
                )

        label = APPENDIX_METRIC_LABELS.get(metric, metric)
        row = (
            f"{label}\n"
            + "\n".join(f"& {cell}" for cell in cells)
            + r" \\"
        )
        rows.append(row)

        if metric == "LiDAR" and "Harmora" in metrics:
            rows.append(r"\midrule")

    table = r'''\begin{table*}[htbp]
\centering
\caption{Within-model results by task family.}
\label{tab:within_model_family_results}
\scriptsize
\setlength{\tabcolsep}{2.25pt}
\renewcommand{\arraystretch}{1.15}
\providecommand{\mv}[2]{$#1 \,\pm\, {\scriptstyle #2}$}

\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccccccccccccc}
\toprule
\multirow{3}{*}{\textbf{Metric}}
& \multicolumn{4}{c}{\textbf{Classification}}
& \multicolumn{4}{c}{\textbf{Clustering}}
& \multicolumn{4}{c}{\textbf{Pair classification}}
& \multicolumn{4}{c}{\textbf{Semantic textual similarity}} \\
\cmidrule(lr){2-5}
\cmidrule(lr){6-9}
\cmidrule(lr){10-13}
\cmidrule(lr){14-17}

& \multicolumn{4}{c}{\scriptsize Banking77, Emotion, HUMEEmotion}
& \multicolumn{4}{c}{\scriptsize ArXivP2P, ArXivS2S, Biorxiv}
& \multicolumn{4}{c}{\scriptsize LegalBenchPC}
& \multicolumn{4}{c}{\scriptsize STS15, STS16, STS-B, SICK-R} \\
\cmidrule(lr){2-5}
\cmidrule(lr){6-9}
\cmidrule(lr){10-13}
\cmidrule(lr){14-17}

& \textbf{Pearson} & \textbf{Spearman} & \textbf{Kendall} & \textbf{Dist.}
& \textbf{Pearson} & \textbf{Spearman} & \textbf{Kendall} & \textbf{Dist.}
& \textbf{Pearson} & \textbf{Spearman} & \textbf{Kendall} & \textbf{Dist.}
& \textbf{Pearson} & \textbf{Spearman} & \textbf{Kendall} & \textbf{Dist.} \\
\midrule
''' + "\n\n".join(rows) + r'''
\bottomrule
\end{tabular}
}

\vspace{1mm}
\begin{minipage}{0.99\textwidth}
\scriptsize
\textit{Note:}
Each cell reports the mean $\pm$ seed variance.
For each seed, models are averaged within each task and tasks are then
averaged within each family.
\end{minipage}
\end{table*}
'''

    path.write_text(table, encoding="utf-8")


def write_overall_appendix_table(
    path: Path,
    overall_summary: pd.DataFrame,
) -> None:
    metrics = ordered_available_metrics(overall_summary["metric"].unique())
    lookup = overall_summary.set_index("metric")

    rows: list[str] = []
    for metric in metrics:
        record = lookup.loc[metric]
        if isinstance(record, pd.DataFrame):
            record = record.iloc[0]

        label = APPENDIX_METRIC_LABELS.get(metric, metric)
        cells = [
            format_mean_variance(
                record[f"mean_{correlation}"],
                record[f"variance_{correlation}"],
            )
            for correlation, _ in APPENDIX_CORRELATIONS
        ]

        positive = (
            f"{int(record['positive_tasks'])}/"
            f"{int(record['n_tasks'])}"
        )

        row = (
            f"{label}\n"
            + "\n".join(f"& {cell}" for cell in cells)
            + f"\n& {positive}"
            + f"\n& {int(record['rank'])}"
            + r" \\"
        )
        rows.append(row)

        if metric == "LiDAR" and "Harmora" in metrics:
            rows.append(r"\midrule")

    table = r'''\begin{table*}[htbp]
\centering
\caption{Overall within-model results across the eleven tasks.}
\label{tab:within_model_overall_results}
\scriptsize
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}
\providecommand{\mv}[2]{$#1 \,\pm\, {\scriptstyle #2}$}

\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccc}
\toprule
\textbf{Metric}
& \textbf{Pearson}
& \textbf{Spearman}
& \textbf{Kendall}
& \textbf{Dist.}
& \textbf{Positive tasks}
& \textbf{Spearman rank} \\
\midrule
''' + "\n\n".join(rows) + r'''
\bottomrule
\end{tabular}
}

\vspace{1mm}
\begin{minipage}{0.99\textwidth}
\scriptsize
\textit{Note:}
Correlation columns report the mean $\pm$ seed variance.
For each seed, models are averaged within each task and the eleven tasks
are then averaged with equal weight. Positive tasks and rank are based
on Spearman correlation.
\end{minipage}
\end{table*}
'''

    path.write_text(table, encoding="utf-8")


def export_tables(
    seed_df: pd.DataFrame,
    model_task: pd.DataFrame,
    task_metric: pd.DataFrame,
    family_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
) -> list[Path]:
    paths = {
        "seed_points": TABLES_DIR / "within_model_seed_level_points.csv",
        "model_task": TABLES_DIR / "within_model_model_task_seed_means.csv",
        "task_values": TABLES_DIR / "within_model_task_level_values.csv",
        "family_summary": TABLES_DIR / "within_model_family_summary.csv",
        "overall_summary": TABLES_DIR / "within_model_overall_summary.csv",
        "family_latex": TABLES_DIR / "appendix_within_model_family.tex",
        "overall_latex": TABLES_DIR / "appendix_within_model_overall.tex",
    }

    seed_df.sort_values(
        ["family", "task", "model", "metric", "seed"]
    ).to_csv(paths["seed_points"], index=False)

    model_task.sort_values(
        ["family", "task", "model", "metric"]
    ).to_csv(paths["model_task"], index=False)

    task_metric.sort_values(
        ["family", "task", "metric"]
    ).to_csv(paths["task_values"], index=False)

    family_summary.sort_values(
        ["family", "metric"]
    ).to_csv(paths["family_summary"], index=False)

    overall_summary.to_csv(paths["overall_summary"], index=False)

    write_family_appendix_table(
        paths["family_latex"],
        family_summary,
    )
    write_overall_appendix_table(
        paths["overall_latex"],
        overall_summary,
    )

    return list(paths.values())


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    validate_paths()

    # Discover and validate the input before removing previous generated files.
    input_csv, column_mapping = discover_input_csv()
    seed_df = load_seed_level_data(input_csv, column_mapping)

    model_task, task_metric, family_summary, overall_summary = build_summaries(
        seed_df
    )

    reset_generated_outputs()

    generated_files: list[Path] = []
    generated_files.extend(
        build_figure_1(
            seed_df=seed_df,
            task_metric=task_metric,
            family_summary=family_summary,
            overall_summary=overall_summary,
        )
    )
    generated_files.extend(
        export_tables(
            seed_df=seed_df,
            model_task=model_task,
            task_metric=task_metric,
            family_summary=family_summary,
            overall_summary=overall_summary,
        )
    )

    run_info = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(resolved(PROJECT_ROOT)),
        "selected_input_csv": str(resolved(input_csv)),
        "recognized_columns": column_mapping,
        "expected_seeds": list(EXPECTED_SEEDS),
        "n_seed_level_rows": int(len(seed_df)),
        "n_tasks": int(seed_df["task"].nunique()),
        "n_models": int(seed_df["model"].nunique()),
        "n_metrics": int(seed_df["metric"].nunique()),
        "families": {
            family: {
                "tasks": sorted(
                    seed_df.loc[seed_df["family"] == family, "task"].unique()
                ),
                "n_tasks": int(
                    seed_df.loc[seed_df["family"] == family, "task"].nunique()
                ),
            }
            for family in FAMILY_ORDER
        },
        "metric_order": overall_summary["metric"].tolist(),
        "aggregation": [
            "all model-task-seed correlations are displayed as points",
            "seeds are averaged within each model-task-metric",
            "models are averaged within each task-metric",
            "tasks are averaged for family and overall task-balanced means",
        ],
        "normalization": "none",
        "generated_files": [str(resolved(path)) for path in generated_files],
    }

    RUN_INFO_PATH.write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[done] Generated:")
    for path in generated_files:
        print(f"       {resolved(path)}")
    print(f"       {resolved(RUN_INFO_PATH)}")

    print("\n[safety] Original experiment results were read only.")
    print("[safety] Recreated only:")
    print(f"         {resolved(FIGURES_DIR)}")
    print(f"         {resolved(TABLES_DIR)}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
