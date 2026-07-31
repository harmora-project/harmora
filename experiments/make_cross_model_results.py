#!/usr/bin/env python3
"""Generate cross-model representation-selection figures and tables.

Input
-----
``outputs/full_experiment/analysis/cross_model/``
``cross_model_candidate_alignment_seed_detail.csv``

Outputs
-------
``artifacts/generated/cross_model/``

The input contains one row per metric, task, and downstream seed. The script
produces the paper's task-, family-, and overall summaries without modifying
the core experiment outputs.
"""

from __future__ import annotations

import argparse
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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_VERSION = "2026-07-25-v5-seed-variance-main"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Set explicitly if automatic discovery selects the wrong file.
INPUT_DETAIL_CSV: Path | None = (
    PROJECT_ROOT
    / "outputs"
    / "full_experiment"
    / "analysis"
    / "cross_model"
    / "cross_model_candidate_alignment_seed_detail.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "generated" / "cross_model"
FIGURES_DIR = OUTPUT_ROOT / "figures"
TABLES_DIR = OUTPUT_ROOT / "tables"
RUN_INFO_PATH = OUTPUT_ROOT / "run_info.json"

EXPECTED_SEEDS = (11, 22, 33, 44)
EXPORT_PNG_DPI = 600

MAIN_MEASURES = (
    "ndcg_at_1",
    "ndcg_at_3",
    "selected_percentile",
    "top1_overlap",
    "top3_overlap",
    "top5_overlap",
)
OPTIONAL_MEASURES = ("ndcg_at_5", "regret_at_1")
LOWER_IS_BETTER = {"regret_at_1"}

MEASURE_LABELS = {
    "ndcg_at_1": "NDCG@1",
    "ndcg_at_3": "NDCG@3",
    "ndcg_at_5": "NDCG@5",
    "selected_percentile": "Selected pct.",
    "top1_overlap": "Top-1 overlap",
    "top3_overlap": "Top-3 overlap",
    "top5_overlap": "Top-5 overlap",
    "regret_at_1": "Regret@1",
}

FAMILY_ORDER = [
    "Core class-organized",
    "Cluster-organized",
    "Pairwise relational",
    "Similarity-dominated",
]
FAMILY_DISPLAY = {
    "Core class-organized": "Classification",
    "Cluster-organized": "Clustering",
    "Pairwise relational": "Pair classification",
    "Similarity-dominated": "Semantic textual similarity",
}
TASK_FAMILY_ALIASES = {
    "Core class-organized": ["banking77", "emotion", "humeemotion"],
    "Cluster-organized": [
        "arxivp2p",
        "arxivs2s",
        "arxivhierarchicalclusteringp2p",
        "arxivhierarchicalclusterings2s",
        "biorxiv",
    ],
    "Pairwise relational": ["legalbenchpc"],
    "Similarity-dominated": ["sts15", "sts16", "stsb", "sickr"],
}

METRIC_ORDER = [
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
METRIC_LABELS = {
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

COLUMN_ALIASES = {
    "metric": ["metric", "metric_name", "method", "feature", "diagnostic"],
    "task": ["task", "task_name", "dataset", "benchmark"],
    "seed": ["seed", "downstream_seed", "probe_seed", "run_seed"],
    "ndcg_at_1": ["ndcg_at_1", "ndcg_1", "ndcg1", "mean_ndcg_at_1"],
    "ndcg_at_3": ["ndcg_at_3", "ndcg_3", "ndcg3", "mean_ndcg_at_3"],
    "ndcg_at_5": ["ndcg_at_5", "ndcg_5", "ndcg5", "mean_ndcg_at_5"],
    "selected_percentile": [
        "selected_percentile",
        "selected_pct",
        "selected_utility_percentile",
        "selected_oracle_percentile",
        "selected_candidate_percentile",
        "selection_percentile",
        "utility_percentile",
        "oracle_percentile",
        "top1_percentile",
        "chosen_percentile",
        "mean_selected_percentile",
    ],
    "top1_overlap": [
        "top1_overlap",
        "top_1_overlap",
        "topk_overlap_at_1",
        "top_k_overlap_at_1",
        "overlap_at_1",
        "overlap_1",
        "top1_hit",
        "top_1_hit",
        "hit_at_1",
        "top1_recall",
        "top_1_recall",
    ],
    "top3_overlap": [
        "top3_overlap",
        "top_3_overlap",
        "topk_overlap_at_3",
        "top_k_overlap_at_3",
        "overlap_at_3",
        "overlap_3",
        "top3_hit",
        "top_3_hit",
        "hit_at_3",
        "top3_recall",
        "top_3_recall",
    ],
    "top5_overlap": [
        "top5_overlap",
        "top_5_overlap",
        "topk_overlap_at_5",
        "top_k_overlap_at_5",
        "overlap_at_5",
        "overlap_5",
        "top5_hit",
        "top_5_hit",
        "hit_at_5",
        "top5_recall",
        "top_5_recall",
    ],
    "regret_at_1": ["regret_at_1", "regret_1", "regret1", "top1_regret"],
}
DESCRIPTOR_ALIASES = {
    "calibration": ["calibration", "score_scale", "normalization"],
    "target_source": ["target_source", "utility_source", "target_name"],
    "setting": [
        "setting", "evaluation", "evaluation_type", "experiment", "analysis", "variant"
    ],
}


# -----------------------------------------------------------------------------
# Generic helpers and safety
# -----------------------------------------------------------------------------

MARKER_FILE = ".generated_by_make_cross_model_results"


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def canonical_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def canonical_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def reset_dir(path: Path) -> None:
    path = resolved(path)
    if path.name not in {"figures", "tables"} or path.parent != resolved(OUTPUT_ROOT):
        raise RuntimeError(f"Unsafe generated directory: {path}")
    marker = path / MARKER_FILE
    if path.exists():
        if not marker.exists():
            raise RuntimeError(
                f"Refusing to delete an unmarked directory:\n{path}\n"
                "Delete or rename it manually once if it is disposable."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)
    marker.write_text("Generated by make_cross_model_results.py\n", encoding="utf-8")


def reset_outputs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reset_dir(FIGURES_DIR)
    reset_dir(TABLES_DIR)


def infer_columns(columns: Iterable[str], aliases: dict[str, list[str]]) -> dict[str, str]:
    normalized = {canonical_column(c): c for c in columns}
    result: dict[str, str] = {}
    for target, variants in aliases.items():
        for variant in variants:
            key = canonical_column(variant)
            if key in normalized:
                result[target] = normalized[key]
                break
    return result


def required_columns() -> set[str]:
    return {"metric", "task", "seed", *MAIN_MEASURES}


# -----------------------------------------------------------------------------
# Input discovery and loading
# -----------------------------------------------------------------------------


def scan_csv_inputs(max_rows: int = 80) -> int:
    """
    Print a diagnostic list of CSV files and the cross-model columns detected
    in each file. This does not create or delete any output files.
    """
    required = required_columns()
    records: list[tuple[int, int, str, list[str], list[str], list[str]]] = []

    for path in resolved(PROJECT_ROOT).rglob("*.csv"):
        if resolved(SCRIPT_DIR) in path.resolve().parents:
            continue

        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue

        columns = list(header.columns)
        mapping = infer_columns(columns, COLUMN_ALIASES)
        recognized = sorted(mapping)
        missing = sorted(required - set(mapping))

        path_text = str(path).lower()
        keyword_score = sum(
            int(token in path_text)
            for token in (
                "cross",
                "model",
                "selection",
                "candidate",
                "shortlist",
                "ranking",
                "ndcg",
                "overlap",
                "seed",
                "task",
            )
        )
        records.append(
            (
                len(recognized),
                keyword_score,
                str(path),
                recognized,
                missing,
                columns,
            )
        )

    records.sort(
        key=lambda item: (item[0], item[1], item[2]),
        reverse=True,
    )

    report_lines = [
        "Cross-model CSV scan",
        "=" * 80,
        f"Project root: {resolved(PROJECT_ROOT)}",
        f"Required canonical fields: {sorted(required)}",
        "",
    ]

    if not records:
        report_lines.append("No readable CSV files were found.")
    else:
        for index, record in enumerate(records[:max_rows], start=1):
            recognized_count, _, path, recognized, missing, columns = record
            report_lines.extend(
                [
                    f"[{index}] {path}",
                    f"    Recognized ({recognized_count}): {recognized}",
                    f"    Missing: {missing}",
                    f"    Actual columns: {columns}",
                    "",
                ]
            )

    report = "\n".join(report_lines)
    print(report)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_ROOT / "input_scan_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[scan] Saved report to:\n       {resolved(report_path)}")
    return 0


def discover_input(
    input_override: str | None = None,
) -> tuple[Path, dict[str, str], dict[str, str]]:
    configured_input: Path | str | None = (
        input_override if input_override is not None else INPUT_DETAIL_CSV
    )

    if configured_input is not None:
        path = Path(configured_input)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = resolved(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Cross-model input CSV does not exist:\n{path}"
            )

        header = pd.read_csv(path, nrows=0)
        mapping = infer_columns(header.columns, COLUMN_ALIASES)
        descriptors = infer_columns(header.columns, DESCRIPTOR_ALIASES)
        missing = required_columns() - set(mapping)

        if missing:
            raise ValueError(
                "The selected CSV does not contain all required fields.\n"
                f"File: {path}\n"
                f"Recognized: {sorted(mapping)}\n"
                f"Missing: {sorted(missing)}\n"
                f"Actual columns: {list(header.columns)}\n\n"
                "Run the script with --scan to inspect the other CSV files."
            )

        return path, mapping, descriptors

    candidates: list[tuple[int, Path, dict[str, str], dict[str, str]]] = []
    for path in resolved(PROJECT_ROOT).rglob("*.csv"):
        if resolved(SCRIPT_DIR) in path.resolve().parents:
            continue

        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue

        mapping = infer_columns(header.columns, COLUMN_ALIASES)
        if not required_columns().issubset(mapping):
            continue

        descriptors = infer_columns(header.columns, DESCRIPTOR_ALIASES)
        path_text = str(path).lower()
        score = 20
        score += 12 if "cross_model" in path_text or "cross-model" in path_text else 0
        score += 9 if "selection" in path_text or "candidate" in path_text else 0
        score += 8 if "detail" in path.name.lower() or "task" in path.name.lower() else 0
        score += 6 if "seed" in path_text else 0
        score -= 15 if "global_summary" in path.name.lower() else 0
        score -= 15 if "within_model" in path_text or "within-model" in path_text else 0
        candidates.append((score, path, mapping, descriptors))

    if not candidates:
        raise FileNotFoundError(
            "No task-by-seed cross-model selection CSV was found.\n\n"
            "Run the core experiment first:\n"
            "  python scripts/run_full_pipeline.py\n\n"
            "Then rerun this analysis, or pass an explicit input CSV with "
            "--input. "
        )

    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    score, path, mapping, descriptors = candidates[0]
    print(f"[input] Selected: {path}")
    print(f"[input] Score: {score}")
    print(f"[input] Columns: {mapping}")
    return path, mapping, descriptors


def normalize_metric(value: object) -> str:
    """Map internal feature/field names to publication-ready metric names."""
    text = str(value).strip()
    key = canonical_token(text)

    # Cross-model outputs may store combined names such as Harmora.Score or
    # Spectral Gap.Lambda2. Match the canonical metric token rather than the
    # complete internal feature/field string.
    if "harmora" in key or "officialrhoenergy" in key:
        return "Harmora"
    if "participationratio" in key:
        return "Participation ratio"
    if "matrixentropy" in key:
        return "Matrix entropy"
    if "spectralgap" in key or key.startswith("lambda2"):
        return "Spectral gap"
    if "anisotropy" in key:
        return "Anisotropy"
    if "curvature" in key:
        return "Curvature"
    if "infonce" in key:
        return "InfoNCE"
    if "dime" in key:
        return "DiME"
    if "lidar" in key:
        return "LiDAR"

    # Keep unknown metrics readable instead of silently discarding them.
    return re.sub(r"[._]+", " ", text).strip().title()


def assign_family(task: object) -> str | None:
    key = canonical_token(task)
    aliases: list[tuple[int, str, str]] = []
    for family, values in TASK_FAMILY_ALIASES.items():
        for value in values:
            alias = canonical_token(value)
            aliases.append((len(alias), alias, family))
    for _, alias, family in sorted(aliases, reverse=True):
        if alias in key:
            return family
    return None


def primary_filter(raw: pd.DataFrame, descriptors: dict[str, str]) -> pd.DataFrame:
    df = raw.copy()
    setting = descriptors.get("setting")
    if setting:
        bad = re.compile(
            r"generalization|generalisation|multi[\s_-]*probe|"
            r"model[\s_-]*logratio|log[\s_-]*ratio|pairwise[\s_-]*preference",
            re.I,
        )
        df = df.loc[~df[setting].astype(str).str.contains(bad, na=False)].copy()

    calibration = descriptors.get("calibration")
    if calibration:
        values = df[calibration].astype(str)
        keep = values.str.contains(r"(^|[\s_-])raw($|[\s_-])", case=False, regex=True, na=False)
        if keep.any():
            df = df.loc[keep].copy()

    target = descriptors.get("target_source")
    if target:
        values = df[target].astype(str)
        keep = values.str.contains(r"task[\s_-]*matched|primary", case=False, regex=True, na=False)
        if keep.any():
            df = df.loc[keep].copy()
    return df


def load_data(path: Path, mapping: dict[str, str], descriptors: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    raw = primary_filter(pd.read_csv(path), descriptors)
    measures = [m for m in (*MAIN_MEASURES, *OPTIONAL_MEASURES) if m in mapping]
    rename = {source: target for target, source in mapping.items()}
    df = raw.rename(columns=rename)[["metric", "task", "seed", *measures]].copy()

    df["metric"] = df["metric"].map(normalize_metric)
    df["task"] = df["task"].astype(str).str.strip()
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    for measure in measures:
        df[measure] = pd.to_numeric(df[measure], errors="coerce")

    df = df.dropna(subset=["metric", "task", "seed", *MAIN_MEASURES])
    df["seed"] = df["seed"].astype(int)
    df = df[df["seed"].isin(EXPECTED_SEEDS)].copy()

    aggregations = {m: (m, "mean") for m in measures}
    df = df.groupby(["metric", "task", "seed"], as_index=False).agg(**aggregations)
    df["family"] = df["task"].map(assign_family)

    unmapped = sorted(df.loc[df["family"].isna(), "task"].unique())
    if unmapped:
        raise ValueError("Unmapped tasks:\n" + "\n".join(f"  - {x}" for x in unmapped))

    if df["task"].nunique() != 11:
        raise ValueError(f"Expected 11 tasks, found {df['task'].nunique()}.")

    missing_seeds = set(EXPECTED_SEEDS) - set(df["seed"].unique())
    if missing_seeds:
        raise ValueError(f"Missing seeds: {sorted(missing_seeds)}")

    print(
        f"[input] Loaded {len(df):,} metric-task-seed rows, "
        f"{df['metric'].nunique()} metrics, 11 tasks, 4 seeds."
    )
    print(f"[input] Measures: {measures}")
    return df, measures


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def build_summaries(df: pd.DataFrame, measures: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggs: dict[str, tuple] = {}
    for measure in measures:
        aggs[f"{measure}_mean"] = (measure, "mean")
        aggs[f"{measure}_variance"] = (measure, lambda x: x.var(ddof=1))

    task = df.groupby(["metric", "family", "task"], as_index=False).agg(
        **aggs, n_seeds=("seed", "nunique")
    )

    family_seed = df.groupby(["metric", "family", "seed"], as_index=False).agg(
        **{m: (m, "mean") for m in measures}, n_tasks=("task", "nunique")
    )
    family = family_seed.groupby(["metric", "family"], as_index=False).agg(
        **aggs, n_seeds=("seed", "nunique"), n_tasks=("n_tasks", "max")
    )

    overall_seed = df.groupby(["metric", "seed"], as_index=False).agg(
        **{m: (m, "mean") for m in measures}, n_tasks=("task", "nunique")
    )
    overall = overall_seed.groupby("metric", as_index=False).agg(
        **aggs, n_seeds=("seed", "nunique"), n_tasks=("n_tasks", "max")
    )
    overall = overall.sort_values(["ndcg_at_3_mean", "metric"], ascending=[False, True]).reset_index(drop=True)
    overall["ndcg3_rank"] = np.arange(1, len(overall) + 1)
    return task, family, overall


# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------


def ordered_metrics(values: Iterable[str]) -> list[str]:
    available = set(values)
    ordered = [m for m in METRIC_ORDER if m in available]
    return ordered + sorted(available - set(ordered))


def fmt(value: object) -> str:
    return "--" if pd.isna(value) else f"{float(value):.3f}"


def fmt_var(value: object) -> str:
    if pd.isna(value):
        return "--"
    value = float(value)
    if abs(value) < 1e-12:
        return "0"
    if abs(value) < 1e-3:
        mantissa, exponent = f"{value:.1e}".split("e")
        return rf"{mantissa}\mathrm{{e}}{{{int(exponent)}}}"
    return f"{value:.3f}"


def mv(mean: object, variance: object) -> str:
    return rf"\mv{{{fmt(mean)}}}{{{fmt_var(variance)}}}"


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def best(frame: pd.DataFrame, measure: str) -> float:
    values = frame[f"{measure}_mean"]
    return float(values.min() if measure in LOWER_IS_BETTER else values.max())


def main_table_cell(
    mean: object,
    variance: object,
    best_mean: float,
) -> str:
    """Format one Main-table cell as mean plus seed variance.

    Only the best mean is bolded; the variance remains in smaller normal type.
    """
    if pd.isna(mean):
        return mv(mean, variance)

    mean_value = float(mean)
    mean_text = fmt(mean_value)
    variance_text = fmt_var(variance)

    if math.isclose(
        mean_value,
        best_mean,
        rel_tol=1e-9,
        abs_tol=5e-7,
    ):
        mean_text = rf"\mathbf{{{mean_text}}}"

    return rf"\mv{{{mean_text}}}{{{variance_text}}}"



# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------


def build_summary_figure(overall: pd.DataFrame) -> list[Path]:
    metrics = overall["metric"].tolist()
    measures = list(MAIN_MEASURES)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(6)])
    markers = ["o", "^", "s", "D", "P", "X"]

    fig, ax = plt.subplots(figsize=(11.8, 6.7))
    y = np.arange(len(metrics))[::-1]
    offsets = np.linspace(-0.28, 0.28, len(measures))

    lookup = overall.set_index("metric")
    for index, (measure, offset) in enumerate(zip(measures, offsets)):
        values = lookup.loc[metrics, f"{measure}_mean"].to_numpy()
        ax.scatter(
            values,
            y + offset,
            s=48,
            marker=markers[index],
            color=cycle[index % len(cycle)],
            label=MEASURE_LABELS[measure],
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=10, fontweight="bold")
    ax.set_xlabel("Task-balanced selection score", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis="x", alpha=0.20, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=False,
        prop={"size": 9.2, "weight": "bold"},
    )
    fig.tight_layout()

    base = FIGURES_DIR / "cross_model_selection_summary"
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".svg"), base.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=EXPORT_PNG_DPI, bbox_inches="tight")
    plt.close(fig)
    return outputs


def build_family_figure(family: pd.DataFrame, overall: pd.DataFrame) -> list[Path]:
    metrics = overall["metric"].tolist()
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [f"C{i}" for i in range(4)])
    markers = ["o", "^", "s", "D"]
    lookup = family.set_index(["metric", "family"])

    fig, ax = plt.subplots(figsize=(11.6, 6.5))
    y = np.arange(len(metrics))[::-1]
    offsets = np.linspace(-0.24, 0.24, len(FAMILY_ORDER))

    for index, (family_name, offset) in enumerate(zip(FAMILY_ORDER, offsets)):
        values = [float(lookup.loc[(m, family_name), "ndcg_at_3_mean"]) for m in metrics]
        ax.scatter(
            values,
            y + offset,
            s=50,
            marker=markers[index],
            color=cycle[index % len(cycle)],
            label=FAMILY_DISPLAY[family_name],
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=10, fontweight="bold")
    ax.set_xlabel("NDCG@3", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis="x", alpha=0.20, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
        frameon=False,
        prop={"size": 9.2, "weight": "bold"},
    )
    fig.tight_layout()

    base = FIGURES_DIR / "appendix_cross_model_ndcg3_by_family"
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".svg"), base.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=EXPORT_PNG_DPI, bbox_inches="tight")
    plt.close(fig)
    return outputs


# -----------------------------------------------------------------------------
# LaTeX tables
# -----------------------------------------------------------------------------


def write_main_table(path: Path, overall: pd.DataFrame) -> None:
    lookup = overall.set_index("metric")
    metrics = ordered_metrics(overall["metric"].unique())
    best_values = {measure: best(overall, measure) for measure in MAIN_MEASURES}
    rows: list[str] = []

    for metric in metrics:
        row = lookup.loc[metric]
        if isinstance(row, pd.DataFrame):
            raise ValueError(
                f"Duplicate overall summary rows were found for metric: {metric}"
            )

        cells = [
            main_table_cell(
                mean=row[f"{measure}_mean"],
                variance=row[f"{measure}_variance"],
                best_mean=best_values[measure],
            )
            for measure in MAIN_MEASURES
        ]

        rows.append(
            metric_label(metric)
            + "\n"
            + "\n".join(f"& {cell}" for cell in cells)
            + r" \\"
        )

        if metric == "LiDAR" and "Harmora" in metrics:
            rows.append(r"\midrule")

    tex = r'''\begin{table*}[t]
\centering
\caption{Cross-model representation-selection performance across the eleven tasks.}
\label{tab:cross_model_selection_main}
\small
\setlength{\tabcolsep}{4.4pt}
\renewcommand{\arraystretch}{1.15}
\providecommand{\mv}[2]{$#1\!\pm\!{\scriptscriptstyle #2}$}

\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
\textbf{Metric}
& \textbf{NDCG@1}
& \textbf{NDCG@3}
& \textbf{Selected pct.}
& \textbf{Top-1 overlap}
& \textbf{Top-3 overlap}
& \textbf{Top-5 overlap} \\
\midrule
''' + "\n\n".join(rows) + r'''
\bottomrule
\end{tabular*}

\vspace{1mm}
\begin{minipage}{0.99\textwidth}
\scriptsize
\textit{Note:}
Each entry reports the task-balanced mean across the four downstream
seeds, with the corresponding seed variance shown in smaller type.
For each seed, the eleven tasks are averaged with equal weight.
Higher is better for all reported measures. The best mean in each
column is shown in bold.
\end{minipage}
\end{table*}
'''
    path.write_text(tex, encoding="utf-8")


def write_overall_appendix(path: Path, overall: pd.DataFrame, measures: list[str]) -> None:
    lookup = overall.set_index("metric")
    metrics = ordered_metrics(overall["metric"].unique())
    rows: list[str] = []
    for metric in metrics:
        row = lookup.loc[metric]
        cells = [mv(row[f"{m}_mean"], row[f"{m}_variance"]) for m in measures]
        rows.append(metric_label(metric) + "\n" + "\n".join(f"& {x}" for x in cells) + r" \\")
        if metric == "LiDAR" and "Harmora" in metrics:
            rows.append(r"\midrule")

    spec = "l" + "c" * len(measures)
    headers = "\n".join(f"& \\textbf{{{MEASURE_LABELS[m]}}}" for m in measures)
    tex = (
        r'''\begin{table*}[htbp]
\centering
\caption{Detailed cross-model selection results across the eleven tasks.}
\label{tab:cross_model_selection_overall}
\scriptsize
\setlength{\tabcolsep}{3.4pt}
\renewcommand{\arraystretch}{1.15}
\providecommand{\mv}[2]{$#1\!\pm\!{\scriptscriptstyle #2}$}
\resizebox{\textwidth}{!}{
'''
        + rf"\begin{{tabular}}{{{spec}}}\n"
        + r'''\toprule
\textbf{Metric}
'''
        + headers
        + r''' \\
\midrule
'''
        + "\n\n".join(rows)
        + r'''
\bottomrule
\end{tabular}
}
\vspace{1mm}
\begin{minipage}{0.99\textwidth}
\scriptsize
\textit{Note:}
Each entry reports the mean across four downstream seeds, with seed variance
shown in smaller type. For each seed, the eleven tasks are averaged with
equal weight. Higher is better except for Regret@1.
\end{minipage}
\end{table*}
'''
    )
    path.write_text(tex, encoding="utf-8")


def write_family_appendix(path: Path, family: pd.DataFrame) -> None:
    metrics = ordered_metrics(family["metric"].unique())
    blocks: list[str] = []

    for family_name in FAMILY_ORDER:
        subset = family[family["family"] == family_name].set_index("metric")
        rows: list[str] = []
        for metric in metrics:
            row = subset.loc[metric]
            cells = [mv(row[f"{m}_mean"], row[f"{m}_variance"]) for m in MAIN_MEASURES]
            rows.append(metric_label(metric) + "\n" + "\n".join(f"& {x}" for x in cells) + r" \\")
            if metric == "LiDAR" and "Harmora" in metrics:
                rows.append(r"\midrule")

        label = canonical_column(FAMILY_DISPLAY[family_name])
        block = (
            r'''\begin{table*}[htbp]
\centering
'''
            + rf"\caption{{Cross-model selection results for {FAMILY_DISPLAY[family_name].lower()}.}}\n"
            + rf"\label{{tab:cross_model_selection_{label}}}\n"
            + r'''\footnotesize
\setlength{\tabcolsep}{4.3pt}
\renewcommand{\arraystretch}{1.15}
\providecommand{\mv}[2]{$#1\!\pm\!{\scriptscriptstyle #2}$}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
\textbf{Metric}
& \textbf{NDCG@1}
& \textbf{NDCG@3}
& \textbf{Selected pct.}
& \textbf{Top-1 overlap}
& \textbf{Top-3 overlap}
& \textbf{Top-5 overlap} \\
\midrule
'''
            + "\n\n".join(rows)
            + r'''
\bottomrule
\end{tabular*}
\vspace{1mm}
\begin{minipage}{0.99\textwidth}
\scriptsize
\textit{Note:}
Each entry reports the mean across four downstream seeds, with seed variance
shown in smaller type. Tasks within this family are averaged equally for
each seed.
\end{minipage}
\end{table*}
'''
        )
        blocks.append(block)

    path.write_text("\n\n".join(blocks), encoding="utf-8")


def write_task_longtable(path: Path, task: pd.DataFrame) -> None:
    metrics = ordered_metrics(task["metric"].unique())
    metric_rank = {m: i for i, m in enumerate(metrics)}
    family_rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    frame = task.copy()
    frame["metric_order"] = frame["metric"].map(metric_rank)
    frame["family_order"] = frame["family"].map(family_rank)
    frame = frame.sort_values(["family_order", "task", "metric_order"])

    rows: list[str] = []
    for row in frame.itertuples(index=False):
        cells = [
            mv(getattr(row, f"{m}_mean"), getattr(row, f"{m}_variance"))
            for m in MAIN_MEASURES
        ]
        rows.append(
            f"{row.task}\n& {metric_label(row.metric)}\n"
            + "\n".join(f"& {x}" for x in cells)
            + r" \\")

    tex = r'''\scriptsize
\setlength{\tabcolsep}{2.8pt}
\renewcommand{\arraystretch}{1.10}
\providecommand{\mv}[2]{$#1\!\pm\!{\scriptscriptstyle #2}$}
\begin{longtable}{llcccccc}
\caption{Task-level cross-model representation-selection results.}
\label{tab:cross_model_selection_task_level}\\
\toprule
\textbf{Task} & \textbf{Metric} & \textbf{NDCG@1} & \textbf{NDCG@3}
& \textbf{Selected pct.} & \textbf{Top-1} & \textbf{Top-3} & \textbf{Top-5} \\
\midrule
\endfirsthead
\toprule
\textbf{Task} & \textbf{Metric} & \textbf{NDCG@1} & \textbf{NDCG@3}
& \textbf{Selected pct.} & \textbf{Top-1} & \textbf{Top-3} & \textbf{Top-5} \\
\midrule
\endhead
''' + "\n".join(rows) + r'''
\bottomrule
\end{longtable}

\noindent\scriptsize
\textit{Note:}
Each entry reports the mean across four downstream seeds, with seed variance
shown in smaller type.
'''
    path.write_text(tex, encoding="utf-8")


# -----------------------------------------------------------------------------
# Exports and main
# -----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_tables(
    detail: pd.DataFrame,
    task: pd.DataFrame,
    family: pd.DataFrame,
    overall: pd.DataFrame,
    measures: list[str],
) -> list[Path]:
    paths = {
        "detail": TABLES_DIR / "cross_model_selection_task_seed_detail.csv",
        "task": TABLES_DIR / "cross_model_selection_task_summary.csv",
        "family": TABLES_DIR / "cross_model_selection_family_summary.csv",
        "overall": TABLES_DIR / "cross_model_selection_overall_summary.csv",
        "main_tex": TABLES_DIR / "table_1_cross_model_selection.tex",
        "overall_tex": TABLES_DIR / "appendix_cross_model_selection_overall.tex",
        "family_tex": TABLES_DIR / "appendix_cross_model_selection_family.tex",
        "task_tex": TABLES_DIR / "appendix_cross_model_selection_task_level.tex",
    }
    detail.sort_values(["family", "task", "metric", "seed"]).to_csv(paths["detail"], index=False)
    task.sort_values(["family", "task", "metric"]).to_csv(paths["task"], index=False)
    family.sort_values(["family", "metric"]).to_csv(paths["family"], index=False)
    overall.to_csv(paths["overall"], index=False)
    write_main_table(paths["main_tex"], overall)
    write_overall_appendix(paths["overall_tex"], overall, measures)
    write_family_appendix(paths["family_tex"], family)
    write_task_longtable(paths["task_tex"], task)
    return list(paths.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cross-model representation-selection figures and tables."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to the task-by-seed cross-model selection CSV. "
            "Relative paths are resolved from the HARMORA project root."
        ),
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help=(
            "List readable CSV files, detected columns, and missing required "
            "fields without generating results."
        ),
    )
    return parser.parse_args()


def main() -> int:
    print(f"[version] {SCRIPT_VERSION}")
    args = parse_args()

    if args.scan:
        return scan_csv_inputs()

    path, mapping, descriptors = discover_input(args.input)
    detail, measures = load_data(path, mapping, descriptors)
    task, family, overall = build_summaries(detail, measures)

    reset_outputs()
    generated: list[Path] = []
    generated.extend(build_summary_figure(overall))
    generated.extend(build_family_figure(family, overall))
    generated.extend(export_tables(detail, task, family, overall, measures))

    run_info = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_input_csv": str(resolved(path)),
        "selected_input_sha256": sha256_file(resolved(path)),
        "recognized_columns": mapping,
        "recognized_descriptor_columns": descriptors,
        "expected_seeds": list(EXPECTED_SEEDS),
        "measures": measures,
        "n_rows": int(len(detail)),
        "n_tasks": int(detail["task"].nunique()),
        "n_metrics": int(detail["metric"].nunique()),
        "metric_order": overall["metric"].tolist(),
        "aggregation": [
            "input unit: metric-task-seed selection result",
            "tasks averaged equally within family for each seed",
            "all eleven tasks averaged equally for each overall seed result",
            "mean and sample variance computed across seeds 11, 22, 33, and 44",
            "main table ordered by mean NDCG@3",
        ],
        "generated_files": [str(resolved(x)) for x in generated],
    }
    RUN_INFO_PATH.write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[done] Generated:")
    for item in generated:
        print(f"       {resolved(item)}")
    print(f"       {resolved(RUN_INFO_PATH)}")
    print("\n[safety] Original experiment outputs were read only.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
