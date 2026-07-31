#!/usr/bin/env python
"""
Harmora spectral-analysis generator used for the paper.

Produces all figures and tables required for two separate paper subsections:

Main
----
7.1 Bandwidth Sensitivity
7.2 Spectral Organization Across Depth

Appendix
--------
A separate detailed subsection for each of the two analyses.

The script uses the project's existing canonical outputs:

1) metric_csv/harmora_k_curves_long.csv
   Required columns:
   model_alias, task, task_type, layer, K, H_K, mass, energy,
   rho, lambda, selected_K, sample_hash

2) csv/downstream_profiles_long.csv
   Required columns:
   model_alias, task, task_type, seed, layer, primary_score, sample_hash

Scientific aggregation for bandwidth sensitivity
-------------------------------------------------
- For every model-task pair, intersect the layers available over the
  primary bandwidth-analysis range.
- Compute Spearman across that identical layer set for every task/model/seed/K.
- Average models within each task and seed.
- Average tasks with equal weight within each family and seed.
- Report the mean and sample variance across downstream seeds.

Spectral-depth summaries
------------------------
For each model-layer-task candidate, normalize the observed per-mode energy:

    p_k = energy_k / sum_j energy_j

Then calculate:

    mean harmonic mode = sum_k k p_k

    harmonic-mode spread =
        sqrt(sum_k p_k (k - mean harmonic mode)^2)

The names used in figures and tables are deliberately descriptive.
No unexplained "centroid" or "effective bandwidth" terminology is required.

K90/K95 policy
--------------
The script uses the existing `mass` column only after validating its semantics.
It accepts either:
- a monotone cumulative mass curve in [0, 1], or
- nonnegative per-mode masses whose candidate-wise sum is approximately 1.

If neither interpretation passes validation, K90/K95 are omitted rather than
being fabricated from a truncated spectrum.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


VERSION = "2026-07-28-v8"

PLOT_RC = {
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 3.5,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plt.rcParams.update(PLOT_RC)


FAMILY_ORDER = [
    "Classification",
    "Clustering",
    "Pair classification",
    "Semantic textual similarity",
]

FAMILY_FILE = {
    "Classification": "classification",
    "Clustering": "clustering",
    "Pair classification": "pair_classification",
    "Semantic textual similarity": "sts",
}


class DataError(RuntimeError):
    pass


def normalize_family(value: object) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    compact = re.sub(r"\s+", "", text)

    if compact in {"classification", "class"}:
        return "Classification"
    if compact in {"clustering", "cluster"}:
        return "Clustering"
    if compact in {"pairclassification", "pair"}:
        return "Pair classification"
    if compact in {
        "sts",
        "semantictextualsimilarity",
        "semanticsimilarity",
    }:
        return "Semantic textual similarity"

    # Conservative keyword fallback.
    if "pair" in text:
        return "Pair classification"
    if "cluster" in text:
        return "Clustering"
    if "sts" in text or "similarity" in text:
        return "Semantic textual similarity"
    if "class" in text:
        return "Classification"

    raise DataError(f"Unrecognized task family: {value!r}")


def require_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataError(
            f"Missing columns in {path}:\n"
            f"  {missing}\n"
            f"Available columns:\n"
            f"  {list(df.columns)}"
        )


def load_curves(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "model_alias",
        "task",
        "task_type",
        "layer",
        "K",
        "H_K",
        "mass",
        "energy",
        "rho",
        "lambda",
        "selected_K",
        "sample_hash",
    ]
    require_columns(df, required, path)

    out = df[required].copy()
    out["family"] = out["task_type"].map(normalize_family)

    numeric = ["layer", "K", "H_K", "mass", "energy", "rho", "lambda", "selected_K"]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["layer", "K", "H_K", "energy"]).copy()
    out["K"] = out["K"].astype(int)

    # Defensive aggregation in case the source contains harmless duplicates.
    keys = [
        "model_alias",
        "task",
        "task_type",
        "family",
        "layer",
        "K",
        "sample_hash",
    ]
    out = (
        out.groupby(keys, as_index=False, observed=True)
        .agg(
            H_K=("H_K", "mean"),
            mass=("mass", "mean"),
            energy=("energy", "mean"),
            rho=("rho", "mean"),
            lambda_value=("lambda", "mean"),
            selected_K=("selected_K", "max"),
        )
    )

    if (out["energy"] < 0).any():
        bad = out.loc[out["energy"] < 0].head()
        raise DataError(f"Negative harmonic energy encountered:\n{bad}")

    return out


def load_downstream(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "model_alias",
        "task",
        "task_type",
        "seed",
        "layer",
        "primary_score",
        "sample_hash",
    ]
    require_columns(df, required, path)

    out = df[required].copy()
    out["family"] = out["task_type"].map(normalize_family)

    for column in ["seed", "layer", "primary_score"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["seed", "layer", "primary_score"]).copy()
    out["seed"] = out["seed"].astype(int)

    keys = [
        "model_alias",
        "task",
        "task_type",
        "family",
        "seed",
        "layer",
        "sample_hash",
    ]
    out = (
        out.groupby(keys, as_index=False, observed=True)
        .agg(primary_score=("primary_score", "mean"))
    )
    return out


def choose_k_values(curves: pd.DataFrame, k_min: Optional[int], k_max: Optional[int]) -> list[int]:
    values = sorted(int(value) for value in curves["K"].dropna().unique())
    if k_min is not None:
        values = [value for value in values if value >= k_min]
    if k_max is not None:
        values = [value for value in values if value <= k_max]
    if not values:
        raise DataError("No K values remain after filtering.")
    return values


def ordered_tasks(df: pd.DataFrame) -> list[str]:
    """Return tasks in a stable order, grouped by task family."""
    tasks: list[str] = []

    for family in FAMILY_ORDER:
        family_tasks = sorted(
            df.loc[df["family"] == family, "task"]
            .dropna()
            .astype(str)
            .unique(),
            key=lambda task: pretty_task_name(task).lower(),
        )
        tasks.extend(family_tasks)

    return tasks


def build_task_colors(df: pd.DataFrame) -> dict[str, object]:
    """Assign distinct, consistent task colors across all panels."""
    tasks = ordered_tasks(df)
    cmap = plt.get_cmap("tab20")

    # Use the darker members of the qualitative pairs first so adjacent
    # legend entries remain visually distinct.
    color_indices = [0, 2, 4, 6, 8, 10, 12, 16, 18, 14,
                     1, 3, 5, 7, 9, 11, 13, 17, 19, 15]

    return {
        task: cmap(color_indices[index % len(color_indices)])
        for index, task in enumerate(tasks)
    }


def build_common_support(
    curves: pd.DataFrame,
    k_values: list[int],
    min_layers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    - retained model-task cases and their exact common layer sets
    - a long table containing one row per retained common layer
    """
    case_keys = ["model_alias", "task", "task_type", "family", "sample_hash"]

    layer_sets = (
        curves.loc[curves["K"].isin(k_values)]
        .groupby(case_keys + ["K"], observed=True)["layer"]
        .apply(lambda values: frozenset(pd.unique(values)))
        .reset_index(name="layer_set")
    )

    retained_rows = []
    layer_rows = []

    for case_values, group in layer_sets.groupby(case_keys, observed=True):
        by_k = {int(row.K): row.layer_set for row in group.itertuples()}

        if any(k not in by_k for k in k_values):
            continue

        common_layers = set.intersection(*(set(by_k[k]) for k in k_values))
        if len(common_layers) < min_layers:
            continue

        record = dict(zip(case_keys, case_values))
        retained_rows.append(
            {
                **record,
                "n_common_layers": len(common_layers),
                "min_common_layer": min(common_layers),
                "max_common_layer": max(common_layers),
            }
        )
        for layer in sorted(common_layers):
            layer_rows.append({**record, "layer": layer})

    retained = pd.DataFrame(retained_rows)
    common_layers = pd.DataFrame(layer_rows)

    if retained.empty:
        raise DataError(
            "No model-task cases have sufficient common layer support across "
            f"all displayed K values ({min(k_values)}--{max(k_values)})."
        )

    return retained, common_layers


def merge_curves_and_downstream(
    curves: pd.DataFrame,
    downstream: pd.DataFrame,
    common_layers: pd.DataFrame,
    k_values: list[int],
) -> pd.DataFrame:
    case_keys = ["model_alias", "task", "task_type", "family", "sample_hash"]

    supported_curves = curves.merge(
        common_layers,
        on=case_keys + ["layer"],
        how="inner",
        validate="many_to_one",
    )
    supported_curves = supported_curves.loc[supported_curves["K"].isin(k_values)].copy()

    downstream_for_merge = downstream.copy()

    # Enforce sample identity whenever both files provide nonempty hashes.
    merge_keys = ["model_alias", "task", "task_type", "family", "layer"]
    curve_hash_available = supported_curves["sample_hash"].notna().all()
    downstream_hash_available = downstream_for_merge["sample_hash"].notna().all()

    if curve_hash_available and downstream_hash_available:
        merge_keys.append("sample_hash")
    else:
        supported_curves = supported_curves.drop(columns=["sample_hash"])
        downstream_for_merge = downstream_for_merge.drop(columns=["sample_hash"])

    merged = supported_curves.merge(
        downstream_for_merge,
        on=merge_keys,
        how="inner",
        validate="many_to_many",
        suffixes=("_metric", "_target"),
    )

    if merged.empty:
        raise DataError(
            "The Harmora curves and downstream profiles did not merge. "
            "Check model/task/layer names and sample hashes."
        )

    return merged


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return np.nan
    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan
    return float(spearmanr(valid["x"], valid["y"]).statistic)


def bandwidth_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["model_alias", "task", "task_type", "family", "seed", "K"]

    for values, group in merged.groupby(group_keys, observed=True):
        group = (
            group.groupby("layer", as_index=False)
            .agg(
                H_K=("H_K", "mean"),
                primary_score=("primary_score", "mean"),
            )
        )
        rho = safe_spearman(group["H_K"], group["primary_score"])
        record = dict(zip(group_keys, values))
        rows.append(
            {
                **record,
                "spearman": rho,
                "n_layers": len(group),
            }
        )

    result = pd.DataFrame(rows).dropna(subset=["spearman"])
    if result.empty:
        raise DataError("No valid bandwidth correlations were produced.")
    return result


def aggregate_bandwidth(case_results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Models are averaged within each task and seed.
    task_seed = (
        case_results.groupby(
            ["K", "family", "task", "seed"],
            as_index=False,
            observed=True,
        )
        .agg(
            spearman=("spearman", "mean"),
            n_models=("model_alias", "nunique"),
        )
    )

    # Tasks receive equal weight within family and seed.
    family_seed = (
        task_seed.groupby(["K", "family", "seed"], as_index=False, observed=True)
        .agg(
            spearman=("spearman", "mean"),
            n_tasks=("task", "nunique"),
        )
    )

    family_curve = (
        family_seed.groupby(["K", "family"], as_index=False, observed=True)
        .agg(
            mean_spearman=("spearman", "mean"),
            seed_variance=(
                "spearman",
                lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else np.nan,
            ),
            n_seeds=("seed", "nunique"),
            n_tasks=("n_tasks", "max"),
        )
    )

    # Overall: all tasks receive equal weight within seed.
    overall_seed = (
        task_seed.groupby(["K", "seed"], as_index=False, observed=True)
        .agg(
            spearman=("spearman", "mean"),
            n_tasks=("task", "nunique"),
        )
    )

    overall_curve = (
        overall_seed.groupby("K", as_index=False)
        .agg(
            mean_spearman=("spearman", "mean"),
            seed_variance=(
                "spearman",
                lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else np.nan,
            ),
            n_seeds=("seed", "nunique"),
            n_tasks=("n_tasks", "max"),
        )
    )

    task_curve = (
        task_seed.groupby(["K", "family", "task"], as_index=False, observed=True)
        .agg(
            mean_spearman=("spearman", "mean"),
            seed_variance=(
                "spearman",
                lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else np.nan,
            ),
            n_seeds=("seed", "nunique"),
        )
    )

    return {
        "task_seed": task_seed,
        "family_seed": family_seed,
        "family_curve": family_curve,
        "overall_curve": overall_curve,
        "task_curve": task_curve,
    }


def bandwidth_summary(
    family_curve: pd.DataFrame,
    fixed_k: int,
) -> pd.DataFrame:
    rows = []

    for family in FAMILY_ORDER:
        curve = family_curve.loc[family_curve["family"] == family].sort_values("K")
        if curve.empty:
            continue

        fixed = curve.loc[curve["K"] == fixed_k]
        if fixed.empty:
            raise DataError(f"K={fixed_k} is absent for family {family}.")

        best = curve.loc[curve["mean_spearman"].idxmax()]
        fixed_row = fixed.iloc[0]

        rows.append(
            {
                "family": family,
                "tasks": int(fixed_row["n_tasks"]),
                "fixed_K": fixed_k,
                "fixed_alignment": float(fixed_row["mean_spearman"]),
                "fixed_seed_variance": float(fixed_row["seed_variance"]),
                "best_displayed_K": int(best["K"]),
                "best_displayed_alignment": float(best["mean_spearman"]),
                "gap_to_best_displayed": float(
                    best["mean_spearman"] - fixed_row["mean_spearman"]
                ),
            }
        )

    return pd.DataFrame(rows)


def task_bandwidth_summary(task_curve: pd.DataFrame, fixed_k: int) -> pd.DataFrame:
    """Create the Appendix task-level bandwidth table."""
    rows = []

    for family in FAMILY_ORDER:
        family_data = task_curve.loc[task_curve["family"] == family]
        for task, curve in family_data.groupby("task", observed=True):
            curve = curve.sort_values("K")
            fixed = curve.loc[curve["K"] == fixed_k]
            if fixed.empty:
                continue

            fixed_row = fixed.iloc[0]
            best = curve.loc[curve["mean_spearman"].idxmax()]
            rows.append(
                {
                    "family": family,
                    "task": str(task),
                    "fixed_K": fixed_k,
                    "fixed_alignment": float(fixed_row["mean_spearman"]),
                    "fixed_seed_variance": float(fixed_row["seed_variance"]),
                    "best_displayed_K": int(best["K"]),
                    "best_displayed_alignment": float(best["mean_spearman"]),
                    "gap_to_best_displayed": float(
                        best["mean_spearman"] - fixed_row["mean_spearman"]
                    ),
                }
            )

    return pd.DataFrame(rows)

def support_audit(
    curves: pd.DataFrame,
    all_k_values: list[int],
    fixed_k: int,
    min_layers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retention as the required maximum K increases.
    This is based on exact candidate-layer availability in the curve file.
    """
    case_keys = ["model_alias", "task", "task_type", "family", "sample_hash"]

    fixed_layer_sets = (
        curves.loc[curves["K"] == fixed_k]
        .groupby(case_keys, observed=True)["layer"]
        .nunique()
        .reset_index(name="n_layers")
    )
    fixed_eligible = fixed_layer_sets.loc[fixed_layer_sets["n_layers"] >= min_layers]
    fixed_cases = len(fixed_eligible)
    fixed_tasks = fixed_eligible["task"].nunique()

    rows = []
    for cutoff in all_k_values:
        required_k = [value for value in all_k_values if value <= cutoff]
        retained, _ = build_common_support(curves, required_k, min_layers)
        rows.append(
            {
                "required_through_K": cutoff,
                "retained_cases": len(retained),
                "retained_tasks": retained["task"].nunique(),
                "case_retention_fraction": (
                    len(retained) / fixed_cases if fixed_cases else np.nan
                ),
                "task_retention_fraction": (
                    retained["task"].nunique() / fixed_tasks if fixed_tasks else np.nan
                ),
            }
        )

    curve = pd.DataFrame(rows)

    final = curve.iloc[-1]
    final_k = int(final["required_through_K"])
    retained_final, common_final = build_common_support(
        curves,
        [value for value in all_k_values if value <= final_k],
        min_layers,
    )

    summary = pd.DataFrame(
        [
            ("Primary bandwidth", fixed_k),
            ("Maximum displayed bandwidth", final_k),
            ("Displayed bandwidth values", len(all_k_values)),
            ("Eligible cases at primary K", fixed_cases),
            ("Retained support-matched cases", int(final["retained_cases"])),
            ("Eligible tasks at primary K", fixed_tasks),
            ("Retained support-matched tasks", int(final["retained_tasks"])),
            ("Case retention", float(final["case_retention_fraction"])),
            ("Task retention", float(final["task_retention_fraction"])),
            (
                "Minimum common layers per retained case",
                int(retained_final["n_common_layers"].min()),
            ),
            (
                "Maximum common layers per retained case",
                int(retained_final["n_common_layers"].max()),
            ),
        ],
        columns=["quantity", "value"],
    )

    return curve, summary


def add_relative_depth(curves: pd.DataFrame) -> pd.DataFrame:
    """
    Relative depth is defined within each architecture from the ordered
    exposed layer indices. The same mapping is then used for all tasks.
    """
    layer_map_rows = []
    for model, group in curves.groupby("model_alias", observed=True):
        layers = sorted(group["layer"].unique())
        if len(layers) == 1:
            mapping = {layers[0]: 0.5}
        else:
            mapping = {
                layer: index / (len(layers) - 1)
                for index, layer in enumerate(layers)
            }
        for layer, relative_depth in mapping.items():
            layer_map_rows.append(
                {
                    "model_alias": model,
                    "layer": layer,
                    "relative_depth": relative_depth,
                }
            )

    layer_map = pd.DataFrame(layer_map_rows)
    return curves.merge(
        layer_map,
        on=["model_alias", "layer"],
        how="left",
        validate="many_to_one",
    )


def infer_mass_semantics(curves: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """
    Returns:
      cumulative | per_mode | unavailable
    """
    keys = ["model_alias", "task", "layer", "sample_hash"]
    diagnostics = []

    for _, group in curves.sort_values("K").groupby(keys, observed=True):
        mass = group["mass"].dropna().to_numpy(dtype=float)
        if len(mass) == 0:
            continue

        within_unit = bool(np.all((mass >= -1e-8) & (mass <= 1.0001)))
        monotone = bool(np.all(np.diff(mass) >= -1e-8))
        final_value = float(mass[-1])
        total = float(np.sum(mass))

        diagnostics.append(
            {
                "within_unit": within_unit,
                "monotone": monotone,
                "final_value": final_value,
                "sum_value": total,
            }
        )

    if not diagnostics:
        return "unavailable", {}

    diag = pd.DataFrame(diagnostics)
    cumulative_fraction = float(
        (diag["within_unit"] & diag["monotone"] & (diag["final_value"] >= 0.80)).mean()
    )
    per_mode_fraction = float(
        (
            diag["within_unit"]
            & (diag["sum_value"] >= 0.90)
            & (diag["sum_value"] <= 1.10)
        ).mean()
    )

    report = {
        "candidate_groups_checked": int(len(diag)),
        "cumulative_compatible_fraction": cumulative_fraction,
        "per_mode_compatible_fraction": per_mode_fraction,
    }

    if cumulative_fraction >= 0.95:
        return "cumulative", report
    if per_mode_fraction >= 0.95:
        return "per_mode", report
    return "unavailable", report


def spectral_candidate_statistics(
    curves: pd.DataFrame,
    depth_bins: int,
) -> tuple[pd.DataFrame, str, dict[str, float]]:
    working = add_relative_depth(curves).copy()

    candidate_keys = [
        "model_alias",
        "task",
        "task_type",
        "family",
        "layer",
        "relative_depth",
        "sample_hash",
    ]

    observed_total = (
        working.groupby(candidate_keys, as_index=False, observed=True)
        .agg(observed_energy=("energy", "sum"))
    )
    working = working.merge(
        observed_total,
        on=candidate_keys,
        how="left",
        validate="many_to_one",
    )
    working = working.loc[working["observed_energy"] > 0].copy()

    working["normalized_mode_energy"] = (
        working["energy"] / working["observed_energy"]
    )

    working["weighted_mode"] = (
        working["K"] * working["normalized_mode_energy"]
    )

    means = (
        working.groupby(candidate_keys, as_index=False, observed=True)
        .agg(mean_harmonic_mode=("weighted_mode", "sum"))
    )
    working = working.merge(
        means,
        on=candidate_keys,
        how="left",
        validate="many_to_one",
    )

    working["weighted_squared_deviation"] = (
        working["normalized_mode_energy"]
        * (working["K"] - working["mean_harmonic_mode"]) ** 2
    )

    candidate_stats = (
        working.groupby(candidate_keys, as_index=False, observed=True)
        .agg(
            mean_harmonic_mode=("mean_harmonic_mode", "first"),
            mode_variance=("weighted_squared_deviation", "sum"),
            n_modes=("K", "nunique"),
            observed_energy=("observed_energy", "first"),
        )
    )
    candidate_stats["harmonic_mode_spread"] = np.sqrt(
        candidate_stats["mode_variance"].clip(lower=0)
    )

    mass_semantics, mass_report = infer_mass_semantics(working)

    if mass_semantics != "unavailable":
        threshold_rows = []

        for values, group in working.sort_values("K").groupby(
            candidate_keys,
            observed=True,
        ):
            record = dict(zip(candidate_keys, values))
            if mass_semantics == "cumulative":
                cumulative = group["mass"].to_numpy(dtype=float)
            else:
                cumulative = np.cumsum(group["mass"].to_numpy(dtype=float))

            k_array = group["K"].to_numpy(dtype=int)

            for threshold, name in [(0.90, "K90"), (0.95, "K95")]:
                indices = np.flatnonzero(cumulative >= threshold)
                if len(indices):
                    record[name] = int(k_array[indices[0]])
                    record[f"{name}_available"] = True
                else:
                    record[name] = np.nan
                    record[f"{name}_available"] = False

            threshold_rows.append(record)

        threshold_df = pd.DataFrame(threshold_rows)
        candidate_stats = candidate_stats.merge(
            threshold_df,
            on=candidate_keys,
            how="left",
            validate="one_to_one",
        )

    edges = np.linspace(0.0, 1.0, depth_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    candidate_stats["depth_bin"] = pd.cut(
        candidate_stats["relative_depth"],
        bins=edges,
        labels=False,
        include_lowest=True,
        right=True,
    )
    candidate_stats["depth_center"] = candidate_stats["depth_bin"].map(
        dict(enumerate(centers))
    )

    working["depth_bin"] = pd.cut(
        working["relative_depth"],
        bins=edges,
        labels=False,
        include_lowest=True,
        right=True,
    )
    working["depth_center"] = working["depth_bin"].map(
        dict(enumerate(centers))
    )

    return candidate_stats, working, mass_semantics, mass_report


def aggregate_depth(
    candidate_stats: pd.DataFrame,
    mode_rows: pd.DataFrame,
    mass_semantics: str,
) -> dict[str, pd.DataFrame]:
    # Candidate summaries -> task/depth -> family/depth.
    task_depth = (
        candidate_stats.groupby(
            ["family", "task", "depth_bin", "depth_center"],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_harmonic_mode=("mean_harmonic_mode", "mean"),
            harmonic_mode_spread=("harmonic_mode_spread", "mean"),
            n_candidates=("layer", "count"),
        )
    )

    family_depth = (
        task_depth.groupby(
            ["family", "depth_bin", "depth_center"],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_harmonic_mode=("mean_harmonic_mode", "mean"),
            harmonic_mode_spread=("harmonic_mode_spread", "mean"),
            n_tasks=("task", "nunique"),
        )
    )

    # Candidate mode rows -> task/depth/K -> family/depth/K.
    task_heat = (
        mode_rows.groupby(
            ["family", "task", "depth_bin", "depth_center", "K"],
            as_index=False,
            observed=True,
        )
        .agg(
            normalized_mode_energy=("normalized_mode_energy", "mean"),
        )
    )

    family_heat = (
        task_heat.groupby(
            ["family", "depth_bin", "depth_center", "K"],
            as_index=False,
            observed=True,
        )
        .agg(
            normalized_mode_energy=("normalized_mode_energy", "mean"),
            n_tasks=("task", "nunique"),
        )
    )

    def region(value: float) -> str:
        if value < 1.0 / 3.0:
            return "Early"
        if value <= 2.0 / 3.0:
            return "Middle"
        return "Late"

    candidate_stats = candidate_stats.copy()
    candidate_stats["depth_region"] = candidate_stats["relative_depth"].map(region)

    task_region = (
        candidate_stats.groupby(
            ["family", "task", "depth_region"],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_harmonic_mode=("mean_harmonic_mode", "mean"),
            harmonic_mode_spread=("harmonic_mode_spread", "mean"),
        )
    )

    family_region = (
        task_region.groupby(
            ["family", "depth_region"],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_harmonic_mode=("mean_harmonic_mode", "mean"),
            harmonic_mode_spread=("harmonic_mode_spread", "mean"),
        )
    )

    summary_rows = []
    for family in FAMILY_ORDER:
        family_candidates = candidate_stats.loc[
            candidate_stats["family"] == family
        ]
        if family_candidates.empty:
            continue

        row = {
            "family": family,
            "tasks": int(family_candidates["task"].nunique()),
            "layer_candidates": int(len(family_candidates)),
        }

        for region_name in ["Early", "Middle", "Late"]:
            region_row = family_region.loc[
                (family_region["family"] == family)
                & (family_region["depth_region"] == region_name)
            ]
            prefix = region_name.lower()
            row[f"{prefix}_mean_mode"] = (
                float(region_row["mean_harmonic_mode"].iloc[0])
                if not region_row.empty
                else np.nan
            )
            row[f"{prefix}_mode_spread"] = (
                float(region_row["harmonic_mode_spread"].iloc[0])
                if not region_row.empty
                else np.nan
            )

        if mass_semantics != "unavailable":
            for name in ["K90", "K95"]:
                available_column = f"{name}_available"
                row[f"{name}_availability"] = float(
                    family_candidates[available_column].mean()
                )
                observed = family_candidates.loc[
                    family_candidates[available_column],
                    name,
                ]
                row[f"median_{name}"] = (
                    float(observed.median()) if len(observed) else np.nan
                )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    return {
        "candidate_stats": candidate_stats,
        "task_depth": task_depth,
        "family_depth": family_depth,
        "family_heat": family_heat,
        "summary": summary,
    }


def save_single_plot(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)



# def plot_main_bandwidth(family_curve: pd.DataFrame, fixed_k: int, out: Path) -> None:
#     """
#     Clean Main-panel curve.

#     Uncertainty is intentionally not drawn as a ribbon. Seed variance is
#     reported numerically in the accompanying table.
#     """
#     fig, ax = plt.subplots(figsize=(5.7, 4.15))

#     for family in FAMILY_ORDER:
#         part = family_curve.loc[family_curve["family"] == family].sort_values("K")
#         if part.empty:
#             continue

#         markevery = max(1, len(part) // 14)
#         display_label = (
#             "STS"
#             if family == "Semantic textual similarity"
#             else family
#         )
#         ax.plot(
#             part["K"],
#             part["mean_spearman"],
#             marker="o",
#             markersize=3.2,
#             markevery=markevery,
#             label=display_label,
#         )

#     ax.axvline(fixed_k, linestyle="--", linewidth=1.0)
#     ax.text(
#         fixed_k,
#         0.985,
#         rf"$K={fixed_k}$",
#         transform=ax.get_xaxis_transform(),
#         ha="left",
#         va="top",
#         fontsize=7,
#     )
#     ax.set_xlabel("Retained harmonic bandwidth $K$", labelpad=2)
#     ax.set_ylabel("Task-balanced Spearman correlation")
#     ax.grid(True, linewidth=0.45, alpha=0.18)
#     ax.set_axisbelow(True)
#     ax.legend(frameon=False, loc="lower right", fontsize=6.6)
#     fig.tight_layout(pad=0.4)
#     save_single_plot(fig, out / "main_bandwidth_sensitivity_panel")

def plot_main_bandwidth(
    task_curve: pd.DataFrame,
    fixed_k: int,
    task_colors: dict[str, object],
    out: Path,
) -> None:
    """
    Task-level bandwidth curves.

    Each line represents one downstream task. Values are averaged across
    models and downstream seeds within that task.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.35))

    for task in ordered_tasks(task_curve):
        part = task_curve.loc[
            task_curve["task"].astype(str) == str(task)
        ].sort_values("K")

        if part.empty:
            continue

        markevery = max(1, len(part) // 14)

        ax.plot(
            part["K"],
            part["mean_spearman"],
            marker="o",
            markersize=3.0,
            markevery=markevery,
            color=task_colors[task],
            label=pretty_task_name(task),
        )

    ax.axvline(fixed_k, linestyle="--", linewidth=1.0)
    ax.text(
        fixed_k,
        0.985,
        rf"$K={fixed_k}$",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=7,
    )

    ax.set_xlabel("Retained harmonic bandwidth $K$")
    ax.set_ylabel("Task-level Spearman correlation")
    ax.grid(True, linewidth=0.45, alpha=0.18)
    ax.set_axisbelow(True)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=6.4,
        columnspacing=1.0,
        handlelength=1.8,
    )

    fig.subplots_adjust(bottom=0.30)
    save_single_plot(fig, out / "main_bandwidth_sensitivity_panel")

# def plot_main_depth(
#     family_depth: pd.DataFrame,
#     value_column: str,
#     ylabel: str,
#     filename: str,
#     out: Path,
# ) -> None:
#     """
#     Clean depth panel with no embedded title or caption.

#     Family colors follow the same plotting order as the bandwidth panel.
#     """
#     fig, ax = plt.subplots(figsize=(5.25, 4.15))

#     for family in FAMILY_ORDER:
#         part = family_depth.loc[
#             family_depth["family"] == family
#         ].sort_values("depth_center")
#         if part.empty:
#             continue
#         ax.plot(
#             part["depth_center"],
#             part[value_column],
#             marker="o",
#             markersize=3.2,
#         )

#     ax.set_xlabel("Relative representation depth")
#     ax.set_ylabel(ylabel)
#     ax.set_xlim(0.0, 1.0)
#     ax.grid(True, linewidth=0.45, alpha=0.18)
#     ax.set_axisbelow(True)
#     fig.tight_layout(pad=0.4)
#     save_single_plot(fig, out / filename)

def plot_main_depth(
    task_depth: pd.DataFrame,
    value_column: str,
    ylabel: str,
    filename: str,
    task_colors: dict[str, object],
    out: Path,
) -> None:
    """
    Task-level spectral-depth curves.

    Each line represents one downstream task. Candidate values are averaged
    within each task and relative-depth bin.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.35))

    for task in ordered_tasks(task_depth):
        part = task_depth.loc[
            task_depth["task"].astype(str) == str(task)
        ].sort_values("depth_center")

        if part.empty:
            continue

        ax.plot(
            part["depth_center"],
            part[value_column],
            marker="o",
            markersize=3.0,
            color=task_colors[task],
            label=pretty_task_name(task),
        )

    ax.set_xlabel("Relative representation depth")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, linewidth=0.45, alpha=0.18)
    ax.set_axisbelow(True)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=6.4,
        columnspacing=1.0,
        handlelength=1.8,
    )

    fig.subplots_adjust(bottom=0.30)
    save_single_plot(fig, out / filename)


def add_early_depth_relative_change(
    task_depth: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """
    Express each task's depth curve as percentage change from its
    mean value over the early third of relative depth.
    """
    result = task_depth.copy()

    early_baseline = (
        result.loc[result["depth_center"] <= (1.0 / 3.0)]
        .groupby(["family", "task"], as_index=False, observed=True)
        .agg(early_baseline=(value_column, "mean"))
    )

    result = result.merge(
        early_baseline,
        on=["family", "task"],
        how="left",
        validate="many_to_one",
    )

    if result["early_baseline"].isna().any():
        raise DataError(
            f"Missing early-depth baseline for {value_column}."
        )

    if (result["early_baseline"] <= 0).any():
        raise DataError(
            f"Non-positive early-depth baseline for {value_column}."
        )

    result["relative_change_pct"] = (
        100.0
        * (result[value_column] - result["early_baseline"])
        / result["early_baseline"]
    )

    return result

def _joint_contraction_span(
    mean_mode_relative: pd.DataFrame,
    spread_relative: pd.DataFrame,
) -> tuple[float, float] | None:
    """Return the longest depth interval where both task medians are below zero."""
    mean_curve = (
        mean_mode_relative.groupby("depth_center", as_index=False)["relative_change_pct"]
        .median()
        .rename(columns={"relative_change_pct": "median_mean_mode_change"})
    )
    spread_curve = (
        spread_relative.groupby("depth_center", as_index=False)["relative_change_pct"]
        .median()
        .rename(columns={"relative_change_pct": "median_spread_change"})
    )

    joint = mean_curve.merge(spread_curve, on="depth_center", how="inner")
    joint = joint.sort_values("depth_center").reset_index(drop=True)
    if joint.empty:
        return None

    contracting = (
        (joint["median_mean_mode_change"] < 0)
        & (joint["median_spread_change"] < 0)
    ).to_numpy()
    if not contracting.any():
        return None

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, active in enumerate(contracting):
        if active and run_start is None:
            run_start = index
        if run_start is not None and (not active or index == len(contracting) - 1):
            run_end = index if active and index == len(contracting) - 1 else index - 1
            runs.append((run_start, run_end))
            run_start = None

    start_index, end_index = max(runs, key=lambda pair: pair[1] - pair[0] + 1)
    centers = joint["depth_center"].to_numpy(dtype=float)
    positive_steps = np.diff(centers)
    positive_steps = positive_steps[positive_steps > 0]
    half_step = 0.5 * float(np.median(positive_steps)) if len(positive_steps) else 0.05

    # Highlight only the core of the contraction interval. Removing one
    # boundary bin on each side avoids shading the transition regions where
    # the curves have only just crossed zero or have already begun to recover.
    if end_index - start_index + 1 >= 4:
        start_index += 1
        end_index -= 1

    left = max(0.0, float(centers[start_index] - half_step))
    right = min(1.0, float(centers[end_index] + half_step))
    return left, right


def plot_main_task_diagnostics(
    task_curve: pd.DataFrame,
    task_depth: pd.DataFrame,
    fixed_k: int,
    out: Path,
) -> None:
    """
    Create the three main task-level diagnostic panels with one shared
    legend. Depth quantities are shown as percentage change from each
    task's early-depth mean.
    """
    tasks = ordered_tasks(task_curve)
    task_colors = build_task_colors(task_curve)

    mean_mode_relative = add_early_depth_relative_change(
        task_depth,
        "mean_harmonic_mode",
    )
    spread_relative = add_early_depth_relative_change(
        task_depth,
        "harmonic_mode_spread",
    )
    contraction_span = _joint_contraction_span(
        mean_mode_relative,
        spread_relative,
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(16.2, 4.15),
    )

    # -----------------------------------------------------------------
    # (a) Bandwidth sensitivity: absolute task-level Spearman correlation
    # -----------------------------------------------------------------
    ax = axes[0]

    for task in tasks:
        part = task_curve.loc[
            task_curve["task"].astype(str) == str(task)
        ].sort_values("K")

        if part.empty:
            continue

        markevery = max(1, len(part) // 14)

        ax.plot(
            part["K"],
            part["mean_spearman"],
            marker="o",
            markersize=3.0,
            markevery=markevery,
            color=task_colors[task],
            label=pretty_task_name(task),
        )

    ax.axvline(
        fixed_k,
        linestyle="--",
        linewidth=1.0,
    )
    ax.text(
        fixed_k,
        0.98,
        rf"$K={fixed_k}$",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8,
    )

    ax.set_xlabel("Retained harmonic bandwidth $K$")
    ax.set_ylabel("Task-level Spearman correlation")
    ax.grid(True, linewidth=0.45, alpha=0.18)
    ax.set_axisbelow(True)

    # -----------------------------------------------------------------
    # (b) Mean harmonic mode: relative change from early depth
    # -----------------------------------------------------------------
    ax = axes[1]

    for task in tasks:
        part = mean_mode_relative.loc[
            mean_mode_relative["task"].astype(str) == str(task)
        ].sort_values("depth_center")

        if part.empty:
            continue

        ax.plot(
            part["depth_center"],
            part["relative_change_pct"],
            marker="o",
            markersize=3.0,
            color=task_colors[task],
        )

    ax.axhline(0.0, linewidth=0.9, linestyle="--")
    if contraction_span is not None:
        ax.axvspan(
            contraction_span[0],
            contraction_span[1],
            alpha=0.08,
            zorder=0,
        )
        ax.text(
            0.5 * (contraction_span[0] + contraction_span[1]),
            0.95,
            "Lower-frequency concentration",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
        )

    mean_median = (
        mean_mode_relative.groupby("depth_center", as_index=False)["relative_change_pct"]
        .median()
        .sort_values("depth_center")
    )
    if len(mean_median) >= 2:
        last = mean_median.iloc[-1]
        ax.annotate(
            "Output re-expansion",
            xy=(float(last["depth_center"]), float(last["relative_change_pct"])),
            xycoords="data",
            xytext=(0.73, 0.79),
            textcoords="axes fraction",
            ha="left",
            va="center",
            fontsize=7.0,
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        )

    ax.set_xlabel("Relative representation depth", labelpad=2)
    ax.set_ylabel("Mean-mode change from early depth (%)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, linewidth=0.45, alpha=0.18)
    ax.set_axisbelow(True)

    # -----------------------------------------------------------------
    # (c) Harmonic-mode spread: relative change from early depth
    # -----------------------------------------------------------------
    ax = axes[2]

    for task in tasks:
        part = spread_relative.loc[
            spread_relative["task"].astype(str) == str(task)
        ].sort_values("depth_center")

        if part.empty:
            continue

        ax.plot(
            part["depth_center"],
            part["relative_change_pct"],
            marker="o",
            markersize=3.0,
            color=task_colors[task],
        )

    ax.axhline(0.0, linewidth=0.9, linestyle="--")
    if contraction_span is not None:
        ax.axvspan(
            contraction_span[0],
            contraction_span[1],
            alpha=0.08,
            zorder=0,
        )
        ax.text(
            0.5 * (contraction_span[0] + contraction_span[1]),
            0.95,
            "Narrower mode range",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
        )

    spread_median = (
        spread_relative.groupby("depth_center", as_index=False)["relative_change_pct"]
        .median()
        .sort_values("depth_center")
    )
    if len(spread_median) >= 2:
        last = spread_median.iloc[-1]
        ax.annotate(
            "Output re-expansion",
            xy=(float(last["depth_center"]), float(last["relative_change_pct"])),
            xycoords="data",
            xytext=(0.73, 0.79),
            textcoords="axes fraction",
            ha="left",
            va="center",
            fontsize=7.0,
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        )

    ax.set_xlabel("Relative representation depth", labelpad=2)
    ax.set_ylabel("Mode-spread change from early depth (%)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, linewidth=0.45, alpha=0.18)
    ax.set_axisbelow(True)

    # Panel labels below the three plots.
    for ax, panel_label in zip(axes, ("(a)", "(b)", "(c)")):
        ax.text(
            0.5,
            -0.18,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            clip_on=False,
        )

    # One legend for the complete three-panel figure.
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=6,
        frameon=False,
        fontsize=7.4,
        handlelength=2.0,
        columnspacing=1.2,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        top=0.97,
        bottom=0.245,
        wspace=0.30,
    )

    output_stem = out / "figure_3_harmonic_diagnostics"

    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def plot_support_audit(support_curve: pd.DataFrame, fixed_k: int, out: Path) -> None:
    """Compact support audit with no embedded title."""
    fig, ax = plt.subplots(figsize=(5.45, 4.15))

    markevery = max(1, len(support_curve) // 14)
    case_values = 100.0 * support_curve["case_retention_fraction"]
    task_values = 100.0 * support_curve["task_retention_fraction"]

    ax.plot(
        support_curve["required_through_K"],
        case_values,
        marker="o",
        markersize=3.0,
        markevery=markevery,
        label="Model--task cases",
    )
    ax.plot(
        support_curve["required_through_K"],
        task_values,
        marker="s",
        markersize=3.0,
        markevery=markevery,
        label="Tasks",
    )
    ax.axvline(fixed_k, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Complete support required through $K$")
    ax.set_ylabel(r"Retention relative to the $K=10$ cohort (\%)")

    observed_min = float(
        min(case_values.min(skipna=True), task_values.min(skipna=True))
    )
    ax.set_ylim(max(0.0, observed_min - 2.0), 101.0)
    ax.grid(True, linewidth=0.45, alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.4)
    save_single_plot(fig, out / "appendix_bandwidth_support_panel")

def pretty_task_name(task: object) -> str:
    text = str(task)
    replacements = {
        "Banking77Classification.v2": "Banking77",
        "Banking77Classification": "Banking77",
        "EmotionClassification.v2": "Emotion",
        "EmotionClassification": "Emotion",
        "HUMEEmotionClassification": "HUMEEmotion",
        "ArXivHierarchicalClusteringP2P": "ArXiv-P2P",
        "ArXivHierarchicalClusteringS2S": "ArXiv-S2S",
        "BiorxivClusteringP2P.v2": "Biorxiv",
        "BiorxivClusteringP2P": "Biorxiv",
        "STSBenchmark": "STS-B",
    }
    return replacements.get(text, text)



def plot_task_family_curves(task_curve: pd.DataFrame, fixed_k: int, out: Path) -> None:
    """
    Task-level curves without full-range uncertainty ribbons.

    Seed variance is retained in CSV/LaTeX tables.
    """
    for family in FAMILY_ORDER:
        part = task_curve.loc[task_curve["family"] == family]
        if part.empty:
            continue

        fig, ax = plt.subplots(figsize=(5.45, 4.15))
        for task, task_part in part.groupby("task", observed=True):
            task_part = task_part.sort_values("K")
            markevery = max(1, len(task_part) // 14)

            ax.plot(
                task_part["K"],
                task_part["mean_spearman"],
                marker="o",
                markersize=3.0,
                markevery=markevery,
                label=pretty_task_name(task),
            )

        ax.axvline(fixed_k, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Retained harmonic bandwidth $K$")
        ax.set_ylabel("Task-level Spearman correlation")
        ax.grid(True, linewidth=0.45, alpha=0.18)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout(pad=0.4)
        save_single_plot(
            fig,
            out / f"appendix_bandwidth_{FAMILY_FILE[family]}_panel",
        )


def plot_family_heatmaps(family_heat: pd.DataFrame, out: Path) -> None:
    """
    Create one compact 2x2 Appendix heatmap with one shared color scale.

    No subplot titles or captions are embedded. The panel order is:
    top-left Classification, top-right Clustering,
    bottom-left Pair classification, bottom-right STS.
    """
    max_mode = int(family_heat["K"].max())
    max_depth_bin = int(family_heat["depth_bin"].max())

    global_vmin = 0.0
    global_vmax = float(family_heat["normalized_mode_energy"].max())
    if not np.isfinite(global_vmax) or global_vmax <= 0:
        raise DataError("Invalid normalized energy range for heatmaps.")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 7.2),
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()
    image = None

    for ax, family in zip(axes_flat, FAMILY_ORDER):
        part = family_heat.loc[family_heat["family"] == family]
        matrix = np.full((max_depth_bin + 1, max_mode), np.nan)

        for row in part.itertuples():
            k = int(row.K)
            depth_bin = int(row.depth_bin)
            if 1 <= k <= max_mode:
                matrix[depth_bin, k - 1] = row.normalized_mode_energy

        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            extent=[1, max_mode, 0, 1],
            vmin=global_vmin,
            vmax=global_vmax,
        )
        ax.grid(False)

    for ax in axes[1, :]:
        ax.set_xlabel("Harmonic mode")
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative representation depth")

    if image is not None:
        fig.subplots_adjust(
            left=0.08,
            right=0.86,
            bottom=0.10,
            top=0.97,
            wspace=0.10,
            hspace=0.10,
        )

        cbar_ax = fig.add_axes([0.89, 0.18, 0.018, 0.64])

        colorbar = fig.colorbar(
            image,
            cax=cbar_ax,
        )

        colorbar.set_label(
            "Normalized mode energy",
            rotation=90,
            labelpad=10,
        )
        colorbar.set_label("Normalized mode energy")

    fig.subplots_adjust(
        left=0.08,
        right=0.86,
        bottom=0.10,
        top=0.98,
        wspace=0.10,
        hspace=0.10,
    )
    save_single_plot(fig, out / "appendix_energy_heatmaps")

def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text



def write_main_bandwidth_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Support-matched bandwidth sensitivity across the four task families. Values at the fixed setting report the mean and sample variance across downstream seeds. Family-specific best bandwidths are descriptive diagnostics only.}",
        r"\label{tab:bandwidth_sensitivity_main}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Tasks} & \textbf{Fixed $K$} & "
        r"\textbf{Alignment} & \textbf{Seed var.} & "
        r"\textbf{Best $K$} & \textbf{Best alignment} & \textbf{Gap} \\",
        r"\midrule",
    ]

    for row in df.itertuples():
        lines.append(
            f"{latex_escape(row.family)} & {int(row.tasks)} & {int(row.fixed_K)} & "
            f"{row.fixed_alignment:.3f} & {row.fixed_seed_variance:.3g} & "
            f"{int(row.best_displayed_K)} & "
            f"{row.best_displayed_alignment:.3f} & "
            f"{row.gap_to_best_displayed:.3f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

def write_task_bandwidth_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Task-level support-matched bandwidth results. Values at $K=10$ report the mean and sample variance across downstream seeds; the best displayed bandwidth is diagnostic only.}",
        r"\label{tab:bandwidth_task_summary}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccc@{}}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Task} & \textbf{$K=10$} & "
        r"\textbf{Seed var.} & \textbf{Best $K$} & "
        r"\textbf{Best alignment} & \textbf{Gap} \\",
        r"\midrule",
    ]

    for row in df.itertuples():
        lines.append(
            f"{latex_escape(row.family)} & "
            f"{latex_escape(pretty_task_name(row.task))} & "
            f"{row.fixed_alignment:.3f} & "
            f"{row.fixed_seed_variance:.3g} & "
            f"{int(row.best_displayed_K)} & "
            f"{row.best_displayed_alignment:.3f} & "
            f"{row.gap_to_best_displayed:.3f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

def format_support_value(quantity: str, value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if "retention" in quantity.lower():
            return f"{100.0 * float(value):.1f}\\%"
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.3f}"
    return latex_escape(value)


def write_support_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Support audit for the fixed-cohort bandwidth analysis.}",
        r"\label{tab:bandwidth_support_audit}",
        r"\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Quantity} & \textbf{Value} \\",
        r"\midrule",
    ]

    for row in df.itertuples():
        lines.append(
            f"{latex_escape(row.quantity)} & "
            f"{format_support_value(row.quantity, row.value)} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_spectral_summary(
    df: pd.DataFrame,
    mass_semantics: str,
    path: Path,
) -> None:
    threshold_available = mass_semantics != "unavailable"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Task-balanced harmonic-energy summaries across early, middle, and late relative depth. Mean mode gives the average energy location; mode spread is the energy-weighted standard deviation across harmonic modes.}",
        r"\label{tab:spectral_depth_summary}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.6pt}",
        r"\renewcommand{\arraystretch}{1.10}",
    ]

    if threshold_available:
        lines += [
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccccc@{}}",
            r"\toprule",
            r"\textbf{Family} & \textbf{Tasks} & \textbf{Layers} & "
            r"\multicolumn{3}{c}{\textbf{Mean harmonic mode}} & "
            r"\multicolumn{3}{c}{\textbf{Mode spread}} & "
            r"\textbf{$K_{90}$ avail.} & \textbf{Med. $K_{90}$} & "
            r"\textbf{$K_{95}$ avail.} & \textbf{Med. $K_{95}$} \\",
            r"& & & \textbf{Early} & \textbf{Middle} & \textbf{Late} & "
            r"\textbf{Early} & \textbf{Middle} & \textbf{Late} & & & & \\",
            r"\midrule",
        ]
    else:
        lines += [
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}",
            r"\toprule",
            r"\textbf{Family} & \textbf{Tasks} & \textbf{Layers} & "
            r"\multicolumn{3}{c}{\textbf{Mean harmonic mode}} & "
            r"\multicolumn{3}{c}{\textbf{Mode spread}} \\",
            r"& & & \textbf{Early} & \textbf{Middle} & \textbf{Late} & "
            r"\textbf{Early} & \textbf{Middle} & \textbf{Late} \\",
            r"\midrule",
        ]

    for row in df.itertuples():
        values = [
            latex_escape(row.family),
            str(int(row.tasks)),
            str(int(row.layer_candidates)),
            f"{row.early_mean_mode:.2f}",
            f"{row.middle_mean_mode:.2f}",
            f"{row.late_mean_mode:.2f}",
            f"{row.early_mode_spread:.2f}",
            f"{row.middle_mode_spread:.2f}",
            f"{row.late_mode_spread:.2f}",
        ]

        if threshold_available:
            values += [
                f"{100.0 * row.K90_availability:.1f}\\%",
                f"{row.median_K90:.1f}" if pd.notna(row.median_K90) else "--",
                f"{100.0 * row.K95_availability:.1f}\\%",
                f"{row.median_K95:.1f}" if pd.notna(row.median_K95) else "--",
            ]

        lines.append(" & ".join(values) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--curves-csv",
        type=Path,
        default=None,
        help="Override harmora_k_curves_long.csv.",
    )
    parser.add_argument(
        "--downstream-csv",
        type=Path,
        default=None,
        help="Override downstream_profiles_long.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/generated/spectral"),
    )
    parser.add_argument("--fixed-k", type=int, default=10)
    parser.add_argument(
        "--bandwidth-k-min",
        type=int,
        default=None,
        help="Minimum K used for the support-matched bandwidth analysis.",
    )
    parser.add_argument(
        "--bandwidth-k-max",
        type=int,
        default=47,
        help=(
            "Maximum K used for the primary bandwidth analysis. The current "
            "dataset retains all 11 tasks through K=47."
        ),
    )
    parser.add_argument(
        "--spectral-k-min",
        type=int,
        default=None,
        help="Minimum mode used for the spectral-depth analysis.",
    )
    parser.add_argument(
        "--spectral-k-max",
        type=int,
        default=80,
        help="Maximum mode used for the spectral-depth analysis.",
    )
    parser.add_argument(
        "--support-k-max",
        type=int,
        default=80,
        help=(
            "Maximum K shown in the support audit. This may exceed the "
            "primary bandwidth-analysis range."
        ),
    )
    parser.add_argument("--min-layers", type=int, default=3)
    parser.add_argument("--depth-bins", type=int, default=11)
    parser.add_argument(
        "--require-all-tasks",
        action="store_true",
        help=(
            "Fail instead of continuing when common support over the displayed "
            "K range removes one or more tasks."
        ),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate canonical inputs and print their basic contents.",
    )
    return parser.parse_args()


def canonical_paths(root: Path) -> tuple[Path, Path]:
    base = root / "outputs" / "full_experiment"
    curves = base / "metric_csv" / "harmora_k_curves_long.csv"
    downstream = base / "csv" / "downstream_profiles_long.csv"
    return curves, downstream


def inspect_inputs(curves_path: Path, downstream_path: Path) -> None:
    print(f"Harmora spectral generator {VERSION}")
    print("\n[CURVES]")
    print(curves_path)
    if not curves_path.exists():
        print("MISSING")
    else:
        curves = pd.read_csv(curves_path, nrows=5)
        print(f"columns={list(curves.columns)}")
        print(curves.to_string(index=False))

    print("\n[DOWNSTREAM]")
    print(downstream_path)
    if not downstream_path.exists():
        print("MISSING")
    else:
        downstream = pd.read_csv(downstream_path, nrows=5)
        print(f"columns={list(downstream.columns)}")
        print(downstream.to_string(index=False))


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()

    canonical_curves, canonical_downstream = canonical_paths(root)
    curves_path = (
        args.curves_csv.resolve()
        if args.curves_csv is not None
        else canonical_curves
    )
    downstream_path = (
        args.downstream_csv.resolve()
        if args.downstream_csv is not None
        else canonical_downstream
    )

    if args.inspect_only:
        inspect_inputs(curves_path, downstream_path)
        return 0

    if not curves_path.exists():
        raise DataError(f"Curves file not found:\n{curves_path}")
    if not downstream_path.exists():
        raise DataError(f"Downstream file not found:\n{downstream_path}")

    out = args.output_dir.resolve()
    figures_dir = out / "figures"
    tables_dir = out / "tables"
    data_dir = out / "data"
    metadata_dir = out / "metadata"
    for directory in (figures_dir, tables_dir, data_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Harmora spectral generator {VERSION}")
    print(f"Curves:     {curves_path}")
    print(f"Downstream: {downstream_path}")
    print(f"Output:     {out}")

    curves = load_curves(curves_path)
    downstream = load_downstream(downstream_path)

    bandwidth_k_values = choose_k_values(
        curves,
        args.bandwidth_k_min,
        args.bandwidth_k_max,
    )
    spectral_k_values = choose_k_values(
        curves,
        args.spectral_k_min,
        args.spectral_k_max,
    )
    support_k_values = choose_k_values(
        curves,
        args.bandwidth_k_min,
        args.support_k_max,
    )

    if args.fixed_k not in bandwidth_k_values:
        raise DataError(
            f"Fixed K={args.fixed_k} is absent from the bandwidth range. "
            f"Available values:\n{bandwidth_k_values}"
        )

    retained_cases, common_layers = build_common_support(
        curves,
        bandwidth_k_values,
        args.min_layers,
    )

    fixed_k_tasks = set(
        curves.loc[curves["K"] == args.fixed_k, "task"].astype(str).unique()
    )
    retained_tasks = set(retained_cases["task"].astype(str).unique())
    excluded_tasks = sorted(fixed_k_tasks - retained_tasks)

    if excluded_tasks:
        message = (
            f"Bandwidth K range {min(bandwidth_k_values)}--"
            f"{max(bandwidth_k_values)} removes "
            f"{len(excluded_tasks)} of {len(fixed_k_tasks)} tasks from the "
            "common-support bandwidth analysis:\n  "
            + "\n  ".join(excluded_tasks)
        )
        if args.require_all_tasks:
            raise DataError(message)
        print(f"\nWARNING: {message}")

    excluded_df = pd.DataFrame(
        {"excluded_task": excluded_tasks}
    )

    merged = merge_curves_and_downstream(
        curves,
        downstream,
        common_layers,
        bandwidth_k_values,
    )
    case_results = bandwidth_correlations(merged)
    bandwidth = aggregate_bandwidth(case_results)
    main_summary = bandwidth_summary(
        bandwidth["family_curve"],
        args.fixed_k,
    )
    task_summary = task_bandwidth_summary(
        bandwidth["task_curve"],
        args.fixed_k,
    )
    support_curve, support_summary = support_audit(
        curves,
        support_k_values,
        args.fixed_k,
        args.min_layers,
    )

    candidate_stats, mode_rows, mass_semantics, mass_report = (
        spectral_candidate_statistics(
            curves.loc[curves["K"].isin(spectral_k_values)].copy(),
            args.depth_bins,
        )
    )
    depth = aggregate_depth(
        candidate_stats,
        mode_rows,
        mass_semantics,
    )

    # Figures: each panel is a separate PDF/PNG; LaTeX combines them.
    # plot_main_bandwidth(
    #     bandwidth["family_curve"],
    #     args.fixed_k,
    #     out,
    # )
    # plot_main_depth(
    #     depth["family_depth"],
    #     "mean_harmonic_mode",
    #     "Mean harmonic mode",
    #     "main_mean_harmonic_mode_panel",
    #     out,
    # )
    # plot_main_depth(
    #     depth["family_depth"],
    #     "harmonic_mode_spread",
    #     "Harmonic-mode spread",
    #     "main_harmonic_mode_spread_panel",
    #     out,
    # )
    
    task_colors = build_task_colors(bandwidth["task_curve"])

    # Main paper figure and appendix diagnostics.
    plot_main_task_diagnostics(
        task_curve=bandwidth["task_curve"],
        task_depth=depth["task_depth"],
        fixed_k=args.fixed_k,
        out=figures_dir,
    )
    plot_support_audit(
        support_curve,
        args.fixed_k,
        figures_dir,
    )
    plot_task_family_curves(
        bandwidth["task_curve"],
        args.fixed_k,
        figures_dir,
    )
    plot_family_heatmaps(
        depth["family_heat"],
        figures_dir,
    )

    # CSVs.
    case_results.to_csv(
        data_dir / "bandwidth_case_level_correlations.csv",
        index=False,
    )
    bandwidth["task_seed"].to_csv(
        data_dir / "bandwidth_task_seed_results.csv",
        index=False,
    )
    bandwidth["family_curve"].to_csv(
        data_dir / "bandwidth_family_curves.csv",
        index=False,
    )
    bandwidth["task_curve"].to_csv(
        data_dir / "bandwidth_task_curves.csv",
        index=False,
    )
    main_summary.to_csv(
        data_dir / "main_bandwidth_summary.csv",
        index=False,
    )
    task_summary.to_csv(
        data_dir / "appendix_bandwidth_task_summary.csv",
        index=False,
    )
    support_curve.to_csv(
        data_dir / "appendix_bandwidth_support_curve.csv",
        index=False,
    )
    support_summary.to_csv(
        data_dir / "appendix_bandwidth_support.csv",
        index=False,
    )
    retained_cases.to_csv(
        data_dir / "bandwidth_retained_cases.csv",
        index=False,
    )
    excluded_df.to_csv(
        data_dir / "bandwidth_excluded_tasks.csv",
        index=False,
    )
    common_layers.to_csv(
        data_dir / "bandwidth_common_layers.csv",
        index=False,
    )
    depth["candidate_stats"].to_csv(
        data_dir / "spectral_candidate_statistics.csv",
        index=False,
    )
    depth["family_depth"].to_csv(
        data_dir / "spectral_family_depth_curves.csv",
        index=False,
    )
    depth["family_heat"].to_csv(
        data_dir / "spectral_family_heatmap_values.csv",
        index=False,
    )
    depth["summary"].to_csv(
        data_dir / "appendix_spectral_summary.csv",
        index=False,
    )

    # LaTeX.
    write_main_bandwidth_table(
        main_summary,
        tables_dir / "table_2_bandwidth_sensitivity.tex",
    )
    write_support_table(
        support_summary,
        tables_dir / "appendix_bandwidth_support.tex",
    )
    write_task_bandwidth_table(
        task_summary,
        tables_dir / "appendix_bandwidth_task_summary.tex",
    )
    write_spectral_summary(
        depth["summary"],
        mass_semantics,
        tables_dir / "appendix_spectral_summary.tex",
    )

    manifest = {
        "version": VERSION,
        "curves_input": str(curves_path),
        "downstream_input": str(downstream_path),
        "fixed_K": args.fixed_k,
        "bandwidth_K_values": bandwidth_k_values,
        "spectral_K_values": spectral_k_values,
        "support_audit_K_values": support_k_values,
        "minimum_layers_per_correlation": args.min_layers,
        "depth_bins": args.depth_bins,
        "tasks": sorted(curves["task"].unique().tolist()),
        "families": sorted(curves["family"].unique().tolist()),
        "models": sorted(curves["model_alias"].unique().tolist()),
        "downstream_seeds": sorted(downstream["seed"].unique().tolist()),
        "retained_support_matched_model_task_cases": int(len(retained_cases)),
        "retained_support_matched_tasks": sorted(retained_tasks),
        "excluded_tasks_from_bandwidth_analysis": excluded_tasks,
        "heatmap_color_scale": {
            "shared_across_families": True,
            "vmin": 0.0,
            "vmax": float(depth["family_heat"]["normalized_mode_energy"].max()),
        },
        "uncertainty_visualization": {
            "main_bandwidth": (
                "No shaded ribbon; sample variance across downstream seeds "
                "is reported in the Main table."
            ),
            "appendix_task_bandwidth": (
                "No shaded ribbon; sample variance across downstream seeds "
                "is reported in the task-level Appendix table."
            ),
            "depth_curves": (
                "None; these are deterministic unlabeled spectral summaries, "
                "not downstream-seed estimates."
            ),
        },
        "mass_semantics": mass_semantics,
        "mass_validation": mass_report,
        "definitions": {
            "bandwidth_case": (
                "Spearman correlation across the exact common layer set for "
                "one model-task-seed case at one K."
            ),
            "bandwidth_aggregation": (
                "Models are averaged within task and seed; tasks are equally "
                "weighted within family and seed; mean and sample variance "
                "are computed across downstream seeds."
            ),
            "mean_harmonic_mode": "sum_k k p_k",
            "harmonic_mode_spread": (
                "sqrt(sum_k p_k (k - mean_harmonic_mode)^2)"
            ),
            "depth_aggregation": (
                "Candidates are averaged within task and depth bin, then "
                "tasks are equally weighted within family."
            ),
        },
    }
    (metadata_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nCompleted successfully.")
    print(f"Mass semantics: {mass_semantics}")
    print("\nCreated files:")
    for path in sorted(item for item in out.rglob("*") if item.is_file()):
        print(f"  {path.relative_to(out)}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DataError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"\nERROR:\n{exc}", file=sys.stderr)
        raise SystemExit(2)
