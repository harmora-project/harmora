from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .io_utils import load_json, safe_name


def collect_seed_results(output_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = Path(output_dir) / "seed_results"
    for path in sorted(root.glob("*/*/seed_*.json")):
        payload = load_json(path)
        scores = payload.get("primary_scores", [])
        layers = payload.get("layer_indices", list(range(len(scores))))
        if len(scores) != len(layers):
            raise RuntimeError(f"Layer/score length mismatch in {path}")
        for layer, score in zip(layers, scores):
            rows.append({
                "model_alias": payload["model_alias"],
                "hf_name": payload.get("hf_name"),
                "task": payload["task"],
                "task_type": payload["task_type"],
                "probe_family": payload["probe_family"],
                "primary_metric": payload["primary_metric"],
                "primary_evaluator": payload["primary_evaluator"],
                "seed": int(payload["seed"]),
                "layer": int(layer),
                "primary_score": float(score),
                "sampling_fingerprint": payload.get("sampling_fingerprint"),
                "sample_hash": payload["sample_hash"],
                "embedding_hash": payload["embedding_hash"],
                "encoder_fingerprint": payload.get("encoder_fingerprint"),
                "evaluation_fingerprint": payload.get("evaluation_fingerprint"),
                "full_config_fingerprint": payload.get("full_config_fingerprint"),
                "split_hash": payload.get("split_hash"),
                "seed_affects_evaluation": bool(payload.get("seed_affects_evaluation", True)),
                "num_items": int(payload.get("num_items", 0)),
                "n_train": payload.get("n_train"),
                "n_test": payload.get("n_test"),
                "source_subset": payload.get("source_subset"),
                "source_split": payload.get("source_split"),
                "result_file": str(path),
            })
    if not rows:
        raise RuntimeError(f"No downstream seed results found under {root}")
    df = pd.DataFrame(rows)
    if not np.isfinite(df["primary_score"].to_numpy(dtype=float)).all():
        bad = df[~np.isfinite(df["primary_score"].to_numpy(dtype=float))]
        raise RuntimeError(f"Non-finite downstream scores found:\n{bad.head(20)}")
    return df.sort_values(["model_alias", "task", "seed", "layer"]).reset_index(drop=True)


def summarize_profiles(long_df: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:
    group_cols = [
        "model_alias", "hf_name", "task", "task_type", "probe_family",
        "primary_metric", "primary_evaluator", "layer",
    ]
    records = []
    for keys, group in long_df.groupby(group_cols, dropna=False):
        values = group["primary_score"].to_numpy(dtype=float)
        n = int(len(values))
        mean = float(np.mean(values))
        variance = float(np.var(values, ddof=1)) if n > 1 else 0.0
        std = float(np.sqrt(variance))
        sem = std / np.sqrt(n) if n > 0 else np.nan
        if n > 1:
            critical = float(student_t.ppf(0.5 + confidence_level / 2.0, df=n - 1))
            half_width = critical * sem
        else:
            half_width = 0.0
        record = dict(zip(group_cols, keys))
        record.update({
            "n_seeds": n,
            "score_mean": mean,
            "score_variance": variance,
            "score_std": std,
            "score_sem": float(sem),
            "ci_low": mean - half_width,
            "ci_high": mean + half_width,
            "score_min": float(np.min(values)),
            "score_max": float(np.max(values)),
            "seeds": ",".join(str(x) for x in sorted(group["seed"].unique())),
            "sampling_fingerprint": group["sampling_fingerprint"].iloc[0],
            "sample_hash": group["sample_hash"].iloc[0],
            "embedding_hash": group["embedding_hash"].iloc[0],
            "encoder_fingerprint": group["encoder_fingerprint"].iloc[0],
            "evaluation_fingerprint": group["evaluation_fingerprint"].iloc[0],
            "seed_affects_evaluation": bool(group["seed_affects_evaluation"].iloc[0]),
        })
        records.append(record)
    return pd.DataFrame(records).sort_values(["model_alias", "task", "layer"]).reset_index(drop=True)


def best_layers_by_seed(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "model_alias", "hf_name", "task", "task_type", "probe_family",
        "primary_metric", "primary_evaluator", "seed",
    ]
    for keys, group in long_df.groupby(group_cols, dropna=False):
        group = group.sort_values("layer")
        best_score = group["primary_score"].max()
        # Stable tie-break: earliest layer.
        best = group[np.isclose(group["primary_score"], best_score)].sort_values("layer").iloc[0]
        final = group.iloc[-1]
        record = dict(zip(group_cols, keys))
        record.update({
            "best_layer": int(best["layer"]),
            "best_score": float(best["primary_score"]),
            "final_layer": int(final["layer"]),
            "final_score": float(final["primary_score"]),
            "best_minus_final": float(best["primary_score"] - final["primary_score"]),
            "sample_hash": best["sample_hash"],
            "embedding_hash": best["embedding_hash"],
            "split_hash": best["split_hash"],
        })
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["model_alias", "task", "seed"]).reset_index(drop=True)


def best_layer_summary(summary_df: pd.DataFrame, by_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task), group in summary_df.groupby(["model_alias", "task"]):
        ordered = group.sort_values(["score_mean", "layer"], ascending=[False, True])
        best = ordered.iloc[0]
        final = group.sort_values("layer").iloc[-1]
        seed_group = by_seed_df[(by_seed_df["model_alias"] == model) & (by_seed_df["task"] == task)]
        mode_values = seed_group["best_layer"].mode()
        gain_values = seed_group["best_minus_final"].to_numpy(dtype=float)
        modal_layer = int(mode_values.min()) if len(mode_values) else int(best["layer"])
        modal_frequency = int((seed_group["best_layer"] == modal_layer).sum())
        rows.append({
            "model_alias": model,
            "hf_name": best["hf_name"],
            "task": task,
            "task_type": best["task_type"],
            "probe_family": best["probe_family"],
            "primary_metric": best["primary_metric"],
            "primary_evaluator": best["primary_evaluator"],
            "best_mean_layer": int(best["layer"]),
            "best_mean_score": float(best["score_mean"]),
            "best_mean_score_variance": float(best["score_variance"]),
            "best_mean_score_std": float(best["score_std"]),
            "best_mean_ci_low": float(best["ci_low"]),
            "best_mean_ci_high": float(best["ci_high"]),
            "final_layer": int(final["layer"]),
            "final_score_mean": float(final["score_mean"]),
            "final_score_variance": float(final["score_variance"]),
            "final_score_std": float(final["score_std"]),
            "best_mean_minus_final": float(best["score_mean"] - final["score_mean"]),
            "best_minus_final_seed_mean": float(np.mean(gain_values)),
            "best_minus_final_seed_variance": float(np.var(gain_values, ddof=1)) if len(gain_values) > 1 else 0.0,
            "best_minus_final_seed_std": float(np.std(gain_values, ddof=1)) if len(gain_values) > 1 else 0.0,
            "mean_seed_best_layer": float(seed_group["best_layer"].mean()),
            "variance_seed_best_layer": float(seed_group["best_layer"].var(ddof=1)) if len(seed_group) > 1 else 0.0,
            "std_seed_best_layer": float(seed_group["best_layer"].std(ddof=1)) if len(seed_group) > 1 else 0.0,
            "modal_seed_best_layer": modal_layer,
            "modal_layer_frequency": modal_frequency,
            "modal_layer_fraction": float(modal_frequency / max(len(seed_group), 1)),
            "n_unique_best_layers": int(seed_group["best_layer"].nunique()),
            "n_seeds": int(best["n_seeds"]),
            "sample_hash": best["sample_hash"],
            "embedding_hash": best["embedding_hash"],
        })
    return pd.DataFrame(rows).sort_values(["model_alias", "task"]).reset_index(drop=True)


def final_vs_best_summary(by_seed_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model_alias", "hf_name", "task", "task_type", "probe_family",
        "primary_metric", "primary_evaluator",
    ]
    rows = []
    for keys, group in by_seed_df.groupby(group_cols, dropna=False):
        values = group["best_minus_final"].to_numpy(dtype=float)
        rows.append({
            **dict(zip(group_cols, keys)),
            "n_seeds": int(len(values)),
            "gain_mean": float(np.mean(values)),
            "gain_variance": float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
            "gain_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "gain_min": float(np.min(values)),
            "gain_max": float(np.max(values)),
            "fraction_positive_gain": float(np.mean(values > 1e-12)),
        })
    return pd.DataFrame(rows).sort_values(["model_alias", "task"]).reset_index(drop=True)


def layer_selection_stability(by_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task), group in by_seed_df.groupby(["model_alias", "task"]):
        counts = group["best_layer"].value_counts().sort_index()
        mode_count = int(counts.max())
        modal_layers = sorted(int(x) for x in counts[counts == mode_count].index)
        rows.append({
            "model_alias": model,
            "task": task,
            "task_type": group["task_type"].iloc[0],
            "primary_metric": group["primary_metric"].iloc[0],
            "n_seeds": int(len(group)),
            "unique_selected_layers": int(counts.size),
            "modal_layer": int(modal_layers[0]),
            "modal_count": mode_count,
            "modal_fraction": float(mode_count / len(group)),
            "selected_layer_mean": float(group["best_layer"].mean()),
            "selected_layer_variance": float(group["best_layer"].var(ddof=1)) if len(group) > 1 else 0.0,
            "selected_layer_std": float(group["best_layer"].std(ddof=1)) if len(group) > 1 else 0.0,
            "selected_layers_by_seed": ";".join(
                f"{int(row.seed)}:{int(row.best_layer)}" for row in group.sort_values("seed").itertuples()
            ),
        })
    return pd.DataFrame(rows).sort_values(["model_alias", "task"]).reset_index(drop=True)


def task_oracle_by_seed(by_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, seed), group in by_seed_df.groupby(["task", "seed"]):
        best_score = group["best_score"].max()
        winner = group[np.isclose(group["best_score"], best_score)].sort_values(
            ["model_alias", "best_layer"]
        ).iloc[0]
        rows.append({
            "task": task,
            "task_type": winner["task_type"],
            "primary_metric": winner["primary_metric"],
            "seed": int(seed),
            "oracle_model": winner["model_alias"],
            "oracle_layer": int(winner["best_layer"]),
            "oracle_score": float(winner["best_score"]),
        })
    return pd.DataFrame(rows).sort_values(["task", "seed"]).reset_index(drop=True)


def task_oracle_summary(best_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, group in best_df.groupby("task"):
        best_score = group["best_mean_score"].max()
        winner = group[np.isclose(group["best_mean_score"], best_score)].sort_values(
            ["model_alias", "best_mean_layer"]
        ).iloc[0]
        rows.append({
            "task": task,
            "task_type": winner["task_type"],
            "primary_metric": winner["primary_metric"],
            "oracle_model": winner["model_alias"],
            "oracle_layer": int(winner["best_mean_layer"]),
            "oracle_score_mean": float(winner["best_mean_score"]),
            "oracle_score_variance": float(winner["best_mean_score_variance"]),
            "oracle_score_std": float(winner["best_mean_score_std"]),
            "oracle_ci_low": float(winner["best_mean_ci_low"]),
            "oracle_ci_high": float(winner["best_mean_ci_high"]),
        })
    return pd.DataFrame(rows).sort_values(["task_type", "task"]).reset_index(drop=True)


def model_task_ranks(best_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    for task, group in best_df.groupby("task"):
        group = group.copy()
        # Higher is better for all selected primary downstream metrics.
        group["task_rank"] = group["best_mean_score"].rank(method="average", ascending=False)
        lo = float(group["best_mean_score"].min())
        hi = float(group["best_mean_score"].max())
        if np.isclose(lo, hi):
            group["task_minmax_score"] = 1.0
        else:
            group["task_minmax_score"] = (group["best_mean_score"] - lo) / (hi - lo)
        for row in group.itertuples():
            detail_rows.append({
                "model_alias": row.model_alias,
                "task": row.task,
                "task_type": row.task_type,
                "primary_metric": row.primary_metric,
                "best_mean_score": float(row.best_mean_score),
                "task_rank": float(row.task_rank),
                "task_minmax_score": float(row.task_minmax_score),
                "best_mean_layer": int(row.best_mean_layer),
            })
    detail = pd.DataFrame(detail_rows).sort_values(["task", "task_rank", "model_alias"]).reset_index(drop=True)
    summary = (
        detail.groupby("model_alias")
        .agg(
            n_tasks=("task", "nunique"),
            average_task_rank=("task_rank", "mean"),
            median_task_rank=("task_rank", "median"),
            task_rank_variance=("task_rank", "var"),
            task_rank_std=("task_rank", "std"),
            mean_task_minmax_score=("task_minmax_score", "mean"),
            task_wins=("task_rank", lambda x: int(np.sum(np.isclose(x, 1.0)))),
        )
        .reset_index()
        .sort_values(["average_task_rank", "mean_task_minmax_score"], ascending=[True, False])
    )
    return detail, summary


def family_summary(best_df: pd.DataFrame) -> pd.DataFrame:
    # Metrics are never mixed across task families without an explicit rank normalization.
    return (
        best_df.groupby(["model_alias", "task_type", "primary_metric"])
        .agg(
            n_tasks=("task", "nunique"),
            mean_best_layer_score=("best_mean_score", "mean"),
            variance_across_tasks_best_score=("best_mean_score", "var"),
            std_across_tasks_best_score=("best_mean_score", "std"),
            mean_final_layer_score=("final_score_mean", "mean"),
            mean_best_minus_final=("best_mean_minus_final", "mean"),
            mean_modal_layer_fraction=("modal_layer_fraction", "mean"),
        )
        .reset_index()
        .sort_values(["task_type", "model_alias"])
    )


def plot_profiles(long_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str | Path, save_pdf: bool = True) -> None:
    figure_root = Path(output_dir) / "figures" / "profiles"
    for (model, task), summary in summary_df.groupby(["model_alias", "task"]):
        seeds = long_df[(long_df["model_alias"] == model) & (long_df["task"] == task)]
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for seed, seed_group in seeds.groupby("seed"):
            seed_group = seed_group.sort_values("layer")
            ax.plot(
                seed_group["layer"],
                seed_group["primary_score"],
                linewidth=1.0,
                alpha=0.45,
                label=f"seed {seed}",
            )
        summary = summary.sort_values("layer")
        x = summary["layer"].to_numpy(dtype=float)
        mean = summary["score_mean"].to_numpy(dtype=float)
        std = summary["score_std"].to_numpy(dtype=float)
        ax.plot(x, mean, linewidth=2.5, marker="o", label="mean")
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, label="±1 std")
        best = summary.sort_values(["score_mean", "layer"], ascending=[False, True]).iloc[0]
        ax.axvline(int(best["layer"]), linestyle="--", linewidth=1.0)
        ax.annotate(
            f"best mean layer={int(best['layer'])}\n"
            f"{best['score_mean']:.4f} ± {best['score_std']:.4f}",
            xy=(int(best["layer"]), float(best["score_mean"])),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
        ax.set_xlabel("Layer index")
        ax.set_ylabel(str(best["primary_metric"]))
        ax.set_title(f"{model} — {task}")
        ax.grid(True, linewidth=0.4)
        ax.legend(frameon=False, ncol=3)
        path_dir = figure_root / safe_name(model)
        path_dir.mkdir(parents=True, exist_ok=True)
        stem = safe_name(task) + "_profile"
        fig.savefig(path_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        if save_pdf:
            fig.savefig(path_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_summary_figures(
    rank_summary: pd.DataFrame,
    gain_summary: pd.DataFrame,
    stability_df: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    output_dir: str | Path,
    save_pdf: bool = True,
) -> None:
    root = Path(output_dir) / "figures" / "summary"
    root.mkdir(parents=True, exist_ok=True)

    # Average task rank; unlike raw scores, ranks are comparable across task families.
    d = rank_summary.sort_values("average_task_rank", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(d["model_alias"], d["average_task_rank"])
    ax.invert_yaxis()
    ax.set_xlabel("Average rank across 12 tasks (lower is better)")
    ax.set_title("Downstream model ranking from best layer mean scores")
    ax.grid(True, axis="x", linewidth=0.4)
    fig.savefig(root / "model_average_task_rank.png", dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(root / "model_average_task_rank.pdf", bbox_inches="tight")
    plt.close(fig)

    # Best-layer improvement over final layer by task family.
    fam = (
        gain_summary.groupby("task_type")
        .agg(gain_mean=("gain_mean", "mean"), gain_std=("gain_mean", "std"), n=("task", "count"))
        .reset_index()
        .sort_values("gain_mean", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(fam["task_type"], fam["gain_mean"], yerr=fam["gain_std"].fillna(0.0), capsize=4)
    ax.set_ylabel("Best-layer minus final-layer downstream score")
    ax.set_title("Layer-selection benefit by downstream task family")
    ax.grid(True, axis="y", linewidth=0.4)
    fig.savefig(root / "best_vs_final_gain_by_family.png", dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(root / "best_vs_final_gain_by_family.pdf", bbox_inches="tight")
    plt.close(fig)

    # Layer-selection stability across seeds.
    st = stability_df.sort_values("modal_fraction", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(st["modal_fraction"], bins=np.linspace(0, 1, 6), edgecolor="black")
    ax.set_xlabel("Fraction of seeds selecting the modal layer")
    ax.set_ylabel("Model–task cases")
    ax.set_title("Best-layer stability across evaluation seeds")
    ax.grid(True, axis="y", linewidth=0.4)
    fig.savefig(root / "best_layer_seed_stability.png", dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(root / "best_layer_seed_stability.pdf", bbox_inches="tight")
    plt.close(fig)

    # Oracle model counts across tasks.
    counts = oracle_summary["oracle_model"].value_counts().reindex(sorted(oracle_summary["oracle_model"].unique()))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(counts.index, counts.values)
    ax.set_xlabel("Number of tasks won by model–layer oracle")
    ax.set_title("Taskwise downstream oracle model counts")
    ax.grid(True, axis="x", linewidth=0.4)
    fig.savefig(root / "taskwise_oracle_model_counts.png", dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(root / "taskwise_oracle_model_counts.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_all(
    output_dir: str | Path,
    confidence_level: float = 0.95,
    save_pdf: bool = True,
    save_summary_figures: bool = True,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    long_df = collect_seed_results(output_dir)
    summary_df = summarize_profiles(long_df, confidence_level=confidence_level)
    by_seed_df = best_layers_by_seed(long_df)
    best_df = best_layer_summary(summary_df, by_seed_df)
    gain_df = final_vs_best_summary(by_seed_df)
    stability_df = layer_selection_stability(by_seed_df)
    oracle_seed_df = task_oracle_by_seed(by_seed_df)
    oracle_df = task_oracle_summary(best_df)
    rank_detail_df, rank_summary_df = model_task_ranks(best_df)
    family_df = family_summary(best_df)

    paths = {
        "long": csv_dir / "downstream_profiles_long.csv",
        "summary": csv_dir / "downstream_profiles_summary.csv",
        "best_by_seed": csv_dir / "best_layers_by_seed.csv",
        "best_summary": csv_dir / "best_layers_summary.csv",
        "final_vs_best_by_seed": csv_dir / "final_vs_best_by_seed.csv",
        "final_vs_best_summary": csv_dir / "final_vs_best_summary.csv",
        "layer_stability": csv_dir / "layer_selection_stability.csv",
        "task_oracle_by_seed": csv_dir / "task_oracle_by_seed.csv",
        "task_oracle_summary": csv_dir / "task_oracle_summary.csv",
        "model_task_ranks": csv_dir / "model_task_ranks.csv",
        "model_rank_summary": csv_dir / "model_rank_summary.csv",
        "family_summary": csv_dir / "task_family_downstream_summary.csv",
    }
    long_df.to_csv(paths["long"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    by_seed_df.to_csv(paths["best_by_seed"], index=False)
    best_df.to_csv(paths["best_summary"], index=False)
    by_seed_df.to_csv(paths["final_vs_best_by_seed"], index=False)
    gain_df.to_csv(paths["final_vs_best_summary"], index=False)
    stability_df.to_csv(paths["layer_stability"], index=False)
    oracle_seed_df.to_csv(paths["task_oracle_by_seed"], index=False)
    oracle_df.to_csv(paths["task_oracle_summary"], index=False)
    rank_detail_df.to_csv(paths["model_task_ranks"], index=False)
    rank_summary_df.to_csv(paths["model_rank_summary"], index=False)
    family_df.to_csv(paths["family_summary"], index=False)

    plot_profiles(long_df, summary_df, output_dir, save_pdf=save_pdf)
    if save_summary_figures:
        plot_summary_figures(
            rank_summary_df,
            gain_df,
            stability_df,
            oracle_df,
            output_dir,
            save_pdf=save_pdf,
        )
    return paths
