#!/usr/bin/env python
"""
Framework-aligned post-embedding runtime and scalability benchmark for Harmora.

This benchmark uses the same task-matched primary evaluators as the validated
Harmora downstream pipeline:

- Classification: train-only standardized logistic regression, accuracy.
- Clustering: MiniBatchKMeans, V-measure.
- Pair classification: cosine similarity, average precision.
- STS: cosine similarity, Spearman correlation.

It never performs transformer inference. It consumes the existing task caches,
embedding caches, split manifests, and downstream profile inventory.

Main outputs
------------
1. Exact real-cache post-embedding benchmark over all 106 model-layer
   candidates for each of the 11 tasks.
2. Hybrid shortlisting cost: Harmora over all candidates, followed by the
   task-matched evaluator on the Harmora top-s shortlist.
3. Controlled sample-size scaling.
4. Controlled output-cardinality scaling with one fixed representation matrix.
5. Partial-versus-full eigensolver validation on real cached candidates.
6. CSV, JSON, PDF/PNG, and LaTeX tables suitable for the paper appendix.

Version: 2026-07-26-framework-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Restrict numerical libraries before importing NumPy/SciPy/sklearn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, v_measure_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

VERSION = "2026-07-26-framework-v1"


class BenchmarkError(RuntimeError):
    pass


@dataclass
class HarmoraRun:
    score: float
    graph_seconds: float
    eig_seconds: float
    projection_seconds: float
    total_seconds: float
    n_samples: int
    dimension: int
    k: int
    matvec_count: int
    ncv: int
    boundary_gap: float
    eig_status: str


class CountedSymmetricOperator(LinearOperator):
    """Dense symmetric matrix exposed as a counted LinearOperator."""

    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=np.float64, order="C")
        self.matvec_count = 0
        super().__init__(dtype=self.matrix.dtype, shape=self.matrix.shape)

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        self.matvec_count += 1
        return self.matrix @ vector

    def _matmat(self, matrix: np.ndarray) -> np.ndarray:
        self.matvec_count += int(matrix.shape[1])
        return self.matrix @ matrix


def percentile_interval(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
    )


def safe_name(value: str) -> str:
    text = str(value).strip()
    for character in '<>:"/\\|?*':
        text = text.replace(character, "_")
    return text


def scalar_from_npz(archive: Any, key: str, default: str = "") -> str:
    if key not in archive.files:
        return default
    value = np.asarray(archive[key])
    return str(value.item() if value.shape == () else value.reshape(-1)[0])


def center_and_build_laplacian(
    representation: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct the exact dense Gaussian normalized Laplacian."""
    z = np.asarray(representation, dtype=np.float64)
    if z.ndim != 2:
        raise BenchmarkError(f"Expected a 2-D representation matrix, found {z.shape}.")
    if not np.isfinite(z).all():
        raise BenchmarkError("Representation contains non-finite values.")

    z_centered = z - z.mean(axis=0, keepdims=True)
    squared_condensed = pdist(z_centered, metric="sqeuclidean")
    positive = squared_condensed[squared_condensed > 0]
    tau2 = float(np.median(positive)) if len(positive) else 1.0
    tau2 = max(tau2, eps)

    squared = squareform(squared_condensed)
    affinity = np.exp(-squared / (2.0 * tau2 + eps))
    np.fill_diagonal(affinity, 0.0)
    affinity = 0.5 * (affinity + affinity.T)

    degree = np.clip(affinity.sum(axis=1), eps, None)
    inv_sqrt = degree ** -0.5
    normalized_affinity = (inv_sqrt[:, None] * affinity) * inv_sqrt[None, :]
    laplacian = np.eye(z.shape[0], dtype=np.float64) - normalized_affinity
    laplacian = 0.5 * (laplacian + laplacian.T)
    return z_centered, laplacian, tau2


def score_from_eigenpairs(
    z_centered: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    k: int,
    sigma2: float,
) -> float:
    # Index 0 is the trivial mode. Retain the next k positive-frequency modes.
    lambdas = np.clip(np.asarray(eigenvalues[1 : k + 1], dtype=np.float64), 0.0, None)
    basis = np.asarray(eigenvectors[:, 1 : k + 1], dtype=np.float64)
    if basis.shape[1] != k:
        raise BenchmarkError(f"Expected {k} retained modes, found {basis.shape[1]}.")
    coefficients = z_centered.T @ basis
    energy = np.sum(coefficients * coefficients, axis=0) / float(z_centered.shape[1])
    precision = (1.0 / float(sigma2)) + lambdas
    return float(np.log1p(np.sum(precision * energy)))


def harmora_partial(
    representation: np.ndarray,
    k: int = 10,
    sigma2: float = 1.0,
    tolerance: float = 1e-8,
    maxiter: int = 5000,
    ncv: Optional[int] = None,
) -> HarmoraRun:
    total_start = time.perf_counter()
    graph_start = total_start
    z_centered, laplacian, _ = center_and_build_laplacian(representation)
    graph_seconds = time.perf_counter() - graph_start

    n = int(laplacian.shape[0])
    if n <= k + 2:
        raise BenchmarkError(f"Need N > K+2, found N={n}, K={k}.")

    requested = k + 2  # trivial + K retained + one boundary diagnostic mode
    basis_size = int(ncv) if ncv is not None else min(n - 1, max(2 * requested + 1, 24))
    if basis_size <= requested:
        basis_size = min(n - 1, requested + 2)

    eig_start = time.perf_counter()
    operator = CountedSymmetricOperator(laplacian)
    initial = np.linspace(1.0, 2.0, n, dtype=np.float64)
    initial /= np.linalg.norm(initial)
    values, vectors = eigsh(
        operator,
        k=requested,
        which="SM",
        tol=float(tolerance),
        maxiter=int(maxiter),
        ncv=int(basis_size),
        v0=initial,
        return_eigenvectors=True,
    )
    order = np.argsort(values)
    values = np.asarray(values[order], dtype=np.float64)
    vectors = np.asarray(vectors[:, order], dtype=np.float64)
    eig_seconds = time.perf_counter() - eig_start

    projection_start = time.perf_counter()
    score = score_from_eigenpairs(z_centered, values, vectors, k=k, sigma2=sigma2)
    projection_seconds = time.perf_counter() - projection_start
    boundary_gap = float(values[k + 1] - values[k])

    return HarmoraRun(
        score=score,
        graph_seconds=float(graph_seconds),
        eig_seconds=float(eig_seconds),
        projection_seconds=float(projection_seconds),
        total_seconds=float(time.perf_counter() - total_start),
        n_samples=int(z_centered.shape[0]),
        dimension=int(z_centered.shape[1]),
        k=int(k),
        matvec_count=int(operator.matvec_count),
        ncv=int(basis_size),
        boundary_gap=boundary_gap,
        eig_status="converged",
    )


def harmora_full(
    representation: np.ndarray,
    k: int = 10,
    sigma2: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    graph_start = time.perf_counter()
    z_centered, laplacian, _ = center_and_build_laplacian(representation)
    graph_seconds = time.perf_counter() - graph_start
    eig_start = time.perf_counter()
    values, vectors = eigh(laplacian, check_finite=False, overwrite_a=False)
    eig_seconds = time.perf_counter() - eig_start
    score = score_from_eigenpairs(z_centered, values, vectors, k=k, sigma2=sigma2)
    return score, values, vectors, float(graph_seconds), float(eig_seconds)


def time_harmora(
    representation: np.ndarray,
    *,
    k: int,
    sigma2: float,
    tolerance: float,
    maxiter: int,
    ncv: Optional[int],
    warmups: int,
    repeats: int,
) -> tuple[dict[str, float], HarmoraRun]:
    for _ in range(int(warmups)):
        harmora_partial(
            representation,
            k=k,
            sigma2=sigma2,
            tolerance=tolerance,
            maxiter=maxiter,
            ncv=ncv,
        )

    runs: list[HarmoraRun] = []
    with threadpool_limits(limits=1):
        for _ in range(int(repeats)):
            runs.append(
                harmora_partial(
                    representation,
                    k=k,
                    sigma2=sigma2,
                    tolerance=tolerance,
                    maxiter=maxiter,
                    ncv=ncv,
                )
            )

    score_spread = np.ptp([run.score for run in runs])
    if score_spread > 1e-8:
        raise BenchmarkError(f"Repeated Harmora scores differ by {score_spread:.3e}.")

    summary: dict[str, float] = {}
    for field in ["total_seconds", "graph_seconds", "eig_seconds", "projection_seconds"]:
        median, q1, q3 = percentile_interval(getattr(run, field) for run in runs)
        summary[f"{field}_median"] = median
        summary[f"{field}_q1"] = q1
        summary[f"{field}_q3"] = q3
    summary["matvec_median"] = float(np.median([run.matvec_count for run in runs]))
    summary["boundary_gap"] = float(np.median([run.boundary_gap for run in runs]))
    return summary, runs[-1]


def time_callable(
    function: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
) -> tuple[float, float, float, Any]:
    for _ in range(int(warmups)):
        function()
    durations = []
    result = None
    with threadpool_limits(limits=1):
        for _ in range(int(repeats)):
            start = time.perf_counter()
            result = function()
            durations.append(time.perf_counter() - start)
    median, q1, q3 = percentile_interval(durations)
    return median, q1, q3, result


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    a_norm = a64 / np.clip(np.linalg.norm(a64, axis=1, keepdims=True), 1e-12, None)
    b_norm = b64 / np.clip(np.linalg.norm(b64, axis=1, keepdims=True), 1e-12, None)
    return np.sum(a_norm * b_norm, axis=1)


def encoded_labels(values: Iterable[Any]) -> np.ndarray:
    serialized = [json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values]
    mapping = {value: index for index, value in enumerate(sorted(set(serialized)))}
    return np.asarray([mapping[value] for value in serialized], dtype=int)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def locate_project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    return resolved.parent.parent


def import_project(project_root: Path) -> None:
    src = project_root / "src"
    if not src.exists():
        raise BenchmarkError(f"Project source directory not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_project_config(config_path: Path) -> tuple[dict[str, Any], Path, Path]:
    project_root = locate_project_root(config_path)
    import_project(project_root)
    from harmora_downstream.config import load_config, resolve_path

    cfg = load_config(config_path)
    output_dir = resolve_path(cfg, "output_dir")
    return cfg, project_root, output_dir


def load_inventory(output_dir: Path, expected_candidates: int) -> pd.DataFrame:
    profile_path = output_dir / "csv" / "downstream_profiles_long.csv"
    if not profile_path.exists():
        raise BenchmarkError(f"Missing downstream profile inventory: {profile_path}")
    profile = pd.read_csv(profile_path)
    required = {"model_alias", "task", "task_type", "probe_family", "layer", "sample_hash"}
    missing = sorted(required - set(profile.columns))
    if missing:
        raise BenchmarkError(f"Inventory is missing columns: {missing}")

    inventory = (
        profile[list(required)]
        .drop_duplicates()
        .sort_values(["task", "model_alias", "layer"])
        .reset_index(drop=True)
    )
    counts = inventory.groupby("task").size()
    bad = counts[counts != int(expected_candidates)]
    if len(bad):
        raise BenchmarkError(
            "Candidate pool is not identical across tasks. Expected "
            f"{expected_candidates}; observed:\n{bad.to_string()}"
        )
    return inventory


def load_task_payload(output_dir: Path, task: str) -> dict[str, Any]:
    path = output_dir / "sample_cache" / f"{safe_name(task)}.json"
    if not path.exists():
        raise BenchmarkError(f"Missing task cache: {path}")
    payload = load_json(path)
    if payload.get("status") != "ok":
        raise BenchmarkError(f"Task cache is not usable: {task}: {payload.get('reason')}")
    return payload


def load_embedding_archive(output_dir: Path, model_alias: str, task: str) -> dict[str, Any]:
    path = output_dir / "embedding_cache" / safe_name(model_alias) / f"{safe_name(task)}.npz"
    if not path.exists():
        raise BenchmarkError(f"Missing embedding cache: {path}")
    with np.load(path, allow_pickle=False) as archive:
        result = {
            "sample_hash": scalar_from_npz(archive, "sample_hash"),
            "embedding_hash": scalar_from_npz(archive, "embedding_hash"),
            "num_layers": int(np.asarray(archive["num_layers"]).item()),
        }
        for key in ["embeddings", "embeddings_a", "embeddings_b"]:
            if key in archive.files:
                result[key] = np.asarray(archive[key], dtype=np.float32)
    return result


def layer_mapping(inventory: pd.DataFrame, task: str, model_alias: str, n_layers: int) -> dict[int, int]:
    labels = sorted(
        inventory.loc[
            (inventory["task"] == task) & (inventory["model_alias"] == model_alias),
            "layer",
        ].astype(int).unique()
    )
    if len(labels) != int(n_layers):
        raise BenchmarkError(
            f"Layer mapping mismatch for {model_alias}/{task}: "
            f"inventory={labels}, tensor_layers={n_layers}."
        )
    return {int(label): position for position, label in enumerate(labels)}


def candidate_arrays(
    payload: dict[str, Any],
    archive: dict[str, Any],
    tensor_position: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    family = payload["probe_family"]
    if family in {"classification", "clustering"}:
        single = np.asarray(archive["embeddings"][tensor_position], dtype=np.float64)
        return single, {"embeddings": single}
    a = np.asarray(archive["embeddings_a"][tensor_position], dtype=np.float64)
    b = np.asarray(archive["embeddings_b"][tensor_position], dtype=np.float64)
    metric = np.concatenate([a, b], axis=0)
    return metric, {"embeddings_a": a, "embeddings_b": b}


def load_split(output_dir: Path, task: str, seed: int) -> dict[str, Any]:
    path = output_dir / "split_manifests" / safe_name(task) / f"seed_{int(seed)}.json"
    if not path.exists():
        raise BenchmarkError(
            f"Missing classification split manifest: {path}. "
            "Run the validated downstream pipeline first."
        )
    return load_json(path)


def primary_evaluator(
    payload: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    seed: int,
    downstream_cfg: dict[str, Any],
    split_manifest: Optional[dict[str, Any]],
) -> float:
    family = payload["probe_family"]
    if family == "classification":
        if split_manifest is None:
            raise BenchmarkError("Classification requires a split manifest.")
        y = encoded_labels(payload["labels"])
        train = np.asarray(split_manifest["train_indices"], dtype=int)
        test = np.asarray(split_manifest["test_indices"], dtype=int)
        settings = downstream_cfg.get("classification", {})
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=int(seed),
                        max_iter=int(settings.get("max_iter", 3000)),
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(arrays["embeddings"][train], y[train])
        return float(np.mean(model.predict(arrays["embeddings"][test]) == y[test]))

    if family == "clustering":
        y = encoded_labels(payload["labels"])
        settings = downstream_cfg.get("clustering", {})
        model = MiniBatchKMeans(
            n_clusters=int(len(np.unique(y))),
            random_state=int(seed),
            n_init=int(settings.get("n_init", 10)),
            batch_size=min(
                int(settings.get("batch_size", 500)),
                max(10, arrays["embeddings"].shape[1]),
            ),
        )
        assignments = model.fit_predict(arrays["embeddings"])
        return float(v_measure_score(y, assignments))

    similarity = cosine_rows(arrays["embeddings_a"], arrays["embeddings_b"])
    if family == "pair_classification":
        y = encoded_labels(payload["targets"])
        normalized = [str(value).strip().lower() for value in payload["targets"]]
        positive_tokens = {
            "1", "true", "yes", "positive", "duplicate",
            "entailment", "match",
        }
        raw_positive = np.asarray(
            [value in positive_tokens for value in normalized], dtype=bool
        )
        if raw_positive.any() and (~raw_positive).any():
            y_binary = raw_positive.astype(int)
        else:
            y_binary = (y == np.max(y)).astype(int)
        return float(average_precision_score(y_binary, similarity))
    if family == "sts":
        statistic = spearmanr(
            np.asarray(payload["targets"], dtype=float), similarity, nan_policy="omit"
        ).statistic
        return float(statistic)
    raise BenchmarkError(f"Unsupported probe family: {family}")


def effective_seeds(payload: dict[str, Any], configured_seeds: list[int]) -> list[int]:
    if payload["probe_family"] in {"classification", "clustering"}:
        return [int(seed) for seed in configured_seeds]
    return [int(configured_seeds[0])]


def checkpoint_frame(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def benchmark_real_pool(
    *,
    cfg: dict[str, Any],
    output_dir: Path,
    inventory: pd.DataFrame,
    result_dir: Path,
    k: int,
    sigma2: float,
    tolerance: float,
    maxiter: int,
    ncv: Optional[int],
    shortlist_size: int,
    harmora_warmups: int,
    harmora_repeats: int,
    evaluator_warmups: int,
    evaluator_repeats: int,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_path = result_dir / "real_candidate_runtime_long.csv"
    existing = pd.read_csv(long_path) if resume and long_path.exists() else pd.DataFrame()
    completed_keys = set()
    rows: list[dict[str, Any]] = []
    if len(existing):
        rows = existing.to_dict("records")
        completed_keys = {
            (str(row["task"]), str(row["model_alias"]), int(row["layer"]))
            for row in rows
            if str(row.get("status", "")) == "ok"
        }

    configured_seeds = [int(value) for value in cfg.get("seeds", [])]
    downstream_cfg = cfg.get("downstream", {})

    for task_index, task in enumerate(sorted(inventory["task"].unique()), start=1):
        payload = load_task_payload(output_dir, task)
        task_inventory = inventory.loc[inventory["task"] == task]
        print(
            f"\n[REAL {task_index}/{inventory['task'].nunique()}] "
            f"task={task} family={payload['probe_family']} candidates={len(task_inventory)}"
        )

        for model_alias in sorted(task_inventory["model_alias"].unique()):
            archive = load_embedding_archive(output_dir, model_alias, task)
            if archive["sample_hash"] != str(payload["sample_hash"]):
                raise BenchmarkError(f"Sample hash mismatch for {model_alias}/{task}.")
            mapping = layer_mapping(inventory, task, model_alias, archive["num_layers"])

            model_rows = task_inventory.loc[task_inventory["model_alias"] == model_alias]
            for candidate_index, candidate in enumerate(model_rows.itertuples(), start=1):
                key = (task, model_alias, int(candidate.layer))
                if key in completed_keys:
                    continue
                metric_matrix, evaluator_arrays = candidate_arrays(
                    payload, archive, mapping[int(candidate.layer)]
                )
                print(
                    f"  {model_alias} layer={int(candidate.layer)} "
                    f"N_metric={metric_matrix.shape[0]} d={metric_matrix.shape[1]}"
                )

                try:
                    h_summary, h_run = time_harmora(
                        metric_matrix,
                        k=k,
                        sigma2=sigma2,
                        tolerance=tolerance,
                        maxiter=maxiter,
                        ncv=ncv,
                        warmups=harmora_warmups,
                        repeats=harmora_repeats,
                    )

                    evaluator_entries = []
                    for seed in effective_seeds(payload, configured_seeds):
                        split = load_split(output_dir, task, seed) if payload["probe_family"] == "classification" else None
                        function = lambda seed=seed, split=split: primary_evaluator(
                            payload,
                            evaluator_arrays,
                            seed=seed,
                            downstream_cfg=downstream_cfg,
                            split_manifest=split,
                        )
                        median, q1, q3, evaluator_score = time_callable(
                            function,
                            warmups=evaluator_warmups,
                            repeats=evaluator_repeats,
                        )
                        evaluator_entries.append(
                            {
                                "seed": int(seed),
                                "median": median,
                                "q1": q1,
                                "q3": q3,
                                "score": float(evaluator_score),
                            }
                        )

                    row = {
                        "task": task,
                        "task_type": candidate.task_type,
                        "probe_family": candidate.probe_family,
                        "model_alias": model_alias,
                        "layer": int(candidate.layer),
                        "sample_hash": str(candidate.sample_hash),
                        "n_metric_samples": int(metric_matrix.shape[0]),
                        "n_evaluator_items": int(
                            evaluator_arrays.get("embeddings", evaluator_arrays.get("embeddings_a")).shape[0]
                        ),
                        "dimension": int(metric_matrix.shape[1]),
                        "harmora_score": float(h_run.score),
                        "harmora_total_median_s": h_summary["total_seconds_median"],
                        "harmora_total_q1_s": h_summary["total_seconds_q1"],
                        "harmora_total_q3_s": h_summary["total_seconds_q3"],
                        "harmora_graph_median_s": h_summary["graph_seconds_median"],
                        "harmora_eig_median_s": h_summary["eig_seconds_median"],
                        "harmora_projection_median_s": h_summary["projection_seconds_median"],
                        "harmora_matvec_median": h_summary["matvec_median"],
                        "harmora_ncv": int(h_run.ncv),
                        "boundary_gap": h_summary["boundary_gap"],
                        "effective_evaluation_count": len(evaluator_entries),
                        "evaluator_total_median_s": float(sum(item["median"] for item in evaluator_entries)),
                        "evaluator_seed_medians_json": json.dumps(evaluator_entries),
                        "status": "ok",
                        "error": "",
                    }
                except Exception as exc:
                    row = {
                        "task": task,
                        "task_type": candidate.task_type,
                        "probe_family": candidate.probe_family,
                        "model_alias": model_alias,
                        "layer": int(candidate.layer),
                        "sample_hash": str(candidate.sample_hash),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    traceback.print_exc()
                rows.append(row)
                completed_keys.add(key)
                checkpoint_frame(rows, long_path)

    long_df = pd.DataFrame(rows)
    failures = long_df.loc[long_df["status"] != "ok"]
    if len(failures):
        raise BenchmarkError(
            f"Real-pool benchmark has {len(failures)} failed candidates. "
            f"Inspect {long_path}."
        )

    summary_rows = []
    for task, group in long_df.groupby("task", observed=True):
        ranked = group.sort_values("harmora_score", ascending=False)
        shortlist = ranked.head(int(shortlist_size))
        exhaustive = float(group["evaluator_total_median_s"].sum())
        screening = float(group["harmora_total_median_s"].sum())
        shortlisted = float(shortlist["evaluator_total_median_s"].sum())
        hybrid = screening + shortlisted
        summary_rows.append(
            {
                "task": task,
                "probe_family": str(group["probe_family"].iloc[0]),
                "n_candidates": int(len(group)),
                "shortlist_size": int(shortlist_size),
                "effective_evaluation_count": int(group["effective_evaluation_count"].iloc[0]),
                "exhaustive_evaluator_s": exhaustive,
                "harmora_screen_all_s": screening,
                "shortlist_evaluator_s": shortlisted,
                "hybrid_total_s": hybrid,
                "hybrid_speedup": exhaustive / hybrid if hybrid > 0 else np.nan,
                "harmora_to_mean_evaluator_ratio": (
                    float(group["harmora_total_median_s"].mean())
                    / float(group["evaluator_total_median_s"].mean())
                ),
                "break_even": bool(hybrid < exhaustive),
                "shortlisted_candidates": "; ".join(
                    f"{row.model_alias}:L{int(row.layer)}" for row in shortlist.itertuples()
                ),
            }
        )
    task_summary = pd.DataFrame(summary_rows)
    task_summary.to_csv(result_dir / "real_task_runtime_summary.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "tasks": int(task_summary["task"].nunique()),
                "candidate_task_cases": int(len(long_df)),
                "exhaustive_evaluator_s": float(task_summary["exhaustive_evaluator_s"].sum()),
                "harmora_screen_all_s": float(task_summary["harmora_screen_all_s"].sum()),
                "shortlist_evaluator_s": float(task_summary["shortlist_evaluator_s"].sum()),
                "hybrid_total_s": float(task_summary["hybrid_total_s"].sum()),
                "hybrid_speedup": (
                    float(task_summary["exhaustive_evaluator_s"].sum())
                    / float(task_summary["hybrid_total_s"].sum())
                ),
                "tasks_with_break_even": int(task_summary["break_even"].sum()),
            }
        ]
    )
    overall.to_csv(result_dir / "real_overall_runtime_summary.csv", index=False)
    return long_df, task_summary


def controlled_matrix(n: int, d: int, classes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    labels = np.arange(int(n), dtype=int) % int(classes)
    rng.shuffle(labels)
    latent_dim = min(32, int(d))
    centers = rng.normal(0.0, 1.0, size=(classes, latent_dim))
    latent = centers[labels] + rng.normal(0.0, 0.85, size=(n, latent_dim))
    projection = rng.normal(0.0, 1.0 / math.sqrt(latent_dim), size=(latent_dim, d))
    matrix = latent @ projection + rng.normal(0.0, 0.25, size=(n, d))
    return matrix.astype(np.float64), labels


def benchmark_sample_scaling(
    *,
    result_dir: Path,
    n_values: list[int],
    d: int,
    classes: int,
    k: int,
    sigma2: float,
    tolerance: float,
    maxiter: int,
    ncv: Optional[int],
    warmups: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for n in n_values:
        print(f"\n[SCALING] N={n} d={d} classes={classes}")
        matrix, labels = controlled_matrix(n, d, classes, seed + n)
        train, test = train_test_split(
            np.arange(n), test_size=0.30, random_state=seed, stratify=labels
        )

        h_summary, _ = time_harmora(
            matrix,
            k=k,
            sigma2=sigma2,
            tolerance=tolerance,
            maxiter=maxiter,
            ncv=ncv,
            warmups=warmups,
            repeats=repeats,
        )

        def classification_call() -> float:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            random_state=seed,
                            max_iter=3000,
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model.fit(matrix[train], labels[train])
            return float(np.mean(model.predict(matrix[test]) == labels[test]))

        def clustering_call() -> float:
            model = MiniBatchKMeans(
                n_clusters=classes,
                random_state=seed,
                n_init=10,
                batch_size=min(500, max(10, d)),
            )
            return float(v_measure_score(labels, model.fit_predict(matrix)))

        cls_median, cls_q1, cls_q3, _ = time_callable(
            classification_call, warmups=warmups, repeats=repeats
        )
        clu_median, clu_q1, clu_q3, _ = time_callable(
            clustering_call, warmups=warmups, repeats=repeats
        )
        rows.append(
            {
                "N": int(n),
                "d": int(d),
                "classes": int(classes),
                "harmora_median_s": h_summary["total_seconds_median"],
                "harmora_q1_s": h_summary["total_seconds_q1"],
                "harmora_q3_s": h_summary["total_seconds_q3"],
                "classification_median_s": cls_median,
                "classification_q1_s": cls_q1,
                "classification_q3_s": cls_q3,
                "clustering_median_s": clu_median,
                "clustering_q1_s": clu_q1,
                "clustering_q3_s": clu_q3,
            }
        )
    frame = pd.DataFrame(rows)
    for column, prefix in [
        ("harmora_median_s", "harmora"),
        ("classification_median_s", "classification"),
        ("clustering_median_s", "clustering"),
    ]:
        slope, intercept = np.polyfit(np.log(frame["N"]), np.log(frame[column]), 1)
        prediction = intercept + slope * np.log(frame["N"])
        residual = np.log(frame[column]) - prediction
        total = np.log(frame[column]) - np.log(frame[column]).mean()
        r2 = 1.0 - float(np.sum(residual**2) / np.sum(total**2))
        frame[f"{prefix}_loglog_slope"] = float(slope)
        frame[f"{prefix}_loglog_r2"] = r2
    frame.to_csv(result_dir / "sample_size_scaling.csv", index=False)
    return frame


def fixed_matrix_cardinality_labels(
    n: int,
    d: int,
    cardinalities: list[int],
    seed: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 1.0, size=(n, d))
    direction = rng.normal(0.0, 1.0, size=d)
    latent = matrix @ direction + 0.1 * rng.normal(size=n)
    ranks = np.argsort(np.argsort(latent))
    labels = {
        int(cardinality): np.minimum(
            (ranks * int(cardinality)) // int(n), int(cardinality) - 1
        ).astype(int)
        for cardinality in cardinalities
    }
    return matrix, labels


def benchmark_cardinality_scaling(
    *,
    result_dir: Path,
    n: int,
    d: int,
    cardinalities: list[int],
    k: int,
    sigma2: float,
    tolerance: float,
    maxiter: int,
    ncv: Optional[int],
    warmups: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    matrix, labels_by_cardinality = fixed_matrix_cardinality_labels(
        n, d, cardinalities, seed
    )
    h_summary, _ = time_harmora(
        matrix,
        k=k,
        sigma2=sigma2,
        tolerance=tolerance,
        maxiter=maxiter,
        ncv=ncv,
        warmups=warmups,
        repeats=repeats,
    )
    rows = []
    for cardinality in cardinalities:
        print(f"\n[CARDINALITY] kappa={cardinality} fixed_Z=True")
        labels = labels_by_cardinality[int(cardinality)]
        train, test = train_test_split(
            np.arange(n), test_size=0.30, random_state=seed, stratify=labels
        )

        def classification_call() -> float:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            random_state=seed,
                            max_iter=3000,
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model.fit(matrix[train], labels[train])
            return float(np.mean(model.predict(matrix[test]) == labels[test]))

        def clustering_call() -> float:
            model = MiniBatchKMeans(
                n_clusters=int(cardinality),
                random_state=seed,
                n_init=10,
                batch_size=min(500, max(10, d)),
            )
            return float(v_measure_score(labels, model.fit_predict(matrix)))

        cls_median, cls_q1, cls_q3, _ = time_callable(
            classification_call, warmups=warmups, repeats=repeats
        )
        clu_median, clu_q1, clu_q3, _ = time_callable(
            clustering_call, warmups=warmups, repeats=repeats
        )
        rows.append(
            {
                "output_cardinality": int(cardinality),
                "N": int(n),
                "d": int(d),
                "fixed_representation_hash": hashlib.sha256(matrix.tobytes()).hexdigest()[:20],
                "harmora_median_s": h_summary["total_seconds_median"],
                "classification_median_s": cls_median,
                "classification_q1_s": cls_q1,
                "classification_q3_s": cls_q3,
                "clustering_median_s": clu_median,
                "clustering_q1_s": clu_q1,
                "clustering_q3_s": clu_q3,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(result_dir / "output_cardinality_scaling.csv", index=False)
    return frame


def select_validation_candidates(inventory: pd.DataFrame, count: int) -> pd.DataFrame:
    candidates = inventory.sort_values(["probe_family", "task", "model_alias", "layer"])
    indices = np.linspace(0, len(candidates) - 1, num=min(count, len(candidates)), dtype=int)
    return candidates.iloc[np.unique(indices)].reset_index(drop=True)


def benchmark_eigensolver_validation(
    *,
    output_dir: Path,
    inventory: pd.DataFrame,
    result_dir: Path,
    count: int,
    validation_max_n: int,
    k: int,
    sigma2: float,
    tolerance: float,
    maxiter: int,
    ncv: Optional[int],
) -> pd.DataFrame:
    rows = []
    selected = select_validation_candidates(inventory, count)
    for index, candidate in enumerate(selected.itertuples(), start=1):
        print(
            f"\n[VALIDATION {index}/{len(selected)}] "
            f"{candidate.task} {candidate.model_alias} L{int(candidate.layer)}"
        )
        payload = load_task_payload(output_dir, candidate.task)
        archive = load_embedding_archive(output_dir, candidate.model_alias, candidate.task)
        mapping = layer_mapping(
            inventory, candidate.task, candidate.model_alias, archive["num_layers"]
        )
        matrix, _ = candidate_arrays(payload, archive, mapping[int(candidate.layer)])
        if matrix.shape[0] > validation_max_n:
            rng = np.random.default_rng(index + 2026)
            chosen = np.sort(
                rng.choice(matrix.shape[0], size=int(validation_max_n), replace=False)
            )
            matrix = matrix[chosen]

        z_centered, laplacian, _ = center_and_build_laplacian(matrix)
        partial_start = time.perf_counter()
        operator = CountedSymmetricOperator(laplacian)
        requested = k + 2
        basis_size = int(ncv) if ncv is not None else min(
            laplacian.shape[0] - 1, max(2 * requested + 1, 24)
        )
        values_p, vectors_p = eigsh(
            operator,
            k=requested,
            which="SM",
            tol=tolerance,
            maxiter=maxiter,
            ncv=basis_size,
            v0=np.linspace(1.0, 2.0, laplacian.shape[0]),
        )
        order = np.argsort(values_p)
        values_p = values_p[order]
        vectors_p = vectors_p[:, order]
        partial_seconds = time.perf_counter() - partial_start

        full_start = time.perf_counter()
        values_f, vectors_f = eigh(laplacian, check_finite=False)
        full_seconds = time.perf_counter() - full_start

        score_p = score_from_eigenpairs(z_centered, values_p, vectors_p, k, sigma2)
        score_f = score_from_eigenpairs(z_centered, values_f, vectors_f, k, sigma2)
        basis_p = vectors_p[:, 1 : k + 1]
        basis_f = vectors_f[:, 1 : k + 1]
        projector_error = float(
            np.linalg.norm(basis_p @ basis_p.T - basis_f @ basis_f.T, ord="fro")
            / math.sqrt(2.0 * k)
        )
        rows.append(
            {
                "task": candidate.task,
                "probe_family": candidate.probe_family,
                "model_alias": candidate.model_alias,
                "layer": int(candidate.layer),
                "N": int(matrix.shape[0]),
                "d": int(matrix.shape[1]),
                "relative_score_error": abs(score_p - score_f) / max(abs(score_f), 1e-15),
                "max_retained_eigenvalue_error": float(
                    np.max(np.abs(values_p[: k + 1] - values_f[: k + 1]))
                ),
                "projector_error": projector_error,
                "boundary_gap_full": float(values_f[k + 1] - values_f[k]),
                "partial_seconds": float(partial_seconds),
                "full_seconds": float(full_seconds),
                "partial_matvecs": int(operator.matvec_count),
                "passed": bool(
                    abs(score_p - score_f) / max(abs(score_f), 1e-15) <= 1e-6
                    and projector_error <= 1e-5
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(result_dir / "partial_eigensolver_validation.csv", index=False)
    if not frame["passed"].all():
        raise BenchmarkError("At least one partial-eigensolver validation case failed.")
    return frame


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_outputs(
    result_dir: Path,
    real_task: Optional[pd.DataFrame],
    scaling: Optional[pd.DataFrame],
    cardinality: Optional[pd.DataFrame],
    validation: Optional[pd.DataFrame],
) -> None:
    if real_task is not None and len(real_task):
        ordered = real_task.sort_values("hybrid_speedup")
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.barh(ordered["task"], ordered["hybrid_speedup"])
        ax.axvline(1.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Hybrid speedup over exhaustive evaluation")
        fig.tight_layout()
        save_figure(fig, result_dir / "fig_real_task_speedup")

    if scaling is not None and len(scaling):
        fig, ax = plt.subplots(figsize=(5.6, 4.1))
        ax.plot(scaling["N"], scaling["harmora_median_s"], marker="o", label="Harmora")
        ax.plot(
            scaling["N"],
            scaling["classification_median_s"],
            marker="o",
            label="Classification evaluator",
        )
        ax.plot(
            scaling["N"],
            scaling["clustering_median_s"],
            marker="o",
            label="Clustering evaluator",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of samples N")
        ax.set_ylabel("Post-embedding runtime (s)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, result_dir / "fig_sample_size_scaling")

    if cardinality is not None and len(cardinality):
        fig, ax = plt.subplots(figsize=(5.6, 4.1))
        ax.plot(
            cardinality["output_cardinality"],
            cardinality["harmora_median_s"],
            marker="o",
            label="Harmora",
        )
        ax.plot(
            cardinality["output_cardinality"],
            cardinality["classification_median_s"],
            marker="o",
            label="Classification evaluator",
        )
        ax.plot(
            cardinality["output_cardinality"],
            cardinality["clustering_median_s"],
            marker="o",
            label="Clustering evaluator",
        )
        ax.set_xlabel("Output cardinality")
        ax.set_ylabel("Post-embedding runtime (s)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, result_dir / "fig_output_cardinality_scaling")

    if validation is not None and len(validation):
        fig, ax = plt.subplots(figsize=(5.6, 4.1))
        ax.scatter(
            np.arange(len(validation)), validation["relative_score_error"], label="Score error"
        )
        ax.scatter(
            np.arange(len(validation)), validation["projector_error"], label="Subspace error"
        )
        ax.set_yscale("log")
        ax.set_xlabel("Validation case")
        ax.set_ylabel("Relative error")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, result_dir / "fig_partial_eigensolver_validation")


def latex_escape(text: Any) -> str:
    result = str(text)
    for source, target in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ]:
        result = result.replace(source, target)
    return result


def write_latex_tables(
    result_dir: Path,
    real_task: Optional[pd.DataFrame],
    scaling: Optional[pd.DataFrame],
    validation: Optional[pd.DataFrame],
) -> None:
    if real_task is not None and len(real_task):
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Measured post-embedding cost of exhaustive task-matched evaluation and Harmora top-$5$ shortlisting.}",
            r"\label{tab:runtime_real_tasks}",
            r"\footnotesize",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}",
            r"\toprule",
            "Task & Family & Exhaustive (s) & Screen all (s) & Probe top-5 (s) & Hybrid (s) & Speedup \\",
            r"\midrule",
        ]
        for row in real_task.itertuples():
            lines.append(
                f"{latex_escape(row.task)} & {latex_escape(row.probe_family)} & "
                f"{row.exhaustive_evaluator_s:.3f} & {row.harmora_screen_all_s:.3f} & "
                f"{row.shortlist_evaluator_s:.3f} & {row.hybrid_total_s:.3f} & "
                f"{row.hybrid_speedup:.2f}$\\times$ \\\\"  # noqa: W605
            )
        lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
        (result_dir / "tab_runtime_real_tasks.tex").write_text("\n".join(lines), encoding="utf-8")

    if scaling is not None and len(scaling):
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Controlled post-embedding runtime versus sample size. Entries are medians in seconds.}",
            r"\label{tab:runtime_sample_scaling}",
            r"\small",
            r"\begin{tabular}{rrrr}",
            r"\toprule",
            "$N$ & Harmora & Classification & Clustering \\",
            r"\midrule",
        ]
        for row in scaling.itertuples():
            lines.append(
                f"{int(row.N)} & {row.harmora_median_s:.4f} & "
                f"{row.classification_median_s:.4f} & {row.clustering_median_s:.4f} \\\\"  # noqa: W605
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        (result_dir / "tab_runtime_sample_scaling.tex").write_text("\n".join(lines), encoding="utf-8")

    if validation is not None and len(validation):
        summary = {
            "cases": len(validation),
            "max_score": validation["relative_score_error"].max(),
            "max_eig": validation["max_retained_eigenvalue_error"].max(),
            "max_proj": validation["projector_error"].max(),
            "failures": int((~validation["passed"]).sum()),
        }
        text = rf"""
\begin{{table}}[t]
\centering
\caption{{Partial-eigensolver validation on real cached representations.}}
\label{{tab:partial_eigensolver_validation}}
\small
\begin{{tabular}}{{lr}}
\toprule
Quantity & Value \\
\midrule
Validation cases & {summary['cases']} \\
Maximum relative score error & {summary['max_score']:.2e} \\
Maximum retained-eigenvalue error & {summary['max_eig']:.2e} \\
Maximum subspace-projector error & {summary['max_proj']:.2e} \\
Failed cases & {summary['failures']} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip() + "\n"
        (result_dir / "tab_partial_eigensolver_validation.tex").write_text(text, encoding="utf-8")


def write_manifest(
    result_dir: Path,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    output_dir: Path,
    inventory: pd.DataFrame,
) -> None:
    manifest = {
        "version": VERSION,
        "created_unix": time.time(),
        "config": str(Path(args.config).resolve()),
        "validated_output_dir": str(output_dir),
        "models": sorted(inventory["model_alias"].unique().tolist()),
        "tasks": sorted(inventory["task"].unique().tolist()),
        "candidate_count_per_task": inventory.groupby("task").size().to_dict(),
        "shortlist_size": int(args.shortlist_size),
        "K": int(args.K),
        "sigma2": float(args.sigma2),
        "eigensolver": {
            "algorithm": "scipy.sparse.linalg.eigsh",
            "which": "SM",
            "tolerance": float(args.eig_tolerance),
            "maxiter": int(args.eig_maxiter),
            "ncv": args.ncv,
            "shift_invert": False,
        },
        "timing": {
            "real_warmups": int(args.real_warmups),
            "real_repeats": int(args.real_repeats),
            "evaluator_warmups": int(args.evaluator_warmups),
            "evaluator_repeats": int(args.evaluator_repeats),
            "scaling_warmups": int(args.scaling_warmups),
            "scaling_repeats": int(args.scaling_repeats),
            "thread_limit": 1,
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "threadpools": threadpool_info(),
        },
        "scope": (
            "Post-embedding only. Transformer inference and representation extraction "
            "are excluded. The real-cache benchmark uses the task-matched primary "
            "evaluator defined by the validated downstream framework."
        ),
    }
    (result_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-candidates", type=int, default=106)
    parser.add_argument("--shortlist-size", type=int, default=5)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--eig-tolerance", type=float, default=1e-8)
    parser.add_argument("--eig-maxiter", type=int, default=5000)
    parser.add_argument("--ncv", type=int, default=None)

    parser.add_argument("--real-warmups", type=int, default=1)
    parser.add_argument("--real-repeats", type=int, default=3)
    parser.add_argument("--evaluator-warmups", type=int, default=1)
    parser.add_argument("--evaluator-repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[100, 200, 300, 450, 600, 900, 1200, 1600],
    )
    parser.add_argument("--scaling-d", type=int, default=768)
    parser.add_argument("--scaling-classes", type=int, default=10)
    parser.add_argument("--scaling-warmups", type=int, default=1)
    parser.add_argument("--scaling-repeats", type=int, default=5)
    parser.add_argument("--scaling-seed", type=int, default=2026)

    parser.add_argument("--cardinality-N", type=int, default=600)
    parser.add_argument(
        "--cardinalities", type=int, nargs="+", default=[2, 5, 10, 20, 40, 77]
    )

    parser.add_argument("--validation-cases", type=int, default=12)
    parser.add_argument("--validation-max-N", type=int, default=300)

    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--skip-cardinality", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, project_root, validated_output = load_project_config(args.config)
    result_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else validated_output / "analysis" / "runtime_framework_v2"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    inventory = load_inventory(validated_output, args.expected_candidates)
    print(f"Harmora runtime benchmark {VERSION}")
    print(f"Project root: {project_root}")
    print(f"Validated outputs: {validated_output}")
    print(f"Result directory: {result_dir}")
    print(
        f"Pool: {inventory['model_alias'].nunique()} models, "
        f"{inventory['task'].nunique()} tasks, "
        f"{inventory.groupby('task').size().iloc[0]} candidates/task"
    )

    write_manifest(result_dir, args, cfg, validated_output, inventory)

    real_task = None
    scaling = None
    cardinality = None
    validation = None

    if not args.skip_real:
        _, real_task = benchmark_real_pool(
            cfg=cfg,
            output_dir=validated_output,
            inventory=inventory,
            result_dir=result_dir,
            k=args.K,
            sigma2=args.sigma2,
            tolerance=args.eig_tolerance,
            maxiter=args.eig_maxiter,
            ncv=args.ncv,
            shortlist_size=args.shortlist_size,
            harmora_warmups=args.real_warmups,
            harmora_repeats=args.real_repeats,
            evaluator_warmups=args.evaluator_warmups,
            evaluator_repeats=args.evaluator_repeats,
            resume=args.resume,
        )

    if not args.skip_scaling:
        scaling = benchmark_sample_scaling(
            result_dir=result_dir,
            n_values=args.n_values,
            d=args.scaling_d,
            classes=args.scaling_classes,
            k=args.K,
            sigma2=args.sigma2,
            tolerance=args.eig_tolerance,
            maxiter=args.eig_maxiter,
            ncv=args.ncv,
            warmups=args.scaling_warmups,
            repeats=args.scaling_repeats,
            seed=args.scaling_seed,
        )

    if not args.skip_cardinality:
        cardinality = benchmark_cardinality_scaling(
            result_dir=result_dir,
            n=args.cardinality_N,
            d=args.scaling_d,
            cardinalities=args.cardinalities,
            k=args.K,
            sigma2=args.sigma2,
            tolerance=args.eig_tolerance,
            maxiter=args.eig_maxiter,
            ncv=args.ncv,
            warmups=args.scaling_warmups,
            repeats=args.scaling_repeats,
            seed=args.scaling_seed,
        )

    if not args.skip_validation:
        validation = benchmark_eigensolver_validation(
            output_dir=validated_output,
            inventory=inventory,
            result_dir=result_dir,
            count=args.validation_cases,
            validation_max_n=args.validation_max_N,
            k=args.K,
            sigma2=args.sigma2,
            tolerance=args.eig_tolerance,
            maxiter=args.eig_maxiter,
            ncv=args.ncv,
        )

    plot_outputs(result_dir, real_task, scaling, cardinality, validation)
    write_latex_tables(result_dir, real_task, scaling, validation)
    print("\nCompleted successfully.")
    print(f"Outputs: {result_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BenchmarkError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
