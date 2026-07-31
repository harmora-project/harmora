#!/usr/bin/env python3
"""
Harmora Finite-Sample Precision: Final Paper Experiment
========================================================

This is the final, paper-facing experiment for the observation-precision term
in Harmora. It uses natural finite-sample uncertainty rather than artificial
feature noise. This revision is restricted by default to the eleven primary
benchmark tasks and validates 106 candidates from seven models per task.

For every task-model pair and analysis fraction, all layers are evaluated on
paired m-out-of-n subsamples.  Independent calibration subsamples estimate a
layer-specific instability u_l; independent evaluation subsamples test whether
that instability-aware precision improves downstream-utility-aligned layer
shortlisting.

Canonical adaptive score (no tuned hyperparameter)
---------------------------------------------------

    H_l = log(1 + p_l A_l + C_l),
    p_l = sigma_0^{-2} / (1 + u_l),
    A_l = sum_m E_lm,
    C_l = sum_m lambda_lm E_lm.

The finite-sample uncertainty u_l is a robust, shrinkage-stabilized dispersion
of retained harmonic energy across calibration subsamples, normalized within
each task-model.  The coefficient is fixed to one; there is no label-based or
post-hoc alpha selection.

Primary comparisons
-------------------
- adaptive_precision: correctly matched layer-specific precision;
- fixed_precision: sigma_l^2 = 1 for every layer;
- no_precision: removes the precision term;
- shuffled_precision: permutes valid precision values across layers.

Primary endpoints
-----------------
- layer-ranking Spearman correlation with downstream utility;
- NDCG@5 for top-layer shortlisting.

Secondary endpoints include pairwise preference accuracy, Regret@1,
Top-k oracle-coverage regret, near-oracle hit rate, and repeated-view ranking
reliability.  Reliability is diagnostic rather than a mandatory endpoint,
because baseline reliability can be near its ceiling.

Inference
---------
Metrics are averaged within model-task, then over models inside each task.  The
task is the inferential unit.  Paired task differences use task-bootstrap 95%
confidence intervals, exact sign-flip tests, and Holm correction across the
three baselines within each fraction/outcome family.

Paper-facing outputs
--------------------
The paper-facing outputs are:

1. figure_10_adaptive_precision_validity.{pdf,png}
   Cross-fitted validity of the instability estimate.
2. figure_11_adaptive_precision_primary.{pdf,png}
   Paired NDCG@5 gains at the 25% analysis fraction.
3. table1_method_performance.{csv,tex}
   Compact absolute performance table.
4. table2_primary_paired_tests.{csv,tex}
   Paired task-level effect sizes, intervals, wins/losses/ties, and Holm p.

The script also writes raw audit files, an automatic verdict, and a concise
paper-ready summary.  Official Harmora reconstruction is checked before any
claim is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, spearmanr


EPS = 1e-12
IMPLEMENTATION_VERSION = "2.3-paper-11tasks-project-layer-map"

PRIMARY_METHOD = "adaptive_precision"
BASELINES = (
    "fixed_precision",
    "no_precision",
    "shuffled_precision",
)

PRIMARY_TASKS = (
    "ArXivHierarchicalClusteringP2P",
    "ArXivHierarchicalClusteringS2S",
    "Banking77Classification.v2",
    "BiorxivClusteringP2P.v2",
    "EmotionClassification.v2",
    "HUMEEmotionClassification",
    "LegalBenchPC",
    "SICK-R",
    "STS15",
    "STS16",
    "STSBenchmark",
)

OUTCOMES = {
    "utility_spearman": True,
    "preference_accuracy": True,
    "ndcg_at_k": True,
    "regret_at_1": False,
    "coverage_regret_at_k": False,
    "near_oracle_hit_5pct": True,
    "rank_reliability": True,
    "shortlist_jaccard_at_k": True,
    "top1_agreement": True,
}


@dataclass(frozen=True)
class Config:
    harmonic_k: int
    sigma0_l2: float
    graph_standardize: bool
    bandwidth: str
    k_nn: Optional[int]
    fractions: Tuple[float, ...]
    max_samples: int
    min_samples: int
    uncertainty_views: int
    selection_views: int
    evaluation_views: int
    subsample_seeds: Tuple[int, ...]
    alpha_grid: Tuple[float, ...]
    top_k: int
    precision_shuffles: int
    shrinkage_strength: float
    seed: int
    validation_tolerance: float
    degenerate_threshold: float
    task_bootstrap: int
    noninferiority_margin: float
    stratified_if_available: bool
    device: str

    def signature(self) -> str:
        payload = json.dumps(
            {"implementation_version": IMPLEMENTATION_VERSION, **asdict(self)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("_") or "item"


def stable_seed(*parts: object, modulus: int = 2**31 - 1) -> int:
    payload = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulus


def scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def add_metrics_package(path: Path) -> None:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Metrics package directory not found: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def safe_spearman(
    x: Iterable[float],
    y: Iterable[float],
    min_n: int = 4,
) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < min_n:
        return np.nan
    if np.std(x[valid]) <= EPS or np.std(y[valid]) <= EPS:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def format_fraction(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_representation_cache(
    representation_dir: Path,
) -> Dict[Tuple[str, str], Dict[str, object]]:
    """Load the project's native embedding-cache format.

    The HARMORA pipeline stores one archive per model/task under
    ``embedding_cache/<model_alias>/<task>.npz``.  Single-text tasks use the
    ``embeddings`` array, while pair-classification and STS tasks use
    ``embeddings_a`` and ``embeddings_b``.  Task and model identifiers are
    therefore recovered from the archive path rather than expected as arrays
    inside the archive.
    """
    representation_dir = Path(representation_dir)
    archives = sorted(representation_dir.rglob("*.npz"))
    if not archives:
        raise FileNotFoundError(
            f"No representation caches found under: {representation_dir}"
        )

    output: Dict[Tuple[str, str], Dict[str, object]] = {}

    for path in archives:
        with np.load(path, allow_pickle=False) as archive:
            files = set(archive.files)
            if "sample_hash" not in files:
                raise ValueError(f"{path} is missing array: sample_hash")

            # Native project layout:
            # embedding_cache/<model_alias>/<task>.npz
            model = str(path.parent.name)
            task = str(path.stem)
            sample_hash = scalar_string(archive["sample_hash"])

            if "embeddings" in files:
                hidden = np.asarray(archive["embeddings"], dtype=np.float32)
                cache_kind = "single"
            elif {"embeddings_a", "embeddings_b"}.issubset(files):
                a = np.asarray(archive["embeddings_a"], dtype=np.float32)
                b = np.asarray(archive["embeddings_b"], dtype=np.float32)
                if a.ndim != 3 or b.ndim != 3:
                    raise ValueError(
                        f"{path}: expected pair embeddings [L,N,D], "
                        f"got {a.shape} and {b.shape}"
                    )
                if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
                    raise ValueError(
                        f"{path}: pair representation shape mismatch: "
                        f"{a.shape} vs {b.shape}"
                    )
                # This matches the official metric pipeline: both members of
                # each pair are treated as graph nodes in one representation.
                hidden = np.concatenate([a, b], axis=1)
                cache_kind = "pair"
            elif "hidden_states" in files:
                # Backward-compatible support for standalone archives used by
                # earlier experimental scripts.
                hidden = np.asarray(archive["hidden_states"], dtype=np.float32)
                task = (
                    scalar_string(archive["task"])
                    if "task" in files
                    else task
                )
                model = (
                    scalar_string(archive["model_alias"])
                    if "model_alias" in files
                    else model
                )
                cache_kind = "legacy_hidden_states"
            else:
                raise ValueError(
                    f"{path} contains none of the supported representation "
                    "arrays: embeddings, embeddings_a/embeddings_b, or "
                    f"hidden_states. Available arrays: {sorted(files)}"
                )

            labels = None
            for candidate in ("labels", "y", "targets"):
                if candidate in files:
                    candidate_labels = np.asarray(archive[candidate])
                    if (
                        candidate_labels.ndim == 1
                        and len(candidate_labels) == hidden.shape[1]
                    ):
                        labels = candidate_labels
                        break

            stored_num_layers = (
                int(np.asarray(archive["num_layers"]).item())
                if "num_layers" in files
                else int(hidden.shape[0])
            )

        if hidden.ndim != 3:
            raise ValueError(
                f"{path}: expected representation [L,N,D], got {hidden.shape}"
            )
        if stored_num_layers != int(hidden.shape[0]):
            raise ValueError(
                f"{path}: num_layers={stored_num_layers} but array has "
                f"{hidden.shape[0]} layers"
            )

        key = (task, model)
        if key in output:
            raise ValueError(f"Duplicate representation cache for {key}")

        output[key] = {
            "hidden_states": hidden,
            "sample_hash": sample_hash,
            "labels": labels,
            "path": str(path),
            "cache_kind": cache_kind,
        }

    return output

def load_utility(
    csv_path: Path,
    target_source: str,
    target_name: str,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)

    required = {
        "target_source",
        "target_name",
        "target_value",
        "task",
        "model_alias",
        "layer",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Utility CSV is missing columns: {sorted(missing)}"
        )

    selected = frame[
        (frame["target_source"].astype(str) == target_source)
        & (frame["target_name"].astype(str) == target_name)
    ].copy()

    if selected.empty:
        raise ValueError(
            "Requested target was not found: "
            f"{target_source} / {target_name}"
        )

    if "gap_regime" in selected.columns:
        regimes = set(selected["gap_regime"].dropna().astype(str))
        if "all" in regimes:
            selected = selected[
                selected["gap_regime"].astype(str) == "all"
            ].copy()

    selected["layer"] = pd.to_numeric(
        selected["layer"],
        errors="coerce",
    )
    selected["target_value"] = pd.to_numeric(
        selected["target_value"],
        errors="coerce",
    )
    selected = selected.dropna(
        subset=["layer", "target_value"]
    ).copy()
    selected["layer"] = selected["layer"].astype(int)

    keys = ["task", "model_alias", "layer"]
    consistency = (
        selected.groupby(keys)["target_value"]
        .agg(["min", "max"])
        .reset_index()
    )
    inconsistent = consistency[
        (consistency["max"] - consistency["min"]).abs() > 1e-12
    ]
    if not inconsistent.empty:
        raise ValueError(
            "The selected utility is not unique per task/model/layer:\n"
            + inconsistent.head(20).to_string(index=False)
        )

    aggregation = {"target_value": "first"}
    if "task_type" in selected.columns:
        aggregation["task_type"] = "first"

    utility = (
        selected.groupby(keys, as_index=False)
        .agg(aggregation)
        .rename(columns={"target_value": "utility"})
    )

    if "task_type" not in utility.columns:
        utility["task_type"] = "Unknown"

    return utility


def load_official_scores(
    metrics_dir: Path,
) -> Dict[Tuple[str, str, int], float]:
    """Load saved Harmora scores using the physical layer labels in each JSON.

    The project does not use one universal zero-based layer convention. For
    CLS-pooled models, the raw embedding layer is removed and the saved physical
    layer labels can start at 1 (for example, 1..12), while the underlying NumPy
    tensor still has positions 0..11. The metric JSON stores the authoritative
    mapping in ``layer_indices``; enumerating the score list would therefore
    shift those models by one layer.
    """
    scores: Dict[Tuple[str, str, int], float] = {}

    for path in sorted(metrics_dir.rglob("*.json")):
        if path.name == "selected_tasks.json":
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        task = payload.get("task")
        model = payload.get("model_alias")
        harmora = payload.get("metrics", {}).get("harmora", {})
        values = harmora.get("score", []) if isinstance(harmora, dict) else []

        if not task or not model or not isinstance(values, list):
            continue

        layer_indices_raw = payload.get("layer_indices")
        if layer_indices_raw is None:
            # Backward compatibility for old/synthetic JSON files only.
            layer_indices = list(range(len(values)))
        else:
            try:
                layer_indices = [int(value) for value in layer_indices_raw]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid layer_indices in {path}: {layer_indices_raw}"
                ) from exc

        if len(layer_indices) != len(values):
            raise ValueError(
                f"Layer-index/score length mismatch in {path}: "
                f"layer_indices={len(layer_indices)}, scores={len(values)}"
            )
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError(
                f"Duplicate physical layer labels in {path}: {layer_indices}"
            )

        for physical_layer, value in zip(layer_indices, values):
            if value is None:
                continue
            try:
                key = (str(task), str(model), int(physical_layer))
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            if key in scores and not np.isclose(
                scores[key], numeric_value, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"Conflicting official Harmora scores for {key}: "
                    f"{scores[key]} vs {numeric_value}"
                )
            scores[key] = numeric_value

    if not scores:
        raise RuntimeError(
            f"No official Harmora scores found under: {metrics_dir}"
        )

    return scores


# ---------------------------------------------------------------------------
# Official Harmora components
# ---------------------------------------------------------------------------

def official_components(
    z: torch.Tensor,
    *,
    harmonic_k: int,
    graph_standardize: bool,
    bandwidth: str,
    k_nn: Optional[int],
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    from metrics.utils import (
        center_features,
        gaussian_affinity,
        normalized_laplacian,
        standardize_features,
    )

    z = z.detach().float()
    z_graph = (
        standardize_features(z, eps=eps)
        if graph_standardize
        else center_features(z)
    )

    affinity = gaussian_affinity(
        z_graph,
        eps=eps,
        standardize=False,
        bandwidth=bandwidth,
        k_nn=k_nn,
    )
    if float(affinity.sum().item()) <= eps:
        raise RuntimeError("Affinity graph has zero total weight.")

    laplacian = normalized_laplacian(affinity, eps=eps)
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    eigenvalues = eigenvalues.clamp_min(0.0)

    k_eff = min(
        int(harmonic_k),
        int(eigenvalues.numel() - 1),
    )
    if k_eff <= 0:
        raise RuntimeError("No nonconstant graph modes are available.")

    lambdas = eigenvalues[1 : 1 + k_eff]
    modes = eigenvectors[:, 1 : 1 + k_eff]
    coefficients = z_graph.T @ modes
    energies = (
        (coefficients ** 2).sum(dim=0)
        / max(int(z_graph.shape[1]), 1)
    )

    return (
        lambdas.detach().cpu().numpy().astype(np.float64),
        energies.detach().cpu().numpy().astype(np.float64),
    )


def score_from_ac(
    precision: np.ndarray | float,
    retained_energy: np.ndarray | float,
    laplacian_energy: np.ndarray | float,
) -> np.ndarray:
    raw = (
        np.asarray(precision, dtype=float)
        * np.asarray(retained_energy, dtype=float)
        + np.asarray(laplacian_energy, dtype=float)
    )
    return np.log1p(np.maximum(raw, 0.0))


# ---------------------------------------------------------------------------
# Paired finite-sample subsampling
# ---------------------------------------------------------------------------

def stratified_subsample(
    indices: np.ndarray,
    labels: np.ndarray,
    n_select: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.asarray(labels)
    cohort_labels = labels[indices]
    classes, counts = np.unique(cohort_labels, return_counts=True)

    expected = n_select * counts / counts.sum()
    allocation = np.floor(expected).astype(int)
    allocation = np.minimum(allocation, counts)

    remainder = int(n_select - allocation.sum())
    fractional_order = np.argsort(-(expected - allocation))

    for position in fractional_order:
        if remainder <= 0:
            break
        if allocation[position] < counts[position]:
            allocation[position] += 1
            remainder -= 1

    while remainder > 0:
        candidates = np.where(allocation < counts)[0]
        if len(candidates) == 0:
            break
        chosen = int(rng.choice(candidates))
        allocation[chosen] += 1
        remainder -= 1

    selected = []
    for class_value, amount in zip(classes, allocation):
        if amount <= 0:
            continue
        class_indices = indices[cohort_labels == class_value]
        selected.extend(
            rng.choice(
                class_indices,
                size=int(amount),
                replace=False,
            ).tolist()
        )

    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)

    if len(selected) != n_select:
        remaining = np.setdiff1d(indices, selected, assume_unique=False)
        needed = n_select - len(selected)
        if needed > 0:
            selected = np.concatenate(
                [
                    selected,
                    rng.choice(
                        remaining,
                        size=needed,
                        replace=False,
                    ),
                ]
            )
        elif needed < 0:
            selected = rng.choice(
                selected,
                size=n_select,
                replace=False,
            )

    return np.sort(selected.astype(int))


def deterministic_cohort(
    n_total: int,
    max_samples: int,
    *,
    labels: Optional[np.ndarray],
    stratified: bool,
    seed: int,
) -> np.ndarray:
    n_cohort = min(int(n_total), int(max_samples))
    all_indices = np.arange(n_total, dtype=int)

    if n_cohort == n_total:
        return all_indices

    rng = np.random.default_rng(seed)
    if stratified and labels is not None:
        return stratified_subsample(
            all_indices,
            labels,
            n_cohort,
            rng,
        )

    return np.sort(
        rng.choice(
            all_indices,
            size=n_cohort,
            replace=False,
        ).astype(int)
    )


def generate_view_indices(
    cohort: np.ndarray,
    *,
    fraction: float,
    n_views: int,
    labels: Optional[np.ndarray],
    stratified: bool,
    seed_prefix: Sequence[object],
    min_samples: int,
    view_keys: Optional[Sequence[object]] = None,
) -> List[np.ndarray]:
    cohort = np.asarray(cohort, dtype=int)
    if abs(float(fraction) - 1.0) <= 1e-12:
        return [cohort.copy() for _ in range(n_views)]

    n_select = int(round(len(cohort) * float(fraction)))
    n_select = max(int(min_samples), n_select)
    n_select = min(len(cohort), n_select)

    if view_keys is None:
        view_keys = tuple(range(n_views))
    if len(view_keys) != n_views:
        raise ValueError(
            f"Expected {n_views} view keys, received {len(view_keys)}."
        )

    output = []
    for view_id, view_key in enumerate(view_keys):
        rng = np.random.default_rng(
            stable_seed(*seed_prefix, "view", view_key)
        )
        if stratified and labels is not None:
            selected = stratified_subsample(
                cohort,
                labels,
                n_select,
                rng,
            )
        else:
            selected = np.sort(
                rng.choice(
                    cohort,
                    size=n_select,
                    replace=False,
                ).astype(int)
            )
        output.append(selected)

    return output


# ---------------------------------------------------------------------------
# Component-bank construction
# ---------------------------------------------------------------------------

def group_fraction_cache_path(
    cache_dir: Path,
    task: str,
    model: str,
    fraction: float,
) -> Path:
    return (
        cache_dir
        / safe_name(task)
        / safe_name(model)
        / f"fraction_{safe_name(format_fraction(fraction))}.csv"
    )


def build_group_fraction_bank(
    *,
    task: str,
    model: str,
    payload: Mapping[str, object],
    group_utility: pd.DataFrame,
    official_scores: Mapping[Tuple[str, str, int], float],
    fraction: float,
    cfg: Config,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hidden = np.asarray(payload["hidden_states"], dtype=np.float32)
    n_layers, n_total, dimension = hidden.shape
    labels = payload.get("labels")

    layer_meta = (
        group_utility[
            ["layer", "task_type", "utility"]
        ]
        .drop_duplicates("layer")
        .set_index("layer")
    )

    # Physical layer labels in the utility/metric tables are not necessarily
    # zero-based tensor indices. The exact project convention is to sort the
    # physical labels and map them positionally to the cached tensor axis.
    physical_layers = sorted(layer_meta.index.astype(int).tolist())
    if len(physical_layers) != int(n_layers):
        raise ValueError(
            f"Layer mapping mismatch for {model}/{task}: "
            f"utility_layers={physical_layers}, tensor_layers={n_layers}. "
            "The utility CSV and embedding cache must come from the same run."
        )
    if len(set(physical_layers)) != len(physical_layers):
        raise ValueError(
            f"Duplicate utility layer labels for {model}/{task}: "
            f"{physical_layers}"
        )
    layer_to_tensor_position = {
        int(layer_label): int(position)
        for position, layer_label in enumerate(physical_layers)
    }

    cohort = deterministic_cohort(
        n_total,
        cfg.max_samples,
        labels=labels,
        stratified=cfg.stratified_if_available,
        seed=stable_seed(
            cfg.seed,
            cfg.signature(),
            task,
            model,
            "cohort",
        ),
    )

    n_all_views = (
        cfg.uncertainty_views
        + cfg.selection_views
        + cfg.evaluation_views
    )
    roles = (
        ["uncertainty"] * cfg.uncertainty_views
        + ["selection"] * cfg.selection_views
        + ["evaluation"] * cfg.evaluation_views
    )
    uncertainty_seed_keys = [
        ("uncertainty", seed)
        for seed in cfg.subsample_seeds[: cfg.uncertainty_views]
    ]
    selection_seed_keys = [
        ("selection", cfg.seed, index)
        for index in range(cfg.selection_views)
    ]
    evaluation_seed_keys = [
        ("evaluation", seed)
        for seed in cfg.subsample_seeds[: cfg.evaluation_views]
    ]
    view_keys = (
        uncertainty_seed_keys
        + selection_seed_keys
        + evaluation_seed_keys
    )
    indices = generate_view_indices(
        cohort,
        fraction=fraction,
        n_views=n_all_views,
        labels=labels,
        stratified=cfg.stratified_if_available,
        seed_prefix=(
            cfg.seed,
            cfg.signature(),
            task,
            model,
            fraction,
        ),
        min_samples=cfg.min_samples,
        view_keys=view_keys,
    )

    role_ids = []
    counters = {"uncertainty": 0, "selection": 0, "evaluation": 0}
    for role in roles:
        role_ids.append(counters[role])
        counters[role] += 1

    bank_rows: List[dict] = []
    validation_rows: List[dict] = []
    base_precision = 1.0 / cfg.sigma0_l2

    for layer in physical_layers:
        tensor_position = layer_to_tensor_position[int(layer)]
        z_full = torch.from_numpy(hidden[tensor_position]).to(
            device=device,
            dtype=torch.float32,
        )

        if abs(float(fraction) - 1.0) <= 1e-12:
            key = (task, model, int(layer))
            if key not in official_scores:
                raise RuntimeError(f"Missing official score for {key}")

            lambdas_full, energies_full = official_components(
                z_full,
                harmonic_k=cfg.harmonic_k,
                graph_standardize=cfg.graph_standardize,
                bandwidth=cfg.bandwidth,
                k_nn=cfg.k_nn,
                eps=EPS,
            )
            reconstructed = float(
                score_from_ac(
                    base_precision,
                    float(np.sum(energies_full)),
                    float(np.dot(lambdas_full, energies_full)),
                )
            )
            saved = float(official_scores[key])
            absolute_error = abs(reconstructed - saved)

            if (
                not np.isfinite(absolute_error)
                or absolute_error > cfg.validation_tolerance
            ):
                raise RuntimeError(
                    "Official validation failed at "
                    f"{key}: reconstructed={reconstructed:.12g}, "
                    f"saved={saved:.12g}, error={absolute_error:.3e}"
                )

            validation_rows.append(
                {
                    "task": task,
                    "task_type": str(layer_meta.loc[layer, "task_type"]),
                    "model_alias": model,
                    "layer": int(layer),
                    "tensor_position": int(tensor_position),
                    "reconstructed": reconstructed,
                    "official_saved": saved,
                    "absolute_error": absolute_error,
                }
            )

        for global_view_id, (sample_indices, role, role_view_id) in enumerate(
            zip(indices, roles, role_ids)
        ):
            z_view = z_full[
                torch.as_tensor(
                    sample_indices,
                    dtype=torch.long,
                    device=device,
                )
            ]

            lambdas, energies = official_components(
                z_view,
                harmonic_k=cfg.harmonic_k,
                graph_standardize=cfg.graph_standardize,
                bandwidth=cfg.bandwidth,
                k_nn=cfg.k_nn,
                eps=EPS,
            )
            retained_energy_raw = float(np.sum(energies))
            laplacian_energy_raw = float(np.dot(lambdas, energies))

            if not np.isfinite(retained_energy_raw) or not np.isfinite(
                laplacian_energy_raw
            ):
                raise RuntimeError(
                    f"Non-finite harmonic components at "
                    f"{task}/{model}/L{layer}, fraction={fraction}, "
                    f"role={role}, view={role_view_id}: "
                    f"A={retained_energy_raw}, C={laplacian_energy_raw}"
                )

            # A near-zero retained energy is a scientifically valid outcome:
            # it indicates that this layer/view carries essentially no energy
            # in the retained nonconstant harmonic subspace. It must receive a
            # correspondingly small score rather than abort the experiment.
            # Tiny negative values can only be numerical round-off because E_m
            # and lambda_m are nonnegative in theory.
            retained_energy = max(retained_energy_raw, 0.0)
            laplacian_energy = max(laplacian_energy_raw, 0.0)
            is_degenerate = bool(
                retained_energy <= cfg.degenerate_threshold
            )

            bank_rows.append(
                {
                    "config_signature": cfg.signature(),
                    "task": task,
                    "task_type": str(layer_meta.loc[layer, "task_type"]),
                    "model_alias": model,
                    "layer": int(layer),
                    "tensor_position": int(tensor_position),
                    "utility": float(layer_meta.loc[layer, "utility"]),
                    "fraction": float(fraction),
                    "cohort_size": len(cohort),
                    "sample_size": len(sample_indices),
                    "n_total_samples": n_total,
                    "dimension": dimension,
                    "sample_hash": payload["sample_hash"],
                    "view_role": role,
                    "role_view_id": int(role_view_id),
                    "global_view_id": int(global_view_id),
                    "sample_index_hash": hashlib.sha256(
                        np.asarray(sample_indices, dtype=np.int64).tobytes()
                    ).hexdigest()[:16],
                    "retained_energy": retained_energy,
                    "laplacian_energy": laplacian_energy,
                    "is_degenerate_retained_energy": is_degenerate,
                    "fixed_score": float(
                        score_from_ac(
                            base_precision,
                            retained_energy,
                            laplacian_energy,
                        )
                    ),
                    "no_precision_score": float(
                        score_from_ac(
                            0.0,
                            retained_energy,
                            laplacian_energy,
                        )
                    ),
                    "mean_lambda": float(np.mean(lambdas)),
                    "lambda_min": float(np.min(lambdas)),
                    "lambda_max": float(np.max(lambdas)),
                }
            )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pd.DataFrame(bank_rows), pd.DataFrame(validation_rows)


def load_or_build_bank(
    cache: Mapping[Tuple[str, str], Dict[str, object]],
    utility: pd.DataFrame,
    official_scores: Mapping[Tuple[str, str, int], float],
    *,
    cfg: Config,
    cache_dir: Path,
    rebuild: bool,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bank_parts = []
    validation_parts = []

    groups = list(
        utility.groupby(["task", "model_alias"], sort=True)
    )
    total = len(groups) * len(cfg.fractions)
    counter = 0

    for (task, model), group_utility in groups:
        key = (str(task), str(model))
        if key not in cache:
            print(f"[skip] representation cache missing for {key}")
            continue

        for fraction in cfg.fractions:
            counter += 1
            path = group_fraction_cache_path(
                cache_dir,
                str(task),
                str(model),
                float(fraction),
            )
            meta_path = path.with_suffix(".meta.json")
            validation_path = path.with_name(
                path.stem + "__validation.csv"
            )

            use_cache = False
            if (
                not rebuild
                and path.exists()
                and meta_path.exists()
                and validation_path.exists()
            ):
                try:
                    meta = json.loads(
                        meta_path.read_text(encoding="utf-8")
                    )
                    use_cache = (
                        meta.get("config_signature")
                        == cfg.signature()
                    )
                except Exception:
                    use_cache = False

            print(
                f"[{counter}/{total}] {task} / {model} / "
                f"fraction={fraction:g}: "
                f"{'cached' if use_cache else 'computing'}"
            )

            if use_cache:
                bank = pd.read_csv(path)
                validation = pd.read_csv(validation_path)
            else:
                bank, validation = build_group_fraction_bank(
                    task=str(task),
                    model=str(model),
                    payload=cache[key],
                    group_utility=group_utility,
                    official_scores=official_scores,
                    fraction=float(fraction),
                    cfg=cfg,
                    device=device,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                bank.to_csv(path, index=False)
                validation.to_csv(validation_path, index=False)
                write_json(
                    meta_path,
                    {
                        "config_signature": cfg.signature(),
                        "n_bank_rows": len(bank),
                        "n_validation_rows": len(validation),
                    },
                )

            bank_parts.append(bank)
            if not validation.empty:
                validation_parts.append(validation)

    if not bank_parts:
        raise RuntimeError("No component bank could be constructed.")

    validation = (
        pd.concat(validation_parts, ignore_index=True)
        if validation_parts
        else pd.DataFrame()
    )

    return pd.concat(bank_parts, ignore_index=True), validation


# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------

def robust_log_energy_variance(values: np.ndarray) -> Tuple[float, float, float]:
    """Scale-invariant robust dispersion of retained harmonic energy.

    The former log(max(A, eps)) transform can turn harmless floating-point
    fluctuations around zero into an artificially enormous uncertainty.  We
    instead normalize each layer by its own typical positive energy and use
    log1p(A/scale).  Consequently:

    * an identically collapsed layer has uncertainty 0;
    * a consistently tiny layer is not penalized merely for its magnitude;
    * genuine relative variation across subsamples is still measured.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.maximum(values, 0.0)

    if len(values) <= 1:
        return 0.0, 0.0, 0.0

    positive = values[values > EPS]
    if len(positive) == 0:
        return 0.0, 0.0, 0.0

    scale = float(np.median(positive))
    if scale <= EPS:
        scale = float(np.mean(positive))
    scale = max(scale, EPS)

    x = np.log1p(values / scale)
    if np.std(x) <= EPS:
        return 0.0, 0.0, 0.0

    classical = float(np.var(x, ddof=1))
    median = float(np.median(x))
    mad_scale = 1.4826 * float(np.median(np.abs(x - median)))
    robust = mad_scale ** 2
    combined = 0.5 * classical + 0.5 * robust

    return combined, classical, robust


def estimate_uncertainty(
    bank: pd.DataFrame,
    *,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    validity_rows = []

    grouping = ["task", "task_type", "model_alias", "fraction"]

    for group_key, group in bank.groupby(grouping, sort=True):
        task, task_type, model, fraction = group_key
        layer_rows = []

        for layer, layer_group in group.groupby("layer", sort=True):
            uncertainty_views = layer_group[
                layer_group["view_role"] == "uncertainty"
            ].sort_values("role_view_id")
            evaluation_views = layer_group[
                layer_group["view_role"] == "evaluation"
            ].sort_values("role_view_id")

            combined, classical, robust = robust_log_energy_variance(
                uncertainty_views["retained_energy"].to_numpy(dtype=float)
            )
            heldout_energy_var, _, _ = robust_log_energy_variance(
                evaluation_views["retained_energy"].to_numpy(dtype=float)
            )
            heldout_score_var = float(
                np.var(
                    evaluation_views["fixed_score"].to_numpy(dtype=float),
                    ddof=1,
                )
            ) if len(evaluation_views) > 1 else 0.0

            layer_rows.append(
                {
                    "task": task,
                    "task_type": task_type,
                    "model_alias": model,
                    "fraction": float(fraction),
                    "layer": int(layer),
                    "utility": float(layer_group["utility"].iloc[0]),
                    "raw_uncertainty": combined,
                    "classical_log_energy_variance": classical,
                    "robust_log_energy_variance": robust,
                    "heldout_log_energy_variance": heldout_energy_var,
                    "heldout_fixed_score_variance": heldout_score_var,
                    "degenerate_view_fraction": float(
                        layer_group[
                            "is_degenerate_retained_energy"
                        ].astype(float).mean()
                    )
                    if "is_degenerate_retained_energy" in layer_group.columns
                    else 0.0,
                }
            )

        layer_frame = pd.DataFrame(layer_rows)
        median_raw = float(np.median(layer_frame["raw_uncertainty"]))
        weight = (
            cfg.uncertainty_views
            / (cfg.uncertainty_views + cfg.shrinkage_strength)
        )

        layer_frame["shrunk_uncertainty"] = (
            weight * layer_frame["raw_uncertainty"]
            + (1.0 - weight) * median_raw
        )

        median_shrunk = float(
            np.median(layer_frame["shrunk_uncertainty"])
        )

        if (
            abs(float(fraction) - 1.0) <= 1e-12
            or median_shrunk <= EPS
        ):
            layer_frame["relative_uncertainty"] = 0.0
        else:
            layer_frame["relative_uncertainty"] = (
                layer_frame["shrunk_uncertainty"]
                / median_shrunk
            )

        rows.extend(layer_frame.to_dict("records"))

        validity_rows.append(
            {
                "task": task,
                "task_type": task_type,
                "model_alias": model,
                "fraction": float(fraction),
                "n_layers": len(layer_frame),
                "spearman_estimated_vs_heldout_energy_variance": safe_spearman(
                    layer_frame["relative_uncertainty"],
                    layer_frame["heldout_log_energy_variance"],
                ),
                "spearman_estimated_vs_heldout_score_variance": safe_spearman(
                    layer_frame["relative_uncertainty"],
                    layer_frame["heldout_fixed_score_variance"],
                ),
                "median_relative_uncertainty": float(
                    np.median(layer_frame["relative_uncertainty"])
                ),
                "relative_uncertainty_cv": float(
                    np.std(layer_frame["relative_uncertainty"])
                    / max(
                        np.mean(layer_frame["relative_uncertainty"]),
                        EPS,
                    )
                ),
                "relative_uncertainty_range": float(
                    layer_frame["relative_uncertainty"].max()
                    - layer_frame["relative_uncertainty"].min()
                ),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(validity_rows)


# ---------------------------------------------------------------------------
# Ranking and shortlist metrics
# ---------------------------------------------------------------------------

def pairwise_preference_accuracy(
    scores: np.ndarray,
    utility: np.ndarray,
    tolerance: float = 1e-12,
) -> float:
    outcomes = []
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            utility_difference = utility[left] - utility[right]
            if abs(utility_difference) <= tolerance:
                continue
            score_difference = scores[left] - scores[right]
            if abs(score_difference) <= tolerance:
                outcomes.append(0.5)
            else:
                outcomes.append(
                    float(
                        np.sign(score_difference)
                        == np.sign(utility_difference)
                    )
                )
    return float(np.mean(outcomes)) if outcomes else np.nan


def ndcg_at_k(
    scores: np.ndarray,
    utility: np.ndarray,
    k: int,
) -> float:
    scores = np.asarray(scores, dtype=float)
    utility = np.asarray(utility, dtype=float)
    k_eff = min(int(k), len(scores))
    utility_range = float(np.max(utility) - np.min(utility))
    if utility_range <= EPS:
        return 1.0

    relevance = (
        utility - float(np.min(utility))
    ) / utility_range
    predicted = np.argsort(-scores)[:k_eff]
    ideal = np.argsort(-utility)[:k_eff]
    discounts = 1.0 / np.log2(
        np.arange(2, k_eff + 2, dtype=float)
    )
    dcg = (
        (2.0 ** relevance[predicted] - 1.0)
        * discounts
    ).sum()
    idcg = (
        (2.0 ** relevance[ideal] - 1.0)
        * discounts
    ).sum()
    return float(dcg / idcg) if idcg > EPS else 1.0


def regret_at_1(
    scores: np.ndarray,
    utility: np.ndarray,
) -> float:
    selected = int(np.argmax(scores))
    best = float(np.max(utility))
    worst = float(np.min(utility))
    scale = best - worst
    if scale <= EPS:
        return 0.0
    return float(
        (best - float(utility[selected])) / scale
    )


def coverage_regret_at_k(
    scores: np.ndarray,
    utility: np.ndarray,
    k: int,
) -> float:
    selected = np.argsort(-scores)[: min(int(k), len(scores))]
    best = float(np.max(utility))
    worst = float(np.min(utility))
    scale = best - worst
    if scale <= EPS:
        return 0.0
    return float(
        (best - float(np.max(utility[selected]))) / scale
    )


def near_oracle_hit(
    scores: np.ndarray,
    utility: np.ndarray,
    k: int,
    tolerance_fraction: float = 0.05,
) -> float:
    selected = np.argsort(-scores)[: min(int(k), len(scores))]
    best = float(np.max(utility))
    worst = float(np.min(utility))
    scale = best - worst
    if scale <= EPS:
        return 1.0
    threshold = best - tolerance_fraction * scale
    return float(np.any(utility[selected] >= threshold))


def mean_pairwise_rank_reliability(
    score_matrix: np.ndarray,
) -> float:
    values = []
    for left in range(score_matrix.shape[0]):
        for right in range(left + 1, score_matrix.shape[0]):
            values.append(
                safe_spearman(
                    score_matrix[left],
                    score_matrix[right],
                )
            )
    return float(np.nanmean(values)) if values else np.nan


def mean_pairwise_jaccard(
    score_matrix: np.ndarray,
    k: int,
) -> float:
    values = []
    k_eff = min(int(k), score_matrix.shape[1])

    for left in range(score_matrix.shape[0]):
        left_set = set(
            np.argsort(-score_matrix[left])[:k_eff].tolist()
        )
        for right in range(left + 1, score_matrix.shape[0]):
            right_set = set(
                np.argsort(-score_matrix[right])[:k_eff].tolist()
            )
            union = left_set | right_set
            values.append(
                len(left_set & right_set) / len(union)
                if union
                else 1.0
            )

    return float(np.mean(values)) if values else np.nan


def top1_agreement(score_matrix: np.ndarray) -> float:
    top = np.argmax(score_matrix, axis=1)
    values = []
    for left in range(len(top)):
        for right in range(left + 1, len(top)):
            values.append(float(top[left] == top[right]))
    return float(np.mean(values)) if values else np.nan


def evaluate_score_matrix(
    score_matrix: np.ndarray,
    utility: np.ndarray,
    *,
    top_k: int,
) -> Dict[str, float]:
    view_rows = []

    for scores in score_matrix:
        view_rows.append(
            {
                "utility_spearman": safe_spearman(scores, utility),
                "preference_accuracy": pairwise_preference_accuracy(
                    scores,
                    utility,
                ),
                "ndcg_at_k": ndcg_at_k(scores, utility, top_k),
                "regret_at_1": regret_at_1(scores, utility),
                "coverage_regret_at_k": coverage_regret_at_k(
                    scores,
                    utility,
                    top_k,
                ),
                "near_oracle_hit_5pct": near_oracle_hit(
                    scores,
                    utility,
                    top_k,
                    tolerance_fraction=0.05,
                ),
            }
        )

    return {
        key: float(np.nanmean([row[key] for row in view_rows]))
        for key in view_rows[0]
    } | {
        "rank_reliability": mean_pairwise_rank_reliability(
            score_matrix
        ),
        "shortlist_jaccard_at_k": mean_pairwise_jaccard(
            score_matrix,
            top_k,
        ),
        "top1_agreement": top1_agreement(score_matrix),
    }


# ---------------------------------------------------------------------------
# Alpha selection and evaluation
# ---------------------------------------------------------------------------

def score_matrix_for_group(
    group: pd.DataFrame,
    uncertainty: pd.DataFrame,
    *,
    role: str,
    alpha: float,
    sigma0_l2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    role_group = group[
        group["view_role"] == role
    ].copy()

    layers = np.sort(role_group["layer"].unique().astype(int))
    view_ids = np.sort(
        role_group["role_view_id"].unique().astype(int)
    )

    uncertainty_lookup = (
        uncertainty.set_index("layer")["relative_uncertainty"]
        .to_dict()
    )
    precision_lookup = {
        int(layer): (
            (1.0 / sigma0_l2)
            / (
                1.0
                + float(alpha)
                * float(uncertainty_lookup[int(layer)])
            )
        )
        for layer in layers
    }

    matrix = np.zeros((len(view_ids), len(layers)), dtype=float)

    for i, view_id in enumerate(view_ids):
        view = (
            role_group[
                role_group["role_view_id"] == view_id
            ]
            .set_index("layer")
            .loc[layers]
        )
        precision = np.asarray(
            [precision_lookup[int(layer)] for layer in layers],
            dtype=float,
        )
        matrix[i] = score_from_ac(
            precision,
            view["retained_energy"].to_numpy(dtype=float),
            view["laplacian_energy"].to_numpy(dtype=float),
        )

    utility = (
        role_group[
            ["layer", "utility"]
        ]
        .drop_duplicates("layer")
        .set_index("layer")
        .loc[layers, "utility"]
        .to_numpy(dtype=float)
    )

    return matrix, layers, utility


def selection_objective(
    score_matrix: np.ndarray,
    top_k: int,
) -> Dict[str, float]:
    rank = mean_pairwise_rank_reliability(score_matrix)
    jaccard = mean_pairwise_jaccard(score_matrix, top_k)
    top1 = top1_agreement(score_matrix)

    return {
        "rank_reliability": rank,
        "shortlist_jaccard_at_k": jaccard,
        "top1_agreement": top1,
        "selection_objective": (
            0.5 * rank
            + 0.4 * jaccard
            + 0.1 * top1
        ),
    }


def compute_selection_metrics(
    bank: pd.DataFrame,
    uncertainty: pd.DataFrame,
    *,
    cfg: Config,
) -> pd.DataFrame:
    rows = []
    grouping = ["task", "task_type", "model_alias", "fraction"]

    for group_key, group in bank.groupby(grouping, sort=True):
        task, task_type, model, fraction = group_key
        uncertainty_group = uncertainty[
            (uncertainty["task"] == task)
            & (uncertainty["model_alias"] == model)
            & (uncertainty["fraction"] == fraction)
        ]

        for alpha in cfg.alpha_grid:
            matrix, _, _ = score_matrix_for_group(
                group,
                uncertainty_group,
                role="selection",
                alpha=float(alpha),
                sigma0_l2=cfg.sigma0_l2,
            )
            objective = selection_objective(
                matrix,
                cfg.top_k,
            )

            rows.append(
                {
                    "task": task,
                    "task_type": task_type,
                    "model_alias": model,
                    "fraction": float(fraction),
                    "alpha": float(alpha),
                    **objective,
                }
            )

    return pd.DataFrame(rows)


def choose_loto_alpha(
    selection_metrics: pd.DataFrame,
    *,
    cfg: Config,
) -> pd.DataFrame:
    task_macro = (
        selection_metrics.groupby(
            ["task", "task_type", "fraction", "alpha"],
            as_index=False,
        )[
            [
                "rank_reliability",
                "shortlist_jaccard_at_k",
                "top1_agreement",
                "selection_objective",
            ]
        ]
        .mean()
    )

    rows = []
    tasks = sorted(task_macro["task"].unique())

    for fraction in sorted(task_macro["fraction"].unique()):
        fraction_frame = task_macro[
            task_macro["fraction"] == fraction
        ]

        for heldout_task in tasks:
            training = fraction_frame[
                fraction_frame["task"] != heldout_task
            ]
            if training.empty:
                raise RuntimeError(
                    "LOTO alpha selection requires at least two tasks."
                )

            summary = (
                training.groupby("alpha", as_index=False)
                .agg(
                    mean_selection_objective=(
                        "selection_objective",
                        "mean",
                    ),
                    mean_rank_reliability=(
                        "rank_reliability",
                        "mean",
                    ),
                    mean_shortlist_jaccard=(
                        "shortlist_jaccard_at_k",
                        "mean",
                    ),
                    n_training_tasks=("task", "nunique"),
                )
            )

            best_value = float(
                summary["mean_selection_objective"].max()
            )
            tolerance = 1e-10
            candidates = summary[
                summary["mean_selection_objective"]
                >= best_value - tolerance
            ].sort_values("alpha")

            chosen = candidates.iloc[0]

            rows.append(
                {
                    "heldout_task": heldout_task,
                    "fraction": float(fraction),
                    "selected_alpha": float(chosen["alpha"]),
                    "training_objective": float(
                        chosen["mean_selection_objective"]
                    ),
                    "training_rank_reliability": float(
                        chosen["mean_rank_reliability"]
                    ),
                    "training_shortlist_jaccard": float(
                        chosen["mean_shortlist_jaccard"]
                    ),
                    "n_training_tasks": int(
                        chosen["n_training_tasks"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def evaluate_methods(
    bank: pd.DataFrame,
    uncertainty: pd.DataFrame,
    alpha_choices: pd.DataFrame,
    *,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    candidate_rows = []
    grouping = ["task", "task_type", "model_alias", "fraction"]

    for group_key, group in bank.groupby(grouping, sort=True):
        task, task_type, model, fraction = group_key
        uncertainty_group = uncertainty[
            (uncertainty["task"] == task)
            & (uncertainty["model_alias"] == model)
            & (uncertainty["fraction"] == fraction)
        ].sort_values("layer")

        choice = alpha_choices[
            (alpha_choices["heldout_task"] == task)
            & (alpha_choices["fraction"] == fraction)
        ]
        if choice.empty:
            raise RuntimeError(
                f"Missing LOTO alpha for task={task}, fraction={fraction}"
            )
        selected_alpha = float(choice["selected_alpha"].iloc[0])

        adaptive_matrix, layers, utility = score_matrix_for_group(
            group,
            uncertainty_group,
            role="evaluation",
            alpha=selected_alpha,
            sigma0_l2=cfg.sigma0_l2,
        )
        alpha1_matrix, _, _ = score_matrix_for_group(
            group,
            uncertainty_group,
            role="evaluation",
            alpha=1.0,
            sigma0_l2=cfg.sigma0_l2,
        )
        fixed_matrix, _, _ = score_matrix_for_group(
            group,
            uncertainty_group,
            role="evaluation",
            alpha=0.0,
            sigma0_l2=cfg.sigma0_l2,
        )

        evaluation_group = group[
            group["view_role"] == "evaluation"
        ]
        view_ids = np.sort(
            evaluation_group["role_view_id"].unique().astype(int)
        )
        no_precision_matrix = np.zeros_like(fixed_matrix)

        for i, view_id in enumerate(view_ids):
            view = (
                evaluation_group[
                    evaluation_group["role_view_id"] == view_id
                ]
                .set_index("layer")
                .loc[layers]
            )
            no_precision_matrix[i] = score_from_ac(
                0.0,
                view["retained_energy"].to_numpy(dtype=float),
                view["laplacian_energy"].to_numpy(dtype=float),
            )

        methods = {
            "adaptive_loto": adaptive_matrix,
            "adaptive_alpha_1": alpha1_matrix,
            "fixed_precision": fixed_matrix,
            "no_precision": no_precision_matrix,
        }

        relative_uncertainty = (
            uncertainty_group.set_index("layer")
            .loc[layers, "relative_uncertainty"]
            .to_numpy(dtype=float)
        )
        adaptive_precision = (
            (1.0 / cfg.sigma0_l2)
            / (1.0 + selected_alpha * relative_uncertainty)
        )

        shuffled_metric_rows = []
        shuffled_matrix_sum = np.zeros_like(adaptive_matrix)

        for shuffle_id in range(cfg.precision_shuffles):
            rng = np.random.default_rng(
                stable_seed(
                    cfg.seed,
                    cfg.signature(),
                    task,
                    model,
                    fraction,
                    "precision_shuffle",
                    shuffle_id,
                )
            )
            shuffled_precision = adaptive_precision[
                rng.permutation(len(adaptive_precision))
            ]
            shuffled_matrix = np.zeros_like(adaptive_matrix)

            for i, view_id in enumerate(view_ids):
                view = (
                    evaluation_group[
                        evaluation_group["role_view_id"] == view_id
                    ]
                    .set_index("layer")
                    .loc[layers]
                )
                shuffled_matrix[i] = score_from_ac(
                    shuffled_precision,
                    view["retained_energy"].to_numpy(dtype=float),
                    view["laplacian_energy"].to_numpy(dtype=float),
                )

            shuffled_matrix_sum += shuffled_matrix
            shuffled_metric_rows.append(
                evaluate_score_matrix(
                    shuffled_matrix,
                    utility,
                    top_k=cfg.top_k,
                )
            )

        methods["shuffled_precision"] = (
            shuffled_matrix_sum / cfg.precision_shuffles
        )

        for method, matrix in methods.items():
            if method == "shuffled_precision":
                metrics = {
                    outcome: float(
                        np.nanmean(
                            [
                                row[outcome]
                                for row in shuffled_metric_rows
                            ]
                        )
                    )
                    for outcome in OUTCOMES
                }
            else:
                metrics = evaluate_score_matrix(
                    matrix,
                    utility,
                    top_k=cfg.top_k,
                )

            metric_rows.append(
                {
                    "task": task,
                    "task_type": task_type,
                    "model_alias": model,
                    "fraction": float(fraction),
                    "sample_size": int(
                        evaluation_group["sample_size"].iloc[0]
                    ),
                    "method": method,
                    "selected_alpha": (
                        selected_alpha
                        if method
                        in (
                            "adaptive_loto",
                            "shuffled_precision",
                        )
                        else (
                            1.0
                            if method == "adaptive_alpha_1"
                            else (
                                0.0
                                if method == "fixed_precision"
                                else np.nan
                            )
                        )
                    ),
                    **metrics,
                }
            )

            mean_scores = np.mean(matrix, axis=0)
            for index, layer in enumerate(layers):
                candidate_rows.append(
                    {
                        "task": task,
                        "task_type": task_type,
                        "model_alias": model,
                        "fraction": float(fraction),
                        "method": method,
                        "layer": int(layer),
                        "utility": float(utility[index]),
                        "mean_score": float(mean_scores[index]),
                        "relative_uncertainty": float(
                            relative_uncertainty[index]
                        ),
                        "selected_alpha": (
                            selected_alpha
                            if method
                            in (
                                "adaptive_loto",
                                "shuffled_precision",
                            )
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(candidate_rows)



# ---------------------------------------------------------------------------
# Final paper method evaluation (fixed alpha=1; no label-based tuning)
# ---------------------------------------------------------------------------

def evaluate_paper_methods(
    bank: pd.DataFrame,
    uncertainty: pd.DataFrame,
    *,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[dict] = []
    candidate_rows: List[dict] = []
    grouping = ["task", "task_type", "model_alias", "fraction"]
    base_precision = 1.0 / cfg.sigma0_l2

    for group_key, group in bank.groupby(grouping, sort=True):
        task, task_type, model, fraction = group_key
        uncertainty_group = uncertainty[
            (uncertainty["task"] == task)
            & (uncertainty["model_alias"] == model)
            & np.isclose(uncertainty["fraction"], fraction)
        ].sort_values("layer")

        evaluation_group = group[
            group["view_role"] == "evaluation"
        ].copy()
        layers = np.sort(evaluation_group["layer"].unique().astype(int))
        view_ids = np.sort(
            evaluation_group["role_view_id"].unique().astype(int)
        )

        if len(view_ids) < 2:
            raise RuntimeError(
                f"At least two evaluation views are required for {task}/{model}, "
                f"fraction={fraction}."
            )

        uncertainty_lookup = (
            uncertainty_group.set_index("layer")["relative_uncertainty"]
            .to_dict()
        )
        relative_uncertainty = np.asarray(
            [float(uncertainty_lookup[int(layer)]) for layer in layers],
            dtype=float,
        )
        adaptive_precision = (
            base_precision / (1.0 + relative_uncertainty)
        )
        fixed_precision = np.full(len(layers), base_precision, dtype=float)
        no_precision = np.zeros(len(layers), dtype=float)

        utility = (
            evaluation_group[["layer", "utility"]]
            .drop_duplicates("layer")
            .set_index("layer")
            .loc[layers, "utility"]
            .to_numpy(dtype=float)
        )

        def matrix_from_precision(precision: np.ndarray) -> np.ndarray:
            matrix = np.zeros((len(view_ids), len(layers)), dtype=float)
            for row_index, view_id in enumerate(view_ids):
                view = (
                    evaluation_group[
                        evaluation_group["role_view_id"] == view_id
                    ]
                    .set_index("layer")
                    .loc[layers]
                )
                matrix[row_index] = score_from_ac(
                    precision,
                    view["retained_energy"].to_numpy(dtype=float),
                    view["laplacian_energy"].to_numpy(dtype=float),
                )
            return matrix

        matrices: Dict[str, np.ndarray] = {
            "adaptive_precision": matrix_from_precision(adaptive_precision),
            "fixed_precision": matrix_from_precision(fixed_precision),
            "no_precision": matrix_from_precision(no_precision),
        }

        shuffled_metric_rows: List[Dict[str, float]] = []
        shuffled_matrix_sum = np.zeros_like(matrices["adaptive_precision"])

        for shuffle_id in range(cfg.precision_shuffles):
            rng = np.random.default_rng(
                stable_seed(
                    cfg.seed,
                    cfg.signature(),
                    task,
                    model,
                    fraction,
                    "paper_precision_shuffle",
                    shuffle_id,
                )
            )
            shuffled = adaptive_precision[
                rng.permutation(len(adaptive_precision))
            ]
            shuffled_matrix = matrix_from_precision(shuffled)
            shuffled_matrix_sum += shuffled_matrix
            shuffled_metric_rows.append(
                evaluate_score_matrix(
                    shuffled_matrix,
                    utility,
                    top_k=cfg.top_k,
                )
            )

        matrices["shuffled_precision"] = (
            shuffled_matrix_sum / cfg.precision_shuffles
        )

        for method, matrix in matrices.items():
            if method == "shuffled_precision":
                metrics = {
                    outcome: float(
                        np.nanmean(
                            [row[outcome] for row in shuffled_metric_rows]
                        )
                    )
                    for outcome in OUTCOMES
                }
            else:
                metrics = evaluate_score_matrix(
                    matrix,
                    utility,
                    top_k=cfg.top_k,
                )

            metric_rows.append(
                {
                    "task": task,
                    "task_type": task_type,
                    "model_alias": model,
                    "fraction": float(fraction),
                    "sample_size": int(
                        evaluation_group["sample_size"].iloc[0]
                    ),
                    "method": method,
                    **metrics,
                }
            )

            mean_scores = np.mean(matrix, axis=0)
            for layer_index, layer in enumerate(layers):
                candidate_rows.append(
                    {
                        "task": task,
                        "task_type": task_type,
                        "model_alias": model,
                        "fraction": float(fraction),
                        "method": method,
                        "layer": int(layer),
                        "utility": float(utility[layer_index]),
                        "mean_score": float(mean_scores[layer_index]),
                        "relative_uncertainty": float(
                            relative_uncertainty[layer_index]
                        ),
                        "adaptive_precision": float(
                            adaptive_precision[layer_index]
                        ),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(candidate_rows)

# ---------------------------------------------------------------------------
# Aggregation and inference
# ---------------------------------------------------------------------------

def aggregate_metrics(
    evaluation: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = list(OUTCOMES)

    model_task = (
        evaluation.groupby(
            [
                "task",
                "task_type",
                "model_alias",
                "fraction",
                "method",
            ],
            as_index=False,
        )[outcomes]
        .mean()
    )

    task_macro = (
        model_task.groupby(
            [
                "task",
                "task_type",
                "fraction",
                "method",
            ],
            as_index=False,
        )[outcomes]
        .mean()
    )

    aggregations = {
        "n_tasks": ("task", "nunique"),
    }
    for outcome in outcomes:
        aggregations[f"mean_{outcome}"] = (
            outcome,
            "mean",
        )
        aggregations[f"median_{outcome}"] = (
            outcome,
            "median",
        )

    method_summary = (
        task_macro.groupby(
            ["fraction", "method"],
            as_index=False,
        )
        .agg(**aggregations)
    )

    return model_task, task_macro, method_summary


def exact_sign_flip_pvalue(
    differences: np.ndarray,
) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    differences = differences[np.abs(differences) > EPS]
    n = len(differences)

    if n == 0:
        return np.nan

    observed = abs(float(np.mean(differences)))
    absolute = np.abs(differences)

    if n <= 20:
        exceed = 0
        total = 2 ** n
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            value = abs(
                float(
                    np.mean(
                        absolute * np.asarray(signs, dtype=float)
                    )
                )
            )
            exceed += int(value >= observed - 1e-15)
        return float(exceed / total)

    rng = np.random.default_rng(20260627)
    n_draws = 200000
    exceed = 0
    for _ in range(n_draws):
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=n,
        )
        value = abs(float(np.mean(absolute * signs)))
        exceed += int(value >= observed - 1e-15)
    return float((exceed + 1) / (n_draws + 1))


def bootstrap_ci(
    differences: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    samples = rng.choice(
        differences,
        size=(n_bootstrap, len(differences)),
        replace=True,
    )
    means = samples.mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    finite = np.where(np.isfinite(p_values))[0]
    if len(finite) == 0:
        return adjusted

    order = finite[np.argsort(p_values[finite])]
    m = len(order)
    running = 0.0

    for rank, original_index in enumerate(order):
        candidate = (
            (m - rank) * p_values[original_index]
        )
        running = max(running, candidate)
        adjusted[original_index] = min(running, 1.0)

    return adjusted


def paired_task_tests(
    task_macro: pd.DataFrame,
    *,
    cfg: Config,
) -> pd.DataFrame:
    rows = []

    for fraction in sorted(task_macro["fraction"].unique()):
        fraction_frame = task_macro[
            task_macro["fraction"] == fraction
        ]

        for outcome, higher_is_better in OUTCOMES.items():
            wide = fraction_frame.pivot(
                index="task",
                columns="method",
                values=outcome,
            )

            for baseline in BASELINES:
                if (
                    PRIMARY_METHOD not in wide.columns
                    or baseline not in wide.columns
                ):
                    continue

                pair = wide[
                    [PRIMARY_METHOD, baseline]
                ].dropna()

                if higher_is_better:
                    difference = (
                        pair[PRIMARY_METHOD].to_numpy(dtype=float)
                        - pair[baseline].to_numpy(dtype=float)
                    )
                else:
                    difference = (
                        pair[baseline].to_numpy(dtype=float)
                        - pair[PRIMARY_METHOD].to_numpy(dtype=float)
                    )

                ci_low, ci_high = bootstrap_ci(
                    difference,
                    n_bootstrap=cfg.task_bootstrap,
                    seed=stable_seed(
                        cfg.seed,
                        fraction,
                        outcome,
                        baseline,
                    ),
                )

                nonzero = difference[
                    np.abs(difference) > EPS
                ]
                wins = int(np.sum(nonzero > 0))
                losses = int(np.sum(nonzero < 0))
                ties = int(len(difference) - len(nonzero))
                sign_test = (
                    float(
                        binomtest(
                            min(wins, losses),
                            n=wins + losses,
                            p=0.5,
                            alternative="two-sided",
                        ).pvalue
                    )
                    if wins + losses > 0
                    else np.nan
                )

                rows.append(
                    {
                        "fraction": float(fraction),
                        "outcome": outcome,
                        "primary_method": PRIMARY_METHOD,
                        "baseline": baseline,
                        "n_tasks": len(difference),
                        "mean_advantage": (
                            float(np.mean(difference))
                            if len(difference)
                            else np.nan
                        ),
                        "median_advantage": (
                            float(np.median(difference))
                            if len(difference)
                            else np.nan
                        ),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "task_wins": wins,
                        "task_losses": losses,
                        "task_ties": ties,
                        "exact_sign_flip_p": (
                            exact_sign_flip_pvalue(difference)
                        ),
                        "sign_test_p": sign_test,
                    }
                )

    result = pd.DataFrame(rows)

    if not result.empty:
        for keys, indices in result.groupby(
            ["fraction", "outcome"]
        ).groups.items():
            indices = list(indices)
            result.loc[
                indices,
                "exact_sign_flip_p_holm",
            ] = holm_adjust(
                result.loc[
                    indices,
                    "exact_sign_flip_p",
                ].to_numpy()
            )
            result.loc[
                indices,
                "sign_test_p_holm",
            ] = holm_adjust(
                result.loc[
                    indices,
                    "sign_test_p",
                ].to_numpy()
            )

    return result


def negative_control_audit(
    method_summary: pd.DataFrame,
) -> pd.DataFrame:
    full = method_summary[
        np.isclose(method_summary["fraction"], 1.0)
    ].copy()

    rows = []
    if full.empty:
        return pd.DataFrame(rows)

    adaptive = full[
        full["method"] == PRIMARY_METHOD
    ]
    fixed = full[
        full["method"] == "fixed_precision"
    ]
    if adaptive.empty or fixed.empty:
        return pd.DataFrame(rows)

    adaptive_row = adaptive.iloc[0]
    fixed_row = fixed.iloc[0]

    for outcome in OUTCOMES:
        adaptive_value = float(
            adaptive_row[f"mean_{outcome}"]
        )
        fixed_value = float(
            fixed_row[f"mean_{outcome}"]
        )
        rows.append(
            {
                "outcome": outcome,
                "adaptive_value": adaptive_value,
                "fixed_value": fixed_value,
                "absolute_difference": abs(
                    adaptive_value - fixed_value
                ),
                "passed": abs(
                    adaptive_value - fixed_value
                ) <= 1e-10,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Automatic verdict
# ---------------------------------------------------------------------------

def task_level_validity_summary(
    validity: pd.DataFrame,
) -> pd.DataFrame:
    return (
        validity.groupby(
            ["task", "task_type", "fraction"],
            as_index=False,
        )[
            [
                "spearman_estimated_vs_heldout_energy_variance",
                "spearman_estimated_vs_heldout_score_variance",
            ]
        ]
        .mean()
    )


def validity_inference(
    validity: pd.DataFrame,
    *,
    cfg: Config,
) -> pd.DataFrame:
    task_level = task_level_validity_summary(validity)
    rows = []

    for fraction, group in task_level.groupby("fraction", sort=True):
        for outcome in (
            "spearman_estimated_vs_heldout_energy_variance",
            "spearman_estimated_vs_heldout_score_variance",
        ):
            values = group[outcome].dropna().to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci(
                values,
                n_bootstrap=cfg.task_bootstrap,
                seed=stable_seed(
                    cfg.seed,
                    "validity",
                    fraction,
                    outcome,
                ),
            )
            rows.append(
                {
                    "fraction": float(fraction),
                    "outcome": outcome,
                    "n_tasks": len(values),
                    "mean_correlation": (
                        float(np.mean(values))
                        if len(values)
                        else np.nan
                    ),
                    "median_correlation": (
                        float(np.median(values))
                        if len(values)
                        else np.nan
                    ),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                }
            )

    return pd.DataFrame(rows)


def build_verdict(
    tests: pd.DataFrame,
    validity_tests: pd.DataFrame,
    negative_control: pd.DataFrame,
    *,
    cfg: Config,
) -> Dict[str, object]:
    non_full = sorted(
        float(value)
        for value in tests["fraction"].unique()
        if float(value) < 1.0 - 1e-12
    )
    if not non_full:
        return {
            "overall_precision_role_supported": False,
            "interpretation": "No finite-sample fraction was available.",
        }

    low_fraction = min(non_full)

    def test_row(fraction: float, outcome: str, baseline: str) -> Optional[pd.Series]:
        subset = tests[
            np.isclose(tests["fraction"], fraction)
            & (tests["outcome"] == outcome)
            & (tests["baseline"] == baseline)
        ]
        return subset.iloc[0] if not subset.empty else None

    validity_checks = []
    for fraction in non_full:
        row = validity_tests[
            np.isclose(validity_tests["fraction"], fraction)
            & (
                validity_tests["outcome"]
                == "spearman_estimated_vs_heldout_energy_variance"
            )
        ]
        passed = bool(
            not row.empty
            and row["mean_correlation"].iloc[0] > 0
            and row["bootstrap_ci_low"].iloc[0] > 0
        )
        validity_checks.append(
            {
                "fraction": fraction,
                "passed": passed,
                "mean_correlation": (
                    float(row["mean_correlation"].iloc[0])
                    if not row.empty else None
                ),
                "ci_low": (
                    float(row["bootstrap_ci_low"].iloc[0])
                    if not row.empty else None
                ),
            }
        )

    low_utility_checks = []
    for baseline in ("fixed_precision", "shuffled_precision"):
        row = test_row(low_fraction, "utility_spearman", baseline)
        passed = bool(
            row is not None
            and row["mean_advantage"] > 0
            and row["bootstrap_ci_low"] > 0
            and row["exact_sign_flip_p_holm"] < 0.05
        )
        low_utility_checks.append(
            {
                "baseline": baseline,
                "passed": passed,
                "mean_advantage": float(row["mean_advantage"]) if row is not None else None,
                "ci_low": float(row["bootstrap_ci_low"]) if row is not None else None,
                "holm_p": float(row["exact_sign_flip_p_holm"]) if row is not None else None,
            }
        )

    ndcg_fraction_support = []
    for fraction in non_full:
        checks = []
        for baseline in BASELINES:
            row = test_row(fraction, "ndcg_at_k", baseline)
            checks.append(
                bool(
                    row is not None
                    and row["mean_advantage"] > 0
                    and row["bootstrap_ci_low"] > 0
                    and row["exact_sign_flip_p_holm"] < 0.05
                )
            )
        ndcg_fraction_support.append(
            {
                "fraction": fraction,
                "all_three_baselines_passed": all(checks),
            }
        )

    no_precision_noninferiority = []
    for fraction in non_full:
        for outcome in ("utility_spearman", "ndcg_at_k"):
            row = test_row(fraction, outcome, "no_precision")
            passed = bool(
                row is not None
                and row["bootstrap_ci_low"] > -cfg.noninferiority_margin
            )
            no_precision_noninferiority.append(
                {
                    "fraction": fraction,
                    "outcome": outcome,
                    "passed": passed,
                    "ci_low": float(row["bootstrap_ci_low"]) if row is not None else None,
                    "margin": cfg.noninferiority_margin,
                }
            )

    uncertainty_validated = bool(
        validity_checks and all(item["passed"] for item in validity_checks)
    )
    low_sample_utility_supported = bool(
        low_utility_checks and all(item["passed"] for item in low_utility_checks)
    )
    ndcg_supported = any(
        item["all_three_baselines_passed"]
        for item in ndcg_fraction_support
    )
    utility_noninferior = bool(
        no_precision_noninferiority
        and all(item["passed"] for item in no_precision_noninferiority)
    )
    negative_control_passed = bool(
        not negative_control.empty
        and negative_control["passed"].all()
    )

    overall = bool(
        uncertainty_validated
        and low_sample_utility_supported
        and ndcg_supported
        and utility_noninferior
        and negative_control_passed
    )

    return {
        "overall_precision_role_supported": overall,
        "canonical_method": PRIMARY_METHOD,
        "adaptive_strength": 1.0,
        "uncertainty_estimator_validated": uncertainty_validated,
        "low_sample_utility_gain_supported": low_sample_utility_supported,
        "ndcg_gain_against_all_baselines_supported": ndcg_supported,
        "no_precision_noninferiority_supported": utility_noninferior,
        "full_fraction_negative_control_passed": negative_control_passed,
        "validity_checks": validity_checks,
        "low_sample_utility_checks": low_utility_checks,
        "ndcg_fraction_checks": ndcg_fraction_support,
        "no_precision_noninferiority_checks": no_precision_noninferiority,
        "interpretation": (
            "Cross-fitted finite-sample uncertainty is valid, and the canonical "
            "alpha=1 adaptive precision improves utility-aligned shortlisting "
            "under limited samples while preserving performance against the "
            "no-precision control."
            if overall
            else
            "The final finite-sample experiment does not satisfy every "
            "pre-specified condition required to justify the adaptive precision term."
        ),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_figure(
    figure: plt.Figure,
    path_without_suffix: Path,
) -> None:
    figure.savefig(
        path_without_suffix.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    figure.savefig(
        path_without_suffix.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_metric_by_fraction(
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.5))

    for method in (
        "adaptive_loto",
        "adaptive_alpha_1",
        "fixed_precision",
        "no_precision",
        "shuffled_precision",
    ):
        subset = summary[
            summary["method"] == method
        ].sort_values("fraction")
        if subset.empty:
            continue
        axis.plot(
            subset["fraction"],
            subset[f"mean_{metric}"],
            marker="o",
            label=method,
        )

    axis.set_xlabel("Analysis-cohort fraction")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    save_figure(figure, path)


def plot_uncertainty_validity(
    validity_tests: pd.DataFrame,
    out_dir: Path,
) -> None:
    subset = validity_tests[
        validity_tests["outcome"]
        == "spearman_estimated_vs_heldout_energy_variance"
    ].sort_values("fraction")

    if subset.empty:
        return

    lower = (
        subset["mean_correlation"]
        - subset["bootstrap_ci_low"]
    ).to_numpy(dtype=float)
    upper = (
        subset["bootstrap_ci_high"]
        - subset["mean_correlation"]
    ).to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.errorbar(
        subset["fraction"],
        subset["mean_correlation"],
        yerr=np.vstack([lower, upper]),
        marker="o",
        capsize=3,
    )
    axis.axhline(0.0, linewidth=1)
    axis.set_xlabel("Analysis-cohort fraction")
    axis.set_ylabel(
        "Task-macro Spearman: estimated vs held-out uncertainty"
    )
    axis.set_title(
        "Cross-fitted validity of finite-sample uncertainty"
    )
    figure.tight_layout()
    save_figure(
        figure,
        out_dir / "fig_uncertainty_validity",
    )


def plot_alpha_choices(
    alpha_choices: pd.DataFrame,
    out_dir: Path,
) -> None:
    if alpha_choices.empty:
        return

    summary = (
        alpha_choices.groupby(
            ["fraction", "selected_alpha"]
        )
        .size()
        .reset_index(name="count")
    )

    figure, axis = plt.subplots(figsize=(7.0, 4.3))
    for fraction, group in summary.groupby("fraction"):
        axis.plot(
            group["selected_alpha"],
            group["count"],
            marker="o",
            label=f"fraction={fraction:g}",
        )

    axis.set_xscale("symlog", linthresh=0.05)
    axis.set_xlabel("LOTO-selected alpha")
    axis.set_ylabel("Number of held-out tasks")
    axis.set_title("Label-free leave-one-task-out alpha choices")
    axis.legend()
    figure.tight_layout()
    save_figure(
        figure,
        out_dir / "fig_alpha_choices",
    )


def plot_tradeoff(
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    subset = summary[
        summary["fraction"]
        == min(
            fraction
            for fraction in summary["fraction"].unique()
            if fraction < 1.0 - 1e-12
        )
    ].copy()

    if subset.empty:
        return

    figure, axis = plt.subplots(figsize=(6.3, 5.0))
    axis.scatter(
        subset["mean_utility_spearman"],
        subset["mean_shortlist_jaccard_at_k"],
        s=65,
    )

    for row in subset.itertuples(index=False):
        axis.annotate(
            row.method,
            (
                row.mean_utility_spearman,
                row.mean_shortlist_jaccard_at_k,
            ),
            xytext=(4, 4),
            textcoords="offset points",
        )

    axis.set_xlabel("Task-macro utility Spearman")
    axis.set_ylabel("Task-macro shortlist Jaccard")
    axis.set_title(
        f"Utility–reliability trade-off at fraction={subset['fraction'].iloc[0]:g}"
    )
    figure.tight_layout()
    save_figure(
        figure,
        out_dir / "fig_tradeoff",
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def create_report(
    validation_summary: pd.DataFrame,
    validity_tests: pd.DataFrame,
    method_summary: pd.DataFrame,
    tests: pd.DataFrame,
    alpha_choices: pd.DataFrame,
    negative_control: pd.DataFrame,
    verdict: Mapping[str, object],
    *,
    cfg: Config,
) -> str:
    lines = [
        "HARMORA FINITE-SAMPLE UNCERTAINTY EXPERIMENT",
        "=" * 50,
        "",
        "Scientific design",
        "-----------------",
        (
            "No feature noise was injected. Graphs and harmonic energies were "
            "recomputed on independent paired m-out-of-n subsamples."
        ),
        (
            f"Fractions: {list(cfg.fractions)}; maximum cohort size: "
            f"{cfg.max_samples}; uncertainty/selection/evaluation views: "
            f"{cfg.uncertainty_views}/{cfg.selection_views}/"
            f"{cfg.evaluation_views}."
        ),
        (
            "Adaptive alpha was selected without labels by leave-one-task-out "
            "calibration stability."
        ),
        "",
        "Official validation",
        "-------------------",
        (
            f"Validated candidates: "
            f"{int(validation_summary['n_candidates'].iloc[0])}"
        ),
        (
            "Maximum absolute reconstruction error: "
            f"{validation_summary['max_abs_error'].iloc[0]:.6e}"
        ),
        (
            "Median absolute reconstruction error: "
            f"{validation_summary['median_abs_error'].iloc[0]:.6e}"
        ),
        (
            "Fraction within tolerance: "
            f"{validation_summary['fraction_within_tolerance'].iloc[0]:.6f}"
        ),
        "",
        "Uncertainty-estimator validity",
        "------------------------------",
    ]

    for row in validity_tests.itertuples(index=False):
        lines.append(
            f"fraction={row.fraction:g} | {row.outcome}: "
            f"mean rho={row.mean_correlation:.4f}, "
            f"95% CI=[{row.bootstrap_ci_low:.4f}, "
            f"{row.bootstrap_ci_high:.4f}]"
        )

    lines.extend(
        [
            "",
            "Method summary",
            "--------------",
        ]
    )

    for row in method_summary.itertuples(index=False):
        lines.append(
            f"fraction={row.fraction:g} | {row.method}: "
            f"utility rho={row.mean_utility_spearman:.4f}, "
            f"rank reliability={row.mean_rank_reliability:.4f}, "
            f"shortlist Jaccard={row.mean_shortlist_jaccard_at_k:.4f}, "
            f"coverage regret={row.mean_coverage_regret_at_k:.4f}, "
            f"near-oracle hit={row.mean_near_oracle_hit_5pct:.4f}"
        )

    lines.extend(
        [
            "",
            "Primary task-level comparisons",
            "------------------------------",
        ]
    )

    for row in tests.itertuples(index=False):
        lines.append(
            f"fraction={row.fraction:g} | {row.outcome}: "
            f"adaptive_loto vs {row.baseline}: "
            f"mean advantage={row.mean_advantage:.4f}, "
            f"95% CI=[{row.bootstrap_ci_low:.4f}, "
            f"{row.bootstrap_ci_high:.4f}], "
            f"wins/losses/ties="
            f"{row.task_wins}/{row.task_losses}/{row.task_ties}, "
            f"Holm p={row.exact_sign_flip_p_holm:.6g}"
        )

    lines.extend(
        [
            "",
            "LOTO alpha choices",
            "------------------",
        ]
    )
    for fraction, group in alpha_choices.groupby("fraction"):
        counts = group["selected_alpha"].value_counts().sort_index()
        lines.append(
            f"fraction={fraction:g}: "
            + ", ".join(
                f"alpha={alpha:g}: {count}"
                for alpha, count in counts.items()
            )
        )

    lines.extend(
        [
            "",
            "Negative control",
            "----------------",
            (
                f"All full-fraction adaptive/fixed outcomes identical: "
                f"{bool(not negative_control.empty and negative_control['passed'].all())}"
            ),
            "",
            "Automatic verdict",
            "-----------------",
            str(verdict["interpretation"]),
            (
                "Overall precision role supported: "
                f"{verdict['overall_precision_role_supported']}"
            ),
            (
                "Uncertainty estimator validated: "
                f"{verdict['uncertainty_estimator_validated']}"
            ),
            (
                "Held-out reliability gain supported: "
                f"{verdict['heldout_reliability_gain_supported']}"
            ),
            (
                "Utility/coverage non-inferiority supported: "
                f"{verdict['utility_and_coverage_noninferiority_supported']}"
            ),
            (
                "Full-fraction negative control passed: "
                f"{verdict['full_fraction_negative_control_passed']}"
            ),
        ]
    )

    return "\n".join(lines) + "\n"



# ---------------------------------------------------------------------------
# Final paper tables and figures
# ---------------------------------------------------------------------------

PAPER_METHODS = (
    "adaptive_precision",
    "fixed_precision",
    "no_precision",
    "shuffled_precision",
)
PAPER_METHOD_LABELS = {
    "adaptive_precision": "Adaptive precision",
    "fixed_precision": "Fixed precision",
    "no_precision": "No precision",
    "shuffled_precision": "Shuffled precision",
}
PRIMARY_ENDPOINTS = ("utility_spearman", "ndcg_at_k")


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values,
        size=(n_bootstrap, len(values)),
        replace=True,
    ).mean(axis=1)
    return (
        float(np.mean(values)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def build_method_ci_table(
    task_macro: pd.DataFrame,
    *,
    cfg: Config,
) -> pd.DataFrame:
    rows = []
    metrics = (
        "utility_spearman",
        "ndcg_at_k",
        "coverage_regret_at_k",
        "rank_reliability",
    )
    for fraction in sorted(task_macro["fraction"].unique()):
        for method in PAPER_METHODS:
            subset = task_macro[
                np.isclose(task_macro["fraction"], fraction)
                & (task_macro["method"] == method)
            ]
            if subset.empty:
                continue
            row = {
                "fraction": float(fraction),
                "method": method,
                "method_label": PAPER_METHOD_LABELS[method],
                "n_tasks": int(subset["task"].nunique()),
            }
            for metric in metrics:
                mean, low, high = bootstrap_mean_interval(
                    subset[metric].to_numpy(dtype=float),
                    n_bootstrap=cfg.task_bootstrap,
                    seed=stable_seed(
                        cfg.seed,
                        "paper_method_ci",
                        fraction,
                        method,
                        metric,
                    ),
                )
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            rows.append(row)
    return pd.DataFrame(rows)


def _latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(old, new)
    return text


def write_table1(
    method_ci: pd.DataFrame,
    out_dir: Path,
) -> None:
    compact = method_ci[
        [
            "fraction",
            "method",
            "method_label",
            "n_tasks",
            "utility_spearman_mean",
            "utility_spearman_ci_low",
            "utility_spearman_ci_high",
            "ndcg_at_k_mean",
            "ndcg_at_k_ci_low",
            "ndcg_at_k_ci_high",
            "coverage_regret_at_k_mean",
            "coverage_regret_at_k_ci_low",
            "coverage_regret_at_k_ci_high",
            "rank_reliability_mean",
            "rank_reliability_ci_low",
            "rank_reliability_ci_high",
        ]
    ].copy()
    compact.to_csv(
        out_dir / "table1_method_performance.csv",
        index=False,
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Finite-sample layer-shortlisting performance. Values are task-macro means with task-bootstrap 95\% confidence intervals. Lower coverage regret is better; all other metrics are higher-is-better.}",
        r"\label{tab:finite_sample_precision_performance}",
        r"\begin{tabular}{clcccc}",
        r"\toprule",
        r"Fraction & Method & Utility $\rho$ & NDCG@5 & Coverage regret@5 & Rank reliability \\",
        r"\midrule",
    ]

    for fraction in sorted(compact["fraction"].unique()):
        block = compact[np.isclose(compact["fraction"], fraction)]
        for index, row in enumerate(block.itertuples(index=False)):
            fraction_text = f"{row.fraction:.2f}" if index == 0 else ""
            utility_text = (
                f"{row.utility_spearman_mean:.3f} "
                f"[{row.utility_spearman_ci_low:.3f}, {row.utility_spearman_ci_high:.3f}]"
            )
            ndcg_text = (
                f"{row.ndcg_at_k_mean:.3f} "
                f"[{row.ndcg_at_k_ci_low:.3f}, {row.ndcg_at_k_ci_high:.3f}]"
            )
            coverage_text = (
                f"{row.coverage_regret_at_k_mean:.3f} "
                f"[{row.coverage_regret_at_k_ci_low:.3f}, {row.coverage_regret_at_k_ci_high:.3f}]"
            )
            reliability_text = (
                f"{row.rank_reliability_mean:.3f} "
                f"[{row.rank_reliability_ci_low:.3f}, {row.rank_reliability_ci_high:.3f}]"
            )
            lines.append(
                f"{fraction_text} & {_latex_escape(row.method_label)} & "
                f"{utility_text} & {ndcg_text} & {coverage_text} & "
                f"{reliability_text} \\\\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    (out_dir / "table1_method_performance.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_table2(
    tests: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    compact = tests[
        (tests["fraction"] < 1.0 - 1e-12)
        & tests["outcome"].isin(PRIMARY_ENDPOINTS)
        & tests["baseline"].isin(BASELINES)
    ].copy()
    compact["outcome_label"] = compact["outcome"].map(
        {
            "utility_spearman": "Utility Spearman",
            "ndcg_at_k": "NDCG@5",
        }
    )
    compact["baseline_label"] = compact["baseline"].map(
        PAPER_METHOD_LABELS
    )
    compact = compact.sort_values(
        ["fraction", "outcome", "baseline"]
    )
    compact.to_csv(
        out_dir / "table2_primary_paired_tests.csv",
        index=False,
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Paired task-level comparison of adaptive precision against controls. Positive $\Delta$ favors adaptive precision. Confidence intervals use task bootstrap; $p_{\mathrm{Holm}}$ is the exact sign-flip test corrected across the three controls within each fraction and endpoint.}",
        r"\label{tab:finite_sample_precision_paired}",
        r"\begin{tabular}{cclccc}",
        r"\toprule",
        r"Fraction & Endpoint & Baseline & $\Delta$ [95\% CI] & W/L/T & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]

    for row in compact.itertuples(index=False):
        effect = (
            f"{row.mean_advantage:+.4f} "
            f"[{row.bootstrap_ci_low:+.4f}, {row.bootstrap_ci_high:+.4f}]"
        )
        wlt = f"{row.task_wins}/{row.task_losses}/{row.task_ties}"
        p_value = (
            "--"
            if not np.isfinite(row.exact_sign_flip_p_holm)
            else f"{row.exact_sign_flip_p_holm:.4f}"
        )
        lines.append(
            f"{row.fraction:.2f} & {_latex_escape(row.outcome_label)} & "
            f"{_latex_escape(row.baseline_label)} & {effect} & {wlt} & "
            f"{p_value} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    (out_dir / "table2_primary_paired_tests.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return compact


def save_paper_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(figure)


def plot_paper_validity(
    validity_tests: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Recreate the appendix validity figure used in the paper."""
    subset = validity_tests[
        (validity_tests["fraction"] < 1.0 - 1e-12)
        & (
            validity_tests["outcome"]
            == "spearman_estimated_vs_heldout_energy_variance"
        )
    ].sort_values("fraction")
    if subset.empty:
        raise ValueError("No restricted-fraction uncertainty-validity results found.")

    mean = subset["mean_correlation"].to_numpy(dtype=float)
    lower = mean - subset["bootstrap_ci_low"].to_numpy(dtype=float)
    upper = subset["bootstrap_ci_high"].to_numpy(dtype=float) - mean
    fractions = subset["fraction"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    axis.errorbar(
        fractions,
        mean,
        yerr=np.vstack([lower, upper]),
        marker="o",
        linewidth=1.5,
        capsize=3,
    )
    axis.axhline(0.0, linewidth=0.9)
    axis.set_xticks(fractions)
    axis.set_xticklabels([f"{int(round(100 * value))}%" for value in fractions])
    axis.set_xlabel("Analysis-cohort fraction")
    axis.set_ylabel("Task-macro Spearman correlation")
    axis.set_title("Cross-fitted validation of the instability estimate")
    figure.tight_layout()
    save_paper_figure(
        figure,
        out_dir / "figure_10_adaptive_precision_validity",
    )


def plot_paper_primary(
    tests: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Recreate the paired NDCG@5 comparison used in the paper."""
    baseline_order = [
        "fixed_precision",
        "shuffled_precision",
        "no_precision",
    ]
    labels = [
        "Fixed precision",
        "Shuffled precision",
        "No precision",
    ]
    subset = tests[
        np.isclose(tests["fraction"], 0.25)
        & (tests["outcome"] == "ndcg_at_k")
    ].set_index("baseline").reindex(baseline_order)
    if subset["mean_advantage"].isna().any():
        raise ValueError("Incomplete NDCG@5 paired results at fraction=0.25.")

    mean = subset["mean_advantage"].to_numpy(dtype=float)
    lower = mean - subset["bootstrap_ci_low"].to_numpy(dtype=float)
    upper = subset["bootstrap_ci_high"].to_numpy(dtype=float) - mean
    positions = np.arange(len(baseline_order), dtype=float)

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.errorbar(
        positions,
        mean,
        yerr=np.vstack([lower, upper]),
        marker="o",
        linestyle="none",
        capsize=3,
    )
    axis.axhline(0.0, linewidth=0.9)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("Control variant")
    axis.set_ylabel(r"Task-macro $\Delta$NDCG@5 (adaptive $-$ baseline)")
    axis.set_title("Adaptive precision at the 25% analysis fraction")

    p_values = subset["exact_sign_flip_p_holm"].to_numpy(dtype=float)
    for position, value, high, p_value in zip(positions, mean, upper, p_values):
        axis.annotate(
            rf"$p_{{\mathrm{{Holm}}}}={p_value:.4f}$",
            xy=(position, value + high),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    save_paper_figure(
        figure,
        out_dir / "figure_11_adaptive_precision_primary",
    )


def create_paper_report(
    validation_summary: pd.DataFrame,
    validity_tests: pd.DataFrame,
    tests: pd.DataFrame,
    verdict: Mapping[str, object],
) -> str:
    lines = [
        "HARMORA FINITE-SAMPLE PRECISION: PAPER SUMMARY",
        "=" * 52,
        "",
        "Canonical method",
        "----------------",
        "Adaptive precision uses p_l = 1 / (1 + u_l), with no tuned alpha.",
        "Calibration and evaluation subsamples are independent.",
        "",
        "Official implementation validation",
        "----------------------------------",
        f"Validated candidates: {int(validation_summary['n_candidates'].iloc[0])}",
        (
            "Maximum absolute reconstruction error: "
            f"{validation_summary['max_abs_error'].iloc[0]:.6e}"
        ),
        (
            "Fraction within tolerance: "
            f"{validation_summary['fraction_within_tolerance'].iloc[0]:.6f}"
        ),
        "",
        "Uncertainty validity",
        "--------------------",
    ]
    for row in validity_tests[
        validity_tests["fraction"] < 1.0 - 1e-12
    ].itertuples(index=False):
        lines.append(
            f"fraction={row.fraction:g}, {row.outcome}: "
            f"rho={row.mean_correlation:.4f}, "
            f"95% CI=[{row.bootstrap_ci_low:.4f}, "
            f"{row.bootstrap_ci_high:.4f}]"
        )

    lines.extend(["", "Primary paired results", "----------------------"])
    primary = tests[
        (tests["fraction"] < 1.0 - 1e-12)
        & tests["outcome"].isin(PRIMARY_ENDPOINTS)
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"fraction={row.fraction:g}, {row.outcome}, vs {row.baseline}: "
            f"Delta={row.mean_advantage:+.5f}, "
            f"95% CI=[{row.bootstrap_ci_low:+.5f}, "
            f"{row.bootstrap_ci_high:+.5f}], "
            f"W/L/T={row.task_wins}/{row.task_losses}/{row.task_ties}, "
            f"Holm p={row.exact_sign_flip_p_holm:.6g}"
        )

    lines.extend(
        [
            "",
            "Verdict",
            "-------",
            str(verdict.get("interpretation", "")),
            (
                "Overall precision role supported: "
                f"{verdict.get('overall_precision_role_supported', False)}"
            ),
            "",
            "Paper-facing artifacts",
            "----------------------",
            "figure_10_adaptive_precision_validity.pdf/png",
            "figure_11_adaptive_precision_primary.pdf/png",
            "table1_method_performance.csv/tex",
            "table2_primary_paired_tests.csv/tex",
        ]
    )
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Final paper experiment for Harmora finite-sample adaptive precision."
        )
    )
    parser.add_argument(
        "--representation-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--utility-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metrics-package-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--official-metrics-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--target-source",
        default="downstream",
    )
    parser.add_argument(
        "--target-name",
        default="downstream_primary_score_mean",
    )
    parser.add_argument(
        "--primary-tasks",
        nargs="+",
        default=list(PRIMARY_TASKS),
        help="Exact task names to include. Defaults to the eleven primary tasks.",
    )
    parser.add_argument(
        "--expected-task-count",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--expected-candidates-per-task",
        type=int,
        default=106,
    )
    parser.add_argument(
        "--expected-model-count",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--harmonic-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--sigma0-l2",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--graph-standardize",
        action="store_true",
    )
    parser.add_argument(
        "--bandwidth",
        choices=("median", "unit"),
        default="median",
    )
    parser.add_argument(
        "--k-nn",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75, 1.0],
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--uncertainty-views",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--selection-views",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--evaluation-views",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--subsample-seeds",
        nargs="+",
        type=int,
        default=[11, 29, 47, 71, 101],
        help=(
            "Base seeds for the five independent uncertainty and evaluation "
            "subsamples. Role-specific hashing keeps calibration and evaluation "
            "views independent even when the same base seeds are used."
        ),
    )
    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=[1.0],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--precision-shuffles",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--shrinkage-strength",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--task-bootstrap",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--noninferiority-margin",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--stratified-if-available",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a specific torch device such as cuda:0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--validation-tolerance",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--degenerate-threshold",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "--rebuild-bank",
        action="store_true",
    )
    parser.add_argument(
        "--bank-cache-dir",
        type=Path,
        default=Path(
            "cache/precision_ablation"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "artifacts/generated/adaptive_precision"
        ),
    )
    args = parser.parse_args()

    fractions = tuple(
        sorted(set(float(value) for value in args.fractions))
    )
    if 1.0 not in fractions:
        fractions = tuple(sorted(fractions + (1.0,)))

    alpha_grid = tuple(
        sorted(set(float(value) for value in args.alpha_grid))
    )
    if 0.0 not in alpha_grid:
        alpha_grid = (0.0,) + alpha_grid
    if 1.0 not in alpha_grid:
        alpha_grid = tuple(sorted(alpha_grid + (1.0,)))

    if args.harmonic_k < 1:
        raise ValueError("--harmonic-k must be positive.")
    if args.sigma0_l2 <= 0:
        raise ValueError("--sigma0-l2 must be positive.")
    if any(value <= 0 or value > 1 for value in fractions):
        raise ValueError("All fractions must lie in (0,1].")
    if args.max_samples < 8:
        raise ValueError("--max-samples is too small.")
    if args.min_samples < args.harmonic_k + 2:
        raise ValueError(
            "--min-samples must exceed harmonic_k + 1."
        )
    if args.min_samples > args.max_samples:
        raise ValueError(
            "--min-samples cannot exceed --max-samples."
        )
    if args.uncertainty_views < 3:
        raise ValueError(
            "--uncertainty-views must be at least 3."
        )
    if args.selection_views < 0:
        raise ValueError(
            "--selection-views must be nonnegative."
        )
    if args.evaluation_views < 2:
        raise ValueError(
            "--evaluation-views must be at least 2."
        )
    if len(set(args.subsample_seeds)) != len(args.subsample_seeds):
        raise ValueError("--subsample-seeds must be unique.")
    if args.uncertainty_views != len(args.subsample_seeds):
        raise ValueError(
            "--uncertainty-views must equal the number of --subsample-seeds."
        )
    if args.evaluation_views != len(args.subsample_seeds):
        raise ValueError(
            "--evaluation-views must equal the number of --subsample-seeds."
        )
    if any(value < 0 for value in alpha_grid):
        raise ValueError("Alpha values must be nonnegative.")
    if args.precision_shuffles < 1:
        raise ValueError(
            "--precision-shuffles must be positive."
        )
    if args.shrinkage_strength < 0:
        raise ValueError(
            "--shrinkage-strength must be nonnegative."
        )
    if args.task_bootstrap < 1000:
        raise ValueError(
            "--task-bootstrap must be at least 1000."
        )
    if args.noninferiority_margin < 0:
        raise ValueError(
            "--noninferiority-margin must be nonnegative."
        )

    device = resolve_device(args.device)

    cfg = Config(
        harmonic_k=args.harmonic_k,
        sigma0_l2=args.sigma0_l2,
        graph_standardize=args.graph_standardize,
        bandwidth=args.bandwidth,
        k_nn=args.k_nn,
        fractions=fractions,
        max_samples=args.max_samples,
        min_samples=args.min_samples,
        uncertainty_views=args.uncertainty_views,
        selection_views=args.selection_views,
        evaluation_views=args.evaluation_views,
        subsample_seeds=tuple(int(x) for x in args.subsample_seeds),
        alpha_grid=alpha_grid,
        top_k=args.top_k,
        precision_shuffles=args.precision_shuffles,
        shrinkage_strength=args.shrinkage_strength,
        seed=args.seed,
        validation_tolerance=args.validation_tolerance,
        degenerate_threshold=args.degenerate_threshold,
        task_bootstrap=args.task_bootstrap,
        noninferiority_margin=args.noninferiority_margin,
        stratified_if_available=args.stratified_if_available,
        device=str(device),
    )

    add_metrics_package(args.metrics_package_dir)

    cache = load_representation_cache(
        args.representation_dir
    )
    utility = load_utility(
        args.utility_csv,
        args.target_source,
        args.target_name,
    )

    requested_tasks = tuple(dict.fromkeys(str(x) for x in args.primary_tasks))
    missing_tasks = sorted(set(requested_tasks) - set(utility["task"].astype(str)))
    if missing_tasks:
        raise ValueError(
            "Primary tasks missing from the selected utility target: "
            f"{missing_tasks}"
        )
    utility = utility[utility["task"].astype(str).isin(requested_tasks)].copy()

    observed_tasks = sorted(utility["task"].astype(str).unique())
    if len(observed_tasks) != args.expected_task_count:
        raise ValueError(
            f"Expected {args.expected_task_count} tasks, found "
            f"{len(observed_tasks)}: {observed_tasks}"
        )

    task_candidate_counts = (
        utility.groupby("task")[["model_alias", "layer"]]
        .apply(lambda frame: len(frame.drop_duplicates()))
        .astype(int)
    )
    bad_candidate_counts = task_candidate_counts[
        task_candidate_counts != args.expected_candidates_per_task
    ]
    if not bad_candidate_counts.empty:
        raise ValueError(
            "Unexpected candidate counts per task; expected "
            f"{args.expected_candidates_per_task}: "
            f"{bad_candidate_counts.to_dict()}"
        )

    task_model_counts = utility.groupby("task")["model_alias"].nunique().astype(int)
    bad_model_counts = task_model_counts[
        task_model_counts != args.expected_model_count
    ]
    if not bad_model_counts.empty:
        raise ValueError(
            "Unexpected model counts per task; expected "
            f"{args.expected_model_count}: {bad_model_counts.to_dict()}"
        )

    cache = {
        key: value for key, value in cache.items()
        if key[0] in set(observed_tasks)
    }

    official_scores = load_official_scores(
        args.official_metrics_dir
    )

    out_root = args.out_dir.resolve()
    data_dir = out_root / "data"
    figures_dir = out_root / "figures"
    tables_dir = out_root / "tables"
    metadata_dir = out_root / "metadata"
    for directory in (data_dir, figures_dir, tables_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    args.bank_cache_dir.mkdir(parents=True, exist_ok=True)

    bank, validation_detail = load_or_build_bank(
        cache,
        utility,
        official_scores,
        cfg=cfg,
        cache_dir=args.bank_cache_dir,
        rebuild=args.rebuild_bank,
        device=device,
    )

    validation_detail.to_csv(
        data_dir / "official_validation_detail.csv",
        index=False,
    )

    validation_summary = pd.DataFrame(
        [
            {
                "n_candidates": len(validation_detail),
                "max_abs_error": float(
                    validation_detail["absolute_error"].max()
                ),
                "median_abs_error": float(
                    validation_detail["absolute_error"].median()
                ),
                "fraction_within_tolerance": float(
                    np.mean(
                        validation_detail["absolute_error"]
                        <= cfg.validation_tolerance
                    )
                ),
            }
        ]
    )
    validation_summary.to_csv(
        data_dir / "official_validation.csv",
        index=False,
    )

    uncertainty, validity = estimate_uncertainty(
        bank,
        cfg=cfg,
    )
    uncertainty.to_csv(
        data_dir / "uncertainty_estimates.csv",
        index=False,
    )
    validity.to_csv(
        data_dir / "uncertainty_validity.csv",
        index=False,
    )

    # Final paper evaluation: fixed alpha=1, no label-based tuning.
    evaluation, candidate_scores = evaluate_paper_methods(
        bank,
        uncertainty,
        cfg=cfg,
    )
    evaluation.to_csv(
        data_dir / "evaluation_trial_metrics.csv",
        index=False,
    )
    candidate_scores.to_csv(
        data_dir / "evaluation_candidate_scores.csv",
        index=False,
    )

    model_task, task_macro, method_summary = aggregate_metrics(
        evaluation
    )
    model_task.to_csv(
        data_dir / "model_task_metrics.csv",
        index=False,
    )
    task_macro.to_csv(
        data_dir / "task_macro_metrics.csv",
        index=False,
    )
    method_summary.to_csv(
        data_dir / "method_summary.csv",
        index=False,
    )

    tests = paired_task_tests(
        task_macro,
        cfg=cfg,
    )
    tests.to_csv(
        data_dir / "paired_task_level_tests.csv",
        index=False,
    )

    validity_tests = validity_inference(
        validity,
        cfg=cfg,
    )
    validity_tests.to_csv(
        data_dir / "uncertainty_validity_inference.csv",
        index=False,
    )

    negative_control = negative_control_audit(
        method_summary
    )
    negative_control.to_csv(
        data_dir / "negative_control_audit.csv",
        index=False,
    )

    verdict = build_verdict(
        tests,
        validity_tests,
        negative_control,
        cfg=cfg,
    )
    write_json(
        metadata_dir / "automatic_verdict.json",
        verdict,
    )

    method_ci = build_method_ci_table(
        task_macro,
        cfg=cfg,
    )
    method_ci.to_csv(
        data_dir / "method_performance_with_ci.csv",
        index=False,
    )
    write_table1(method_ci, tables_dir)
    write_table2(tests, tables_dir)

    plot_paper_validity(
        validity_tests,
        figures_dir,
    )
    plot_paper_primary(
        tests,
        figures_dir,
    )

    report = create_paper_report(
        validation_summary,
        validity_tests,
        tests,
        verdict,
    )
    (
        metadata_dir / "paper_ready_summary.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    write_json(
        metadata_dir / "effective_config.json",
        {
            "implementation_version": IMPLEMENTATION_VERSION,
            "config_signature": cfg.signature(),
            **asdict(cfg),
            "representation_dir": str(
                args.representation_dir.resolve()
            ),
            "utility_csv": str(args.utility_csv.resolve()),
            "metrics_package_dir": str(
                args.metrics_package_dir.resolve()
            ),
            "official_metrics_dir": str(
                args.official_metrics_dir.resolve()
            ),
            "target_source": args.target_source,
            "target_name": args.target_name,
            "primary_tasks": observed_tasks,
            "expected_task_count": args.expected_task_count,
            "expected_candidates_per_task": args.expected_candidates_per_task,
            "expected_model_count": args.expected_model_count,
            "candidate_counts_per_task": task_candidate_counts.to_dict(),
            "model_counts_per_task": task_model_counts.to_dict(),
            "bank_cache_dir": str(
                args.bank_cache_dir.resolve()
            ),
            "out_dir": str(out_root),
        },
    )

    print("\nDevice")
    print(device)

    print("\nOfficial validation")
    print(validation_summary.to_string(index=False))

    print("\nUncertainty validity")
    print(validity_tests.to_string(index=False))

    print("\nMethod summary")
    print(method_summary.to_string(index=False))

    print("\nPrimary paired task-level tests (adaptive precision)")
    print(
        tests[
            (tests["fraction"] < 1.0 - 1e-12)
            & tests["outcome"].isin(PRIMARY_ENDPOINTS)
        ][
            [
                "fraction",
                "outcome",
                "baseline",
                "n_tasks",
                "mean_advantage",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "task_wins",
                "task_losses",
                "task_ties",
                "exact_sign_flip_p_holm",
            ]
        ].to_string(index=False)
    )

    print("\nAutomatic verdict")
    print(json.dumps(verdict, indent=2))

    print(
        "\nComplete output written to:\n"
        f"{out_root}"
    )


if __name__ == "__main__":
    main()
