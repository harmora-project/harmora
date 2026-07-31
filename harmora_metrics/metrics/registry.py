"""
Metric registry and unified runner.

Use this file when you want to compute a selected subset or all registered metrics
with one API call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch

from .single_view import (
    compute_anisotropy,
    compute_curvature,
    compute_harmora,
    compute_intrinsic_dimension,
    compute_matrix_entropy,
    compute_participation_ratio,
    compute_spectral_gap,
)
from .augmentation import compute_dime, compute_infonce, compute_lidar


@dataclass
class MetricConfig:
    """Central configuration for metric computation."""

    # Main Harmora settings
    harmora_sigma_l2: float = 1.0
    harmora_K_l: Optional[int] = None
    harmora_K_max: Optional[int] = 80
    graph_bandwidth: str = "median"
    graph_k_nn: Optional[int] = None
    graph_standardize: bool = False

    # Entropy-style settings
    entropy_alpha: float = 1.0
    entropy_normalizations: Sequence[str] = ("maxEntropy",)

    # Augmentation-based settings
    infonce_temperature: float = 0.1
    infonce_normalize: bool = True

    # Curvature settings
    curvature_k: int = 1

    # Numerical stability
    eps: float = 1e-12


SINGLE_VIEW_METRICS = {
    "harmora": compute_harmora,
    "matrix_entropy": compute_matrix_entropy,
    "participation_ratio": compute_participation_ratio,
    "anisotropy": compute_anisotropy,
    "intrinsic_dimension": compute_intrinsic_dimension,
    "curvature": compute_curvature,
    "spectral_gap": compute_spectral_gap,
}

AUGMENTATION_METRICS = {
    "infonce": compute_infonce,
    "dime": compute_dime,
    "lidar": compute_lidar,
}

ALL_METRICS = {**SINGLE_VIEW_METRICS, **AUGMENTATION_METRICS}

DEFAULT_SINGLE_VIEW = [
    "harmora",
    "matrix_entropy",
    "participation_ratio",
    "anisotropy",
    "intrinsic_dimension",
    "curvature",
    "spectral_gap",
]

DEFAULT_AUGMENTATION = ["infonce", "dime", "lidar"]
DEFAULT_ALL = DEFAULT_SINGLE_VIEW + DEFAULT_AUGMENTATION


def compute_metric(name: str, hidden_states=None, augmented_states=None, config: Optional[MetricConfig] = None) -> Dict[str, Any]:
    """Compute a single metric by name."""
    cfg = config or MetricConfig()
    name = name.lower()

    if name == "harmora":
        return compute_harmora(
            hidden_states,
            sigma_l2=cfg.harmora_sigma_l2,
            K_l=cfg.harmora_K_l,
            K_max=cfg.harmora_K_max,
            eps=cfg.eps,
            standardize=cfg.graph_standardize,
            bandwidth=cfg.graph_bandwidth,
            k_nn=cfg.graph_k_nn,
        )
    if name == "matrix_entropy":
        return compute_matrix_entropy(hidden_states, alpha=cfg.entropy_alpha, normalizations=cfg.entropy_normalizations, eps=cfg.eps)
    if name == "participation_ratio":
        return compute_participation_ratio(hidden_states, eps=cfg.eps)
    if name == "anisotropy":
        return compute_anisotropy(hidden_states, eps=cfg.eps)
    if name == "intrinsic_dimension":
        return compute_intrinsic_dimension(hidden_states, eps=cfg.eps)
    if name == "curvature":
        return compute_curvature(hidden_states, k=cfg.curvature_k, eps=cfg.eps)
    if name == "spectral_gap":
        return compute_spectral_gap(
            hidden_states,
            eps=cfg.eps,
            standardize=cfg.graph_standardize,
            bandwidth=cfg.graph_bandwidth,
            k_nn=cfg.graph_k_nn,
        )
    # if name == "infonce":
    #     return compute_infonce(augmented_states, temperature=cfg.infonce_temperature, normalize=cfg.infonce_normalize)
    # if name == "dime":
    #     return compute_dime(augmented_states, alpha=cfg.entropy_alpha, normalizations=cfg.entropy_normalizations)
    if name == "infonce":
        if augmented_states is None:
            raise ValueError("InfoNCE requires augmented_states [L,N,A,D].")
        if augmented_states.shape[2] < 2:
            raise ValueError(f"InfoNCE requires at least 2 views, got A={augmented_states.shape[2]}.")
        aug2 = augmented_states[:, :, :2, :]
        return compute_infonce(
            aug2,
            temperature=cfg.infonce_temperature,
            normalize=cfg.infonce_normalize,
        )

    if name == "dime":
        if augmented_states is None:
            raise ValueError("DiME requires augmented_states [L,N,A,D].")
        if augmented_states.shape[2] < 2:
            raise ValueError(f"DiME requires at least 2 views, got A={augmented_states.shape[2]}.")
        aug2 = augmented_states[:, :, :2, :]
        return compute_dime(
            aug2,
            alpha=cfg.entropy_alpha,
            normalizations=cfg.entropy_normalizations,
        )
    
    if name == "lidar":
        return compute_lidar(augmented_states, alpha=cfg.entropy_alpha, normalizations=cfg.entropy_normalizations)

    raise KeyError(f"Unknown metric '{name}'. Available metrics: {sorted(ALL_METRICS.keys())}")


def compute_all_metrics(
    hidden_states: Optional[torch.Tensor] = None,
    augmented_states: Optional[torch.Tensor] = None,
    metrics: Optional[Sequence[str]] = None,
    config: Optional[MetricConfig] = None,
    skip_errors: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute selected unsupervised representation metrics.

    Args:
        hidden_states: [L, N, D] for single-view metrics.
        augmented_states: [L, N, A, D] for augmentation-based metrics.
        metrics: list of metric names. If None, choose available defaults based on inputs.
        config: MetricConfig.
        skip_errors: if True, store metric errors instead of raising.

    Returns:
        Nested dictionary: results[metric_name][score_name] -> layer-wise list.
    """
    cfg = config or MetricConfig()

    if metrics is None:
        metrics = []
        if hidden_states is not None:
            metrics.extend(DEFAULT_SINGLE_VIEW)
        if augmented_states is not None:
            metrics.extend(DEFAULT_AUGMENTATION)

    results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    for name in metrics:
        try:
            if name in SINGLE_VIEW_METRICS:
                if hidden_states is None:
                    raise ValueError(f"Metric '{name}' requires hidden_states [L,N,D].")
                results[name] = compute_metric(name, hidden_states=hidden_states, config=cfg)
            elif name in AUGMENTATION_METRICS:
                if augmented_states is None:
                    raise ValueError(f"Metric '{name}' requires augmented_states [L,N,A,D].")
                results[name] = compute_metric(name, augmented_states=augmented_states, config=cfg)
            else:
                raise KeyError(f"Unknown metric '{name}'.")
        except Exception as e:
            if not skip_errors:
                raise
            errors[name] = repr(e)

    if errors:
        results["_errors"] = errors
    return results
