"""
Utility functions for unsupervised representation metrics.

The package assumes layer-wise representations in the canonical shape:
    hidden_states: torch.Tensor [L, N, D]
where L is number of layers, N is number of samples, and D is representation dimension.

For augmentation-based metrics, the canonical shape is:
    augmented_states: torch.Tensor [L, N, A, D]
where A is the number of augmentations/views.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

Tensor = torch.Tensor


_ALLOWED_NORMALIZATIONS = {"maxEntropy", "logN", "logD", "logNlogD", "raw", "length"}


def as_float(x) -> float:
    """Safely convert a scalar tensor or scalar value to Python float."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return float(x.item())
        return float(x.reshape(-1)[0].item())
    return float(x)


def nan_list(length: int) -> List[float]:
    return [float("nan")] * length


def validate_hidden_states(hidden_states: Tensor) -> Tuple[int, int, int]:
    """Validate [L, N, D] single-view representation tensor."""
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("hidden_states must be a torch.Tensor with shape [L, N, D].")
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must have shape [L, N, D], got {tuple(hidden_states.shape)}.")
    L, N, D = hidden_states.shape
    if L < 1 or N < 1 or D < 1:
        raise ValueError(f"Invalid hidden_states shape {tuple(hidden_states.shape)}.")
    return int(L), int(N), int(D)


def validate_augmented_states(augmented_states: Tensor) -> Tuple[int, int, int, int]:
    """Validate [L, N, A, D] augmentation-view representation tensor."""
    if not isinstance(augmented_states, torch.Tensor):
        raise TypeError("augmented_states must be a torch.Tensor with shape [L, N, A, D].")
    if augmented_states.ndim != 4:
        raise ValueError(f"augmented_states must have shape [L, N, A, D], got {tuple(augmented_states.shape)}.")
    L, N, A, D = augmented_states.shape
    if L < 1 or N < 1 or A < 1 or D < 1:
        raise ValueError(f"Invalid augmented_states shape {tuple(augmented_states.shape)}.")
    return int(L), int(N), int(A), int(D)


def entropy_normalization(entropy: float, normalization: str, N: int, D: int) -> float:
    """
    Normalize entropy-like scores using the conventions used in the prior metric code.
    """
    if normalization not in _ALLOWED_NORMALIZATIONS:
        raise ValueError(f"Unknown normalization '{normalization}'. Allowed: {sorted(_ALLOWED_NORMALIZATIONS)}")

    if entropy is None or np.isnan(entropy):
        return float("nan")

    if normalization == "maxEntropy":
        denom = min(math.log(max(N, 2)), math.log(max(D, 2)))
        return entropy / denom if denom > 0 else float("nan")
    if normalization == "logN":
        denom = math.log(max(N, 2))
        return entropy / denom if denom > 0 else float("nan")
    if normalization == "logD":
        denom = math.log(max(D, 2))
        return entropy / denom if denom > 0 else float("nan")
    if normalization == "logNlogD":
        denom = math.log(max(N, 2)) * math.log(max(D, 2))
        return entropy / denom if denom > 0 else float("nan")
    if normalization == "length":
        return float(N)
    return float(entropy)


def center_features(X: Tensor) -> Tensor:
    """Feature-wise centering for [N, D] representations."""
    return X - X.mean(dim=0, keepdim=True)


def standardize_features(X: Tensor, eps: float = 1e-6) -> Tensor:
    """Feature-wise standardization for [N, D] representations."""
    return (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + eps)


def l2_normalize_rows(X: Tensor, eps: float = 1e-12) -> Tensor:
    """L2-normalize each row vector."""
    return X / (torch.linalg.norm(X, dim=1, keepdim=True) + eps)


def covariance_eigenvalues_from_svd(X: Tensor, center: bool = True, eps: float = 1e-12) -> Tensor:
    """
    Stable covariance eigenvalues using SVD.

    For centered X in R^{N x D}, nonzero eigenvalues of covariance X^T X/(N-1)
    are s_i^2/(N-1), where s_i are singular values of X.
    """
    X = X.detach().float()
    if center:
        X = center_features(X)
    N, _ = X.shape
    if N < 2:
        return torch.empty(0, device=X.device, dtype=X.dtype)
    s = torch.linalg.svdvals(X)
    eigvals = (s ** 2) / max(N - 1, 1)
    eigvals = eigvals.clamp_min(0.0)
    return eigvals[eigvals > eps]


def gaussian_affinity(
    X: Tensor,
    eps: float = 1e-12,
    standardize: bool = True,
    bandwidth: str = "median",
    bandwidth_scale: float = 0.5,
    k_nn: Optional[int] = None,
) -> Tensor:
    """
    Construct a symmetric Gaussian affinity graph from representations.

    Args:
        X: [N, D] representation matrix.
        bandwidth:
            'median'             -> tau2 = median(D2)
            'legacy_median_half' -> tau2 = median(D2) / 2
            'unit'               -> tau2 = 1
        k_nn: optional kNN sparsification before symmetrization.
    """
    X = X.detach().float()

    if standardize:
        X = standardize_features(X)
    else:
        X = center_features(X)

    N = X.shape[0]
    D2 = torch.cdist(X, X, p=2) ** 2

    upper = D2.triu(diagonal=1)
    positive = upper[upper > 0]

    if bandwidth == "median":
        upper = D2.triu(diagonal=1)
        positive = upper[upper > 0]
        med = positive.median() if positive.numel() > 0 else torch.tensor(1.0, device=X.device, dtype=X.dtype)
        tau2 = torch.clamp(float(bandwidth_scale) * med, min=eps)

    elif bandwidth == "unit":
        tau2 = torch.tensor(1.0, device=X.device, dtype=X.dtype)

    else:
        raise ValueError(
            "bandwidth must be 'median', 'legacy_median_half', or 'unit'."
        )

    W = torch.exp(-D2 / (2.0 * tau2 + eps))
    W.fill_diagonal_(0.0)

    if k_nn is not None and 0 < k_nn < N - 1:
        D2_masked = D2.clone()
        D2_masked.fill_diagonal_(float("inf"))

        nn_idx = torch.topk(D2_masked, k=k_nn, largest=False, dim=1).indices

        mask = torch.zeros_like(W, dtype=torch.bool)
        rows = torch.arange(N, device=X.device).view(-1, 1).expand_as(nn_idx)
        mask[rows, nn_idx] = True

        W = W * mask.float()

    return 0.5 * (W + W.T)


def normalized_laplacian(W: Tensor, eps: float = 1e-12) -> Tensor:
    """Symmetric normalized graph Laplacian L = I - D^{-1/2} W D^{-1/2}."""
    N = W.shape[0]
    deg = W.sum(dim=1).clamp_min(eps)
    d_inv_sqrt = deg.pow(-0.5)
    S = (d_inv_sqrt[:, None] * W) * d_inv_sqrt[None, :]
    L = torch.eye(N, device=W.device, dtype=W.dtype) - S
    return 0.5 * (L + L.T)
