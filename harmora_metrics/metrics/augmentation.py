"""
Augmentation-based unsupervised representation metrics.

Canonical input:
    augmented_states: torch.Tensor [L, N, A, D]

Implemented metrics:
    - InfoNCE proxy
    - DiME
    - LiDAR
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch

from .utils import entropy_normalization, validate_augmented_states

try:
    import repitl.matrix_itl as itl
    import repitl.difference_of_entropies as dent
except Exception:  # pragma: no cover
    itl = None
    dent = None


def compute_infonce(augmented_states: torch.Tensor, temperature: float = 0.1, normalize: bool = True) -> Dict[str, List[float]]:
    """
    Compute InfoNCE loss/proxy for two augmented views.

    Args:
        augmented_states: [L, N, A, D], requires A == 2.
        temperature: softmax temperature.
        normalize: L2-normalize representations before logits.

    Returns:
        raw: cross-entropy InfoNCE loss per layer.
        mi_lower_bound: 1 - loss/log(N), matching the prior code convention.
    """
    L, N, A, D = validate_augmented_states(augmented_states)
    if A != 2:
        raise ValueError(f"InfoNCE requires exactly 2 views, got A={A}.")

    losses, normalized_scores = [], []
    labels = torch.arange(N, device=augmented_states.device)

    for layer in augmented_states:
        A_view = layer[:, 0, :].detach().float()
        B_view = layer[:, 1, :].detach().float()
        if normalize:
            A_view = A_view / (torch.linalg.norm(A_view, dim=1, keepdim=True) + 1e-12)
            B_view = B_view / (torch.linalg.norm(B_view, dim=1, keepdim=True) + 1e-12)
        logits = A_view @ B_view.T
        loss = torch.nn.functional.cross_entropy(logits / temperature, labels, reduction="mean")
        loss_val = float(loss.detach().cpu().item())
        losses.append(loss_val)
        normalized_scores.append(1.0 - loss_val / math.log(max(N, 2)))

    return {"raw": losses, "mi_lower_bound": normalized_scores}


def compute_dime(
    augmented_states: torch.Tensor,
    alpha: float = 1,
    normalizations: Sequence[str] = ("maxEntropy",),
) -> Dict[str, List[float]]:
    """
    Compute DiME for two augmented views using repitl.difference_of_entropies.

    Args:
        augmented_states: [L, N, A, D], requires A == 2.
    """
    if dent is None:
        raise ImportError("compute_dime requires repitl. Install repitl or remove this metric.")

    L, N, A, D = validate_augmented_states(augmented_states)
    if A != 2:
        raise ValueError(f"DiME requires exactly 2 views, got A={A}.")

    # Convert to [L, A, N, D] to match the original code logic.
    X = augmented_states.permute(0, 2, 1, 3).detach().float()

    if N > D:
        cov = torch.matmul(X.transpose(-1, -2), X)  # [L, A, D, D]
    else:
        cov = torch.matmul(X, X.transpose(-1, -2))  # [L, A, N, N]

    dimes: List[float] = []
    for idx in range(L):
        try:
            C_a = cov[idx, 0].double()
            C_b = cov[idx, 1].double()
            dimes.append(float(dent.doe(C_a, C_b, alpha=alpha, n_iters=10).item()))
        except Exception:
            dimes.append(np.nan)

    return {norm: [entropy_normalization(x, norm, N, D) for x in dimes] for norm in normalizations}


def compute_lda_matrix(augmented_layer: torch.Tensor, delta: float = 1e-4, return_within_class_scatter: bool = False) -> torch.Tensor:
    """
    Compute the LDA matrix used by LiDAR.

    Args:
        augmented_layer: [N, A, D] augmented representations for one layer.
    """
    N, A, D = augmented_layer.shape
    X = augmented_layer.detach().float()

    dataset_mean = X.mean(dim=(0, 1)).squeeze()      # [D]
    class_means = X.mean(dim=1)                     # [N, D]

    centered_class = class_means - dataset_mean
    between = (centered_class.T @ centered_class) / max(N, 1)

    residual = X - class_means[:, None, :]
    residual = residual.reshape(N * A, D)
    within = (residual.T @ residual) / max(N * A, 1)
    within = within + delta * torch.eye(D, device=X.device, dtype=X.dtype)

    if return_within_class_scatter:
        return within

    eigs, eigvecs = torch.linalg.eigh(within)
    eigs = eigs.clamp_min(delta)
    within_inv_sqrt = eigvecs @ torch.diag(eigs.pow(-0.5)) @ eigvecs.T
    return within_inv_sqrt @ between @ within_inv_sqrt


def compute_lidar(
    augmented_states: torch.Tensor,
    alpha: float = 1,
    normalizations: Sequence[str] = ("maxEntropy",),
    return_within_scatter: bool = False,
) -> Dict[str, List[float]]:
    """
    Compute LiDAR entropy score for augmented representations.

    Args:
        augmented_states: [L, N, A, D], with A >= 2 recommended.
    """
    if itl is None:
        raise ImportError("compute_lidar requires repitl. Install repitl or remove this metric.")

    L, N, A, D = validate_augmented_states(augmented_states)
    scores: List[float] = []

    for layer in augmented_states:
        try:
            lda = compute_lda_matrix(layer.double(), return_within_class_scatter=return_within_scatter)
            scores.append(float(itl.matrixAlphaEntropy(lda.double(), alpha=alpha).item()))
        except Exception:
            scores.append(np.nan)

    return {norm: [entropy_normalization(x, norm, N, D) for x in scores] for norm in normalizations}
