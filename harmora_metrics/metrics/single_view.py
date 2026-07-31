"""
Single-view unsupervised representation metrics.

Canonical input:
    hidden_states: torch.Tensor [L, N, D]

Implemented metrics:
    - Harmora
    - Matrix Entropy
    - Participation Ratio
    - Anisotropy, Ethayarajh-style random-pair cosine
    - Intrinsic Dimension / TwoNN
    - Curvature
    - Spectral Gap / lambda_2
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .utils import (
    as_float,
    center_features,
    covariance_eigenvalues_from_svd,
    entropy_normalization,
    gaussian_affinity,
    l2_normalize_rows,
    normalized_laplacian,
    standardize_features,
    validate_hidden_states,
)

try:
    import repitl.matrix_itl as itl
except Exception:  # pragma: no cover
    itl = None

try:
    from dadapy.data import Data as ID_DATA
except Exception:  # pragma: no cover
    ID_DATA = None


def compute_harmora(
    hidden_states: torch.Tensor,
    sigma_l2: float = 1.0,
    K_l: Optional[int] = None,
    K_max: Optional[int] = None,
    eps: float = 1e-12,
    standardize: bool = False,
    bandwidth: str = "median",
    k_nn: Optional[int] = None,
) -> Dict[str, List[float]]:
    """
    Compute Harmora and an internally consistent K-curve.

    Official score:
        H_l(K_l) = log(1 + sum_{m=1}^{K_l} rho_m E_m)

    where m indexes the non-constant graph harmonic modes, i.e. the
    original Laplacian modes {2, ..., K_l+1}.

    If K_max is provided, the function also stores the full diagnostic curve
    H_l(K), K=1,...,K_max, computed inside the same graph/preprocessing path.

    Consistency guarantee:
        score[layer] == score_curve[layer][selected_K[layer]-1]
    for every valid layer.
    """
    L, N, D = validate_hidden_states(hidden_states)

    scores: List[float] = []
    lambda2s: List[float] = []
    lambda10s: List[float] = []
    energy_sums: List[float] = []
    precision_means: List[float] = []

    score_curves: List[List[float]] = []
    rho_energy_curves: List[List[float]] = []
    energy_curves: List[List[float]] = []
    rho_curves: List[List[float]] = []
    lambda_curves: List[List[float]] = []
    mass_curves: List[List[float]] = []
    mode_indices: List[List[int]] = []
    selected_Ks: List[int] = []
    n_modes_available: List[int] = []
    consistency_abs_diff: List[float] = []

    inv_sigma2 = 1.0 / max(float(sigma_l2), eps)

    for layer in hidden_states:
        Z = layer.detach().float()
        Z_graph = standardize_features(Z, eps=eps) if standardize else center_features(Z)
        n, d_l = Z_graph.shape

        def append_nan():
            scores.append(np.nan)
            lambda2s.append(np.nan)
            lambda10s.append(np.nan)
            energy_sums.append(np.nan)
            precision_means.append(np.nan)
            score_curves.append([])
            rho_energy_curves.append([])
            energy_curves.append([])
            rho_curves.append([])
            lambda_curves.append([])
            mass_curves.append([])
            mode_indices.append([])
            selected_Ks.append(0)
            n_modes_available.append(0)
            consistency_abs_diff.append(np.nan)

        if n < 3:
            append_nan()
            continue

        # Z_graph is already centered/standardized, so do not standardize again.
        W = gaussian_affinity(Z_graph, eps=eps, standardize=False, bandwidth=bandwidth, k_nn=k_nn)
        if W.sum() <= eps:
            append_nan()
            continue

        Lsym = normalized_laplacian(W, eps=eps)
        lambdas_all, U_all = torch.linalg.eigh(Lsym)
        lambdas_all = lambdas_all.clamp_min(0.0)

        # Store enough modes for diagnostics. Official K_l is selected from this curve.
        if K_max is None:
            K_store = n - 1
        else:
            K_store = min(int(K_max), n - 1)
        if K_store <= 0:
            append_nan()
            continue

        lambdas = lambdas_all[1:K_store + 1]
        U = U_all[:, 1:K_store + 1]

        coeff = Z_graph.T @ U
        E_m = (coeff ** 2).sum(dim=0) / max(d_l, 1)
        rho_m = inv_sigma2 + lambdas
        rho_energy = rho_m * E_m

        cumulative = torch.cumsum(rho_energy, dim=0)
        H_curve = torch.log1p(cumulative)
        mass_curve = cumulative / (rho_energy.sum() + eps)

        if K_l is None:
            K_selected = K_store
        else:
            K_selected = min(int(K_l), K_store)

        if K_selected <= 0:
            H_l = torch.tensor(float("nan"), device=Z_graph.device, dtype=Z_graph.dtype)
        else:
            H_l = H_curve[K_selected - 1]

        scores.append(as_float(H_l))
        lambda2s.append(as_float(lambdas_all[1]) if lambdas_all.numel() > 1 else np.nan)
        lambda10s.append(as_float(lambdas_all[9]) if lambdas_all.numel() > 9 else np.nan)

        energy_sums.append(as_float(E_m[:K_selected].sum()) if K_selected > 0 else np.nan)
        precision_means.append(as_float(rho_m[:K_selected].mean()) if K_selected > 0 else np.nan)

        curve_list = [as_float(x) for x in H_curve]
        score_curves.append(curve_list)
        rho_energy_curves.append([as_float(x) for x in rho_energy])
        energy_curves.append([as_float(x) for x in E_m])
        rho_curves.append([as_float(x) for x in rho_m])
        lambda_curves.append([as_float(x) for x in lambdas])
        mass_curves.append([as_float(x) for x in mass_curve])
        mode_indices.append(list(range(2, 2 + K_store)))
        selected_Ks.append(int(K_selected))
        n_modes_available.append(int(K_store))

        if K_selected > 0 and len(curve_list) >= K_selected:
            consistency_abs_diff.append(abs(as_float(H_l) - float(curve_list[K_selected - 1])))
        else:
            consistency_abs_diff.append(np.nan)

    return {
        "score": scores,
        "lambda2": lambda2s,
        "lambda10": lambda10s,
        "energy_sum": energy_sums,
        "precision_mean": precision_means,

        # Internally consistent K-diagnostic outputs.
        "score_curve": score_curves,
        "rho_energy_curve": rho_energy_curves,
        "energy_curve": energy_curves,
        "rho_curve": rho_curves,
        "lambda_curve": lambda_curves,
        "mass_curve": mass_curves,
        "mode_index": mode_indices,
        "selected_K": selected_Ks,
        "n_modes_available": n_modes_available,
        "curve_consistency_abs_diff": consistency_abs_diff,

        # Reproducibility metadata.
        "K_l": K_l,
        "K_max": K_max,
        "sigma_l2": sigma_l2,
        "standardize": standardize,
        "bandwidth": bandwidth,
        "k_nn": k_nn,
    }

def compute_matrix_entropy(
    hidden_states: torch.Tensor,
    alpha: float = 1,
    normalizations: Sequence[str] = ("maxEntropy",),
    eps: float = 1e-12,
) -> Dict[str, List[float]]:
    """
    Matrix-based entropy over representation Gram/covariance matrices.

    This follows the existing code style: build the smaller Gram/covariance matrix,
    trace-normalize it, then apply matrixAlphaEntropy from repitl.
    """
    if itl is None:
        raise ImportError("compute_matrix_entropy requires repitl. Install repitl or remove this metric.")

    L, N, D = validate_hidden_states(hidden_states)
    X = hidden_states.detach().float()

    if N > D:
        cov = torch.matmul(X.transpose(1, 2), X)  # [L, D, D]
    else:
        cov = torch.matmul(X, X.transpose(1, 2))  # [L, N, N]

    cov = torch.clamp(cov, min=0)
    entropies: List[float] = []

    for C in cov:
        try:
            C = C.double()
            tr = torch.trace(C)
            if tr <= eps:
                entropies.append(np.nan)
            else:
                C = C / tr
                entropies.append(float(itl.matrixAlphaEntropy(C, alpha=alpha).item()))
        except Exception:
            entropies.append(np.nan)

    return {norm: [entropy_normalization(x, norm, N, D) for x in entropies] for norm in normalizations}


def compute_participation_ratio(hidden_states: torch.Tensor, eps: float = 1e-12) -> Dict[str, List[float]]:
    """
    Participation Ratio effective dimensionality.

        PR = (sum_i lambda_i)^2 / sum_i lambda_i^2

    where lambda_i are covariance eigenvalues. This is a soft effective rank,
    not a graph metric and not supervised.
    """
    L, N, D = validate_hidden_states(hidden_states)
    rank_upper = max(1, min(N - 1, D))
    raw, normalized, inverse = [], [], []

    for layer in hidden_states:
        X = layer.detach().float()
        eigvals = covariance_eigenvalues_from_svd(X, center=True, eps=eps)
        if eigvals.numel() == 0 or eigvals.sum() <= eps:
            pr = float("nan")
        else:
            pr = as_float((eigvals.sum() ** 2) / ((eigvals ** 2).sum() + eps))
        raw.append(pr)
        normalized.append(pr / rank_upper if not np.isnan(pr) else np.nan)
        inverse.append(1.0 / pr if pr > eps else np.nan)

    return {"raw": raw, "normalized_rank": normalized, "inverse": inverse}


def compute_anisotropy(hidden_states: torch.Tensor, eps: float = 1e-12) -> Dict[str, List[float]]:
    """
    Ethayarajh-style anisotropy.

    For each layer, compute mean off-diagonal cosine similarity between uniformly
    sampled representation vectors. We do not mean-center features, because the
    common mean direction is part of the cone-effect anisotropy.
    """
    L, N, D = validate_hidden_states(hidden_states)
    cosine_mean, cosine_abs_mean = [], []

    for layer in hidden_states:
        X = layer.detach().float()
        if X.shape[0] < 2:
            cosine_mean.append(np.nan)
            cosine_abs_mean.append(np.nan)
            continue
        Xn = l2_normalize_rows(X, eps=eps)
        S = Xn @ Xn.T
        mask = ~torch.eye(X.shape[0], dtype=torch.bool, device=X.device)
        off = S[mask]
        cosine_mean.append(as_float(off.mean()))
        cosine_abs_mean.append(as_float(off.abs().mean()))

    return {"anisotropy": cosine_mean, "anisotropy_abs": cosine_abs_mean}


def compute_covariance_concentration(hidden_states: torch.Tensor, eps: float = 1e-12) -> Dict[str, List[float]]:
    """
    Auxiliary spectral concentration score: lambda_max / trace(covariance).

    This is not the Ethayarajh anisotropy metric. It is useful as an auxiliary
    spectral dominance diagnostic.
    """
    L, N, D = validate_hidden_states(hidden_states)
    raw, isotropy = [], []
    for layer in hidden_states:
        eigvals = covariance_eigenvalues_from_svd(layer, center=True, eps=eps)
        if eigvals.numel() == 0 or eigvals.sum() <= eps:
            score = float("nan")
        else:
            score = as_float(eigvals.max() / (eigvals.sum() + eps))
        raw.append(score)
        isotropy.append(1.0 - score if not np.isnan(score) else np.nan)
    return {"covariance_concentration": raw, "covariance_isotropy": isotropy}


def compute_intrinsic_dimension(hidden_states, eps=1e-12):
    """
    TwoNN intrinsic dimension without dadapy.
    Input: hidden_states [L, N, D]
    """
    import math
    import numpy as np
    import torch
    from sklearn.neighbors import NearestNeighbors

    raw = []
    logN = []

    for layer in hidden_states:
        X = layer.detach().float().cpu().numpy()

        if X.shape[0] < 3:
            raw.append(np.nan)
            logN.append(np.nan)
            continue

        X = X - X.mean(axis=0, keepdims=True)

        try:
            nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
            nn.fit(X)
            distances, _ = nn.kneighbors(X)

            r1 = distances[:, 1]
            r2 = distances[:, 2]

            valid = (r1 > eps) & (r2 > eps)
            mu = r2[valid] / r1[valid]

            if len(mu) < 3:
                dim = np.nan
            else:
                dim = 1.0 / (np.mean(np.log(mu + eps)) + eps)

            raw.append(float(dim))
            logN.append(float(dim / math.log(X.shape[0])) if X.shape[0] > 1 else np.nan)

        except Exception:
            raw.append(np.nan)
            logN.append(np.nan)

    return {"raw": raw, "logN": logN}



def compute_curvature(hidden_states: torch.Tensor, k: int = 1, eps: float = 1e-12) -> Dict[str, List[float]]:
    """
    Average discrete curvature along the sample order within each layer.

    This metric is meaningful when sample order has semantic structure, e.g., a trajectory,
    sequence, or time series. For shuffled iid samples, interpret with caution.
    """
    L, N, D = validate_hidden_states(hidden_states)
    curvatures: List[float] = []

    step = max(1, int(k))
    for layer in hidden_states:
        X = layer.detach().float()
        if X.shape[0] < 2 * step + 1:
            curvatures.append(np.nan)
            continue
        vals = []
        for i in range(step, X.shape[0] - step):
            v1 = X[i] - X[i - step]
            v2 = X[i + step] - X[i]
            denom = torch.linalg.norm(v1) * torch.linalg.norm(v2) + eps
            cos = torch.clamp(torch.dot(v1, v2) / denom, min=-1.0, max=1.0)
            vals.append(torch.arccos(cos))
        curvatures.append(as_float(torch.stack(vals).mean()) if vals else np.nan)

    return {"raw": curvatures, "logD": [x / math.log(max(D, 2)) if not np.isnan(x) else np.nan for x in curvatures]}


def compute_spectral_gap(
    hidden_states: torch.Tensor,
    eps: float = 1e-12,
    standardize: bool = False,
    bandwidth: str = "median",
    k_nn: Optional[int] = None,
    zero_tol: float = 1e-6,
) -> Dict[str, List[float]]:
    """
    Graph spectral gap for each representation layer.

    Builds the same Gaussian representation graph used by Harmora and reports
    lambda_2 of the symmetric normalized Laplacian.
    """
    L, N, D = validate_hidden_states(hidden_states)
    lambda2, lambda3, lambda10, num_components, normalized_lambda2, spectral_entropy = [], [], [], [], [], []

    for layer in hidden_states:
        X = layer.detach().float()
        if X.shape[0] < 3:
            lambda2.append(np.nan); lambda3.append(np.nan); lambda10.append(np.nan)
            num_components.append(np.nan); normalized_lambda2.append(np.nan); spectral_entropy.append(np.nan)
            continue

        W = gaussian_affinity(X, eps=eps, standardize=standardize, bandwidth=bandwidth, k_nn=k_nn)
        if W.sum() <= eps:
            lambda2.append(np.nan); lambda3.append(np.nan); lambda10.append(np.nan)
            num_components.append(np.nan); normalized_lambda2.append(np.nan); spectral_entropy.append(np.nan)
            continue

        Lsym = normalized_laplacian(W, eps=eps)
        eigvals = torch.linalg.eigvalsh(Lsym).clamp_min(0.0)
        nonconstant = eigvals[1:]
        denom = nonconstant.sum() + eps
        p = nonconstant / denom

        lambda2.append(as_float(eigvals[1]) if eigvals.numel() > 1 else np.nan)
        lambda3.append(as_float(eigvals[2]) if eigvals.numel() > 2 else np.nan)
        lambda10.append(as_float(eigvals[9]) if eigvals.numel() > 9 else np.nan)
        num_components.append(int((eigvals <= zero_tol).sum().detach().cpu().item()))
        normalized_lambda2.append(as_float(eigvals[1] / denom) if eigvals.numel() > 1 else np.nan)
        spectral_entropy.append(as_float(-(p * torch.log(p + eps)).sum()))

    return {
        "lambda2": lambda2,
        "lambda3": lambda3,
        "lambda10": lambda10,
        "num_components": num_components,
        "normalized_lambda2": normalized_lambda2,
        "spectral_entropy": spectral_entropy,
    }
