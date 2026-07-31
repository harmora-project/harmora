from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

EPS = 1e-12


def finite_pairs(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(x, dtype=float).reshape(-1)
    ya = np.asarray(y, dtype=float).reshape(-1)
    n = min(len(xa), len(ya))
    xa, ya = xa[:n], ya[:n]
    mask = np.isfinite(xa) & np.isfinite(ya)
    return xa[mask], ya[mask]


def distance_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = finite_pairs(x, y)
    if len(xa) < 3:
        return float("nan")
    xa = xa.reshape(-1, 1)
    ya = ya.reshape(-1, 1)
    a = np.abs(xa - xa.T)
    b = np.abs(ya - ya.T)
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = float(np.mean(A * B))
    dvarx = float(np.mean(A * A))
    dvary = float(np.mean(B * B))
    if dvarx <= EPS or dvary <= EPS:
        return float("nan")
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(np.sqrt(dvarx * dvary)))


def all_correlations(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    xa, ya = finite_pairs(x, y)
    if len(xa) < 3 or np.std(xa) <= EPS or np.std(ya) <= EPS:
        return {
            "pearson": float("nan"),
            "spearman": float("nan"),
            "kendall": float("nan"),
            "distance": distance_correlation(xa, ya),
            "n_layers": int(len(xa)),
        }
    return {
        "pearson": float(stats.pearsonr(xa, ya).statistic),
        "spearman": float(stats.spearmanr(xa, ya).statistic),
        "kendall": float(stats.kendalltau(xa, ya).statistic),
        "distance": distance_correlation(xa, ya),
        "n_layers": int(len(xa)),
    }


def sample_summary(values: Iterable[float], confidence_level: float = 0.95) -> dict[str, float]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = int(len(arr))
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "variance": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
        }
    mean = float(np.mean(arr))
    variance = float(np.var(arr, ddof=1)) if n > 1 else 0.0
    std = float(np.sqrt(variance))
    sem = float(std / np.sqrt(n)) if n > 0 else float("nan")
    if n > 1:
        critical = float(stats.t.ppf(0.5 + confidence_level / 2.0, n - 1))
        half = critical * sem
    else:
        half = 0.0
    return {
        "n": n,
        "mean": mean,
        "variance": variance,
        "std": std,
        "sem": sem,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 5000, confidence_level: float = 0.95, seed: int = 2025) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(arr) == 1:
        value = float(arr[0])
        return value, value, value
    rng = np.random.default_rng(int(seed))
    boot = rng.choice(arr, size=(int(n_boot), len(arr)), replace=True).mean(axis=1)
    alpha = 1.0 - float(confidence_level)
    return float(arr.mean()), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def rankdata_average(values: Sequence[float], descending: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    work = -arr if descending else arr
    return stats.rankdata(work, method="average")


def pairwise_preference_accuracy(score: Sequence[float], utility: Sequence[float], groups: Sequence[str] | None = None, mode: str = "all") -> tuple[float, int]:
    s = np.asarray(score, dtype=float)
    u = np.asarray(utility, dtype=float)
    g = np.asarray(groups if groups is not None else [""] * len(s), dtype=str)
    ok = 0
    total = 0
    for i in range(len(s) - 1):
        for j in range(i + 1, len(s)):
            if not all(np.isfinite([s[i], s[j], u[i], u[j]])):
                continue
            same = g[i] == g[j]
            if mode == "within" and not same:
                continue
            if mode == "cross" and same:
                continue
            if abs(s[i] - s[j]) <= EPS or abs(u[i] - u[j]) <= EPS:
                continue
            total += 1
            ok += int((s[i] - s[j]) * (u[i] - u[j]) > 0)
    return (float(ok / total), int(total)) if total else (float("nan"), 0)


def ndcg_at_k(score: Sequence[float], utility: Sequence[float], k: int) -> float:
    s, u = finite_pairs(score, utility)
    if len(s) < 2:
        return float("nan")
    relevance = u - np.min(u)
    if np.max(relevance) <= EPS:
        return float("nan")
    k = min(int(k), len(s))
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    order = np.argsort(-s, kind="mergesort")[:k]
    ideal = np.argsort(-relevance, kind="mergesort")[:k]
    dcg = float(np.sum(relevance[order] * discount))
    idcg = float(np.sum(relevance[ideal] * discount))
    return dcg / idcg if idcg > EPS else float("nan")


def topk_overlap(score: Sequence[float], utility: Sequence[float], k: int) -> float:
    s, u = finite_pairs(score, utility)
    if len(s) < 2:
        return float("nan")
    k = min(int(k), len(s))
    a = set(np.argsort(-s, kind="mergesort")[:k].tolist())
    b = set(np.argsort(-u, kind="mergesort")[:k].tolist())
    return float(len(a & b) / k)


def regret_at_1(score: Sequence[float], utility: Sequence[float]) -> dict[str, float]:
    s, u = finite_pairs(score, utility)
    if len(s) < 2:
        return {"selected_utility": float("nan"), "best_utility": float("nan"), "regret_at_1": float("nan"), "selected_percentile": float("nan")}
    idx = int(np.argmax(s))
    selected = float(u[idx])
    best = float(np.max(u))
    ranks = stats.rankdata(u, method="average")
    percentile = float((ranks[idx] - 1) / max(len(ranks) - 1, 1))
    return {"selected_utility": selected, "best_utility": best, "regret_at_1": best - selected, "selected_percentile": percentile}



def exact_sign_flip_pvalue(values: Sequence[float]) -> float:
    """Exact two-sided sign-flip test for a paired task-level effect.

    The input values must already represent independent experimental units
    (tasks in this package). Repeated seeds must be aggregated within task
    before this function is called.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(len(arr))
    if n == 0:
        return float("nan")
    observed = float(abs(np.mean(arr)))
    extreme = 0
    total = 1 << n
    for mask in range(total):
        signs = np.fromiter(
            (1.0 if (mask >> i) & 1 else -1.0 for i in range(n)),
            dtype=float,
            count=n,
        )
        statistic = float(abs(np.mean(arr * signs)))
        extreme += int(statistic >= observed - EPS)
    return float(extreme / total)


def paired_rank_biserial(values: Sequence[float]) -> float:
    """Paired rank-biserial correlation for non-zero paired differences."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (np.abs(arr) > EPS)]
    if len(arr) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(arr), method="average")
    positive = float(np.sum(ranks[arr > 0]))
    negative = float(np.sum(ranks[arr < 0]))
    denominator = positive + negative
    return float((positive - negative) / denominator) if denominator > 0 else 0.0

def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    finite_idx = np.where(np.isfinite(p))[0]
    if len(finite_idx) == 0:
        return out
    ordered = finite_idx[np.argsort(p[finite_idx])]
    m = len(ordered)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(ordered):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[rank] = running
    for rank, idx in enumerate(ordered):
        out[idx] = adjusted[rank]
    return out
