from __future__ import annotations

import warnings
from typing import Any, Dict, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, v_measure_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import set_global_seed
from .sampling import encode_labels


def _cosine_pairwise_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return np.sum(a_norm * b_norm, axis=1)


def classification_profile(
    embeddings: np.ndarray,
    labels: Sequence[Any],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    seed: int,
    max_iter: int = 3000,
) -> list[float]:
    """One primary evaluator: train-only standardized logistic-regression accuracy."""
    y = encode_labels(labels)
    train_idx = np.asarray(train_indices, dtype=int)
    test_idx = np.asarray(test_indices, dtype=int)
    if len(set(train_idx.tolist()) & set(test_idx.tolist())):
        raise RuntimeError("Train/test overlap detected inside classification evaluator.")
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Classification evaluator received an empty train or test split.")

    scores: list[float] = []
    for layer in range(embeddings.shape[0]):
        set_global_seed(seed, deterministic=True)
        pipeline = Pipeline(
            steps=[
                # Fitted only on X_train by Pipeline.fit; no full-data leakage.
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=int(seed),
                        max_iter=int(max_iter),
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            pipeline.fit(embeddings[layer, train_idx], y[train_idx])
        predictions = pipeline.predict(embeddings[layer, test_idx])
        scores.append(float(accuracy_score(y[test_idx], predictions)))
    return scores


def clustering_profile(
    embeddings: np.ndarray,
    labels: Sequence[Any],
    seed: int,
    n_init: int = 10,
    batch_size: int = 500,
) -> list[float]:
    """One primary evaluator: MiniBatchKMeans V-measure."""
    y = encode_labels(labels)
    n_clusters = int(len(np.unique(y)))
    if n_clusters < 2:
        raise RuntimeError("Clustering task has fewer than two classes.")
    scores: list[float] = []
    for layer in range(embeddings.shape[0]):
        set_global_seed(seed, deterministic=True)
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=int(seed),
            n_init=int(n_init),
            batch_size=min(int(batch_size), max(10, embeddings.shape[1])),
        )
        assignments = model.fit_predict(embeddings[layer])
        scores.append(float(v_measure_score(y, assignments)))
    return scores


def pair_classification_profile(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    labels: Sequence[Any],
) -> list[float]:
    """One primary evaluator: cosine-similarity average precision."""
    y = encode_labels(labels)
    if len(np.unique(y)) != 2:
        raise RuntimeError("PairClassification primary AP evaluator requires binary labels.")

    # Preserve conventional positive labels when possible. The fallback matches
    # the deterministic encoded-label order used by the rest of the pipeline.
    normalized = [str(x).strip().lower() for x in labels]
    positive_tokens = {"1", "true", "yes", "positive", "duplicate", "entailment", "match"}
    raw_positive = np.asarray([value in positive_tokens for value in normalized], dtype=bool)
    if raw_positive.any() and (~raw_positive).any():
        y_binary = raw_positive.astype(int)
    else:
        y_binary = (y == np.max(y)).astype(int)
    scores: list[float] = []
    for layer in range(embeddings_a.shape[0]):
        similarity = _cosine_pairwise_rows(embeddings_a[layer], embeddings_b[layer])
        scores.append(float(average_precision_score(y_binary, similarity)))
    return scores


def sts_profile(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    gold_scores: Sequence[float],
) -> list[float]:
    """One primary evaluator: cosine-similarity Spearman correlation."""
    gold = np.asarray(gold_scores, dtype=float)
    scores: list[float] = []
    for layer in range(embeddings_a.shape[0]):
        similarity = _cosine_pairwise_rows(embeddings_a[layer], embeddings_b[layer])
        result = spearmanr(gold, similarity, nan_policy="omit")
        statistic = getattr(result, "statistic", result[0])
        scores.append(float(statistic))
    return scores


def primary_metric_for_family(family: str) -> str:
    mapping = {
        "classification": "accuracy",
        "clustering": "v_measure",
        "pair_classification": "average_precision",
        "sts": "spearman",
    }
    try:
        return mapping[family]
    except KeyError as exc:
        raise ValueError(f"Unsupported probe family: {family}") from exc


def evaluate_primary_profile(
    task_payload: Dict[str, Any],
    embeddings: Dict[str, np.ndarray],
    seed: int,
    downstream_cfg: Dict[str, Any],
    split_manifest: Dict[str, Any] | None,
) -> tuple[list[float], dict[str, Any]]:
    family = task_payload["probe_family"]
    if family == "classification":
        if split_manifest is None:
            raise RuntimeError("Classification evaluation requires a split manifest.")
        settings = downstream_cfg.get("classification", {})
        profile = classification_profile(
            embeddings["embeddings"],
            task_payload["labels"],
            split_manifest["train_indices"],
            split_manifest["test_indices"],
            seed=seed,
            max_iter=int(settings.get("max_iter", 3000)),
        )
        meta = {
            "primary_evaluator": "logistic_regression",
            "primary_metric": "accuracy",
            "seed_affects_evaluation": True,
            "standardization": "fit_on_train_only",
            "split_hash": split_manifest["split_hash"],
        }
    elif family == "clustering":
        settings = downstream_cfg.get("clustering", {})
        profile = clustering_profile(
            embeddings["embeddings"],
            task_payload["labels"],
            seed=seed,
            n_init=int(settings.get("n_init", 10)),
            batch_size=int(settings.get("batch_size", 500)),
        )
        meta = {
            "primary_evaluator": "minibatch_kmeans",
            "primary_metric": "v_measure",
            "seed_affects_evaluation": True,
            "split_hash": None,
        }
    elif family == "pair_classification":
        profile = pair_classification_profile(
            embeddings["embeddings_a"],
            embeddings["embeddings_b"],
            task_payload["targets"],
        )
        meta = {
            "primary_evaluator": "cosine_similarity",
            "primary_metric": "average_precision",
            "seed_affects_evaluation": False,
            "split_hash": None,
        }
    elif family == "sts":
        profile = sts_profile(
            embeddings["embeddings_a"],
            embeddings["embeddings_b"],
            task_payload["targets"],
        )
        meta = {
            "primary_evaluator": "cosine_similarity",
            "primary_metric": "spearman",
            "seed_affects_evaluation": False,
            "split_hash": None,
        }
    else:
        raise ValueError(f"Unsupported family: {family}")

    if not profile:
        raise RuntimeError(f"No layerwise scores were generated for {task_payload['task']}.")
    return profile, meta
