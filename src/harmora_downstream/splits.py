from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .io_utils import load_json, safe_name, save_json
from .sampling import encode_labels, stable_hash


def split_manifest_path(output_dir: str | Path, task_name: str, seed: int) -> Path:
    return Path(output_dir) / "split_manifests" / safe_name(task_name) / f"seed_{int(seed)}.json"


def _label_counts(y: np.ndarray, indices: Sequence[int]) -> dict[str, int]:
    counts = Counter(int(y[int(i)]) for i in indices)
    return {str(k): int(v) for k, v in sorted(counts.items())}


def deterministic_stratified_split(
    labels: Sequence[Any],
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Deterministic per-class split.

    Every class with at least two samples contributes to both train and test.
    Singleton classes remain in train. This avoids silent sample loss and makes
    class coverage explicit in the manifest.
    """
    y = encode_labels(labels)
    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    singleton_classes: list[int] = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0].astype(int)
        cls_idx = rng.permutation(cls_idx)
        n = len(cls_idx)
        if n == 1:
            singleton_classes.append(int(cls))
            train_parts.append(cls_idx)
            continue
        n_test = int(round(n * float(test_size)))
        n_test = max(1, min(n - 1, n_test))
        test_parts.append(cls_idx[:n_test])
        train_parts.append(cls_idx[n_test:])

    train_idx = np.concatenate(train_parts) if train_parts else np.asarray([], dtype=int)
    test_idx = np.concatenate(test_parts) if test_parts else np.asarray([], dtype=int)
    train_idx = rng.permutation(train_idx).astype(int)
    test_idx = rng.permutation(test_idx).astype(int)

    all_indices = set(range(len(y)))
    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    if train_set & test_set:
        raise RuntimeError("Train/test overlap detected while creating split.")
    if train_set | test_set != all_indices:
        missing = sorted(all_indices - (train_set | test_set))
        raise RuntimeError(f"Split does not cover all examples; missing indices={missing[:20]}")
    if len(test_idx) == 0:
        raise RuntimeError("Test split is empty. Increase the task sample size.")

    metadata = {
        "strategy": "deterministic_per_class_holdout",
        "singleton_classes_kept_in_train": singleton_classes,
        "n_singleton_classes": len(singleton_classes),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "actual_test_fraction": float(len(test_idx) / max(len(y), 1)),
        "train_label_counts": _label_counts(y, train_idx),
        "test_label_counts": _label_counts(y, test_idx),
    }
    return train_idx, test_idx, metadata


def get_or_create_split_manifest(
    output_dir: str | Path,
    task_payload: Dict[str, Any],
    seed: int,
    test_size: float,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if task_payload.get("probe_family") != "classification":
        raise ValueError("Train/test manifests are created only for Classification tasks.")

    path = split_manifest_path(output_dir, task_payload["task"], seed)
    if path.exists() and not overwrite:
        manifest = load_json(path)
        sample_ok = manifest.get("sample_hash") == task_payload.get("sample_hash")
        seed_ok = int(manifest.get("seed", -1)) == int(seed)
        size_ok = np.isclose(
            float(manifest.get("requested_test_size", -1.0)),
            float(test_size),
            rtol=0.0,
            atol=1e-12,
        )
        if not (sample_ok and seed_ok and size_ok):
            raise RuntimeError(
                f"Existing split manifest settings mismatch for {task_payload['task']} seed={seed}: "
                f"sample_ok={sample_ok}, seed_ok={seed_ok}, test_size_ok={size_ok}. "
                "Rerun with --overwrite-splits and --overwrite-results."
            )
        return manifest

    train_idx, test_idx, metadata = deterministic_stratified_split(
        task_payload["labels"],
        test_size=float(test_size),
        seed=int(seed),
    )
    manifest = {
        "task": task_payload["task"],
        "task_type": task_payload["task_type"],
        "probe_family": task_payload["probe_family"],
        "sample_hash": task_payload["sample_hash"],
        "seed": int(seed),
        "requested_test_size": float(test_size),
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        **metadata,
    }
    manifest["split_hash"] = stable_hash(
        {
            "task": manifest["task"],
            "sample_hash": manifest["sample_hash"],
            "seed": manifest["seed"],
            "train_indices": manifest["train_indices"],
            "test_indices": manifest["test_indices"],
        }
    )
    save_json(path, manifest)
    return manifest
