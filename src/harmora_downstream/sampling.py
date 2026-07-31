from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from .io_utils import load_json, safe_name, save_json
from .tasks import (
    LABEL_CANDIDATES,
    TEXT_A_CANDIDATES,
    TEXT_B_CANDIDATES,
    TEXT_CANDIDATES,
    classify_probe_family,
    find_column,
    load_task_dataset,
    task_name,
    task_type,
    text_from_row,
)


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_hash(value: Any, length: int = 20) -> str:
    text = json.dumps(canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _encoded_labels(labels: Sequence[Any]) -> np.ndarray:
    keys = [json.dumps(canonical(v), ensure_ascii=False, sort_keys=True) for v in labels]
    mapping = {value: i for i, value in enumerate(sorted(set(keys)))}
    return np.asarray([mapping[value] for value in keys], dtype=int)


def _sample_indices(
    n_rows: int,
    max_samples: int,
    seed: int,
    labels: Sequence[Any] | None = None,
) -> tuple[np.ndarray, str]:
    if n_rows <= 0:
        return np.asarray([], dtype=int), "empty"
    if max_samples <= 0 or n_rows <= max_samples:
        return np.arange(n_rows, dtype=int), "all_rows"

    all_indices = np.arange(n_rows, dtype=int)
    if labels is not None:
        encoded = _encoded_labels(labels)
        try:
            selected, _ = train_test_split(
                all_indices,
                train_size=max_samples,
                random_state=int(seed),
                stratify=encoded,
                shuffle=True,
            )
            return np.sort(np.asarray(selected, dtype=int)), "stratified_random_sample"
        except Exception:
            pass

    rng = np.random.default_rng(int(seed))
    selected = rng.choice(all_indices, size=max_samples, replace=False)
    return np.sort(selected.astype(int)), "uniform_random_sample"


def cache_path(output_dir: str | Path, task: str) -> Path:
    return Path(output_dir) / "sample_cache" / f"{safe_name(task)}.json"


def build_task_cache(task: Any, cfg: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
    output_dir = Path(cfg["_resolved_output_dir"])
    name = task_name(task)
    ttype = task_type(task)
    family = classify_probe_family(ttype, name)
    sampling_cfg = cfg.get("sampling", {})
    max_samples = int(sampling_cfg.get("max_samples_per_task", 300))
    sample_seed = int(cfg.get("sample_seed", 2025))
    priorities = (
        sampling_cfg.get(
            "labeled_split_priority",
            ["test", "validation", "dev", "train"],
        )
        if family in {"classification", "clustering"}
        else sampling_cfg.get(
            "pair_split_priority",
            ["test", "validation", "dev", "train"],
        )
    )
    sampling_fingerprint = stable_hash({
        "task": name,
        "task_type": ttype,
        "family": family,
        "sample_seed": sample_seed,
        "max_samples": max_samples,
        "split_priority": list(priorities),
    })
    path = cache_path(output_dir, name)
    if path.exists() and not overwrite:
        cached = load_json(path)
        if cached.get("sampling_fingerprint") != sampling_fingerprint:
            raise RuntimeError(
                f"Existing sample cache settings mismatch for {name}. "
                "Rerun 02_build_sample_cache.py with --overwrite, then rebuild embeddings/results."
            )
        return cached

    dataset, meta = load_task_dataset(task, priorities)
    columns = list(meta["columns"])

    payload: Dict[str, Any] = {
        "task": name,
        "task_type": ttype,
        "probe_family": family,
        "sample_seed": sample_seed,
        "max_samples": max_samples,
        "sampling_fingerprint": sampling_fingerprint,
        "source_subset": meta.get("subset"),
        "source_split": meta.get("split"),
        "source_num_rows": int(meta.get("num_rows", len(dataset))),
        "columns": columns,
        "status": "ok",
        "reason": "",
    }

    if family in {"classification", "clustering"}:
        text_col = find_column(columns, TEXT_CANDIDATES)
        label_col = find_column(columns, LABEL_CANDIDATES)
        if label_col is None:
            raise RuntimeError(f"No label column found for {name}; columns={columns}")
        labels_all = dataset[label_col]
        selected, strategy = _sample_indices(len(dataset), max_samples, sample_seed, labels_all)
        texts, labels, source_indices = [], [], []
        for idx in selected.tolist():
            row = dataset[int(idx)]
            text = text_from_row(row, columns, text_col)
            label = row.get(label_col)
            if text is None or label is None:
                continue
            texts.append(str(text))
            labels.append(canonical(label))
            source_indices.append(int(idx))
        payload.update({
            "sampling_strategy": strategy,
            "text_column": text_col,
            "label_column": label_col,
            "texts": texts,
            "labels": labels,
            "source_indices": source_indices,
            "n_items": len(texts),
            "n_classes": int(len(np.unique(_encoded_labels(labels)))) if labels else 0,
        })
        if len(texts) < 10 or payload["n_classes"] < 2:
            payload["status"] = "warning"
            payload["reason"] = "Too few usable labeled examples or classes."

    elif family in {"sts", "pair_classification"}:
        a_col = find_column(columns, TEXT_A_CANDIDATES)
        b_col = find_column(columns, TEXT_B_CANDIDATES)
        label_candidates = (
            ["score", "similarity_score", "relatedness_score", "label", "labels"]
            if family == "sts"
            else ["label", "labels", "target", "score"]
        )
        label_col = find_column(columns, label_candidates)
        if a_col is None or b_col is None or label_col is None:
            raise RuntimeError(
                f"Could not identify pair columns for {name}; columns={columns}, "
                f"a={a_col}, b={b_col}, label={label_col}"
            )
        labels_all = dataset[label_col]
        stratify_labels = labels_all if family == "pair_classification" else None
        selected, strategy = _sample_indices(len(dataset), max_samples, sample_seed, stratify_labels)
        sentence1, sentence2, targets, source_indices = [], [], [], []
        for idx in selected.tolist():
            row = dataset[int(idx)]
            a, b, target = row.get(a_col), row.get(b_col), row.get(label_col)
            if a is None or b is None or target is None:
                continue
            try:
                value = float(target) if family == "sts" else canonical(target)
            except Exception:
                continue
            sentence1.append(str(a))
            sentence2.append(str(b))
            targets.append(value)
            source_indices.append(int(idx))
        payload.update({
            "sampling_strategy": strategy,
            "sentence1_column": a_col,
            "sentence2_column": b_col,
            "target_column": label_col,
            "sentence1": sentence1,
            "sentence2": sentence2,
            "targets": targets,
            "source_indices": source_indices,
            "n_items": len(targets),
        })
        if len(targets) < 3:
            payload["status"] = "warning"
            payload["reason"] = "Too few usable pair examples."

    else:
        raise AssertionError(family)

    hash_payload = {
        key: payload[key]
        for key in payload
        if key not in {"sample_hash", "status", "reason"}
    }
    payload["sample_hash"] = stable_hash(hash_payload)
    save_json(path, payload)
    return payload


def load_task_cache(output_dir: str | Path, task: str) -> Dict[str, Any]:
    return load_json(cache_path(output_dir, task))


def encode_labels(labels: Sequence[Any]) -> np.ndarray:
    return _encoded_labels(labels)
