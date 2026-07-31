from __future__ import annotations

import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from .config import ensure_output_dirs, resolve_device, resolve_path, set_global_seed
from .embedding_cache import build_or_load_embeddings
from .encoder import EncoderConfig, LayerwiseEncoder, array_hash
from .io_utils import load_json, safe_name, save_json
from .models import get_model_specs
from .sampling import build_task_cache, stable_hash
from .tasks import get_selected_tasks, task_name


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def add_metrics_package(cfg: Dict[str, Any]) -> Path:
    path = resolve_path(cfg, "metrics_package_dir")
    if not (path / "metrics" / "registry.py").exists():
        raise FileNotFoundError(
            f"Harmora metrics package is missing at {path}. Expected metrics/registry.py."
        )
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    return path


def build_metric_config(cfg: Dict[str, Any]):
    add_metrics_package(cfg)
    from metrics import MetricConfig

    raw = cfg.get("metrics", {}).get("config", {})
    graph_k_nn = raw.get("graph_k_nn", raw.get("spectral_gap_k_nn"))
    return MetricConfig(
        harmora_sigma_l2=float(raw.get("harmora_sigma_l2", 1.0)),
        harmora_K_l=raw.get("harmora_K_l"),
        harmora_K_max=raw.get("harmora_K_max", 80),
        graph_bandwidth=str(raw.get("graph_bandwidth", "median")),
        graph_k_nn=graph_k_nn,
        graph_standardize=bool(raw.get("graph_standardize", False)),
        entropy_alpha=float(raw.get("entropy_alpha", 1.0)),
        entropy_normalizations=tuple(raw.get("entropy_normalizations", ["maxEntropy"])),
        infonce_temperature=float(raw.get("infonce_temperature", 0.1)),
        infonce_normalize=bool(raw.get("infonce_normalize", True)),
        curvature_k=int(raw.get("curvature_k", 1)),
        eps=float(raw.get("eps", 1e-12)),
    )


def metric_json_path(output_dir: str | Path, model_alias: str, task: str) -> Path:
    return Path(output_dir) / "metrics" / safe_name(model_alias) / f"{safe_name(task)}.json"


def augmentation_cache_path(output_dir: str | Path, model_alias: str, task: str) -> Path:
    return Path(output_dir) / "augmentation_cache" / safe_name(model_alias) / f"{safe_name(task)}.npz"


def _metric_texts(payload: Dict[str, Any]) -> list[str]:
    family = payload["probe_family"]
    if family in {"classification", "clustering"}:
        return [str(x) for x in payload["texts"]]
    if family in {"pair_classification", "sts"}:
        # Same sampled rows as downstream. Both members of every sampled pair are used.
        return [str(x) for x in payload["sentence1"]] + [str(x) for x in payload["sentence2"]]
    raise ValueError(f"Unsupported family: {family}")


def _metric_hidden_states(payload: Dict[str, Any], embeddings: Dict[str, Any]) -> np.ndarray:
    family = payload["probe_family"]
    if family in {"classification", "clustering"}:
        return np.asarray(embeddings["embeddings"], dtype=np.float32)
    if family in {"pair_classification", "sts"}:
        a = np.asarray(embeddings["embeddings_a"], dtype=np.float32)
        b = np.asarray(embeddings["embeddings_b"], dtype=np.float32)
        if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
            raise RuntimeError(f"Pair representation shape mismatch: {a.shape} vs {b.shape}")
        return np.concatenate([a, b], axis=1)
    raise ValueError(f"Unsupported family: {family}")


def _drop_words(text: str, drop_prob: float, rng: random.Random) -> str:
    words = str(text).split()
    if len(words) <= 3:
        return str(text)
    kept = [word for word in words if rng.random() > drop_prob]
    if not kept:
        kept = [rng.choice(words)]
    return " ".join(kept)


def make_text_views(texts: Sequence[str], num_views: int, seed: int, drop_prob: float) -> list[list[str]]:
    rng = random.Random(int(seed))
    views: list[list[str]] = []
    for text in texts:
        row = [str(text)]
        for _ in range(1, int(num_views)):
            row.append(_drop_words(str(text), drop_prob=drop_prob, rng=rng))
        views.append(row)
    return views


def _augmentation_fingerprint(
    model_alias: str,
    task: str,
    sample_hash: str,
    encoder_fingerprint: str,
    num_views: int,
    seed: int,
    drop_prob: float,
) -> str:
    return stable_hash({
        "model_alias": model_alias,
        "task": task,
        "sample_hash": sample_hash,
        "encoder_fingerprint": encoder_fingerprint,
        "num_views": int(num_views),
        "seed": int(seed),
        "drop_prob": float(drop_prob),
    })


def build_or_load_augmented_embeddings(
    cfg: Dict[str, Any],
    output_dir: Path,
    payload: Dict[str, Any],
    model_spec: Any,
    encoder_cfg: EncoderConfig,
    encoder_fingerprint: str,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    extraction = cfg.get("metric_extraction", {})
    views_cfg = extraction.get("augmentations", {})
    num_views = max(
        int(views_cfg.get("infonce_views", 2)),
        int(views_cfg.get("dime_views", 2)),
        int(views_cfg.get("lidar_views", 16)),
    )
    seed = int(cfg.get("augmentation_seed", 2026))
    drop_prob = float(extraction.get("augmentation_drop_probability", 0.12))
    path = augmentation_cache_path(output_dir, model_spec.alias, payload["task"])
    fingerprint = _augmentation_fingerprint(
        model_spec.alias,
        payload["task"],
        payload["sample_hash"],
        encoder_fingerprint,
        num_views,
        seed,
        drop_prob,
    )
    if path.exists() and not overwrite:
        with np.load(path, allow_pickle=False) as archive:
            stored = str(np.asarray(archive["augmentation_fingerprint"]).item())
            if stored != fingerprint:
                raise RuntimeError(
                    f"Augmentation cache settings mismatch for {model_spec.alias}/{payload['task']}. "
                    "Use --overwrite-augmentations."
                )
            return {
                "augmented_embeddings": np.asarray(archive["augmented_embeddings"], dtype=np.float32),
                "augmentation_hash": str(np.asarray(archive["augmentation_hash"]).item()),
                "augmentation_fingerprint": stored,
                "num_views": int(np.asarray(archive["num_views"]).item()),
                "cache_hit": True,
            }

    texts = _metric_texts(payload)
    text_views = make_text_views(texts, num_views=num_views, seed=seed, drop_prob=drop_prob)
    flattened = [view for row in text_views for view in row]
    encoder = LayerwiseEncoder(model_spec, encoder_cfg)
    encoded = encoder.encode(flattened, show_progress=show_progress)

    # The model is no longer needed once pooled layer embeddings are available.
    # Releasing it before hashing/compression lowers peak memory considerably.
    del encoder, flattened, text_views
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    layers, total, dim = encoded.shape
    if total != len(texts) * num_views:
        raise RuntimeError(
            f"Augmentation reshape mismatch: encoded={total}, expected={len(texts) * num_views}"
        )
    augmented = encoded.reshape(layers, len(texts), num_views, dim).astype(np.float32, copy=False)
    aug_hash = array_hash(augmented)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        augmented_embeddings=augmented,
        augmentation_hash=np.asarray(aug_hash),
        augmentation_fingerprint=np.asarray(fingerprint),
        num_views=np.asarray(num_views),
        sample_hash=np.asarray(payload["sample_hash"]),
    )
    # ``augmented`` owns the encoded array view returned to metric computation.
    # The encoder/model has already been released above.
    gc.collect()
    return {
        "augmented_embeddings": augmented,
        "augmentation_hash": aug_hash,
        "augmentation_fingerprint": fingerprint,
        "num_views": num_views,
        "cache_hit": False,
    }


def _harmora_consistency(results: Dict[str, Any], tolerance: float = 1e-8) -> Dict[str, Any]:
    harmora = results.get("harmora", {}) if isinstance(results, dict) else {}
    score = harmora.get("score", [])
    curves = harmora.get("score_curve", [])
    selected = harmora.get("selected_K", [])
    diffs = []
    for layer, value in enumerate(score):
        try:
            k = int(selected[layer])
            curve_value = float(curves[layer][k - 1]) if k > 0 else float("nan")
            diffs.append(abs(float(value) - curve_value))
        except Exception:
            diffs.append(float("nan"))
    finite = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    return {
        "tolerance": tolerance,
        "n_layers": len(score),
        "n_checked": int(len(finite)),
        "max_abs_diff": float(np.max(finite)) if len(finite) else float("nan"),
        "n_bad_layers": int(np.sum(finite > tolerance)) if len(finite) else 0,
    }


def extract_metrics(
    cfg: Dict[str, Any],
    model_aliases: Iterable[str] | None = None,
    requested_tasks: Iterable[str] | None = None,
    overwrite_metrics: bool = False,
    overwrite_embeddings: bool = False,
    overwrite_augmentations: bool = False,
    show_progress: bool = True,
) -> Dict[str, Path]:
    ensure_output_dirs(cfg)
    add_metrics_package(cfg)
    from metrics import compute_all_metrics

    output_dir = resolve_path(cfg, "output_dir")
    model_cache_dir = resolve_path(cfg, "model_cache_dir")
    models = get_model_specs(cfg, model_aliases)
    tasks = get_selected_tasks(cfg, requested_tasks)
    metric_names = list(cfg.get("metrics", {}).get("names", []))
    primary_fields = dict(cfg.get("metrics", {}).get("primary_fields", {}))
    directions = {str(k): int(v) for k, v in cfg.get("metrics", {}).get("directions", {}).items()}
    metric_cfg = build_metric_config(cfg)
    extraction_cfg = cfg.get("metric_extraction", {})
    compute_aug = bool(extraction_cfg.get("compute_augmentation_metrics", True))
    skip_errors = bool(extraction_cfg.get("skip_metric_errors", False))

    enc_raw = cfg.get("encoding", {})
    encoder_cfg = EncoderConfig(
        max_length=int(enc_raw.get("max_length", 256)),
        batch_size=int(enc_raw.get("batch_size", 16)),
        device=resolve_device(enc_raw.get("device", "auto")),
        dtype=str(enc_raw.get("dtype", "float32")),
        include_embedding_layer=bool(enc_raw.get("include_embedding_layer", True)),
        model_cache_dir=str(model_cache_dir),
    )

    long_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for model in models:
        for task in tasks:
            name = task_name(task)
            payload = build_task_cache(task, cfg, overwrite=False)
            if payload.get("status") != "ok":
                raise RuntimeError(f"Task cache not usable for metrics: {name}: {payload.get('reason')}")
            path = metric_json_path(output_dir, model.alias, name)
            if path.exists() and not overwrite_metrics:
                obj = load_json(path)
                if obj.get("sample_hash") != payload["sample_hash"]:
                    raise RuntimeError(
                        f"Metric sample hash mismatch for {model.alias}/{name}; use --overwrite-metrics."
                    )
                results = obj["metrics"]
                runtime_rows.append({
                    "model_alias": model.alias,
                    "task": name,
                    "stage": "metric_extraction",
                    "seconds": 0.0,
                    "status": "reused",
                })
            else:
                print(f"\n[METRICS] model={model.alias} task={name}")
                start = time.perf_counter()
                embeddings = build_or_load_embeddings(
                    output_dir,
                    payload,
                    model,
                    encoder_cfg,
                    overwrite=overwrite_embeddings,
                    show_progress=show_progress,
                )
                hidden_np = _metric_hidden_states(payload, embeddings)
                raw_embedding_layer_retained = (
                    encoder_cfg.include_embedding_layer
                    and model.pooling.lower() != "cls"
                )

                layer_start = 0 if raw_embedding_layer_retained else 1
                layer_indices = list(
                    range(layer_start, layer_start + int(hidden_np.shape[0]))
                )
                hidden = torch.from_numpy(hidden_np)
                metric_representation_hash = array_hash(hidden_np)

                augmented = None
                augmentation_meta: Dict[str, Any] = {}
                if compute_aug and any(x in metric_names for x in ["infonce", "dime", "lidar"]):
                    aug_obj = build_or_load_augmented_embeddings(
                        cfg,
                        output_dir,
                        payload,
                        model,
                        encoder_cfg,
                        embeddings["encoder_fingerprint"],
                        overwrite=overwrite_augmentations,
                        show_progress=show_progress,
                    )
                    augmented = torch.from_numpy(aug_obj["augmented_embeddings"])
                    augmentation_meta = {
                        "augmentation_hash": aug_obj["augmentation_hash"],
                        "augmentation_fingerprint": aug_obj["augmentation_fingerprint"],
                        "num_augmentation_views": aug_obj["num_views"],
                    }

                results = compute_all_metrics(
                    hidden_states=hidden,
                    augmented_states=augmented,
                    metrics=metric_names,
                    config=metric_cfg,
                    skip_errors=skip_errors,
                )
                consistency = _harmora_consistency(results)
                if consistency["n_bad_layers"]:
                    raise RuntimeError(
                        f"Harmora K-curve consistency failed for {model.alias}/{name}: {consistency}"
                    )
                elapsed = time.perf_counter() - start
                obj = {
                    "experiment_name": cfg.get("experiment_name"),
                    "model_alias": model.alias,
                    "hf_name": model.hf_name,
                    "task": name,
                    "task_type": payload["task_type"],
                    "probe_family": payload["probe_family"],
                    "sample_hash": payload["sample_hash"],
                    "embedding_hash": embeddings["embedding_hash"],
                    "metric_representation_hash": metric_representation_hash,
                    "encoder_fingerprint": embeddings["encoder_fingerprint"],
                    "n_task_rows": int(payload["n_items"]),
                    "n_metric_representations": int(hidden_np.shape[1]),
                    "num_layers": int(hidden_np.shape[0]),
                    "layer_indices": layer_indices,
                    "dimension": int(hidden_np.shape[2]),
                    "pair_metric_policy": "concatenate_both_sides" if payload["probe_family"] in {"pair_classification", "sts"} else "single_text",
                    "metric_names": metric_names,
                    "primary_fields": primary_fields,
                    "directions": directions,
                    "metric_config": _jsonable(cfg.get("metrics", {}).get("config", {})),
                    "harmora_curve_consistency": consistency,
                    "runtime_seconds": float(elapsed),
                    "metrics": _jsonable(results),
                    **augmentation_meta,
                }
                save_json(path, obj)
                runtime_rows.append({
                    "model_alias": model.alias,
                    "task": name,
                    "stage": "metric_extraction",
                    "seconds": float(elapsed),
                    "status": "computed",
                })
                del hidden, augmented, hidden_np, embeddings
                if "aug_obj" in locals():
                    del aug_obj
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Recover the physical layer numbers saved in the metric JSON.
            # This is essential for CLS-pooled models, whose raw embedding
            # layer has been removed and whose valid layers start from 1.
            layer_indices_raw = obj.get("layer_indices")

            if layer_indices_raw is None:
                raise RuntimeError(
                    f"Metric JSON has no layer_indices for "
                    f"{model.alias}/{name}. Recompute this metric JSON."
                )

            layer_indices = [int(value) for value in layer_indices_raw]
            expected_num_layers = int(
                obj.get("num_layers", len(layer_indices))
            )

            if len(layer_indices) != expected_num_layers:
                raise RuntimeError(
                    f"Layer-index length mismatch for {model.alias}/{name}: "
                    f"num_layers={expected_num_layers}, "
                    f"layer_indices={len(layer_indices)}"
                )

            if len(set(layer_indices)) != len(layer_indices):
                raise RuntimeError(
                    f"Duplicate layer indices for {model.alias}/{name}: "
                    f"{layer_indices}"
                )

            if layer_indices != sorted(layer_indices):
                raise RuntimeError(
                    f"Layer indices are not ordered for "
                    f"{model.alias}/{name}: {layer_indices}"
                )

            # Flatten primary metric profiles while preserving the actual
            # physical layer numbers.
            for metric_name in metric_names:
                result = results.get(metric_name, {})
                if not isinstance(result, dict):
                    continue

                field = primary_fields[metric_name]
                values = result.get(field)

                if values is None:
                    raise RuntimeError(
                        f"Primary field {metric_name}.{field} missing for "
                        f"{model.alias}/{name}. Available={list(result)}"
                    )

                values = list(values)
                feature = f"{metric_name}.{field}"
                direction = int(directions[feature])

                if len(values) != len(layer_indices):
                    raise RuntimeError(
                        f"Metric profile length mismatch for "
                        f"{model.alias}/{name}/{feature}: "
                        f"values={len(values)}, "
                        f"layer_indices={len(layer_indices)}"
                    )

                for layer, raw_value in zip(layer_indices, values):
                    value = float(raw_value)

                    if not np.isfinite(value):
                        raise RuntimeError(
                            f"Non-finite metric value for "
                            f"{model.alias}/{name}/{feature}/"
                            f"layer={layer}: {value}"
                        )

                    long_rows.append({
                        "model_alias": model.alias,
                        "hf_name": model.hf_name,
                        "task": name,
                        "task_type": obj.get("task_type"),
                        "probe_family": obj.get("probe_family"),
                        "layer": int(layer),
                        "metric": metric_name,
                        "field": field,
                        "feature": feature,
                        "direction": direction,
                        "raw_value": value,
                        "oriented_value": direction * value,
                        "sample_hash": obj.get("sample_hash"),
                        "embedding_hash": obj.get("embedding_hash"),
                        "metric_representation_hash": obj.get(
                            "metric_representation_hash"
                        ),
                    })

            # Flatten Harmora K-curves. Position indexes the internal arrays;
            # layer is the actual physical Transformer layer number.
            harmora = results.get("harmora", {})
            score_curves = harmora.get("score_curve", [])
            mass_curves = harmora.get("mass_curve", [])
            energy_curves = harmora.get("energy_curve", [])
            rho_curves = harmora.get("rho_curve", [])
            lambda_curves = harmora.get("lambda_curve", [])
            selected_ks = harmora.get("selected_K", [])

            if len(score_curves) != len(layer_indices):
                raise RuntimeError(
                    f"Harmora curve/layer mismatch for "
                    f"{model.alias}/{name}: "
                    f"score_curves={len(score_curves)}, "
                    f"layer_indices={len(layer_indices)}"
                )

            for position, (layer, score_curve) in enumerate(
                zip(layer_indices, score_curves)
            ):
                for index, h_value in enumerate(score_curve):
                    mass_value = (
                        float(mass_curves[position][index])
                        if position < len(mass_curves)
                        and index < len(mass_curves[position])
                        else float("nan")
                    )

                    energy_value = (
                        float(energy_curves[position][index])
                        if position < len(energy_curves)
                        and index < len(energy_curves[position])
                        else float("nan")
                    )

                    rho_value = (
                        float(rho_curves[position][index])
                        if position < len(rho_curves)
                        and index < len(rho_curves[position])
                        else float("nan")
                    )

                    lambda_value = (
                        float(lambda_curves[position][index])
                        if position < len(lambda_curves)
                        and index < len(lambda_curves[position])
                        else float("nan")
                    )

                    selected_k = (
                        int(selected_ks[position])
                        if position < len(selected_ks)
                        else 0
                    )

                    curve_rows.append({
                        "model_alias": model.alias,
                        "task": name,
                        "task_type": obj.get("task_type"),
                        "layer": int(layer),
                        "K": int(index + 1),
                        "H_K": float(h_value),
                        "mass": mass_value,
                        "energy": energy_value,
                        "rho": rho_value,
                        "lambda": lambda_value,
                        "selected_K": selected_k,
                        "sample_hash": obj.get("sample_hash"),
                    })

    metric_csv = output_dir / "metric_csv" / "metrics_layerwise_long.csv"
    curve_csv = output_dir / "metric_csv" / "harmora_k_curves_long.csv"
    runtime_csv = output_dir / "logs" / "metric_runtime.csv"
    pd.DataFrame(long_rows).sort_values(["model_alias", "task", "feature", "layer"]).to_csv(metric_csv, index=False)
    pd.DataFrame(curve_rows).sort_values(["model_alias", "task", "layer", "K"]).to_csv(curve_csv, index=False)
    pd.DataFrame(runtime_rows).to_csv(runtime_csv, index=False)
    save_json(output_dir / "manifests" / "metric_extraction_manifest.json", {
        "models": [m.alias for m in models],
        "tasks": [task_name(t) for t in tasks],
        "metric_names": metric_names,
        "same_sample_as_downstream": True,
        "seed_independent_metrics": True,
        "pair_metric_policy": "concatenate_both_sides",
        "metrics_layerwise_csv": str(metric_csv),
        "harmora_k_curves_csv": str(curve_csv),
    })
    return {"metrics_long": metric_csv, "harmora_curves": curve_csv, "runtime": runtime_csv}
