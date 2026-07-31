from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable
import numpy as np

import torch

from .config import (
    ensure_output_dirs,
    resolve_device,
    resolve_path,
    selected_seeds,
    set_global_seed,
)
from .embedding_cache import build_or_load_embeddings, cache_path as embedding_cache_path
from .encoder import EncoderConfig
from .evaluators import evaluate_primary_profile
from .io_utils import load_json, safe_name, save_json
from .models import get_model_specs
from .sampling import build_task_cache, load_task_cache, stable_hash
from .splits import get_or_create_split_manifest
from .tasks import get_selected_tasks, task_name


def result_path(output_dir: str | Path, model_alias: str, task: str, seed: int) -> Path:
    return (
        Path(output_dir)
        / "seed_results"
        / safe_name(model_alias)
        / safe_name(task)
        / f"seed_{int(seed)}.json"
    )


def full_config_fingerprint(cfg: Dict[str, Any]) -> str:
    ignored = {"_config_path", "_root_dir", "_resolved_output_dir"}
    payload = {k: v for k, v in cfg.items() if k not in ignored}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def evaluation_fingerprint(cfg: Dict[str, Any]) -> str:
    """Fingerprint settings that affect a per-seed downstream score.

    The seed list and aggregation settings are deliberately excluded so new seeds
    can be appended without invalidating completed seed files.
    """
    return stable_hash({
        "deterministic": bool(cfg.get("deterministic", True)),
        "downstream": cfg.get("downstream", {}),
    })


def prepare_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ensure_output_dirs(cfg)
    cfg["_resolved_output_dir"] = str(resolve_path(cfg, "output_dir"))
    return cfg


def build_all_task_caches(
    cfg: Dict[str, Any],
    requested_tasks: Iterable[str] | None = None,
    overwrite: bool = False,
) -> list[Dict[str, Any]]:
    prepare_cfg(cfg)
    tasks = get_selected_tasks(cfg, requested_tasks)
    return [build_task_cache(task, cfg, overwrite=overwrite) for task in tasks]


def _verify_existing_result(
    existing: Dict[str, Any],
    *,
    path: Path,
    model_alias: str,
    task: str,
    seed: int,
    sample_hash: str,
    embedding_hash: str,
    encoder_fingerprint: str,
    eval_fingerprint: str,
    split_hash: str | None,
) -> None:
    checks = {
        "model_alias": existing.get("model_alias") == model_alias,
        "task": existing.get("task") == task,
        "seed": int(existing.get("seed", -1)) == int(seed),
        "sample_hash": existing.get("sample_hash") == sample_hash,
        "embedding_hash": existing.get("embedding_hash") == embedding_hash,
        "encoder_fingerprint": existing.get("encoder_fingerprint") == encoder_fingerprint,
        "evaluation_fingerprint": existing.get("evaluation_fingerprint") == eval_fingerprint,
        "split_hash": existing.get("split_hash") == split_hash,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            f"Existing result is incompatible with the current experiment: {path}. "
            f"Failed checks={failed}. Rerun with --overwrite-results; if embeddings or splits changed, "
            "also use the corresponding overwrite flag."
        )


def run_downstream(
    cfg: Dict[str, Any],
    model_aliases: Iterable[str] | None = None,
    requested_tasks: Iterable[str] | None = None,
    seed_override: Iterable[int] | None = None,
    overwrite_results: bool = False,
    overwrite_embeddings: bool = False,
    overwrite_splits: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    prepare_cfg(cfg)
    output_dir = resolve_path(cfg, "output_dir")
    model_cache_dir = resolve_path(cfg, "model_cache_dir")
    models = get_model_specs(cfg, model_aliases)
    tasks = get_selected_tasks(cfg, requested_tasks)
    seeds = selected_seeds(cfg, seed_override)
    deterministic = bool(cfg.get("deterministic", True))
    eval_fingerprint = evaluation_fingerprint(cfg)
    set_global_seed(int(cfg.get("sample_seed", 2025)), deterministic=deterministic)

    task_payloads: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        name = task_name(task)
        try:
            payload = load_task_cache(output_dir, name)
            # Calling build_task_cache without overwrite validates its settings fingerprint.
            payload = build_task_cache(task, cfg, overwrite=False)
        except FileNotFoundError:
            payload = build_task_cache(task, cfg, overwrite=False)
        if payload.get("status") != "ok":
            raise RuntimeError(
                f"Task cache is not fully usable for {name}: status={payload.get('status')} "
                f"reason={payload.get('reason')}"
            )
        task_payloads[name] = payload

    downstream_cfg = cfg.get("downstream", {})
    test_size = float(downstream_cfg.get("test_size", 0.30))

    # A Classification split is a task+seed object shared by every model and every layer.
    split_manifests: Dict[tuple[str, int], Dict[str, Any] | None] = {}
    for name, payload in task_payloads.items():
        for seed in seeds:
            if payload["probe_family"] == "classification":
                split_manifests[(name, seed)] = get_or_create_split_manifest(
                    output_dir,
                    payload,
                    seed=seed,
                    test_size=test_size,
                    overwrite=overwrite_splits,
                )
            else:
                split_manifests[(name, seed)] = None

    enc_raw = cfg.get("encoding", {})
    encoder_cfg = EncoderConfig(
        max_length=int(enc_raw.get("max_length", 256)),
        batch_size=int(enc_raw.get("batch_size", 16)),
        device=resolve_device(enc_raw.get("device", "auto")),
        dtype=str(enc_raw.get("dtype", "float32")),
        include_embedding_layer=bool(enc_raw.get("include_embedding_layer", True)),
        model_cache_dir=str(model_cache_dir),
    )

    run_rows = []
    runtime_rows = []
    for model in models:
        for task in tasks:
            name = task_name(task)
            payload = task_payloads[name]
            print(f"\n[MODEL={model.alias}] [TASK={name}] family={payload['probe_family']}")
            emb_path = embedding_cache_path(output_dir, model.alias, name)
            emb_existed = emb_path.exists() and not overwrite_embeddings
            emb_start = time.perf_counter()
            embeddings = build_or_load_embeddings(
                output_dir,
                payload,
                model,
                encoder_cfg,
                overwrite=overwrite_embeddings,
                show_progress=show_progress,
            )
            emb_seconds = time.perf_counter() - emb_start
            runtime_rows.append({
                "model_alias": model.alias,
                "task": name,
                "seed": None,
                "stage": "embedding",
                "seconds": float(emb_seconds),
                "status": "reused" if emb_existed else "computed",
            })
            if embeddings["sample_hash"] != payload["sample_hash"]:
                raise RuntimeError(f"Sample/embedding hash mismatch for {model.alias}/{name}")

            for seed in seeds:
                path = result_path(output_dir, model.alias, name, seed)
                split_manifest = split_manifests[(name, seed)]
                split_hash = split_manifest["split_hash"] if split_manifest is not None else None

                if path.exists() and not overwrite_results:
                    existing = load_json(path)
                    _verify_existing_result(
                        existing,
                        path=path,
                        model_alias=model.alias,
                        task=name,
                        seed=seed,
                        sample_hash=payload["sample_hash"],
                        embedding_hash=embeddings["embedding_hash"],
                        encoder_fingerprint=embeddings["encoder_fingerprint"],
                        eval_fingerprint=eval_fingerprint,
                        split_hash=split_hash,
                    )
                    print(f"  seed={seed}: compatible existing result reused")
                    runtime_rows.append({
                        "model_alias": model.alias,
                        "task": name,
                        "seed": int(seed),
                        "stage": "downstream_evaluation",
                        "seconds": float(existing.get("evaluation_runtime_seconds", 0.0)),
                        "status": "reused",
                    })
                    run_rows.append({
                        "model_alias": model.alias,
                        "task": name,
                        "seed": int(seed),
                        "status": "reused",
                        "result_path": str(path),
                    })
                    continue

                set_global_seed(seed, deterministic=deterministic)
                eval_start = time.perf_counter()
                scores, evaluator_meta = evaluate_primary_profile(
                    payload,
                    embeddings,
                    seed=seed,
                    downstream_cfg=downstream_cfg,
                    split_manifest=split_manifest,
                )

                raw_embedding_layer_retained = (
                    encoder_cfg.include_embedding_layer
                    and model.pooling.lower() != "cls"
                )

                layer_start = 0 if raw_embedding_layer_retained else 1
                layer_indices = list(range(layer_start, layer_start + len(scores)))

                score_array = np.asarray(scores, dtype=float)

                if not np.isfinite(score_array).all():
                    bad_positions = np.flatnonzero(~np.isfinite(score_array))
                    bad_layers = [layer_indices[int(i)] for i in bad_positions]

                    raise RuntimeError(
                        f"Non-finite downstream scores for {model.alias}/{name}/seed={seed}; "
                        f"layers={bad_layers}"
                    )

                eval_seconds = time.perf_counter() - eval_start
                runtime_rows.append({
                    "model_alias": model.alias,
                    "task": name,
                    "seed": int(seed),
                    "stage": "downstream_evaluation",
                    "seconds": float(eval_seconds),
                    "status": "computed",
                })
                result = {
                    "experiment_name": cfg.get("experiment_name"),
                    "full_config_fingerprint": full_config_fingerprint(cfg),
                    "evaluation_fingerprint": eval_fingerprint,
                    "model_alias": model.alias,
                    "hf_name": model.hf_name,
                    "task": name,
                    "task_type": payload["task_type"],
                    "probe_family": payload["probe_family"],
                    "seed": int(seed),
                    "sample_seed": int(cfg.get("sample_seed", 2025)),
                    "sampling_fingerprint": payload.get("sampling_fingerprint"),
                    "sample_hash": payload["sample_hash"],
                    "embedding_hash": embeddings["embedding_hash"],
                    "encoder_fingerprint": embeddings["encoder_fingerprint"],
                    "num_layers": int(len(scores)),
                    "num_items": int(payload.get("n_items", 0)),
                    "source_subset": payload.get("source_subset"),
                    "source_split": payload.get("source_split"),
                    "layer_indices": layer_indices,
                    "primary_scores": [float(value) for value in scores],
                    "evaluation_runtime_seconds": float(eval_seconds),
                    **evaluator_meta,
                }
                if split_manifest is not None:
                    result.update({
                        "n_train": int(split_manifest["n_train"]),
                        "n_test": int(split_manifest["n_test"]),
                        "train_indices_hash": hashlib.sha256(
                            json.dumps(split_manifest["train_indices"]).encode("utf-8")
                        ).hexdigest()[:20],
                        "test_indices_hash": hashlib.sha256(
                            json.dumps(split_manifest["test_indices"]).encode("utf-8")
                        ).hexdigest()[:20],
                    })
                save_json(path, result)

                best_position = int(np.argmax(score_array))
                best_layer = int(layer_indices[best_position])
                best_score = float(score_array[best_position])

                print(
                    f"  seed={seed}: layers={len(scores)} best_layer={best_layer} "
                    f"best_{result['primary_metric']}={best_score:.6f}"
                )
                run_rows.append({
                    "model_alias": model.alias,
                    "task": name,
                    "seed": int(seed),
                    "status": "computed",
                    "result_path": str(path),
                })

            del embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    expected_runs = len(models) * len(tasks) * len(seeds)
    if len(run_rows) != expected_runs:
        raise RuntimeError(f"Run manifest count mismatch: expected={expected_runs}, observed={len(run_rows)}")

    manifest = {
        "experiment_name": cfg.get("experiment_name"),
        "full_config_fingerprint": full_config_fingerprint(cfg),
        "evaluation_fingerprint": eval_fingerprint,
        "models": [model.alias for model in models],
        "tasks": [task_name(task) for task in tasks],
        "seeds": seeds,
        "sample_seed": int(cfg.get("sample_seed", 2025)),
        "downstream_only": True,
        "expected_model_task_seed_runs": expected_runs,
        "runs": run_rows,
    }
    save_json(output_dir / "manifests" / "run_manifest.json", manifest)
    import pandas as pd
    pd.DataFrame(runtime_rows).to_csv(output_dir / "logs" / "downstream_runtime.csv", index=False)
    return manifest
