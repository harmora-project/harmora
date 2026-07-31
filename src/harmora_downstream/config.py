from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml

LOCKED_MODEL_ALIASES = [
    "minilm_l6",
    "mpnet_base",
    "e5_base_v2",
    "e5_large_v2",
    "bge_base_en_v15",
    "bge_large_en_v15",
    "snowflake_arctic_embed_m",
]

LOCKED_TASK_NAMES = [
    "Banking77Classification.v2",
    "EmotionClassification.v2",
    "HUMEEmotionClassification",
    "ArXivHierarchicalClusteringP2P",
    "ArXivHierarchicalClusteringS2S",
    "BiorxivClusteringP2P.v2",
    "LegalBenchPC",
    "STS15",
    "STS16",
    "STSBenchmark",
    "SICK-R",
]


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    cfg["_config_path"] = str(path)
    cfg["_root_dir"] = str(path.parent.parent.resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    seeds = cfg.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Config must define a non-empty integer list under `seeds`.")
    seeds = [int(s) for s in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be unique.")

    models = cfg.get("models", [])
    if not models:
        raise ValueError("No models configured.")
    aliases = [str(item.get("alias")) for item in models]
    if len(set(aliases)) != len(aliases):
        raise ValueError("Model aliases must be unique.")

    task_names = list(cfg.get("mteb", {}).get("include_task_names", []))
    if not task_names:
        raise ValueError("No tasks configured under mteb.include_task_names.")
    if len(set(task_names)) != len(task_names):
        raise ValueError("Configured task names must be unique.")

    expected_models = cfg.get("expected_model_count")
    if expected_models is not None and len(models) != int(expected_models):
        raise ValueError(
            f"Configured model count changed: expected {expected_models}, found {len(models)}."
        )
    expected_tasks = cfg.get("expected_task_count")
    if expected_tasks is not None and len(task_names) != int(expected_tasks):
        raise ValueError(
            f"Configured task count changed: expected {expected_tasks}, found {len(task_names)}."
        )

    if bool(cfg.get("lock_exact_experiment_set", False)):
        if aliases != LOCKED_MODEL_ALIASES:
            raise ValueError(
                "The exact seven-model experiment set changed. "
                f"Expected {LOCKED_MODEL_ALIASES}, found {aliases}."
            )
        if task_names != LOCKED_TASK_NAMES:
            raise ValueError(
                "The exact eleven-task experiment set changed. "
                f"Expected {LOCKED_TASK_NAMES}, found {task_names}."
            )

    test_size = float(cfg.get("downstream", {}).get("test_size", 0.30))
    if not (0.0 < test_size < 1.0):
        raise ValueError("downstream.test_size must be strictly between 0 and 1.")

    dtype = str(cfg.get("encoding", {}).get("dtype", "float32")).lower()
    if dtype != "float32":
        raise ValueError("This validated package currently supports encoding.dtype=float32 only.")

    # Generalization-related targets are intentionally forbidden. Representation
    # metrics and their correlations are allowed, but their only target is the
    # task-matched downstream score produced by this package.
    forbidden_top_level = {"generalization", "generalization_gap"}
    present = sorted(forbidden_top_level.intersection(cfg.keys()))
    if present:
        raise ValueError(
            "Generalization targets are not part of this package. Remove: "
            + ", ".join(present)
        )

    metric_names = list(cfg.get("metrics", {}).get("names", []))
    primary_fields = dict(cfg.get("metrics", {}).get("primary_fields", {}))
    directions = dict(cfg.get("metrics", {}).get("directions", {}))
    if not metric_names:
        raise ValueError("Config must define metrics.names.")
    missing_fields = [name for name in metric_names if name not in primary_fields]
    if missing_fields:
        raise ValueError(f"Missing metrics.primary_fields entries: {missing_fields}")
    missing_directions = [
        f"{name}.{primary_fields[name]}"
        for name in metric_names
        if f"{name}.{primary_fields[name]}" not in directions
    ]
    if missing_directions:
        raise ValueError(f"Missing metrics.directions entries: {missing_directions}")


def root_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["_root_dir"]).resolve()


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    raw = cfg.get("paths", {}).get(key)
    if raw is None:
        raise KeyError(f"Missing paths.{key}")
    path = Path(raw)
    if not path.is_absolute():
        path = root_dir(cfg) / path
    return path.resolve()


def ensure_output_dirs(cfg: Dict[str, Any]) -> None:
    """Create only the directories required by the paper pipeline."""
    out = resolve_path(cfg, "output_dir")
    resolve_path(cfg, "model_cache_dir").mkdir(parents=True, exist_ok=True)
    resolve_path(cfg, "data_cache_dir").mkdir(parents=True, exist_ok=True)
    for name in [
        "manifests",
        "sample_cache",
        "split_manifests",
        "embedding_cache",
        "seed_results",
        "csv",
        "figures/profiles",
        "figures/summary",
        "metrics",
        "metric_csv",
        "augmentation_cache",
        "correlations/per_seed",
        "correlations/summary",
        "analysis/cross_model",
        "logs",
    ]:
        (out / name).mkdir(parents=True, exist_ok=True)


def resolve_device(value: str | None) -> str:
    if value is None or value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(value)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def selected_seeds(cfg: Dict[str, Any], override: Iterable[int] | None = None) -> list[int]:
    values = list(override) if override is not None else cfg.get("seeds", [])
    seeds = [int(x) for x in values]
    if not seeds:
        raise ValueError("At least one evaluation seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed override contains duplicates.")
    return seeds

