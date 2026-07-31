from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from .aggregation import collect_seed_results
from .io_utils import load_json
from .models import get_model_specs
from .sampling import load_task_cache
from .splits import split_manifest_path
from .tasks import get_selected_tasks, task_name


def _record(rows, severity: str, check: str, status: str, detail: str, **context):
    rows.append({
        "severity": severity,
        "check": check,
        "status": status,
        "detail": detail,
        **context,
    })


def validate_integrity(
    cfg: Dict[str, Any],
    output_dir: str | Path,
    model_aliases: Iterable[str] | None = None,
    requested_tasks: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    expected_models = [spec.alias for spec in get_model_specs(cfg, model_aliases)]
    expected_tasks = [task_name(task) for task in get_selected_tasks(cfg, requested_tasks)]
    expected_seeds = [int(x) for x in (seeds if seeds is not None else cfg["seeds"])]
    rows = []

    expected_group_count = len(expected_models) * len(expected_tasks) * len(expected_seeds)
    _record(
        rows,
        "error",
        "expected_experiment_size",
        "pass",
        f"models={len(expected_models)} tasks={len(expected_tasks)} seeds={len(expected_seeds)} "
        f"expected_model_task_seed_runs={expected_group_count}",
    )

    sample_hashes: dict[str, str] = {}
    families: dict[str, str] = {}
    n_items_by_task: dict[str, int] = {}
    for task in expected_tasks:
        try:
            payload = load_task_cache(output_dir, task)
            sample_hashes[task] = payload["sample_hash"]
            families[task] = payload["probe_family"]
            n_items_by_task[task] = int(payload["n_items"])
            ok = payload.get("status") in {"ok", "warning"} and int(payload["n_items"]) > 0
            _record(
                rows,
                "error",
                "sample_cache",
                "pass" if ok else "fail",
                f"status={payload.get('status')} n_items={payload.get('n_items')} "
                f"sample_hash={payload.get('sample_hash')}",
                task=task,
            )
        except Exception as exc:
            _record(rows, "error", "sample_cache", "fail", str(exc), task=task)

    # Classification split integrity and seed correspondence.
    for task in expected_tasks:
        if families.get(task) != "classification":
            continue
        n_items = n_items_by_task[task]
        task_split_hashes = []
        for seed in expected_seeds:
            path = split_manifest_path(output_dir, task, seed)
            if not path.exists():
                _record(rows, "error", "split_manifest", "fail", f"missing {path}", task=task, seed=seed)
                continue
            manifest = load_json(path)
            train = set(int(x) for x in manifest["train_indices"])
            test = set(int(x) for x in manifest["test_indices"])
            overlap = train & test
            complete = train | test == set(range(n_items))
            hash_ok = manifest.get("sample_hash") == sample_hashes.get(task)
            seed_ok = int(manifest.get("seed")) == int(seed)
            nonempty = len(train) > 0 and len(test) > 0
            ok = not overlap and complete and hash_ok and seed_ok and nonempty
            task_split_hashes.append(manifest.get("split_hash"))
            _record(
                rows,
                "error",
                "split_integrity",
                "pass" if ok else "fail",
                f"train={len(train)} test={len(test)} overlap={len(overlap)} complete={complete} "
                f"sample_hash_ok={hash_ok} seed_ok={seed_ok}",
                task=task,
                seed=seed,
                split_hash=manifest.get("split_hash"),
            )
        # Split manifests should usually differ across evaluation seeds.
        unique_count = len(set(task_split_hashes))
        _record(
            rows,
            "warning",
            "classification_seed_split_diversity",
            "pass" if unique_count == len(task_split_hashes) else "warn",
            f"unique_split_hashes={unique_count}/{len(task_split_hashes)}",
            task=task,
        )

    try:
        long_df = collect_seed_results(output_dir)
    except Exception as exc:
        _record(rows, "error", "seed_results", "fail", str(exc))
        return pd.DataFrame(rows)

    observed_models = sorted(long_df["model_alias"].unique().tolist())
    observed_tasks = sorted(long_df["task"].unique().tolist())
    observed_seeds = sorted(int(x) for x in long_df["seed"].unique())
    exact_axes = (
        observed_models == sorted(expected_models)
        and observed_tasks == sorted(expected_tasks)
        and observed_seeds == sorted(expected_seeds)
    )
    _record(
        rows,
        "error",
        "exact_experiment_axes",
        "pass" if exact_axes else "fail",
        f"models={observed_models}; tasks={observed_tasks}; seeds={observed_seeds}",
    )

    observed_groups = long_df[["model_alias", "task", "seed"]].drop_duplicates()
    _record(
        rows,
        "error",
        "model_task_seed_group_count",
        "pass" if len(observed_groups) == expected_group_count else "fail",
        f"expected={expected_group_count} observed={len(observed_groups)}",
    )

    for model in expected_models:
        for task in expected_tasks:
            subset = long_df[(long_df["model_alias"] == model) & (long_df["task"] == task)]
            model_task_seeds = sorted(int(x) for x in subset["seed"].unique())
            coverage_ok = model_task_seeds == sorted(expected_seeds)
            _record(
                rows,
                "error",
                "seed_coverage",
                "pass" if coverage_ok else "fail",
                f"expected={sorted(expected_seeds)} observed={model_task_seeds}",
                model_alias=model,
                task=task,
            )
            if subset.empty:
                continue

            sample_unique = subset["sample_hash"].dropna().unique()
            embedding_unique = subset["embedding_hash"].dropna().unique()
            encoder_fp_unique = subset["encoder_fingerprint"].dropna().unique()
            evaluation_fp_unique = subset["evaluation_fingerprint"].dropna().unique()
            sample_ok = len(sample_unique) == 1 and sample_unique[0] == sample_hashes.get(task)
            embedding_ok = len(embedding_unique) == 1
            encoder_fp_ok = len(encoder_fp_unique) == 1
            evaluation_fp_ok = len(evaluation_fp_unique) == 1
            _record(
                rows,
                "error",
                "sample_hash_consistency",
                "pass" if sample_ok else "fail",
                f"unique={sample_unique.tolist()} expected={sample_hashes.get(task)}",
                model_alias=model,
                task=task,
            )
            _record(
                rows,
                "error",
                "embedding_hash_consistency",
                "pass" if embedding_ok else "fail",
                f"unique={embedding_unique.tolist()}",
                model_alias=model,
                task=task,
            )
            _record(
                rows,
                "error",
                "encoder_fingerprint_consistency",
                "pass" if encoder_fp_ok else "fail",
                f"unique={encoder_fp_unique.tolist()}",
                model_alias=model,
                task=task,
            )
            _record(
                rows,
                "error",
                "evaluation_fingerprint_consistency",
                "pass" if evaluation_fp_ok else "fail",
                f"unique={evaluation_fp_unique.tolist()}",
                model_alias=model,
                task=task,
            )

            layer_sets = {
                int(seed): tuple(sorted(int(x) for x in group["layer"].unique()))
                for seed, group in subset.groupby("seed")
            }
            same_layers = len(set(layer_sets.values())) == 1
            no_duplicate_layers = all(
                len(group) == group["layer"].nunique()
                for _, group in subset.groupby("seed")
            )
            _record(
                rows,
                "error",
                "layer_alignment",
                "pass" if same_layers and no_duplicate_layers else "fail",
                f"same_layer_sets={same_layers} no_duplicate_layers={no_duplicate_layers} {layer_sets}",
                model_alias=model,
                task=task,
            )

            family = families.get(task)
            seed_affects_values = subset["seed_affects_evaluation"].unique().tolist()
            expected_seed_affects = family in {"classification", "clustering"}
            meta_ok = len(seed_affects_values) == 1 and bool(seed_affects_values[0]) == expected_seed_affects
            _record(
                rows,
                "error",
                "seed_effect_metadata",
                "pass" if meta_ok else "fail",
                f"family={family} expected={expected_seed_affects} observed={seed_affects_values}",
                model_alias=model,
                task=task,
            )

            if family == "classification":
                for seed in expected_seeds:
                    seed_rows = subset[subset["seed"] == seed]
                    if seed_rows.empty:
                        continue
                    observed_hashes = seed_rows["split_hash"].dropna().unique()
                    manifest = load_json(split_manifest_path(output_dir, task, seed))
                    expected_hash = manifest["split_hash"]
                    ok = len(observed_hashes) == 1 and observed_hashes[0] == expected_hash
                    _record(
                        rows,
                        "error",
                        "split_hash_alignment",
                        "pass" if ok else "fail",
                        f"observed={observed_hashes.tolist()} expected={expected_hash}",
                        model_alias=model,
                        task=task,
                        seed=seed,
                    )
            else:
                split_empty = subset["split_hash"].isna().all()
                _record(
                    rows,
                    "error",
                    "nonclassification_no_split",
                    "pass" if split_empty else "fail",
                    f"all_split_hashes_null={split_empty}",
                    model_alias=model,
                    task=task,
                )

            # Deterministic task families should reproduce exactly across seed files.
            if family in {"pair_classification", "sts"}:
                pivot = subset.pivot(index="layer", columns="seed", values="primary_score")
                deterministic_ok = bool(np.allclose(pivot.to_numpy(), pivot.iloc[:, [0]].to_numpy(), atol=0.0, rtol=0.0))
                _record(
                    rows,
                    "error",
                    "deterministic_family_reproducibility",
                    "pass" if deterministic_ok else "fail",
                    "Scores should be identical because these evaluators have no random component.",
                    model_alias=model,
                    task=task,
                )

    # Cross-model split identity for every classification task and seed.
    for task in expected_tasks:
        if families.get(task) != "classification":
            continue
        for seed in expected_seeds:
            subset = long_df[(long_df["task"] == task) & (long_df["seed"] == seed)]
            values = subset["split_hash"].dropna().unique()
            ok = len(values) == 1
            _record(
                rows,
                "error",
                "cross_model_split_identity",
                "pass" if ok else "fail",
                f"unique_split_hashes={values.tolist()}",
                task=task,
                seed=seed,
            )

    return pd.DataFrame(rows)
