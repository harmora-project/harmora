from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, Optional, Tuple


TEXT_CANDIDATES = [
    "text", "sentence", "sent", "content", "query", "document", "title",
    "abstract", "body", "premise", "hypothesis", "input", "question",
]
TEXT_A_CANDIDATES = [
    "sentence1", "sentence_a", "sentencea", "sent1", "sent_a", "text1",
    "query", "question1", "premise", "anchor", "s1", "a",
]
TEXT_B_CANDIDATES = [
    "sentence2", "sentence_b", "sentenceb", "sent2", "sent_b", "text2",
    "document", "question2", "hypothesis", "positive", "candidate", "s2", "b",
]
LABEL_CANDIDATES = [
    "label", "labels", "category", "target", "class", "gold", "score",
    "similarity_score", "relatedness_score",
]


def task_name(task: Any) -> str:
    description = getattr(task, "description", None)
    if isinstance(description, dict) and description.get("name"):
        return str(description["name"])
    metadata = getattr(task, "metadata", None)
    value = getattr(metadata, "name", None) if metadata is not None else None
    return str(value or task.__class__.__name__)


def task_type(task: Any) -> str:
    metadata = getattr(task, "metadata", None)
    value = getattr(metadata, "type", None) if metadata is not None else None
    return str(value or "unknown")


def task_category(task: Any) -> str:
    metadata = getattr(task, "metadata", None)
    value = getattr(metadata, "category", None) if metadata is not None else None
    return str(value or "unknown")


def get_selected_tasks(cfg: Dict[str, Any], requested: Iterable[str] | None = None) -> list[Any]:
    import mteb

    configured = list(cfg.get("mteb", {}).get("include_task_names", []))
    names = list(requested) if requested is not None else configured
    if not names:
        raise ValueError("No task names were requested.")

    languages = cfg.get("mteb", {}).get("languages", ["eng"])
    tasks = []
    direct_error = None
    try:
        tasks = list(mteb.get_tasks(tasks=names, languages=languages))
    except Exception as exc:
        direct_error = exc

    if not tasks:
        try:
            all_tasks = list(
                mteb.get_tasks(
                    languages=languages,
                    modalities=["text"],
                    exclusive_modality_filter=True,
                    exclude_aggregate=True,
                )
            )
        except Exception:
            benchmark = mteb.get_benchmark("MTEB(eng)")
            all_tasks = list(benchmark)
        wanted = set(names)
        tasks = [task for task in all_tasks if task_name(task) in wanted]

    by_name = {task_name(task): task for task in tasks}
    missing = [name for name in names if name not in by_name]
    allow_missing = bool(cfg.get("mteb", {}).get("allow_missing_tasks", False))
    if missing and not allow_missing:
        extra = f" Direct MTEB error: {direct_error}" if direct_error else ""
        raise RuntimeError(
            "The following configured tasks were not found in the installed MTEB version: "
            f"{missing}.{extra}"
        )
    return [by_name[name] for name in names if name in by_name]


def describe_tasks(tasks: Iterable[Any]) -> list[dict[str, str]]:
    return [
        {
            "name": task_name(task),
            "type": task_type(task),
            "category": task_category(task),
        }
        for task in tasks
    ]


def _keys(obj: Any) -> list[str]:
    if isinstance(obj, Mapping):
        return [str(k) for k in obj.keys()]
    try:
        return [str(k) for k in obj.keys()]
    except Exception:
        return []


def _getitem(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[key]
    return obj[key]


def resolve_dataset_split(dataset_obj: Any, priorities: Sequence[str]) -> tuple[str | None, str | None, Any]:
    """Resolve optional subset and split from MTEB/HF dataset containers."""
    keys = _keys(dataset_obj)
    for split in priorities:
        if split in keys:
            return None, split, _getitem(dataset_obj, split)

    # Common nested shape: dataset[subset][split]. Prefer default/English-like subsets.
    subset_order = ["default", "en", "eng", "en-en"] + sorted(k for k in keys if k not in {"default", "en", "eng", "en-en"})
    for subset in subset_order:
        if subset not in keys:
            continue
        child = _getitem(dataset_obj, subset)
        child_keys = _keys(child)
        for split in priorities:
            if split in child_keys:
                return subset, split, _getitem(child, split)

    # Last fallback: first leaf-like object.
    if keys:
        first = keys[0]
        child = _getitem(dataset_obj, first)
        child_keys = _keys(child)
        if child_keys:
            second = child_keys[0]
            return first, second, _getitem(child, second)
        return None, first, child
    return None, None, None


def load_task_dataset(task: Any, split_priority: Sequence[str]) -> tuple[Any, dict[str, Any]]:
    name = task_name(task)
    try:
        task.load_data()
    except Exception as exc:
        raise RuntimeError(f"Could not load data for {name}: {exc}") from exc
    dataset_obj = getattr(task, "dataset", None)
    subset, split, dataset = resolve_dataset_split(dataset_obj, split_priority)
    if dataset is None:
        raise RuntimeError(f"No usable dataset split found for {name}.")
    columns = list(getattr(dataset, "column_names", []) or [])
    return dataset, {
        "task": name,
        "task_type": task_type(task),
        "subset": subset,
        "split": split,
        "columns": columns,
        "num_rows": len(dataset),
    }


def find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    exact = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    for column in columns:
        lowered = str(column).lower()
        for candidate in candidates:
            candidate_lower = candidate.lower()
            # Avoid accidental matches for single-letter fallback names.
            if len(candidate_lower) <= 2:
                continue
            if candidate_lower in lowered:
                return str(column)
    return None


def text_from_row(row: Dict[str, Any], columns: Sequence[str], preferred: str | None) -> str | None:
    if preferred is not None:
        value = row.get(preferred)
        if isinstance(value, str) and value.strip():
            return value.strip()
    parts = []
    for column in columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts[:4]) if parts else None


def classify_probe_family(ttype: str, name: str) -> str:
    if ttype == "Classification":
        return "classification"
    if ttype == "Clustering":
        return "clustering"
    if ttype == "PairClassification":
        return "pair_classification"
    if ttype == "STS" or "STS" in name:
        return "sts"
    raise ValueError(
        f"Task {name} has unsupported type {ttype}. This downstream-only package supports "
        "Classification, Clustering, PairClassification, and STS."
    )
