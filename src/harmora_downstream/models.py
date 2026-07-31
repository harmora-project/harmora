from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    hf_name: str
    trust_remote_code: bool = False
    pooling: str = "mean"
    prompt: str | None = None


def get_model_specs(cfg: Dict[str, Any], aliases: Iterable[str] | None = None) -> list[ModelSpec]:
    specs = [ModelSpec(**item) for item in cfg.get("models", [])]
    if aliases is None:
        return specs
    requested = set(str(x) for x in aliases)
    selected = [s for s in specs if s.alias in requested or s.hf_name in requested]
    found = {s.alias for s in selected} | {s.hf_name for s in selected}
    missing = requested - found
    if missing:
        raise ValueError(f"Unknown model aliases/names: {sorted(missing)}")
    return selected
