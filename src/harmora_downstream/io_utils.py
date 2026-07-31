from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_name(value: str) -> str:
    return (
        str(value)
        .replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
        .replace(":", "_")
    )
