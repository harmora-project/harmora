#!/usr/bin/env python3
"""Validate the local environment and the locked Harmora paper configuration."""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmora_downstream.config import load_config, resolve_device, resolve_path

REQUIRED_IMPORTS = {
    "numpy": "NumPy",
    "pandas": "pandas",
    "scipy": "SciPy",
    "sklearn": "scikit-learn",
    "torch": "PyTorch",
    "transformers": "Transformers",
    "sentence_transformers": "Sentence-Transformers",
    "mteb": "MTEB",
    "yaml": "PyYAML",
    "matplotlib": "Matplotlib",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)

    print(f"Project: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Device: {resolve_device(cfg['encoding']['device'])}")
    print(f"Models: {len(cfg['models'])}")
    print(f"Tasks: {len(cfg['mteb']['include_task_names'])}")
    print(f"Seeds: {cfg['seeds']}")
    print(f"Output: {resolve_path(cfg, 'output_dir')}")

    missing: list[str] = []
    for module_name, display_name in REQUIRED_IMPORTS.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"[ok] {display_name}: {version}")
        except Exception as exc:
            missing.append(display_name)
            print(f"[missing] {display_name}: {exc}")

    metrics_dir = resolve_path(cfg, "metrics_package_dir")
    if not (metrics_dir / "metrics" / "registry.py").exists():
        missing.append("bundled Harmora metrics")
        print(f"[missing] bundled metrics at {metrics_dir}")
    else:
        print(f"[ok] bundled metrics: {metrics_dir}")

    if missing:
        print("\nEnvironment check failed. Missing: " + ", ".join(missing))
        return 2
    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
