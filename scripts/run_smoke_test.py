#!/usr/bin/env python3
"""Run the lightweight unit and reference-reproduction checks."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    run(sys.executable, "-m", "pytest", "-q", "tests")
    run(sys.executable, "scripts/reproduce_paper.py", "--force")
    run(sys.executable, "scripts/validate_reproduction.py")
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
