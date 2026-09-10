"""Verify the shared runtime using distribution metadata, without fitting models."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys


RUNTIME_PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")


def check_runtime(root: Path, *, python_only: bool = False) -> dict[str, str]:
    expected_python = (root / ".python-version").read_text().strip()
    if expected_python != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise ValueError(f"Expected Python {expected_python}.x; found {platform.python_version()}")
    report = {"Python": platform.python_version()}
    if python_only:
        return report
    pins = {}
    for line in (root / "python-constraints.txt").read_text().splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        name, separator, pin = line.partition("==")
        if not separator or not name or not pin or name in pins:
            raise ValueError(f"Expected one exact dependency pin per line: {line!r}")
        pins[name] = pin
    mismatches = []
    for name in RUNTIME_PACKAGES:
        if name not in pins:
            raise ValueError(f"Missing shared pin for {name}")
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "not installed"
        report[name] = installed
        if installed != pins[name]:
            mismatches.append(f"{name}: expected {pins[name]}, found {installed}")
    if mismatches:
        raise ValueError("; ".join(mismatches))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python-only", action="store_true")
    args = parser.parse_args()
    try:
        report = check_runtime(args.root, python_only=args.python_only)
    except (OSError, ValueError) as error:
        print(f"Runtime check failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
