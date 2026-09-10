"""Verify dependency pins and an optional numerical runtime, without fitting."""
from __future__ import annotations

import argparse
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys


RUNTIME_PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")
BLAS_CORE_CPU_FLAGS = {"Haswell": frozenset(("avx2", "fma"))}
# The pinned Discovery NumPy 2.5.3 Linux wheel was probed with this exact
# feature-group layout. A different wheel build requires a new explicit audit.
NUMPY_CPU_BASELINE = ("X86_V2",)
NUMPY_CPU_DISPATCH = ("X86_V3", "X86_V4", "AVX512_ICL", "AVX512_SPR")
NUMPY_DISABLED_CPU_FEATURES = ",".join(NUMPY_CPU_DISPATCH)
NUMERICAL_ENVIRONMENT = (
    "OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NPY_DISABLE_CPU_FEATURES",
)


def check_blas_runtime(core: str, *, cpuinfo_path: Path = Path("/proc/cpuinfo")) -> dict:
    """Verify the requested CPU policy before loading its numerical libraries."""
    if core not in BLAS_CORE_CPU_FLAGS:
        raise ValueError(f"Unsupported fixed BLAS core policy: {core}")
    required = BLAS_CORE_CPU_FLAGS[core]
    flags = [set(line.partition(":")[2].split())
             for line in cpuinfo_path.read_text().splitlines()
             if line.partition(":")[0].strip() == "flags"]
    if not flags or any(not required <= features for features in flags):
        raise ValueError(f"Fixed BLAS core {core} requires CPU flags {', '.join(sorted(required))} "
                         f"on every processor reported by {cpuinfo_path}; numerical imports skipped")
    if os.environ.get("OPENBLAS_CORETYPE") != core:
        raise ValueError(f"OPENBLAS_CORETYPE must be {core} before numerical imports")
    if os.environ.get("NPY_DISABLE_CPU_FEATURES") != NUMPY_DISABLED_CPU_FEATURES:
        raise ValueError(f"NPY_DISABLE_CPU_FEATURES must be {NUMPY_DISABLED_CPU_FEATURES} "
                         "before numerical imports")
    try:
        numpy = import_module("numpy")
        cpu = import_module("numpy._core._multiarray_umath")
        baseline = getattr(cpu, "__cpu_baseline__", None)
        dispatch = getattr(cpu, "__cpu_dispatch__", None)
        features = getattr(cpu, "__cpu_features__", None)
        if baseline != list(NUMPY_CPU_BASELINE) or dispatch != list(NUMPY_CPU_DISPATCH):
            raise ValueError(f"Unexpected NumPy SIMD build: baseline={baseline!r}, dispatch={dispatch!r}; "
                             "expected the audited NumPy 2.5.3 Linux wheel")
        if not isinstance(features, dict) or any(features.get(name) is not True for name in NUMPY_CPU_BASELINE):
            raise ValueError("NumPy's required X86_V2 SIMD baseline is unavailable")
        if any(features.get(name) is not False for name in NUMPY_CPU_DISPATCH):
            raise ValueError("NumPy SIMD dispatch groups must all be disabled; "
                             f"found {None if features is None else {name: features.get(name) for name in NUMPY_CPU_DISPATCH}}")
        scipy_linalg = import_module("scipy.linalg")
        pools = import_module("threadpoolctl").threadpool_info()
    except ImportError as error:
        raise ValueError(f"Could not load the numerical runtime: {error}") from error
    blas = [pool for pool in pools if pool.get("user_api") == "blas"]
    if any(pool.get("internal_api") != "openblas" or pool.get("architecture") != core
           or type(pool.get("num_threads")) is not int or pool["num_threads"] != 1 for pool in blas):
        raise ValueError(f"Every loaded BLAS pool must use OpenBLAS {core} with one thread; found {blas!r}")
    # NumPy and SciPy wheels can load separate OpenBLAS libraries. Verify both,
    # including the common sibling package.libs wheel layout.
    package_roots = {"numpy": Path(numpy.__file__).resolve().parent,
                     "scipy": Path(scipy_linalg.__file__).resolve().parents[1]}
    for name, root in package_roots.items():
        roots = (root, root.with_name(f"{name}.libs"))
        if not any(pool.get("filepath") and any(Path(pool["filepath"]).resolve().is_relative_to(base)
                                               for base in roots) for pool in blas):
            raise ValueError(f"Could not identify the loaded {name} OpenBLAS pool; found {blas!r}")
    return dict(blas_core=core, cpu_flags_verified=sorted(required),
                cpuinfo_path=str(cpuinfo_path), processors_checked=len(flags),
                environment={name: os.environ.get(name) for name in NUMERICAL_ENVIRONMENT},
                numpy_simd=dict(baseline=baseline, dispatch=dispatch,
                                active_features=sorted(name for name, enabled in features.items() if enabled)),
                threadpools=pools)


def check_runtime(root: Path, *, python_only: bool = False, blas_core: str | None = None) -> dict:
    if python_only and blas_core is not None:
        raise ValueError("--blas-core cannot be combined with --python-only")
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
    if blas_core is not None:
        report["numerical_runtime"] = check_blas_runtime(blas_core)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--blas-core", choices=sorted(BLAS_CORE_CPU_FLAGS),
                        help="Require a fixed OpenBLAS core on a compatible Linux CPU and report loaded pools")
    args = parser.parse_args()
    try:
        report = check_runtime(args.root, python_only=args.python_only, blas_core=args.blas_core)
    except (OSError, ValueError) as error:
        print(f"Runtime check failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
