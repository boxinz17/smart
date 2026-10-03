#!/usr/bin/env python3
"""Check whether this runtime regenerates confirmation-campaign data byte for byte.

Diagnostic for run park-grid-20261002-v1: prints the library versions, the
BLAS/SIMD environment and, for the given task indices, the regenerated fit-data
and truth fingerprints against the saved per-replication records. No fitting.

    python park_grid_fingerprint_check.py --plan plan.json --reference per_replication.csv --tasks 900,901
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
for _path in (CODE_ROOT / "simulation", CODE_ROOT / "sparse-smart" / "src",
              CODE_ROOT / "sparse-smart-v2" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from reviewer_revision_20260913 import design  # noqa: E402
from reviewer_revision_20260913.park_grid_extension import load_reference  # noqa: E402
from reviewer_revision_20260913.runner import fingerprint  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--tasks", required=True)
    a = p.parse_args()
    print(json.dumps(dict(numpy=np.__version__, scipy=scipy.__version__, python=sys.version.split()[0],
                          OPENBLAS_CORETYPE=os.environ.get("OPENBLAS_CORETYPE"),
                          NPY_DISABLE_CPU_FEATURES=os.environ.get("NPY_DISABLE_CPU_FEATURES"))))
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats  # type: ignore
        print("cpu features enabled:", sorted(k for k, v in feats.items() if v)[:40])
    except Exception as exc:  # informational only
        print("cpu features unavailable:", exc)
    plan = json.loads(a.plan.read_text())
    ref = load_reference(a.reference)
    ok = True
    for index in [int(t) for t in a.tasks.split(",")]:
        task = plan["tasks"][index]
        case = plan["cases"][task["case_index"]]
        bundle = design.generate_case(case, task["seed"])
        fd, tr = fingerprint(bundle["fit_data"]), fingerprint(bundle["evaluation"])
        same = (fd == ref[index]["fit_data_fingerprint"], tr == ref[index]["truth_fingerprint"])
        ok &= all(same)
        print(json.dumps(dict(task=index, case=case["case_id"], fit_data_identical=same[0], truth_identical=same[1])))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
