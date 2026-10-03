#!/usr/bin/env python3
"""Rerun the published Park et al. comparator with a wider pooled-penalty grid.

Audit finding M-9 (response_r1/audit/AUDIT_REPORT.md): in the confirmation
campaign the selected pooled penalty lambda_w sat at the grid maximum (10) in
178 of the 180 fitted-source tasks. This script regenerates every raw-source
task of that campaign from its frozen plan and seed, checks that the fit data
are byte-identical to the original run (fit_data_fingerprint), and runs the
same validation-selected two-stage fit twice:

  original  lambda_w = lambda_delta = (.01, .03, .1, .3, 1, 3, 10)
            -> must reproduce the original selection and risk;
  extended  lambda_w = original + (30, 100, 300, 1000), lambda_delta unchanged.

Nothing else in the campaign changes: same data generator, same source fit,
same solver settings (1000 iterations, tolerance 1e-6, convergence required),
same validation rule and tie order, same evaluation. Each task writes
<out>/task-<index>.json atomically; completed tasks are skipped, so a requeue
or rerun redoes only missing work.

Usage (Slurm only; the bridge refuses to run outside a job):
    python park_grid_extension.py --plan plan.json --reference per_replication.csv \
        --out OUTDIR --tasks 900,901,...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
for _path in (CODE_ROOT / "simulation", CODE_ROOT / "sparse-smart" / "src",
              CODE_ROOT / "sparse-smart-v2" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from reviewer_revision_20260913 import design  # noqa: E402
from reviewer_revision_20260913.published_park import fit_park_validation  # noqa: E402
from reviewer_revision_20260913.runner import evaluate, fingerprint, jsonable  # noqa: E402

ORIGINAL = (.01, .03, .1, .3, 1., 3., 10.)
EXTENDED_W = ORIGINAL + (30., 100., 300., 1000.)
METHOD = "park_two_stage_nr_external_validation"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as fh:
        json.dump(value, fh, indent=1, sort_keys=True)
        fh.write("\n")
        tmp = fh.name
    os.replace(tmp, path)


def load_reference(path: Path) -> dict[int, dict]:
    rows = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["method"] == METHOD:
                rows[int(row["task_index"])] = row
    return rows


def run_park(data, evaluation, case, lambda_w):
    started = time.perf_counter()
    selected = fit_park_validation(
        data["X"], data["Y"], data["X_validation"], data["Y_validation"],
        data["source_X"], data["source_Y"],
        lambda_w=lambda_w, lambda_delta=ORIGINAL,
        max_iter=1000, tolerance=1e-6, require_convergence=True, timeout_seconds=1800)
    fit = selected.fit
    def scalar(value):
        return value[0] if isinstance(value, list) and len(value) == 1 else value
    candidates = [dict(lambda_w=scalar(c.get("lambda_w")), lambda_delta=scalar(c.get("lambda_delta")),
                       validation_loss=scalar(c.get("validation_loss")), status=scalar(c.get("status")))
                  for c in jsonable(selected.candidate_results)]
    chosen = candidates[selected.selected_index] if candidates else {}
    return dict(metrics=evaluate(fit.coefficient, data, evaluation, fit.intercept),
                selected_index=selected.selected_index,
                selected_lambda_w=chosen.get("lambda_w"),
                selected_lambda_delta=chosen.get("lambda_delta"),
                validation_score=2 * selected.validation_loss / case["q"],
                n_candidates=len(candidates),
                n_eligible=sum(c.get("status") == "ok" for c in candidates),
                candidates=candidates,
                seconds=time.perf_counter() - started)


def run_task(index: int, plan: dict, reference: dict, out: Path) -> None:
    target = out / f"task-{index:06d}.json"
    if target.exists() and json.loads(target.read_text()).get("complete"):
        print(json.dumps(dict(event="skip", task=index)), flush=True)
        return
    task = plan["tasks"][index]
    case = plan["cases"][task["case_index"]]
    if case.get("source_mode") != "fitted":
        raise ValueError(f"task {index} is not a raw-source (fitted) task")
    result = dict(task_index=index, case_id=case["case_id"], seed=task["seed"],
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), complete=False)
    began = time.perf_counter()
    try:
        bundle = design.generate_case(case, task["seed"])
        data, evaluation = bundle["fit_data"], bundle["evaluation"]
        ref = reference.get(index, {})
        result["fit_data_fingerprint"] = fingerprint(data)
        result["truth_fingerprint"] = fingerprint(evaluation)
        result["reference_fit_data_fingerprint"] = ref.get("fit_data_fingerprint")
        result["data_identical"] = result["fit_data_fingerprint"] == ref.get("fit_data_fingerprint")
        result["truth_identical"] = result["truth_fingerprint"] == ref.get("truth_fingerprint")
        result["reference_population_prediction_excess"] = (
            float(ref["population_prediction_excess"]) if ref.get("population_prediction_excess") else None)
        result["reference_candidate_index"] = (
            int(ref["candidate_index"]) if ref.get("candidate_index") else None)
        # Titan cannot regenerate the Discovery bits (its OpenBLAS kernels differ, and
        # forcing the Discovery Haswell kernels crashes intermittently on this CPU), so
        # fingerprint equality is recorded, not required. Agreement is measured instead:
        # the original grid must select the saved pair and give the saved risk within
        # a relative 1e-3. The grid comparison itself is paired on the same data.
        result["original"] = run_park(data, evaluation, case, ORIGINAL)
        if ref:
            original_risk = result["original"]["metrics"]["population_prediction_excess"]
            reference_risk = result["reference_population_prediction_excess"]
            result["same_selection_as_saved"] = (
                result["original"]["selected_index"] == result["reference_candidate_index"])
            result["relative_risk_difference_vs_saved"] = (
                abs(original_risk - reference_risk) / max(abs(reference_risk), 1e-12))
            result["reproduced"] = bool(result["same_selection_as_saved"]
                                        and result["relative_risk_difference_vs_saved"] <= 1e-3)
        result["extended"] = run_park(data, evaluation, case, EXTENDED_W)
        result["complete"] = True
    except Exception as exc:  # recorded, never silently dropped
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    result["elapsed_seconds"] = time.perf_counter() - began
    atomic_json(target, result)
    print(json.dumps(dict(event="task_done", task=index, complete=result["complete"],
                          seconds=round(result["elapsed_seconds"], 1))), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tasks", required=True, help="comma- or plus-separated task indices")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    reference = load_reference(args.reference)
    indices = [int(t) for t in args.tasks.replace("+", ",").split(",") if t]
    failures = 0
    for index in indices:
        run_task(index, plan, reference, args.out)
        done = json.loads((args.out / f"task-{index:06d}.json").read_text())
        failures += not done.get("complete")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
