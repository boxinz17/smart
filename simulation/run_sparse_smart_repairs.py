"""Recheck prior SparseSMART failures with the same data and tuning budget.

Only failed applicable cells and six seed-zero controls (exact/low-noise source
in each model) are fitted. Historical artifacts remain untouched. This is a
solver regression experiment, not an unbiased new comparison with the paper.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from run_restricted_rrr import SimulationSetting
from run_sparse_smart import _atomic_json_dump
from run_sparse_smart_external import RunnerConfig, run_setting

HERE = Path(__file__).resolve().parent


def selected_records(baseline_root):
    records = []
    for path in sorted(Path(baseline_root).glob("model*/exp*/SparseSMARTExternal_result_*.json")):
        old = json.loads(path.read_text())
        failed = old["applicable"] and not old["success"]
        control = (old["experiment"] == "exp4" and old["rd_seed_id"] == 0
                   and old["setting"]["sigma0"] in (0., .01))
        if failed or control:
            records.append((path, "previous_failure" if failed else "control"))
    return records


def fit_record(task):
    old_path, group, baseline_root, output_root = task
    old = json.loads(old_path.read_text())
    options = dict(old["configuration"]["runner"])
    for name in ("init_penalties", "penalties_u", "penalties_v"):
        options[name] = tuple(options[name])
    if options["support_limits"] is not None:
        options["support_limits"] = tuple(tuple(v) for v in options["support_limits"])
    options.update(initialization_spectrum="projected", refinement_solver="anchor_projected")
    config = RunnerConfig(**options)
    destination = output_root / old_path.relative_to(baseline_root)
    outcome, new = run_setting(setting=SimulationSetting(**old["setting"]),
        model=old["model"], experiment=old["experiment"], seed_id=old["rd_seed_id"],
        random_seed=old["random_seed"], destination=destination, config=config)
    for key in ("training_observed_input_fingerprint", "validation_observed_input_fingerprint",
                "evaluation_truth_fingerprint", "validation_seed_metadata", "split"):
        if old[key] != new[key]:
            raise ValueError(f"Before/after data protocol mismatch: {key}: {old_path}")
    if len(old["selection_history"]) != len(new["selection_history"]):
        raise ValueError(f"Candidate budget changed: {old_path}")
    for a, b in zip(old["selection_history"], new["selection_history"]):
        if a["params"] != b["params"]:
            raise ValueError(f"Candidate parameters changed: {old_path}")
    if new["success"]:
        history = new["history"]
        objectives = [v["objective"] for v in history]
        if not np.isfinite(objectives).all() or np.any(np.diff(objectives) > 1e-10):
            raise ValueError(f"Nonmonotone or nonfinite training objective: {destination}")
        floor = new["configuration"]["margins"]["anchor_min"]
        if any(min(v["anchor_min_u"], v["anchor_min_v"]) < floor - 1e-12 for v in history):
            raise ValueError(f"Infeasible accepted anchor: {destination}")
    return dict(model=old["model"], experiment=old["experiment"], setting=old["setting"]["suffix"],
        seed_id=old["rd_seed_id"], group=group, outcome=outcome,
        old_status=old["status"], status=new["status"], old_error=old["avg_err"], error=new["avg_err"],
        old_validation_loss=old["validation_loss"], validation_loss=new["validation_loss"],
        termination_reason=new["termination_reason"], selected_iteration=new["selected_iteration"],
        n_iter=new["n_iter"], candidate_failures=len(new["fit_errors"]),
        candidate_statuses=dict(Counter(v["status"] for v in new["selection_history"])),
        initialization_repaired=sum(v.get("diagnostics", {}).get("initialization_spectrum_repaired", False)
            for v in new["selection_history"]),
        data_and_budget_verified=True, path=str(destination))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=HERE / "result/sparse_smart_external")
    parser.add_argument("--output-root", type=Path, default=HERE / "result/sparse_smart_repairs")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    baseline, output = args.baseline_root.resolve(), args.output_root.resolve()
    if baseline == output or baseline in output.parents or output in baseline.parents:
        parser.error("Baseline and repair result roots must be separate directories")
    if args.workers < 1:
        parser.error("workers must be positive")
    selected = selected_records(baseline)
    if not selected:
        parser.error("No baseline failure or control records found")
    tasks = [(p, group, baseline, output) for p, group in selected]
    manifest = dict(started=datetime.now(timezone.utc).isoformat(),
        baseline_root=str(baseline), output_root=str(output), expected_cells=len(tasks),
        groups=dict(Counter(g for _, g in selected)), workers=args.workers, cells=[], errors=[])
    print(json.dumps(manifest), flush=True)
    if args.dry_run:
        return 0
    path = output / "repair_manifest.json"
    _atomic_json_dump(manifest, path)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(fit_record, task): task[0] for task in tasks}
        for future in as_completed(futures):
            try:
                cell = future.result()
                manifest["cells"].append(cell)
                print(json.dumps(dict(done=len(manifest["cells"]), total=len(tasks), **cell)), flush=True)
            except Exception as error:
                failure = dict(path=str(futures[future]), exception=type(error).__name__, message=str(error))
                manifest["errors"].append(failure)
                print(json.dumps(failure), flush=True)
            manifest["status_counts"] = dict(Counter(v["status"] for v in manifest["cells"]))
            _atomic_json_dump(manifest, path)
    manifest.update(finished=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started)
    _atomic_json_dump(manifest, path)
    print(json.dumps({k:manifest[k] for k in ("status_counts", "errors", "wall_seconds")}), flush=True)
    return int(bool(manifest["errors"]) or len(manifest["cells"]) != len(tasks))


if __name__ == "__main__":
    raise SystemExit(main())
