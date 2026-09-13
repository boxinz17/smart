#!/usr/bin/env python3
"""Training-only grouped BIC fits on Discovery, with immutable reference inputs.

One dispatched task shares an initializer across free counts and penalty pairs.
The terminal fit is scored; validation and truth enter evaluation only after
fitting and scoring. Only possible fixed/automatic winners retain model states.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
from itertools import product
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

import run_sparse_smart_v2_pilot as legacy

legacy._paths()
from sparse_smart_v2_bic_plan import METHOD, digest

read, sha, write = legacy._read, legacy._sha, legacy._json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_plan(root, *, full_source=False):
    plan = read(root / "plan.json")
    require(plan["method"] == METHOD and plan["schema_version"] == 1, "unsupported BIC plan")
    require(plan["root"] == str(root), "plan belongs to another root")
    require(digest({k: v for k, v in plan.items() if k != "plan_fingerprint"}) == plan["plan_fingerprint"],
            "BIC plan fingerprint mismatch")
    require(plan["configuration"]["selection_rule"] == "bic_terminal", "unsupported selection policy")
    require(plan["configuration"]["validation_patience"] is None, "BIC fitting cannot use validation stopping")
    manifest, _ = legacy._manifest(root, full=full_source, expected=plan["source_manifest_sha256"])
    return plan, manifest


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def prepare(root):
    """Bind the paired old observations/source frames without copying them."""
    legacy._require_slurm(root)
    plan, _ = load_plan(root, full_source=True)
    reference = Path(plan["reference_root"]).resolve()
    require(reference.is_relative_to(Path("/scratch2")), "reference must be on scratch2")
    require(reference != root, "new BIC root must differ from reference")
    summaries, preparations, records = {}, {}, []
    for group in plan["groups"]:
        folder = reference / f"model{group['model_id'] + 1}-exp{group['experiment_id'] + 1}"
        if folder not in preparations:
            prep = read(folder / "preparation.json")
            require(prep["success"] and prep["status"] == "complete", "reference preparation incomplete")
            require(prep["plan_sha256"] == sha(folder / "plan.json"), "reference plan changed")
            preparations[folder] = {c["case_id"]: c for c in prep["cases"]}
            summaries[folder] = read(folder / "summary/summary.json")
            require(summaries[folder]["success"], "reference summary incomplete")
        old = folder / "cases" / group["group_id"] / "case.json"
        require(sha(old) == preparations[folder][group["group_id"]]["case_json_sha256"],
                "reference case metadata hash changed")
        meta = read(old)
        for key in ("n_train", "p", "q", "sigma0", "seed_id", "random_seed", "model_id"):
            require(meta["case"][key] == group[key], f"reference design mismatch: {key}")
        files = {}
        for name in ("data.npz", "source.npz"):
            path = old.parent / name
            require(sha(path) == meta["files"][name], f"reference artifact hash mismatch: {path}")
            files[name] = dict(path=str(path), sha256=meta["files"][name])
        data = _load_npz(files["data.npz"]["path"])
        require(legacy._fingerprints(data) == meta["fingerprints"], "reference observation fingerprint mismatch")
        record = dict(group=group, plan_fingerprint=plan["plan_fingerprint"], files=files,
                      fingerprints=meta["fingerprints"], source_metadata=meta["source_metadata"],
                      source_fingerprint=meta["source_fingerprint"],
                      reference_case_json_path=str(old), reference_case_json_sha256=sha(old))
        target = root / "groups" / group["group_id"] / "group.json"
        if target.exists():
            require(read(target) == record, "existing BIC group metadata differs")
        else:
            write(target, record)
        records.append(dict(group_id=group["group_id"], group_json_sha256=sha(target)))
        if len(records) % 100 == 0:
            print(json.dumps(dict(prepared_groups=len(records), n_groups=plan["n_groups"])), flush=True)
    ready = dict(schema_version=1, status="complete", success=True, plan_fingerprint=plan["plan_fingerprint"],
                 plan_sha256=sha(root / "plan.json"), source_manifest_sha256=plan["source_manifest_sha256"],
                 n_groups=plan["n_groups"], n_tasks=plan["n_tasks"], groups=records,
                 finished_at=legacy._now(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                 storage_policy="read reference observations/source frames in place; no raw-data copies")
    write(root / "preparation.json", ready)
    return ready


def load_group(root, plan, group):
    ready = read(root / "preparation.json")
    require(ready["success"] and ready["plan_fingerprint"] == plan["plan_fingerprint"] and
            ready["plan_sha256"] == sha(root / "plan.json"), "BIC preparation identity mismatch")
    item = next(r for r in ready["groups"] if r["group_id"] == group["group_id"])
    path = root / "groups" / group["group_id"] / "group.json"
    require(sha(path) == item["group_json_sha256"], "group metadata changed")
    meta = read(path)
    require(meta["group"] == group and meta["plan_fingerprint"] == plan["plan_fingerprint"], "group identity changed")
    for name, record in meta["files"].items():
        require(sha(record["path"]) == record["sha256"], f"reference input changed: {name}")
    data = _load_npz(meta["files"]["data.npz"]["path"])
    require(legacy._fingerprints(data) == meta["fingerprints"], "data fingerprints changed")
    return data, _load_npz(meta["files"]["source.npz"]["path"]), meta


def _source(api, arrays, metadata, source_rank):
    # The observed full frame is unchanged. Only initializer prefixes differ.
    return api.SourceBases(left=arrays["left"], right=arrays["right"],
                           leading_left=arrays["left"][:, :source_rank],
                           leading_right=arrays["right"][:, :source_rank],
                           source_singular_values=arrays["source_singular_values"], **metadata)


def _options(api, group, task, config, free, left, right):
    rank, p, q = task["rank"], group["p"], group["q"]
    counts = (p, q) if free is None else (free, free)
    return dict(rank=rank, source_rank=task["initializer_source_rank"],
                free_directions=counts, margins=api.Margins(**group["margins"]),
                calibration=api.PracticalCalibration(task["init_penalty"], (left, right),
                    config["inverse_step"], ((p-counts[0])*rank, (q-counts[1])*rank)),
                iterations=config["iterations"], max_backtracks=config["max_backtracks"],
                stationarity_tol=config["stationarity_tol"], checkpoint_iterations=(),
                validation_patience=None, adaptive_anchors=config["adaptive_anchors"],
                max_anchor_switches=config["max_anchor_switches"],
                refinement_solver=config["refinement_solver"], rrr_shortcut=True)


def fit_candidate(api, group, task, config, data, source, cache, free, left, right, candidate_id):
    """No held-out data or truth are passed to fitting or BIC scoring."""
    from sparse_smart_v2.selection import bic_score
    started = time.monotonic()
    counts = [group["p"], group["q"]] if free is None else [free, free]
    row = dict(candidate_id=candidate_id, rank=task["rank"], free_directions=counts,
               init_penalty=task["init_penalty"], penalty_u=left, penalty_v=right,
               success=False, execution_success=False, fit_method="target_rrr" if free is None else "sparse_smart_v2",
               selection=None, metrics=None, state_key=None)
    model, arrays, history = None, {}, None
    try:
        model = api.SparseSMARTv2(**_options(api, group, task, config, free, left, right))
        model.fit(data["X"], data["Y"], source=None if free is None else source,
                  validation_data=None, _initialization_cache=cache)
        require(not model.validation_history_, "validation was evaluated during BIC fitting")
        row.update(execution_success=True, success=bool(model.success_), fit_status=model.status_,
                   termination_reason=model.termination_reason_, n_iter=int(model.n_iter_),
                   selected_iteration=model.selected_iteration_, optimization_converged=bool(model.optimization_converged_),
                   fit_method=model.method_, message=model.message_,
                   numerical_work=getattr(model, "numerical_work_", None),
                   chart_transitions=len(getattr(model, "anchor_switches_", [])),
                   initialization_singular_values=model.metadata_.get("initialization_singular_values"),
                   initialization_diagnostics=model.metadata_.get("lasso"))
        if model.success_:
            require(model.selected_iteration_ == model.n_iter_, "BIC candidate is not the terminal fit")
            coefficient = model.coefficient_
            arrays = dict(coefficient=coefficient)
            if model.method_ == "target_rrr":
                score = bic_score(data["X"], data["Y"], coefficient, rank=task["rank"], direct_rrr=True,
                                  support_tolerance=config["support_tolerance"],
                                  design_rank=model.metadata_.get("design_rank"))
                row["rrr_certificate"] = model.rrr_certificate_
            else:
                chart, state, free_rows = model.chart_, model.last_state_, model.free_rows_
                P, d, Q = chart.reconstruct(state)
                zu, zv = chart.unpack(state)[-2:]
                score = bic_score(data["X"], data["Y"], coefficient, rank=task["rank"],
                    free_directions=counts, weighted_u=zu, weighted_v=zv,
                    penalized_u=free_rows.penalized_u, penalized_v=free_rows.penalized_v,
                    support_tolerance=config["support_tolerance"])
                arrays.update(P=P, d=d, Q=Q, state=state, weighted_u=zu, weighted_v=zv,
                    penalized_u=free_rows.penalized_u, penalized_v=free_rows.penalized_v,
                    free_rows_u=free_rows.rows_u, free_rows_v=free_rows.rows_v,
                    anchors_u=chart.anchors_u, anchors_v=chart.anchors_v,
                    center_u=chart.center_u, center_v=chart.center_v)
            row["selection"] = score.as_dict()
            # Evaluation is deliberately downstream of training-only fitting/scoring.
            row["metrics"] = legacy._metrics(coefficient, data)
            history = dict(history=model.history_, terminal_record=getattr(model, "terminal_record_", None),
                           history_chart_epochs=getattr(model, "history_chart_epochs_", []),
                           chart_transitions=getattr(model, "anchor_switches_", []))
    except Exception as error:
        row.update(success=False, execution_success=False, fit_status="execution_failed",
                   error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    row["elapsed_seconds"] = time.monotonic() - started
    return row, arrays, history


def run_task(root, task_id):
    legacy._require_slurm(root)
    plan, manifest = load_plan(root)
    require(type(task_id) is int and 0 <= task_id < plan["n_tasks"], "invalid task ID")
    task = plan["tasks"][task_id]
    require(task["task_id"] == task_id, "task index mismatch")
    group = next(g for g in plan["groups"] if g["group_id"] == task["group_id"])
    destination = root / "tasks" / f"{task_id:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    identity = dict(schema_version=1, method=METHOD, task=task, group_id=group["group_id"],
                    plan_fingerprint=plan["plan_fingerprint"], source_manifest_sha256=plan["source_manifest_sha256"])
    with (destination / ".fit.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (destination / "result.json").exists():
            result = read(destination / "result.json")
            require(all(result.get(k) == v for k, v in identity.items()), "existing task belongs to another plan")
            if result["execution_success"]:
                for name, expected in result["files"].items():
                    require(sha(destination / name) == expected, "existing task artifact changed")
                return result
        started, tick = legacy._now(), time.monotonic()
        write(destination / "status.json", dict(identity, status="running", started_at=started,
              slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_step_id=os.environ.get("SLURM_STEP_ID")))
        data, source_arrays, meta = load_group(root, plan, group)
        api = legacy._load_api()
        legacy._verify_imports(manifest)
        source = _source(api, source_arrays, meta["source_metadata"], task["initializer_source_rank"])
        cache, outcomes, retained, histories = {}, [], {}, {}
        blocks = ([None] if task["include_rrr"] else []) + task["free_counts"]
        config = plan["configuration"]
        for block_index, free in enumerate(blocks):
            label = "rrr" if free is None else str(free)
            block_path = destination / f"block-{label}.json"
            reusable = block_path.exists()
            if reusable:
                block = read(block_path)
                require(all(block.get(k) == v for k, v in identity.items()), "saved block identity mismatch")
                require(block["free"] == free, "saved block free count changed")
                for name, expected in block["files"].items():
                    require(sha(destination / name) == expected, "saved block artifact changed")
                reusable = all(row["execution_success"] for row in block["outcomes"])
                if not reusable:
                    write(destination / f"block-{label}.failed-{time.time_ns()}.json", block)
            if reusable:
                rows = block["outcomes"]
                arrays = _load_npz(destination / f"block-{label}.npz") if block["retained"] else {}
            else:
                rows, best, arrays, history = [], None, {}, None
                pairs = [(0, 0., 0, 0.)] if free is None else [
                    (ui, u, vi, v) for (ui, u), (vi, v) in product(
                        enumerate(config["penalties_u"]), enumerate(config["penalties_v"])) if u != 0 or v != 0]
                for ui, u, vi, v in pairs:
                    cid = f"t{task_id}_{label}_u{ui}_v{vi}"
                    row, state, trace = fit_candidate(api, group, task, config, data, source, cache, free, u, v, cid)
                    if row["success"] and (best is None or (row["selection"]["score"], cid) < best):
                        best = (row["selection"]["score"], cid)
                        arrays, history = state, trace
                    rows.append(row)
                    write(destination / "progress.json", dict(identity, finished_free_blocks=block_index,
                          current_free=free, completed_in_block=len(rows), block_candidates=len(pairs),
                          completed_candidates=len(outcomes)+len(rows),
                          last_candidate=cid, elapsed_seconds=time.monotonic()-tick, updated_at=legacy._now()))
                best_id = None if best is None else best[1]
                files = {}
                if best_id is not None:
                    legacy._npz(destination / f"block-{label}.npz", arrays)
                    legacy._history(destination / f"block-{label}.history.json.gz", history)
                    files = {name: sha(destination / name) for name in
                             (f"block-{label}.npz", f"block-{label}.history.json.gz")}
                block = dict(identity, free=free, outcomes=rows, retained=best_id, files=files)
                write(block_path, block)
            if block["retained"] is not None:
                prefix = f"c{len(outcomes) + next(i for i, row in enumerate(rows) if row['candidate_id'] == block['retained']):04d}_"
                for row in rows:
                    row["state_key"] = prefix if row["candidate_id"] == block["retained"] else None
                retained.update({prefix + name: value for name, value in arrays.items()})
                histories[block["retained"]] = f"block-{label}.history.json.gz"
            outcomes.extend(rows)
        files = {}
        if retained:
            legacy._npz(destination / "states.npz", retained)
            files["states.npz"] = sha(destination / "states.npz")
        for free in blocks:
            label = "rrr" if free is None else str(free)
            block_path = destination / f"block-{label}.json"
            files[block_path.name] = sha(block_path)
            files.update(read(block_path)["files"])
        result = dict(identity, status="complete", execution_success=all(r["execution_success"] for r in outcomes),
                      success=any(r["success"] for r in outcomes), outcomes=outcomes, files=files,
                      fingerprints=meta["fingerprints"], started_at=started, finished_at=legacy._now(),
                      elapsed_seconds=time.monotonic()-tick, retained_histories=histories,
                      outcome_counts=dict(Counter("eligible" if r["success"] else r["fit_status"] for r in outcomes)),
                      validation_used_for_fit=False, selection_rule="bic_terminal", initializer_cache="one per worker; reused across free counts and penalties",
                      slurm=dict(job_id=os.environ.get("SLURM_JOB_ID"), step_id=os.environ.get("SLURM_STEP_ID")))
        write(destination / "result.json", result)
        write(destination / "status.json", dict(identity, status="finished", execution_success=result["execution_success"],
              success=result["success"], result_sha256=sha(destination / "result.json"), finished_at=result["finished_at"]))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", type=int)
    args = parser.parse_args(argv)
    require(args.root.is_absolute(), "root must be absolute")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        result = prepare(args.root.resolve()) if args.command == "prepare" else run_task(args.root.resolve(), args.task)
        print(json.dumps({k: result[k] for k in ("status", "success", "execution_success", "n_groups", "n_tasks", "outcome_counts") if k in result}))
        return 0 if result.get("execution_success", True) else 1
    except (Exception, KeyboardInterrupt) as error:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
