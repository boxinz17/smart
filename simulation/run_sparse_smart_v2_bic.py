#!/usr/bin/env python3
"""Training-only grouped BIC fits on Discovery, with immutable reference inputs.

One dispatched task shares an initializer across free counts and penalty pairs.
The terminal fit or saved early checkpoints are scored by training-only BIC;
validation and truth enter evaluation only after fitting and scoring. Only
possible fixed/automatic winners retain model states, including terminal controls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
from itertools import product
import json
import numbers
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


def _penalty_pairs(config):
    """Resolve optional paired candidates without renumbering the legacy axes."""
    left_axis, right_axis = config["penalties_u"], config["penalties_v"]
    if "penalty_pairs" not in config:
        return [(ui, u, vi, v) for (ui, u), (vi, v) in product(
            enumerate(left_axis), enumerate(right_axis)) if u != 0 or v != 0]
    requested = config["penalty_pairs"]
    require(isinstance(requested, list) and bool(requested), "penalty_pairs must be a nonempty list")
    pairs, seen = [], set()
    for pair in requested:
        require(isinstance(pair, (list, tuple)) and len(pair) == 2,
                "each penalty_pairs entry must contain two values")
        require(all(isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
                    and np.isfinite(value) and value >= 0 for value in pair),
                "penalty_pairs values must be finite nonnegative numbers")
        u, v = pair
        require(u != 0 or v != 0, "penalty_pairs cannot contain (0, 0); RRR is handled separately")
        require((u, v) not in seen, "penalty_pairs must contain distinct pairs")
        require(u in left_axis and v in right_axis, "penalty_pairs values must belong to their existing axes")
        seen.add((u, v))
        pairs.append((left_axis.index(u), u, right_axis.index(v), v))
    return pairs


def _selection_policy(config):
    policy = config.get("selection_rule", "bic_terminal")
    require(policy in ("bic_terminal", "bic_checkpoint"), "unsupported selection policy")
    if policy == "bic_checkpoint":
        checkpoints = config.get("checkpoint_iterations")
        require(isinstance(checkpoints, (list, tuple)) and bool(checkpoints),
                "checkpoint_iterations must be a nonempty sequence")
        require(all(type(t) is int and 0 <= t <= config["iterations"] for t in checkpoints),
                "checkpoint iterations must be integers within the update budget")
        require(list(checkpoints) == sorted(set(checkpoints)) and checkpoints[0] == 0,
                "checkpoint iterations must be increasing, distinct, and include zero")
    return policy


def load_plan(root, *, full_source=False):
    plan = read(root / "plan.json")
    require(plan["method"] == METHOD and plan["schema_version"] == 1, "unsupported BIC plan")
    require(plan["root"] == str(root), "plan belongs to another root")
    require(digest({k: v for k, v in plan.items() if k != "plan_fingerprint"}) == plan["plan_fingerprint"],
            "BIC plan fingerprint mismatch")
    _selection_policy(plan["configuration"])
    require(plan["configuration"]["validation_patience"] is None, "BIC fitting cannot use validation stopping")
    _penalty_pairs(plan["configuration"])
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
                stationarity_tol=config["stationarity_tol"],
                checkpoint_iterations=(tuple(config["checkpoint_iterations"])
                    if _selection_policy(config) == "bic_checkpoint" else ()),
                validation_patience=None, adaptive_anchors=config["adaptive_anchors"],
                max_anchor_switches=config["max_anchor_switches"],
                refinement_solver=config["refinement_solver"], rrr_shortcut=True)


def _score_snapshot(model, group, task, config, data, counts):
    """Use the snapshot's chart and masks, never the trajectory's final chart."""
    from sparse_smart_v2.selection import bic_score
    coefficient = model.coefficient_
    arrays = dict(coefficient=coefficient)
    arguments = dict(rank=task["rank"], support_tolerance=config["support_tolerance"])
    if model.method_ == "target_rrr":
        arguments.update(direct_rrr=True, design_rank=model.metadata_.get("design_rank"))
    else:
        chart, state, free_rows = model.chart_, model.last_state_, model.free_rows_
        P, d, Q = chart.reconstruct(state)
        zu, zv = chart.unpack(state)[-2:]
        arguments.update(free_directions=counts, weighted_u=zu, weighted_v=zv,
                         penalized_u=free_rows.penalized_u, penalized_v=free_rows.penalized_v)
        arrays.update(P=P, d=d, Q=Q, state=state, weighted_u=zu, weighted_v=zv,
            penalized_u=free_rows.penalized_u, penalized_v=free_rows.penalized_v,
            free_rows_u=free_rows.rows_u, free_rows_v=free_rows.rows_v,
            anchors_u=chart.anchors_u, anchors_v=chart.anchors_v,
            center_u=chart.center_u, center_v=chart.center_v)
    return bic_score(data["X"], data["Y"], coefficient, **arguments).as_dict(), arrays


def fit_candidate(api, group, task, config, data, source, cache, free, left, right, candidate_id):
    """No held-out data or truth are passed to fitting or BIC scoring."""
    policy = _selection_policy(config)
    started, cpu_started = time.monotonic(), time.process_time()
    counts = [group["p"], group["q"]] if free is None else [free, free]
    row = dict(candidate_id=candidate_id, rank=task["rank"], free_directions=counts,
               init_penalty=task["init_penalty"], penalty_u=left, penalty_v=right,
               success=False, execution_success=False, fit_method="target_rrr" if free is None else "sparse_smart_v2",
               selection=None, metrics=None, state_key=None)
    if policy == "bic_checkpoint":
        row.update(terminal_selection=None, terminal_metrics=None, terminal_state_key=None,
                   checkpoint_scores=[])
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
            if model.method_ == "target_rrr":
                row["rrr_certificate"] = model.rrr_certificate_
            if policy == "bic_terminal":
                row["selection"], arrays = _score_snapshot(model, group, task, config, data, counts)
                row["metrics"] = legacy._metrics(arrays["coefficient"], data)
            else:
                iterations = ({0} if model.method_ == "target_rrr" else
                    {0, model.n_iter_, *(t for t in config["checkpoint_iterations"] if t <= model.n_iter_)})
                best, terminal_arrays = None, None
                for iteration in sorted(iterations):
                    snapshot = model if iteration == model.n_iter_ else model.checkpoint_model(iteration)
                    require(snapshot.success_ and snapshot.n_iter_ == iteration and
                            snapshot.selected_iteration_ == iteration,
                            "BIC checkpoint is not a successful training-only prefix")
                    score, state_arrays = _score_snapshot(snapshot, group, task, config, data, counts)
                    row["checkpoint_scores"].append(dict(iteration=iteration, selection=score))
                    key = (score["score"], iteration)
                    if best is None or key < best:
                        best, arrays = key, state_arrays
                        row.update(selection=score, selected_iteration=iteration)
                    if iteration == model.n_iter_:
                        row["terminal_selection"], terminal_arrays = score, state_arrays
                require(terminal_arrays is not None, "successful terminal snapshot was not scored")
                # All scores and the choice are frozen before any evaluation.
                row["metrics"] = legacy._metrics(arrays["coefficient"], data)
                row["terminal_metrics"] = legacy._metrics(terminal_arrays["coefficient"], data)
                arrays = dict(arrays, **{"terminal_" + name: value for name, value in terminal_arrays.items()})
            history = dict(history=model.history_, terminal_record=getattr(model, "terminal_record_", None),
                           history_chart_epochs=getattr(model, "history_chart_epochs_", []),
                           chart_transitions=getattr(model, "anchor_switches_", []))
    except Exception as error:
        row.update(success=False, execution_success=False, fit_status="execution_failed",
                   error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        row.update(selection=None, metrics=None, state_key=None)
        if policy == "bic_checkpoint":
            row.update(terminal_selection=None, terminal_metrics=None, terminal_state_key=None,
                       checkpoint_scores=[])
        arrays, history = {}, None
    row["elapsed_seconds"] = time.monotonic() - started
    row["process_cpu_seconds"] = time.process_time() - cpu_started
    return row, arrays, history


def run_task(root, task_id):
    legacy._require_slurm(root)
    plan, manifest = load_plan(root)
    policy = _selection_policy(plan["configuration"])
    if plan["configuration"].get("cheap_bic_phase") == "assessment":
        from sparse_smart_v2_cheap_bic_reuse import verify_assessment_gate
        verify_assessment_gate(root, plan)
    positive_pairs = _penalty_pairs(plan["configuration"])
    rrr_reuse = plan.get("rrr_reuse", {})
    require(isinstance(rrr_reuse, dict), "rrr_reuse must be a group-to-descriptor mapping")
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
        started, tick, cpu_tick = legacy._now(), time.monotonic(), time.process_time()
        write(destination / "status.json", dict(identity, status="running", started_at=started,
              slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_step_id=os.environ.get("SLURM_STEP_ID")))
        data, source_arrays, meta = load_group(root, plan, group)
        api = legacy._load_api()
        legacy._verify_imports(manifest)
        source = _source(api, source_arrays, meta["source_metadata"], task["initializer_source_rank"])
        cache, outcomes, retained, histories, terminal_histories = {}, [], {}, {}, {}
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
                terminal_arrays = (_load_npz(destination / f"block-{label}-terminal.npz")
                    if block.get("terminal_retained") else {})
            else:
                rows, best, arrays, history = [], None, {}, None
                terminal_best, terminal_arrays, terminal_history = None, {}, None
                pairs = [(0, 0., 0, 0.)] if free is None else positive_pairs
                for ui, u, vi, v in pairs:
                    cid = f"t{task_id}_{label}_u{ui}_v{vi}"
                    if free is None and group["group_id"] in rrr_reuse:
                        from sparse_smart_v2_cheap_bic_reuse import reuse_rrr
                        # An explicitly registered endpoint must pass its audit;
                        # integrity failures propagate instead of triggering a refit.
                        row, state, trace = reuse_rrr(rrr_reuse[group["group_id"]], group, task, config, data, cid)
                    else:
                        row, state, trace = fit_candidate(api, group, task, config, data, source, cache, free, u, v, cid)
                    if policy == "bic_checkpoint" and row["success"] and row["fit_method"] == "target_rrr":
                        # Registered legacy endpoints have no checkpoint policy
                        # fields; their single certified state serves both rules.
                        row.update(terminal_selection=row["selection"], terminal_metrics=row["metrics"],
                                   terminal_state_key=None,
                                   checkpoint_scores=[dict(iteration=0, selection=row["selection"])])
                        state = dict(state, **{"terminal_" + name: value for name, value in state.items()
                                             if not name.startswith("terminal_")})
                    if row["success"] and (best is None or (row["selection"]["score"], cid) < best):
                        best = (row["selection"]["score"], cid)
                        arrays = {name: value for name, value in state.items() if not name.startswith("terminal_")}
                        history = trace
                    if (policy == "bic_checkpoint" and row["success"] and
                            (terminal_best is None or (row["terminal_selection"]["score"], cid) < terminal_best)):
                        terminal_best = (row["terminal_selection"]["score"], cid)
                        terminal_arrays = {name[len("terminal_"):]: value for name, value in state.items()
                                           if name.startswith("terminal_")}
                        require(bool(terminal_arrays), "terminal control has no model state")
                        terminal_history = trace
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
                if policy == "bic_checkpoint":
                    block["terminal_retained"] = None if terminal_best is None else terminal_best[1]
                    if terminal_best is not None:
                        terminal_names = (f"block-{label}-terminal.npz", f"block-{label}-terminal.history.json.gz")
                        legacy._npz(destination / terminal_names[0], terminal_arrays)
                        legacy._history(destination / terminal_names[1], terminal_history)
                        files.update({name: sha(destination / name) for name in terminal_names})
                write(block_path, block)
            if block["retained"] is not None:
                prefix = f"c{len(outcomes) + next(i for i, row in enumerate(rows) if row['candidate_id'] == block['retained']):04d}_"
                for row in rows:
                    row["state_key"] = prefix if row["candidate_id"] == block["retained"] else None
                retained.update({prefix + name: value for name, value in arrays.items()})
                histories[block["retained"]] = f"block-{label}.history.json.gz"
            if block.get("terminal_retained") is not None:
                prefix = f"tc{len(outcomes) + next(i for i, row in enumerate(rows) if row['candidate_id'] == block['terminal_retained']):04d}_"
                for row in rows:
                    row["terminal_state_key"] = prefix if row["candidate_id"] == block["terminal_retained"] else None
                retained.update({prefix + name: value for name, value in terminal_arrays.items()})
                terminal_histories[block["terminal_retained"]] = f"block-{label}-terminal.history.json.gz"
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
                      elapsed_seconds=time.monotonic()-tick, process_cpu_seconds=time.process_time()-cpu_tick,
                      retained_histories=histories,
                      outcome_counts=dict(Counter("eligible" if r["success"] else r["fit_status"] for r in outcomes)),
                      validation_used_for_fit=False, selection_rule=policy, initializer_cache="one per worker; reused across free counts and penalties",
                      slurm=dict(job_id=os.environ.get("SLURM_JOB_ID"), step_id=os.environ.get("SLURM_STEP_ID")))
        if policy == "bic_checkpoint":
            result["retained_terminal_histories"] = terminal_histories
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
