"""Independently audit saved rank-study artifacts; never refit a simulation.

Every planned task/method retains a denominator entry. Candidate coefficients
are independently rescored on the saved validation observations, both within
rank/source cells and across cells. This verifies the recorded finite-library
selection and archive separation, not a proof that arbitrary external code
never accessed truth. Production array processing is Slurm-only; local opt-in
is confined to tiny saved-artifact fixtures.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import sys

import numpy as np

_code_root = Path(__file__).resolve().parents[2]
for _path in (_code_root / "simulation", _code_root / "sparse-smart/src", _code_root / "sparse-smart-v2/src"):
    sys.path.insert(0, str(_path))
from reviewer_revision_20260913 import audit as common


LABELS = ("v2", "v2_transfer_only", "initializer_only")


def _score(coefficient, data):
    return float(np.mean((data["Y_validation"] - data["X_validation"] @ coefficient)**2))


def _best(candidates, data):
    """Reconstruct the validation winner without importing selection code."""
    winner, prediction = None, None
    for candidate in candidates:
        if not candidate.get("eligible"):
            continue
        current = data["X_validation"] @ candidate["coefficient"]
        difference = None if prediction is None else float(np.mean(
            (current - prediction) * (current + prediction - 2 * data["Y_validation"])))
        if winner is None or difference < 0:
            winner, prediction = candidate, current
    return winner


def _seconds(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value >= 0


def _coefficient(path, shape):
    value = common._arrays(path)["coefficient"]
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Incorrect shape or nonfinite coefficient")
    return value


def _archive(directory, filename, expected_specs, data, shape, issue, *, kind="candidate", benchmark=False):
    """Check every terminal candidate outcome and every saved coefficient."""
    archive = common._read_json(directory / filename)
    records = archive.get("candidates", [])
    if archive.get("complete") is not True:
        issue("incomplete_candidates", str(directory / filename))
    index_key, spec_key = ("candidate_index", "candidate") if benchmark else ("index", "spec")
    if [row.get(index_key) for row in records] != list(range(len(expected_specs))):
        issue("candidate_index_or_count_mismatch", str(directory))
    if [row.get(spec_key) for row in records] != expected_specs:
        issue("candidate_specs_mismatch", str(directory))
    if kind != "initializer":
        if archive.get("expected_n_candidates") != len(expected_specs):
            issue("declared_candidate_count_mismatch", str(directory))
        if archive.get("specs_sha256") != common._specs_digest(expected_specs):
            issue("candidate_specs_hash_mismatch", str(directory))
    enriched, saved = [], set()
    for position, row in enumerate(records):
        item = dict(row, index=position, spec=row.get(spec_key))
        if benchmark:
            status = row.get("status")
            if status not in ("ok", "failed", "uncertified"):
                issue("missing_candidate_outcome", str(directory))
            eligible = status == "ok"
            has_coefficient = row.get("coefficient_saved") is True
            if has_coefficient != (status in ("ok", "uncertified")):
                issue("benchmark_coefficient_status_mismatch", f"{directory}: {position}")
        else:
            eligible = row.get("eligible") is True
            has_coefficient = eligible
            if not isinstance(row.get("eligible"), bool):
                issue("missing_candidate_outcome", f"{directory}: {position}")
            if eligible and kind == "candidate" and row.get("exception") is not None:
                issue("eligible_candidate_exception", f"{directory}: {position}")
        item["eligible"] = eligible
        if not _seconds(row.get("elapsed_seconds")):
            issue("invalid_candidate_runtime", f"{directory}: {position}")
        if has_coefficient:
            path = directory / f"{kind}-{position:04d}.npz"
            saved.add(path.name)
            coefficient = _coefficient(path, shape)
            item["coefficient"] = coefficient
            item["recomputed_validation_mse"] = _score(coefficient, data)
            if not common._close(item["recomputed_validation_mse"], row.get("validation_mse")):
                issue("candidate_validation_mismatch", f"{directory}: {position}")
            if benchmark and not common._close(row.get("validation_loss"), item["recomputed_validation_mse"] * shape[1] / 2):
                issue("benchmark_validation_normalization", f"{directory}: {position}")
        enriched.append(item)
    if {path.name for path in directory.glob(f"{kind}-*.npz")} != saved:
        issue("coefficient_inventory_mismatch", str(directory))
    count = sum(row["eligible"] for row in enriched)
    if kind != "initializer" and archive.get("n_eligible") != count:
        issue("eligible_count_mismatch", str(directory))
    total = sum(row.get("elapsed_seconds", 0.) for row in records if _seconds(row.get("elapsed_seconds")))
    if not _seconds(archive.get("total_seconds")):
        issue("invalid_archive_runtime", str(directory))
    elif kind == "initializer" and archive["total_seconds"] + 1e-8 < total:
        issue("initializer_wall_less_than_calls", str(directory))
    return enriched, archive, total


def _initial_specs(specs, config):
    seen, result = set(), []
    for spec in specs:
        penalty = spec["init_penalty"]
        if penalty in config["init_penalties"] and penalty not in seen:
            result.append(spec)
            seen.add(penalty)
    return result


def _benchmark_specs(method, config):
    """Independent finite-grid reconstruction; no generator truth is read."""
    ranks = list(dict.fromkeys(config["rank_grid"]))
    sources = list(dict.fromkeys(config["source_truncation_grid"]))
    if method == "nuclear_contrast":
        return [dict(method=method, nuclear_penalty=value) for value in config["nuclear_penalties"]]
    if method == "ridge_to_source":
        return [dict(method=method, ridge=value) for value in config["benchmark_ridges"]]
    ridges = [0.] if method in ("target_rrr", "source_subspace_rrr", "oracle_subspace_rrr") else config["benchmark_ridges"]
    specs = [dict(method=method, rank=rank, ridge=ridge) for rank, ridge in product(ranks, ridges)]
    if method.startswith("source_subspace"):
        specs = [dict(spec, source_rank=source_rank) for spec, source_rank in product(specs, sources)]
    if method == "source_target_mixture":
        specs = [dict(spec, alpha=alpha) for spec, alpha in product(specs, config["mixture_weights"])]
    return specs


def _v2_evidence(directory, case, config, cells, libraries, data, issue):
    shape = (case["p"], case["q"])
    archive = common._read_json(directory / "v2" / "rank-cells.json")
    if archive.get("complete") is not True:
        issue("incomplete_rank_cells", "rank-cells.json is not complete")
    recorded_cells = archive.get("cells", [])
    if len(recorded_cells) != len(cells):
        issue("rank_cell_count_mismatch", "Missing/repeated cell records")
    pool = {name: [] for name in LABELS}
    totals = dict(fitting_call_seconds=0., cell_wall_seconds=0., initializer_pass_wall_seconds=0.,
                  n_candidates=0, n_eligible=0, n_initializer_candidates=0, n_initializer_eligible=0)
    for position, cell in enumerate(cells):
        recorded = recorded_cells[position] if position < len(recorded_cells) else {}
        if any(recorded.get(key) != value for key, value in cell.items()):
            issue("rank_cell_identity_mismatch", str(position))
        if not cell["admissible"]:
            if recorded.get("status") != "inadmissible":
                issue("inadmissible_cell_executed", str(position))
            for label in LABELS:
                pool[label].append(None)
            continue
        cell_dir = directory / "v2" / f"cell-{cell['cell_index']:03d}"
        if recorded.get("status") != "completed":
            issue("rank_cell_incomplete", f"Cell {position} has status {recorded.get('status')}")
        specs = libraries[position]["candidates"]
        operational = dict(case, fitted_rank=cell["fitted_rank"], source_rank=cell["source_rank"])
        if specs != common._declared_specs(operational, config, "full_caps"):
            issue("frozen_rank_specs_mismatch", str(position))
        candidates, candidate_archive, candidate_time = _archive(
            cell_dir, "candidates.json", specs, data, shape, issue)
        initializers, initial_archive, initial_time = _archive(
            cell_dir, "initializers.json", _initial_specs(specs, config), data, shape, issue, kind="initializer")
        elapsed = candidate_time + initial_time
        if not common._close(elapsed, candidate_archive.get("total_seconds")) or not common._close(elapsed, recorded.get("fitting_call_seconds")):
            issue("cell_tuning_time_mismatch", str(position))
        if not _seconds(recorded.get("wall_seconds")) or recorded.get("wall_seconds", 0.) + 1e-8 < elapsed:
            issue("cell_wall_less_than_calls", str(position))
        for label, eligible in (("v2", candidates),
                ("v2_transfer_only", [row for row in candidates if row.get("method") != "target_rrr"]),
                ("initializer_only", initializers)):
            selected = _best(eligible, data)
            pool[label].append(None if selected is None else dict(selected, cell_index=cell["cell_index"]))
        totals["fitting_call_seconds"] += elapsed
        totals["cell_wall_seconds"] += recorded.get("wall_seconds", 0.)
        totals["initializer_pass_wall_seconds"] += initial_archive.get("total_seconds", 0.)
        totals["n_candidates"] += len(candidates)
        totals["n_eligible"] += sum(row["eligible"] for row in candidates)
        totals["n_initializer_candidates"] += len(initializers)
        totals["n_initializer_eligible"] += sum(row["eligible"] for row in initializers)
        if recorded.get("n_candidates") != len(candidates) or recorded.get("n_eligible") != sum(row["eligible"] for row in candidates):
            issue("cell_candidate_counts_mismatch", str(position))
    runtime = archive.get("runtime", {})
    for key, value in totals.items():
        if not common._close(value, runtime.get(key)):
            issue("rank_tuning_total_mismatch", key)
    if runtime.get("shared_computation_group") != "v2_rank_source_grid" or runtime.get("cold_selected_fit_seconds") is not None:
        issue("runtime_scope_mismatch", "Shared/cached rank timings mislabeled")
    return pool, runtime


def _method_selection(directory, method, recorded, pool, runtime, coefficients, data, issue):
    outer = common._read_json(directory / f"{method}-rank-selection.json")
    if outer.get("complete") is not True or outer.get("selection") != recorded:
        issue("rank_selection_record_mismatch", method)
    cells = outer.get("candidates", [])
    trace = recorded.get("selection_audit", [])
    if len(cells) != len(pool) or len(trace) != len(pool):
        issue("rank_selection_grid_count", method)
    for index, expected in enumerate(pool):
        row = cells[index] if index < len(cells) else {}
        audit = trace[index] if index < len(trace) else {}
        if row.get("cell_index") != index or audit.get("cell_index") != index or audit.get("record_index") != index:
            issue("rank_selection_cell_index", f"{method}: {index}")
        if row.get("eligible") is not (expected is not None) or audit.get("eligible") is not (expected is not None):
            issue("rank_cell_eligibility_mismatch", f"{method}: {index}")
        if expected is not None:
            if row.get("candidate_index") != expected["index"] or row.get("spec") != expected["spec"]:
                issue("nonminimal_within_cell_selection", f"{method}: {index}")
            if not common._close(audit.get("validation_mse"), expected["recomputed_validation_mse"]):
                issue("rank_selection_validation_mismatch", f"{method}: {index}")
    eligible = [row for row in pool if row is not None]
    winner = _best(eligible, data)
    if recorded.get("runtime") != runtime:
        issue("method_shared_runtime_mismatch", method)
    if recorded.get("n_grid_cells") != len(pool) or recorded.get("n_successful_cells") != len(eligible):
        issue("method_cell_counts_mismatch", method)
    if (recorded.get("success") is True) != (winner is not None):
        issue("rank_success_eligibility_mismatch", method)
    if winner is None:
        if not recorded.get("error"):
            issue("unexplained_failure", method)
        return
    if recorded.get("selected_index") != winner["cell_index"] or recorded.get("selected_cell_index") != winner["cell_index"] or recorded.get("selected_inner_candidate_index") != winner["index"]:
        issue("nonminimal_joint_rank_selection", method)
    if recorded.get("parameters") != winner["spec"] or recorded.get("selected_fitted_rank") != winner["spec"]["rank"]:
        issue("selected_rank_parameters_mismatch", method)
    actual_method = "lasso_svd_initializer" if method == "initializer_only" else winner.get("method")
    source_dimension = None if actual_method == "target_rrr" else winner["spec"]["source_rank"]
    if recorded.get("selected_method") != actual_method or recorded.get("selected_source_dimension") != source_dimension:
        issue("selected_source_or_endpoint_mismatch", method)
    expected_iteration = 0 if method == "initializer_only" else winner.get("selected_iteration")
    if recorded.get("selected_iteration") != expected_iteration:
        issue("selected_checkpoint_mismatch", method)
    if not np.array_equal(coefficients.get(method), winner["coefficient"]):
        issue("selected_coefficient_mismatch", method)


def _load_plan(root, *, allow_tiny_local=False):
    root = Path(root).resolve()
    plan_bytes = (root / "rank-plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    production = bool(os.environ.get("SLURM_JOB_ID")) and root.is_relative_to(Path("/scratch2"))
    tiny = allow_tiny_local and len(plan.get("tasks", [])) <= 8 and all(
        max(case.get("p", 999), case.get("q", 999)) <= 32 and
        max(case.get("n_train", 999), case.get("n_validation", 999), case.get("n_test", 999)) <= 64
        for case in plan.get("cases", []))
    if not production and not tiny:
        raise RuntimeError("Production rank auditing requires Slurm and /scratch2; local opt-in is restricted to tiny fixtures")
    return root, plan, hashlib.sha256(plan_bytes).hexdigest()


def _task_indices(plan, task_indices):
    if task_indices is None:
        return list(range(len(plan["tasks"])))
    indices = list(task_indices)
    if any(type(index) is not int or not 0 <= index < len(plan["tasks"]) for index in indices):
        raise ValueError("Task indices must be integer positions in the original rank plan")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate task indices")
    return sorted(indices)


def audit(root, *, write=True, allow_tiny_local=False, task_indices=None):
    """Audit original-plan tasks; subsets are read-only worker results.

    A subset can report its own validity, but cannot claim a complete study or
    publish final study artifacts. The full plan remains unchanged and retains
    the original task indices and all production/local size guards.
    """
    if task_indices is not None and write:
        raise ValueError("Subset rank audits require write=False")
    root, plan, digest = _load_plan(root, allow_tiny_local=allow_tiny_local)
    indices = _task_indices(plan, task_indices)
    issues, rows, tasks = [], [], []
    def issue(code, message, task_index=None, method=None):
        issues.append(dict(code=code, message=str(message), task_index=task_index, method=method, severity="error"))
    if len(plan["cases"]) != plan.get("n_cases") or len(plan["tasks"]) != plan.get("n_tasks"):
        issue("plan_count_mismatch", "Declared case/task count differs")
    if len({(task["case_index"], task["seed"]) for task in plan["tasks"]}) != len(plan["tasks"]):
        issue("duplicate_planned_task", "Repeated case/seed pair")
    config = plan["configuration"]
    if config.get("tuning_uses_truth") is not False or config.get("refit_on_validation") is not False:
        issue("selection_policy_mismatch", "Expected validation-only selection without refitting")
    for case in plan["cases"]:
        expected_pairs = list(product(dict.fromkeys(config["rank_grid"]), dict.fromkeys(config["source_truncation_grid"])))
        cells = plan["rank_cells"][case["case_id"]]
        if [(row.get("fitted_rank"), row.get("source_rank")) for row in cells] != expected_pairs or any(
                row.get("cell_index") != i or row.get("admissible") != (row["fitted_rank"] <= row["source_rank"])
                for i, row in enumerate(cells)):
            issue("operational_rank_grid_mismatch", case["case_id"])
        for method, library in plan["competitor_libraries"][case["case_id"]].items():
            if library != _benchmark_specs(method, config):
                issue("operational_benchmark_grid_mismatch", f"{case['case_id']}: {method}")
    plan_invalid = bool(issues)
    for task_index in indices:
        task = plan["tasks"][task_index]
        case = plan["cases"][task["case_index"]]
        expected_methods = task["expected_methods"]
        directory = root / "rank_tasks" / f"task-{task_index:06d}"
        base = dict(task_index=task_index, case_id=case["case_id"], seed=task["seed"],
                    family=case.get("family", ""), level=case.get("level", ""), n_train=case["n_train"],
                    p=case["p"], q=case["q"], variant="rank_selected")
        report = dict(task_index=task_index, case_id=case["case_id"], state="missing")
        tasks.append(report)
        def task_issue(code, message):
            issue(code, message, task_index)
        if not (directory / "result.json").exists():
            status = "incomplete_task" if directory.exists() else "missing_task"
            task_issue(status, "No completed result.json")
            report["state"] = status
            rows.extend(dict(base, method=method, status=status) for method in expected_methods)
            continue
        task_start = len(issues)
        try:
            result = common._read_json(directory / "result.json")
            if result.get("complete") is not True or result.get("rank_evidence_schema") != 2:
                task_issue("incomplete_or_legacy_rank_evidence", "Expected complete evidence schema 2")
            if result.get("plan_sha256") != digest or result.get("task") != task or result.get("case") != case:
                task_issue("task_identity_or_plan_hash", "Task does not match frozen plan")
            if result.get("executed_code_sha256") != plan.get("code_sha256"):
                task_issue("executed_code_mismatch", "Executed source hashes differ from the frozen plan")
            data, truth = common._arrays(directory / "data.npz"), common._arrays(directory / "truth.npz")
            coefficients = common._arrays(directory / "coefficients.npz")
            if common._fingerprint(data) != result.get("fit_data_fingerprint") or common._fingerprint(truth) != result.get("truth_fingerprint"):
                task_issue("data_or_truth_fingerprint_mismatch", "Saved data differs from fitting/evaluation record")
            forbidden = {"C_star", "U0", "V0", "U_star", "V_star", "Sigma_x", "clean_source", "X_test", "Y_test"}
            if forbidden.intersection(data):
                task_issue("truth_in_fit_archive", "Truth/test objects present in fitting data archive")
            shapes = {"X": (case["n_train"], case["p"]), "Y": (case["n_train"], case["q"]),
                      "X_validation": (case["n_validation"], case["p"]),
                      "Y_validation": (case["n_validation"], case["q"]), "C0": (case["p"], case["q"])}
            if any(data.get(key, np.empty(0)).shape != shape for key, shape in shapes.items()) or not all(np.isfinite(value).all() for value in [*data.values(), *truth.values()]):
                task_issue("invalid_data_arrays", "Shape or finiteness mismatch")
            frames = common._arrays(directory / "observed-source-frames.npz")
            if common._fingerprint(frames) != result.get("observed_source_frames_fingerprint"):
                task_issue("source_frame_fingerprint_mismatch", "Saved common frames changed")
            left, right, values = frames["left"], frames["right"], frames["singular_values"]
            p, q = case["p"], case["q"]
            if left.shape != (p, p) or right.shape != (q, q) or values.shape != (min(p, q),):
                task_issue("source_frame_shape_mismatch", "Expected full completed source frames")
            elif not (np.allclose(left.T @ left, np.eye(p), atol=1e-9, rtol=1e-9)
                      and np.allclose(right.T @ right, np.eye(q), atol=1e-9, rtol=1e-9)
                      and np.allclose((left[:, :len(values)] * values) @ right[:, :len(values)].T,
                                      data["C0"], atol=1e-9, rtol=1e-9)):
                task_issue("source_frames_not_observed_svd", "Common frames do not reconstruct supplied source matrix")
            # Only the source-only frame construction is repeated, not a fit.
            # This also verifies deterministic completion when fitted rank < r0.
            from sparse_smart_v2 import ObservedSource, prepare_source
            expected_frames = prepare_source(ObservedSource(data["C0"]), p=p, q=q,
                source_rank=max(config["source_truncation_grid"]))
            if not (np.allclose(left, expected_frames.left, rtol=1e-10, atol=1e-10)
                    and np.allclose(right, expected_frames.right, rtol=1e-10, atol=1e-10)):
                task_issue("source_frames_differ_from_v2", "Saved frames differ from v2's source-only deterministic convention")
            for key in ("generation_and_source_fit_seconds", "shared_source_decomposition_seconds", "wall_seconds"):
                if not _seconds(result.get(key)):
                    task_issue("invalid_task_runtime", key)
        except Exception as error:
            task_issue("unreadable_task_evidence", str(error))
            report["state"] = "invalid"
            rows.extend(dict(base, method=method, status="audit_invalid") for method in expected_methods)
            continue
        task_invalid = plan_invalid or len(issues) > task_start
        methods = result.get("methods", {})
        if set(methods) != set(expected_methods):
            task_issue("method_inventory_mismatch", "Expected method outcomes differ from saved records")
        successes = {name for name, row in methods.items() if row.get("success") is True}
        if set(coefficients) != successes:
            task_issue("selected_coefficient_inventory", "Success/failure flags do not match coefficient archive")
            task_invalid = True
        group_start = len(issues)
        try:
            pool, runtime = _v2_evidence(directory, case, config, plan["rank_cells"][case["case_id"]],
                plan["v2_libraries"][case["case_id"]], data, task_issue)
        except Exception as error:
            task_issue("unreadable_rank_cell_evidence", str(error))
            pool, runtime = {}, {}
        group_invalid = len(issues) > group_start
        for method in expected_methods:
            start = len(issues)
            recorded = methods.get(method, {})
            row = dict(base, method=method, selected_fitted_rank=recorded.get("selected_fitted_rank"),
                       selected_source_dimension=json.dumps(recorded.get("selected_source_dimension")),
                       selected_method=recorded.get("selected_method"), error=recorded.get("error"))
            def method_issue(code, message):
                issue(code, message, task_index, method)
            if not recorded:
                method_issue("missing_method", method)
                rows.append(dict(row, status="missing_method"))
                continue
            if not isinstance(recorded.get("success"), bool):
                method_issue("missing_success_flag", method)
            try:
                if method in LABELS:
                    if group_invalid:
                        method_issue("invalid_rank_cell_group", "Cannot validate joint winner with invalid cell evidence")
                    else:
                        _method_selection(directory, method, recorded, pool[method], runtime, coefficients, data, method_issue)
                    row["total_seconds"] = runtime.get("initializer_pass_wall_seconds") if method == "initializer_only" else runtime.get("cell_wall_seconds")
                    row["runtime_scope"] = "initializer_passes" if method == "initializer_only" else "shared_v2_grid_wall_time"
                else:
                    library = plan["competitor_libraries"][case["case_id"]][method]
                    candidates, archive, elapsed = _archive(directory / "benchmarks" / method, "candidates.json",
                        library, data, (case["p"], case["q"]), method_issue, benchmark=True)
                    if recorded.get("audit") != archive.get("candidates"):
                        method_issue("benchmark_audit_mismatch", method)
                    if not common._close(elapsed, archive.get("total_seconds")) or not common._close(elapsed, recorded.get("fitting_call_seconds")):
                        method_issue("benchmark_tuning_time_mismatch", method)
                    if not _seconds(recorded.get("wall_seconds")) or recorded.get("wall_seconds", 0.) + 1e-8 < elapsed:
                        method_issue("benchmark_wall_less_than_calls", method)
                    if recorded.get("n_candidates") != len(candidates) or recorded.get("n_eligible") != sum(item["eligible"] for item in candidates):
                        method_issue("benchmark_counts_mismatch", method)
                    winner = _best(candidates, data)
                    if (recorded.get("success") is True) != (winner is not None):
                        method_issue("benchmark_success_eligibility_mismatch", method)
                    if winner is not None:
                        if recorded.get("selected_index") != winner["index"] or archive.get("selected_index") != winner["index"]:
                            method_issue("nonminimal_benchmark_rank_selection", method)
                        if not np.array_equal(coefficients.get(method), winner["coefficient"]):
                            method_issue("selected_coefficient_mismatch", method)
                        expected_rank = None if method in ("ridge_to_source", "nuclear_contrast") else winner["spec"]["rank"]
                        if recorded.get("selected_fitted_rank") != expected_rank:
                            method_issue("selected_rank_parameters_mismatch", method)
                        if method.startswith("source_subspace"):
                            expected_source = [winner["spec"]["source_rank"]] * 2
                            if recorded.get("selected_source_dimension") != expected_source:
                                method_issue("selected_source_dimension_mismatch", method)
                            if any(item.get("diagnostics", {}).get("source_frame_convention") != "supplied_common_observed_source_decomposition"
                                   for item in candidates if item.get("coefficient_saved")):
                                method_issue("source_frame_convention_mismatch", method)
                    row["total_seconds"] = recorded.get("wall_seconds")
                    row["runtime_scope"] = "benchmark_grid_wall_time_shared_source_decomposition_reported_separately"
                if recorded.get("success"):
                    C = coefficients[method]
                    if C.shape != (case["p"], case["q"]) or not np.isfinite(C).all():
                        method_issue("invalid_selected_coefficient", method)
                    metrics = common._recompute(C, data, truth)
                    row.update(metrics)
                    for key, value in metrics.items():
                        if not common._close(value, recorded.get("metrics", {}).get(key)):
                            method_issue("metric_mismatch", f"{method}: {key}")
                elif not recorded.get("error"):
                    method_issue("unexplained_failure", method)
            except Exception as error:
                method_issue("unreadable_method_evidence", str(error))
            invalid = task_invalid or len(issues) > start
            row["status"] = "audit_invalid" if invalid else ("success" if recorded.get("success") else "failed")
            rows.append(row)
        report["state"] = "invalid" if any(row["status"] == "audit_invalid" for row in rows if row["task_index"] == task_index) else "complete"
    return _assemble(root, plan, digest, issues, tasks, rows, write=write,
                     task_indices=None if task_indices is None else indices)


def _assemble(root, plan, digest, issues, tasks, rows, *, write=True, task_indices=None):
    """Summarize audited rows identically for serial and complete parallel runs."""
    aggregate, paired = common._summaries(rows)
    runtime_rows = []
    for case_id, method in sorted({(row["case_id"], row["method"]) for row in rows}):
        members = [row for row in rows if row["case_id"] == case_id and row["method"] == method]
        values = [row["total_seconds"] for row in members if row["status"] in ("success", "failed")
                  and _seconds(row.get("total_seconds"))]
        runtime_rows.append(dict(case_id=case_id, method=method, n_planned=len(members),
            n_success=sum(row["status"] == "success" for row in members),
            n_failure=sum(row["status"] == "failed" for row in members),
            n_unavailable=len(members) - len(values), runtime_scope=members[0].get("runtime_scope"),
            **common._moments(values), median=float(np.median(values)) if values else None,
            q25=float(np.quantile(values, .25)) if values else None,
            q75=float(np.quantile(values, .75)) if values else None))
    rank_counts = Counter((row["case_id"], row["method"], row.get("selected_fitted_rank"),
                           row.get("selected_source_dimension")) for row in rows if row["status"] == "success")
    rank_selection = [dict(case_id=case_id, method=method, selected_fitted_rank=rank,
        selected_source_dimension=source, count=count,
        n_planned=sum(row["case_id"] == case_id and row["method"] == method for row in rows),
        n_success=sum(row["case_id"] == case_id and row["method"] == method and row["status"] == "success" for row in rows))
        for (case_id, method, rank, source), count in sorted(rank_counts.items(), key=lambda item: str(item[0]))]
    audit_result = dict(audit_passed=not issues, all_planned_tasks_complete=all(task["state"] == "complete" for task in tasks),
        issues=issues, tasks=tasks, plan_sha256=digest,
        scope="Saved data separation, complete finite libraries, coefficient/validation/metric recomputation and timing reconciliation; no refitting and no proof of absence of external data access.")
    summary = dict(n_planned_tasks=len(plan["tasks"]), status_counts=dict(Counter(row["status"] for row in rows)),
        n_planned_method_outcomes=len(rows), runtime_note="Shared v2 times repeat across v2 labels; do not sum them. Source-data generation/fitting and shared source decomposition remain separately recorded in task results.")
    if task_indices is not None:
        audit_result.update(audit_passed=False, all_planned_tasks_complete=False,
            coverage_scope="task_subset", subset_valid=not issues,
            audited_task_indices=list(task_indices), n_audited_tasks=len(tasks))
    if write:
        common._atomic_json(root / "rank-audit.json", audit_result)
        common._atomic_json(root / "rank-summary.json", summary)
        for filename, output in (("rank-per-replication.csv", rows), ("rank-aggregate.csv", aggregate),
                                 ("rank-paired.csv", paired), ("rank-selection.csv", rank_selection),
                                 ("rank-runtime.csv", runtime_rows)):
            fields = list(dict.fromkeys(key for row in output for key in row))
            common._csv(root / filename, output, fields)
    return dict(audit=audit_result, summary=summary, per_replication=rows, aggregate=aggregate,
                paired=paired, rank_selection=rank_selection, runtime=runtime_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = audit(args.root, write=not args.no_write)
    print(json.dumps(dict(audit_passed=result["audit"]["audit_passed"],
                          status_counts=result["summary"]["status_counts"]), sort_keys=True))
    raise SystemExit(0 if result["audit"]["audit_passed"] else 1)


if __name__ == "__main__":
    main()
