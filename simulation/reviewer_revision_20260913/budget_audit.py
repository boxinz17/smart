"""Independently audit saved iteration-budget evidence without fitting models.

The audit imports neither the budget runner nor the solver. It reconstructs
finite-library validation decisions and all saved coefficient/prediction losses,
retains a row for every planned task and method, and reports objective changes
within chart epochs separately from chart transitions. Missing and failed larger
budgets never remove independently valid smaller-budget outcomes.

Production array processing is Slurm-only. ``allow_local`` is a Python-only
opt-in for deterministic tiny saved-artifact fixtures, not simulation fitting.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reviewer_revision_20260913 import audit as common


CASE_IDS = ("reference_exact_n80", "containment_both30_n80",
            "fitted_source_n0100_n80", "sparse_stress_exact_n40")
SEEDS = (2000, 2001, 2002, 2003, 2004)
BUDGETS = (100, 500, 1000, 2000)
METHODS = ("v2_reduced_grid", "raw_initializer_selected")
SUCCESS_STATUSES = ("completed", "converged")


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _seconds(value):
    return _finite_number(value) and value >= 0


def _specs(case):
    rank, source = case["target_rank"], case["source_rank"]
    free = max(rank, min(source, 10))
    return [dict(rank=rank, source_rank=source, free_directions=[free, free],
                 support_limits=[(case["p"] - free) * rank, (case["q"] - free) * rank],
                 init_penalty=initial, penalty=[penalty, penalty])
            for initial, penalty in product((.1, .3), (.01, .04, .16))]


def _planned(plan, issue):
    cases, seeds, budgets = plan["cases"], plan["seeds"], plan["budgets"]
    ids = [case["case_id"] for case in cases]
    if not ids or len(set(ids)) != len(ids) or any(name not in CASE_IDS for name in ids):
        issue("undeclared_case_grid", "Cases differ from the four declared representative cases")
    for values, allowed, name in ((seeds, SEEDS, "seeds"), (budgets, BUDGETS, "budgets")):
        if not values or len(set(values)) != len(values) or any(value not in allowed or isinstance(value, bool) for value in values):
            issue("undeclared_" + name, str(values))
    expected = [dict(case_index=index, seed=seed, budget=budget, expected_candidates=6)
                for index in range(len(cases)) for seed in seeds for budget in budgets]
    if plan.get("tasks") != expected:
        issue("planned_task_inventory_mismatch", "Task grid must include every declared case/seed/budget exactly once in order")
    if plan.get("n_tasks") != len(expected) or plan.get("expected_fits") != 6 * len(expected) or plan.get("n_cases") != len(cases):
        issue("plan_denominator_mismatch", "Plan task/candidate counts differ from its complete Cartesian grid")
    for case in cases:
        if plan["candidate_libraries"].get(case["case_id"]) != _specs(case):
            issue("candidate_library_mismatch", case["case_id"])
    for budget in budgets:
        config = plan["configurations"].get(str(budget), {})
        controls = dict(iterations=budget, validation_interval=50,
            validation_patience=None, validation_min_iterations=0,
            validation_min_relative_improvement=0., stationarity_tol=1e-6,
            rrr_shortcut=False, initialization_cache=False, partial_fit_selection=False,
            refit_on_validation=False, init_penalties=[.1, .3],
            factor_penalty_pairs=[[.01, .01], [.04, .04], [.16, .16]])
        if any(key not in config or config[key] != value for key, value in controls.items()):
            issue("budget_configuration_mismatch", str(budget))
    return expected


def _prediction(coefficient, data):
    with np.errstate(over="raise", invalid="raise"):
        prediction = data["X_validation"] @ coefficient
        loss = float(np.mean((data["Y_validation"] - prediction)**2))
    if not np.isfinite(loss):
        raise FloatingPointError("Nonfinite validation loss")
    return prediction, loss


def _difference(prediction, incumbent, response):
    # Independent implementation, with extended accumulation and an incumbent
    # reference to preserve the same earliest strict-improvement convention.
    first, second, response = (np.asarray(value, dtype=np.longdouble)
                               for value in (prediction, incumbent, response))
    value = float(np.mean((first - second) * ((first - response) + (second - response)),
                          dtype=np.longdouble))
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite paired validation difference")
    return value


def _best(rows, arrays, data, *, initializer=False):
    key = "raw_initializer" if initializer else "selected_coefficient"
    eligibility = "raw_initializer_eligible" if initializer else "eligible"
    winner, incumbent, seen, trace = None, None, set(), []
    for index, (row, saved) in enumerate(zip(rows, arrays, strict=True)):
        decision = dict(candidate_index=index, eligible=False)
        initial = row["spec"]["init_penalty"]
        if initializer and initial in seen:
            decision["exclusion_reason"] = "duplicate_initializer_penalty"
        elif not row.get(eligibility):
            decision["exclusion_reason"] = "raw_initializer_ineligible" if initializer else "full_budget_fit_ineligible"
        elif key not in saved:
            decision["exclusion_reason"] = "missing_or_nonfinite_coefficient"
        else:
            try:
                prediction, loss = _prediction(saved[key], data)
                difference = None if incumbent is None else _difference(prediction, incumbent, data["Y_validation"])
            except (ValueError, FloatingPointError):
                decision["exclusion_reason"] = "nonfinite_validation_arithmetic"
            else:
                decision.update(eligible=True, validation_mse=loss)
                seen.add(initial)
                if winner is None or difference < 0:
                    winner, incumbent = index, prediction
        trace.append(decision)
    return winner, trace


def objective_diagnostics(candidate):
    """Compare objectives only inside the same chart epoch; no convergence claim."""
    history = candidate.get("history", [])
    epochs = candidate.get("history_chart_epochs", [])
    if len(history) != len(epochs):
        raise ValueError("History and chart-epoch labels have different lengths")
    groups, transitions = {}, []
    for index, (record, epoch) in enumerate(zip(history, epochs, strict=True)):
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("Invalid chart epoch")
        if not _finite_number(record.get("objective")):
            raise ValueError("Accepted history objective must be finite")
        group = groups.setdefault(epoch, dict(chart_epoch=epoch, records=[], differences=[], changes=[]))
        group["records"].append(record)
        if _finite_number(record.get("objective_change")):
            group["changes"].append(record["objective_change"])
        if index:
            previous = history[index - 1]
            difference = record["objective"] - previous["objective"]
            if epochs[index - 1] == epoch:
                group["differences"].append(difference)
            else:
                transitions.append(dict(from_epoch=epochs[index - 1], to_epoch=epoch,
                    iteration=record.get("iteration"), objective_jump=difference,
                    included_in_descent_check=False))
    output = []
    for epoch, group in groups.items():
        records, differences, changes = group["records"], group["differences"], group["changes"]
        scale = max(1., *(abs(row["objective"]) for row in records))
        tolerance = 2e-10 + 2e-8 * scale
        output.append(dict(chart_epoch=epoch, n_records=len(records),
            first_iteration=records[0].get("iteration"), last_iteration=records[-1].get("iteration"),
            first_objective=records[0]["objective"], last_objective=records[-1]["objective"],
            within_epoch_comparisons=len(differences),
            positive_objective_difference_count=sum(value > 0 for value in differences),
            objective_increase_above_tolerance_count=sum(value > tolerance for value in differences),
            maximum_within_epoch_objective_difference=max(differences) if differences else None,
            positive_reported_change_count=sum(value > 0 for value in changes),
            maximum_reported_change=max(changes) if changes else None,
            comparison_tolerance=tolerance))
    return dict(epochs=output, chart_transitions=transitions,
        cross_chart_objectives_compared_for_descent=False,
        interpretation="Within-chart objective evidence and hard-prox mapping diagnostics do not certify a global optimum or chart KKT conditions")


def _metric_check(coefficient, recorded, data, truth, issue, description):
    recomputed = common._recompute(coefficient, data, truth)
    for name, value in recomputed.items():
        if not common._close(value, recorded.get(name)):
            issue("metric_mismatch", description + ": " + name)
    return recomputed


def _candidate(row, saved, case, config, data, truth, issue):
    index = row["candidate_index"]
    for key, coefficient in saved.items():
        if coefficient.shape != (case["p"], case["q"]) or not np.isfinite(coefficient).all() or np.iscomplexobj(coefficient):
            raise ValueError(f"Candidate {index} has invalid coefficient {key}")
    if row.get("coefficient_keys") != sorted(saved):
        issue("candidate_array_inventory_mismatch", str(index))
    if row.get("fresh_fit") is not True or row.get("initialization_cache_used") is not False:
        issue("candidate_not_fresh_fit", str(index))
    if not _seconds(row.get("fitting_call_seconds")):
        issue("invalid_candidate_runtime", str(index))
    eligible = row.get("eligible")
    if not isinstance(eligible, bool) or not isinstance(row.get("raw_initializer_eligible"), bool):
        issue("missing_candidate_outcome", str(index))
    reconstructed_eligible = (row.get("status") in SUCCESS_STATUSES and row.get("exception") is None
                              and {"selected_coefficient", "terminal_coefficient"}.issubset(saved))
    if eligible is not reconstructed_eligible:
        issue("full_fit_eligibility_mismatch", str(index))
    if row.get("retained_failed_arrays_are_diagnostic_only") is not (not eligible):
        issue("failed_output_label_mismatch", str(index))
    if row.get("raw_initializer_eligible") is not (row.get("raw_initializer_converged") is True and "raw_initializer" in saved):
        issue("raw_initializer_eligibility_mismatch", str(index))
    if not isinstance(row.get("n_iter"), int) or isinstance(row.get("n_iter"), bool) or not 0 <= row["n_iter"] <= config["iterations"]:
        issue("candidate_iterations_outside_budget", str(index))
    if set(row.get("diagnostic_metrics", {})) != set(saved):
        issue("diagnostic_metric_inventory_mismatch", str(index))
    reduced_truth = {name: value for name, value in truth.items() if name not in ("X_test", "Y_test")}
    metrics = {key: _metric_check(value, row.get("diagnostic_metrics", {}).get(key, {}),
                                 data, reduced_truth, issue, f"candidate {index}/{key}")
               for key, value in saved.items()}
    history = row.get("history", [])
    changes = [record["objective_change"] for record in history if _finite_number(record.get("objective_change"))]
    if row.get("accepted_update_backtracks") != sum(record.get("backtracks", 0) for record in history):
        issue("backtracking_count_mismatch", str(index))
    if row.get("positive_reported_objective_change_count") != sum(value > 0 for value in changes):
        issue("objective_change_count_mismatch", str(index))
    maximum = max(changes) if changes else None
    if maximum != row.get("max_reported_objective_change"):
        issue("objective_change_maximum_mismatch", str(index))
    objective = objective_diagnostics(row)
    stationary = eligible and row.get("termination_reason") == "stationarity"
    if row.get("optimization_converged") is not stationary:
        issue("terminal_convergence_flag_mismatch", str(index))
    if row.get("selected_output_converged") is not (stationary and row.get("selected_iteration") == row.get("n_iter")):
        issue("selected_convergence_flag_mismatch", str(index))
    endpoint = row.get("terminal_record") or {}
    if stationary and (not _finite_number(endpoint.get("projected_gradient_norm"))
                       or endpoint["projected_gradient_norm"] > config["stationarity_tol"]
                       or endpoint.get("mapping_domain_reason") is not None):
        issue("unsubstantiated_fixed_point_stop", str(index))
    checkpoints = row.get("checkpoints", [])
    checkpoint_indices = [checkpoint.get("iteration") for checkpoint in checkpoints]
    if checkpoint_indices != sorted(set(checkpoint_indices)):
        issue("checkpoint_iteration_inventory", str(index))
    expected_checkpoint_keys = set()
    by_iteration = {}
    for checkpoint in checkpoints:
        iteration = checkpoint["iteration"]
        if not isinstance(iteration, int) or not 0 <= iteration <= config["iterations"]:
            issue("checkpoint_outside_budget", str(index))
        if checkpoint.get("extraction_error"):
            if checkpoint.get("eligible_for_this_budget"):
                issue("failed_checkpoint_extraction_eligible", str(index))
            continue
        a, b = f"checkpoint_{iteration:06d}", f"prefix_selected_{iteration:06d}"
        expected_checkpoint_keys.update((a, b))
        if checkpoint.get("coefficient_key") != a or checkpoint.get("selected_coefficient_key") != b or a not in saved or b not in saved:
            issue("checkpoint_array_inventory", f"{index}/{iteration}")
            continue
        if checkpoint.get("diagnostic_only") is not (not eligible) or checkpoint.get("eligible_for_this_budget") is not eligible:
            issue("checkpoint_failure_scope_mismatch", f"{index}/{iteration}")
        by_iteration[iteration] = (checkpoint, saved[a], saved[b])
    if {key for key in saved if key.startswith(("checkpoint_", "prefix_selected_"))} != expected_checkpoint_keys:
        issue("unexpected_checkpoint_coefficient", str(index))
    # Replay only the saved target-validation observations. Checkpoint arrays
    # are measured artifacts; no reconstruction invokes a fitted model or truth.
    winner_iteration, incumbent = None, None
    validation_rows = row.get("validation_history", [])
    for observation in validation_rows:
        iteration = observation["iteration"]
        if iteration not in by_iteration:
            issue("validation_checkpoint_missing", f"{index}/{iteration}")
            continue
        _, coefficient, _ = by_iteration[iteration]
        prediction, loss = _prediction(coefficient, data)
        if not common._close(loss, observation.get("loss")):
            issue("checkpoint_validation_mismatch", f"{index}/{iteration}")
        difference = None if incumbent is None else _difference(prediction, incumbent, data["Y_validation"])
        if incumbent is None or difference < 0:
            winner_iteration, incumbent = iteration, prediction
        checkpoint, _, prefix = by_iteration[iteration]
        prefix_prediction, _ = _prediction(prefix, data)
        if checkpoint.get("selected_iteration") != winner_iteration or not np.allclose(prefix_prediction, incumbent, rtol=2e-10, atol=2e-10):
            issue("nonminimal_checkpoint_prefix", f"{index}/{iteration}")
    if eligible:
        terminal_iteration = row.get("n_iter")
        if terminal_iteration not in by_iteration:
            issue("terminal_checkpoint_missing", str(index))
        else:
            checkpoint, terminal, selected = by_iteration[terminal_iteration]
            if not np.array_equal(saved["terminal_coefficient"], terminal) or not np.array_equal(saved["selected_coefficient"], selected):
                issue("terminal_or_selected_checkpoint_mismatch", str(index))
            if row.get("selected_iteration") != checkpoint.get("selected_iteration"):
                issue("selected_iteration_mismatch", str(index))
        if row.get("selected_iteration") != winner_iteration:
            issue("nonminimal_within_fit_validation_selection", str(index))
    return dict(candidate_index=index, status=row.get("status"), eligible=bool(eligible),
                raw_initializer_eligible=bool(row.get("raw_initializer_eligible")),
                n_iter=row.get("n_iter"), selected_iteration=row.get("selected_iteration"),
                terminal_record=endpoint, objective_diagnostics=objective, metrics=metrics,
                fitting_call_seconds=row.get("fitting_call_seconds"))


def _selection(method, reported, candidates, arrays, selected, data, truth, issue):
    initial = method == "raw_initializer_selected"
    winner, trace = _best(candidates, arrays, data, initializer=initial)
    key = "raw_initializer" if initial else "selected_coefficient"
    if reported.get("success") is not (winner is not None) or reported.get("selected_index") != winner:
        issue("nonminimal_between_candidate_selection", method)
    if reported.get("coefficient_key") != key or reported.get("selection_uses_truth") is not False:
        issue("selection_policy_mismatch", method)
    recorded_trace = reported.get("audit", [])
    if len(recorded_trace) != len(trace):
        issue("selection_trace_count_mismatch", method)
    for expected, recorded in zip(trace, recorded_trace):
        for field in ("candidate_index", "eligible", "exclusion_reason"):
            if expected.get(field) != recorded.get(field):
                issue("selection_eligibility_trace_mismatch", f"{method}/{expected['candidate_index']}")
        if expected["eligible"] and not common._close(expected["validation_mse"], recorded.get("validation_mse")):
            issue("selection_trace_loss_mismatch", f"{method}/{expected['candidate_index']}")
    output = dict(success=winner is not None, selected_index=winner)
    if winner is not None:
        coefficient = arrays[winner][key]
        if method not in selected or not np.array_equal(selected[method], coefficient):
            issue("selected_archive_mismatch", method)
        if reported.get("parameters") != candidates[winner]["spec"]:
            issue("selected_parameters_mismatch", method)
        if not common._close(reported.get("validation_mse"), trace[winner]["validation_mse"]):
            issue("selected_validation_loss_mismatch", method)
        metrics = _metric_check(coefficient, reported.get("metrics", {}), data, truth, issue, method)
        if not common._close(reported.get("selected_candidate_full_fit_seconds"), candidates[winner].get("fitting_call_seconds")):
            issue("selected_candidate_runtime_mismatch", method)
        output.update(metrics=metrics, selected_iteration=0 if initial else candidates[winner].get("selected_iteration"),
                      selected_candidate_full_fit_seconds=candidates[winner].get("fitting_call_seconds"))
        if reported.get("selected_iteration") != output["selected_iteration"]:
            issue("reported_selected_iteration_mismatch", method)
        if not initial:
            output["terminal_metrics"] = _metric_check(arrays[winner]["terminal_coefficient"],
                reported.get("terminal_metrics", {}), data, truth, issue, method + "/terminal")
            output["optimization_converged"] = candidates[winner].get("optimization_converged")
            output["selected_output_converged"] = candidates[winner].get("selected_output_converged")
    return output


def _aggregate(rows):
    aggregate, paired = [], []
    for case_id, budget, method in sorted({(row["case_id"], row["budget"], row["method"]) for row in rows}):
        group = [row for row in rows if (row["case_id"], row["budget"], row["method"]) == (case_id, budget, method)]
        successful = [row for row in group if row["status"] == "success"]
        aggregate.append(dict(case_id=case_id, budget=budget, method=method,
            n_planned=len(group), n_success=len(successful),
            status_counts=dict(Counter(row["status"] for row in group)),
            risk=common._moments([row["metrics"]["population_prediction_excess"] for row in successful]),
            fitting_call_seconds=common._moments([row["fitting_call_seconds"] for row in group if row["status"] in ("success", "failed")]),
            selected_iterations=[row["selected_iteration"] for row in successful],
            terminal_fixed_point_stops=sum(bool(row.get("optimization_converged")) for row in successful),
            selected_fixed_point_stops=sum(bool(row.get("selected_output_converged")) for row in successful)))
    for case_id, method in sorted({(row["case_id"], row["method"]) for row in rows}):
        case_rows = [row for row in rows if (row["case_id"], row["method"]) == (case_id, method)]
        budgets = sorted({row["budget"] for row in case_rows})
        for lower, upper in zip(budgets[:-1], budgets[1:]):
            lows = {row["seed"]: row for row in case_rows if row["budget"] == lower}
            highs = {row["seed"]: row for row in case_rows if row["budget"] == upper}
            seeds = sorted(set(lows) | set(highs))
            common_success, mismatches, transitions = [], [], []
            for seed in seeds:
                low, high = lows.get(seed, {}), highs.get(seed, {})
                transitions.append(dict(seed=seed, lower_status=low.get("status", "unplanned"), upper_status=high.get("status", "unplanned")))
                if low.get("status") == high.get("status") == "success":
                    if low.get("fit_data_fingerprint") != high.get("fit_data_fingerprint") or low.get("truth_fingerprint") != high.get("truth_fingerprint"):
                        mismatches.append(seed)
                    else:
                        common_success.append(seed)
            paired.append(dict(case_id=case_id, method=method, lower_budget=lower, upper_budget=upper,
                n_planned=len(seeds), common_success_seeds=common_success, fingerprint_mismatch_seeds=mismatches,
                status_transitions=transitions,
                higher_minus_lower_risk=common._moments([highs[seed]["metrics"]["population_prediction_excess"] - lows[seed]["metrics"]["population_prediction_excess"] for seed in common_success]),
                earlier_success_later_failure_kept=[seed for seed in seeds if lows.get(seed, {}).get("status") == "success" and highs.get(seed, {}).get("status") == "failed"],
                earlier_success_later_unavailable_kept=[seed for seed in seeds if lows.get(seed, {}).get("status") == "success" and highs.get(seed, {}).get("status") not in ("success", "failed")]))
    return aggregate, paired


def audit(root, *, allow_local=False, write=True):
    root = Path(root).resolve()
    if not allow_local and (not os.environ.get("SLURM_JOB_ID") or not root.is_relative_to(Path("/scratch2"))):
        raise RuntimeError("Production budget auditing requires Slurm and a /scratch2 root")
    plan_path = root / "budget-plan.json"
    plan = common._read_json(plan_path)
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    issues, reports, rows = [], [], []
    def issue(code, message, task_index=None):
        issues.append(dict(code=code, message=str(message), task_index=task_index))
    planned = _planned(plan, issue)
    plan_invalid = bool(issues)
    for index, task in enumerate(planned):
        case = plan["cases"][task["case_index"]]
        config = plan["configurations"][str(task["budget"])]
        base = dict(task_index=index, case_id=case["case_id"], seed=task["seed"], budget=task["budget"])
        report = dict(base, status="missing_task", candidates=[])
        reports.append(report)
        directory = root / "budget_tasks" / f"task-{index:06d}"
        path = directory / "result.json"
        if not path.exists():
            issue("missing_task", path, index)
            rows.extend(dict(base, method=method, status="missing_task") for method in METHODS)
            continue
        start = len(issues)
        def task_issue(code, message):
            issue(code, message, index)
        try:
            result = common._read_json(path)
            if result.get("complete") is not True or result.get("plan_sha256") != digest or result.get("task") != task or result.get("case") != case or result.get("configuration") != config:
                task_issue("task_identity_mismatch", "Receipt differs from its frozen plan")
            if result.get("code_sha256") != plan.get("code_sha256"):
                task_issue("code_snapshot_receipt_mismatch", "Saved source hashes differ from plan")
            expected_files = {"data.npz", "truth.npz", "selected-coefficients.npz"} | {f"candidate-{j:04d}.npz" for j in range(6)}
            evidence = result.get("evidence_sha256", {})
            if set(evidence) != expected_files or {file.name for file in directory.glob("*.npz")} != expected_files:
                task_issue("evidence_inventory_mismatch", "Expected data, truth, selected outputs and six complete candidate archives")
            for name, expected_hash in evidence.items():
                if Path(name).name != name or name not in expected_files:
                    task_issue("unexpected_evidence_path", name)
                    continue
                if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected_hash:
                    task_issue("evidence_hash_mismatch", name)
            data, truth = common._arrays(directory / "data.npz"), common._arrays(directory / "truth.npz")
            if {"C_star", "Sigma_x", "X_test", "Y_test"} & set(data):
                task_issue("truth_in_fitting_archive", "Truth/test objects appear in the fitting-data archive")
            for names, archive in ((("X", "Y", "X_validation", "Y_validation", "C0"), data), (("C_star", "Sigma_x"), truth)):
                if any(name not in archive or not np.isfinite(archive[name]).all() for name in names):
                    raise ValueError("Required data/truth arrays are missing or nonfinite")
            dimensions = {"X": (case["n_train"], case["p"]), "Y": (case["n_train"], case["q"]),
                          "X_validation": (case["n_validation"], case["p"]),
                          "Y_validation": (case["n_validation"], case["q"]), "C0": (case["p"], case["q"])}
            if any(data[name].shape != shape for name, shape in dimensions.items()):
                task_issue("fitting_data_dimensions_mismatch", "Fitting arrays differ from planned dimensions")
            if truth["C_star"].shape != (case["p"], case["q"]) or truth["Sigma_x"].shape != (case["p"], case["p"]):
                task_issue("truth_dimensions_mismatch", "Evaluation arrays differ from planned dimensions")
            data_hash, truth_hash = common._fingerprint(data), common._fingerprint(truth)
            if data_hash != result.get("fit_data_fingerprint") or truth_hash != result.get("truth_fingerprint"):
                task_issue("data_fingerprint_mismatch", "Saved arrays differ from receipt fingerprints")
            report.update(fit_data_fingerprint=data_hash, truth_fingerprint=truth_hash)
            candidates = result.get("candidates", [])
            if [row.get("candidate_index") for row in candidates] != list(range(6)) or [row.get("spec") for row in candidates] != _specs(case):
                raise ValueError("Candidate count/order/specifications differ from the six declared fits")
            arrays = []
            for row in candidates:
                filename = f"candidate-{row['candidate_index']:04d}.npz"
                if row.get("coefficient_archive") != filename:
                    task_issue("candidate_archive_path_mismatch", row["candidate_index"])
                saved = common._arrays(directory / filename)
                arrays.append(saved)
                report["candidates"].append(_candidate(row, saved, case, config, data, truth, task_issue))
            for indices in ((0, 1, 2), (3, 4, 5)):
                available = [j for j in indices if "raw_initializer" in arrays[j]]
                if available and any(not np.allclose(arrays[j]["raw_initializer"], arrays[available[0]]["raw_initializer"],
                                                     rtol=2e-10, atol=2e-10) for j in available[1:]):
                    task_issue("repeated_initializer_mismatch", "Identical initialization penalty and observations produced inconsistent saved coefficients")
            if result.get("expected_candidates") != 6 or result.get("n_eligible") != sum(row["eligible"] for row in candidates) or result.get("n_raw_initializer_eligible") != sum(row["raw_initializer_eligible"] for row in candidates):
                task_issue("candidate_denominator_mismatch", "All six candidate outcomes must be accounted for")
            for key, subset in (("candidate_status_counts", candidates), ("failed_candidate_status_counts", [row for row in candidates if not row["eligible"]])):
                if result.get(key) != dict(Counter(row["status"] for row in subset)):
                    task_issue("candidate_status_count_mismatch", key)
            total_seconds = sum(row["fitting_call_seconds"] for row in candidates)
            if not common._close(total_seconds, result.get("fitting_call_seconds")) or not _seconds(result.get("wall_seconds")) or result["wall_seconds"] + 1e-8 < total_seconds:
                task_issue("task_runtime_mismatch", "Full fresh-fit costs do not reconcile")
            if result.get("later_budget_failures_censor_this_result") is not False or result.get("partial_fit_selection") is not False:
                task_issue("budget_failure_policy_mismatch", "Later failures and partial fits must not alter eligibility")
            selected = common._arrays(directory / "selected-coefficients.npz")
            methods = result.get("methods", {})
            if set(methods) != set(METHODS) or set(selected) != {method for method, value in methods.items() if value.get("success") is True}:
                task_issue("method_inventory_mismatch", "Every planned method must have an explicit outcome")
            outputs = {method: _selection(method, methods.get(method, {}), candidates, arrays, selected, data, truth, task_issue) for method in METHODS}
            invalid = plan_invalid or len(issues) > start
            report.update(status="audit_invalid" if invalid else "complete", fitting_call_seconds=total_seconds,
                          candidate_status_counts=result["candidate_status_counts"],
                          failed_candidate_status_counts=result["failed_candidate_status_counts"])
            for method, output in outputs.items():
                rows.append(dict(base, method=method, **output,
                    status="audit_invalid" if invalid else ("success" if output["success"] else "failed"),
                    fitting_call_seconds=total_seconds, fit_data_fingerprint=data_hash, truth_fingerprint=truth_hash))
        except Exception as error:
            task_issue("unreadable_task_evidence", f"{type(error).__name__}: {error}")
            report["status"] = "audit_invalid"
            rows.extend(dict(base, method=method, status="audit_invalid") for method in METHODS)
    aggregate, paired = _aggregate(rows)
    for pair in paired:
        if pair["fingerprint_mismatch_seeds"]:
            issue("cross_budget_data_mismatch", f"{pair['case_id']}/{pair['method']}/{pair['lower_budget']}/{pair['upper_budget']}: {pair['fingerprint_mismatch_seeds']}")
    full_design = set(case["case_id"] for case in plan["cases"]) == set(CASE_IDS) and set(plan["seeds"]) == set(SEEDS) and set(plan["budgets"]) == set(BUDGETS)
    audit_result = dict(schema=1, audit_passed=not issues, plan_sha256=digest,
        all_planned_tasks_complete=all(report["status"] == "complete" for report in reports),
        full_predeclared_study=full_design, full_predeclared_task_count=80,
        planned_task_count=len(planned), planned_fit_count=6 * len(planned),
        planned_method_outcomes=2 * len(planned), issues=issues, tasks=reports,
        scope="Saved-data separation, hashes, six-fit inventory, independent validation selection and coefficient/prediction loss reconstruction. Objective diagnostics compare only within chart epochs. No refits and no claim of global convergence or proof that external code never accessed truth.")
    summary = dict(schema=1, plan_sha256=digest, full_predeclared_study=full_design,
        n_planned_tasks=len(planned), n_planned_method_outcomes=len(rows),
        task_status_counts=dict(Counter(report["status"] for report in reports)),
        method_status_counts=dict(Counter(row["status"] for row in rows)),
        aggregate=aggregate, paired_budget_comparisons=paired,
        runtime_scope="Total six-fit wall-call costs repeat across method labels; raw initialization is a side product, with no separate runtime claim. Source fitting and evidence I/O are separately timed in task receipts.",
        inference_scope="Five paired seeds per full-design cell; descriptive Monte Carlo uncertainty conditional on valid successful fits, with every failure/missing denominator retained")
    if write:
        common._atomic_json(root / "budget-audit.json", audit_result)
        common._atomic_json(root / "budget-audited-summary.json", summary)
        common._atomic_json(root / "budget-audited-per-replication.json", rows)
    return dict(audit=audit_result, summary=summary, per_replication=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    output = audit(args.root, write=not args.no_write)
    print(json.dumps({key: output["audit"][key] for key in ("audit_passed", "all_planned_tasks_complete", "planned_task_count", "full_predeclared_study")}, sort_keys=True))
    raise SystemExit(0 if output["audit"]["audit_passed"] else 1)


if __name__ == "__main__":
    main()
