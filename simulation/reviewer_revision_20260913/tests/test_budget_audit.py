"""Tiny deterministic saved-artifact fixtures; no generation or solver fits."""
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "simulation"))
from reviewer_revision_20260913 import budget_audit as audit


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def _plan(root, *, budgets=(100,), full=False):
    cases = [dict(case_id=name, p=2, q=2, target_rank=1, source_rank=1,
                  n_train=2, n_validation=2, n_test=2)
             for name in (audit.CASE_IDS if full else audit.CASE_IDS[:1])]
    seeds = list(audit.SEEDS if full else (2000,))
    budgets = list(audit.BUDGETS if full else budgets)
    tasks = [dict(case_index=i, seed=seed, budget=budget, expected_candidates=6)
             for i in range(len(cases)) for seed in seeds for budget in budgets]
    configs = {str(budget): dict(iterations=budget, validation_interval=50,
        validation_patience=None, validation_min_iterations=0,
        validation_min_relative_improvement=0., stationarity_tol=1e-6,
        rrr_shortcut=False, initialization_cache=False, partial_fit_selection=False,
        refit_on_validation=False, init_penalties=[.1, .3],
        factor_penalty_pairs=[[.01, .01], [.04, .04], [.16, .16]]) for budget in budgets}
    specs = [dict(rank=1, source_rank=1, free_directions=[1, 1], support_limits=[1, 1],
                  init_penalty=initial, penalty=[penalty, penalty])
             for initial, penalty in product((.1, .3), (.01, .04, .16))]
    plan = dict(cases=cases, seeds=seeds, budgets=budgets, tasks=tasks, n_cases=len(cases),
        n_tasks=len(tasks), expected_fits=6*len(tasks), configurations=configs,
        candidate_libraries={case["case_id"]: deepcopy(specs) for case in cases}, code_sha256={})
    _write_json(root / "budget-plan.json", plan)
    return plan


def _metrics(C, data, truth):
    delta = C - truth["C_star"]
    result = dict(coefficient_rmse=float(np.sqrt(np.mean(delta**2))),
        coefficient_squared_error=float(np.sum(delta**2)),
        population_prediction_excess=float(np.sum(delta * (truth["Sigma_x"] @ delta)) / 2),
        training_mse=float(np.mean((data["Y"] - data["X"] @ C)**2)),
        validation_mse=float(np.mean((data["Y_validation"] - data["X_validation"] @ C)**2)),
        numerical_rank=int(np.linalg.matrix_rank(C)))
    if "X_test" in truth:
        result["test_mse"] = float(np.mean((truth["Y_test"] - truth["X_test"] @ C)**2))
    return result


def _write_task(root, plan, index=0, *, failed=False):
    task = plan["tasks"][index]
    case = plan["cases"][task["case_index"]]
    directory = root / "budget_tasks" / f"task-{index:06d}"
    directory.mkdir(parents=True)
    C = lambda value: np.diag([value, 0.])
    data = dict(X=np.eye(2), Y=C(1), X_validation=np.eye(2), Y_validation=C(1), C0=C(.5))
    truth = dict(C_star=C(2), Sigma_x=np.eye(2), X_test=np.eye(2), Y_test=C(2))
    reduced = {key: value for key, value in truth.items() if key not in ("X_test", "Y_test")}
    np.savez(directory / "data.npz", **data)
    np.savez(directory / "truth.npz", **truth)
    candidates, archives = [], []
    for j, spec in enumerate(plan["candidate_libraries"][case["case_id"]]):
        initial = C(.2 if j < 3 else .4)
        selected = C((.8, 1., 1.2, .6, .5, .4)[j])
        budget = task["budget"]
        saved = dict(raw_initializer=initial)
        checkpoints, history, validation = [], [], []
        if not failed:
            saved.update(selected_coefficient=selected, terminal_coefficient=selected,
                         checkpoint_000000=initial, prefix_selected_000000=initial)
            saved[f"checkpoint_{budget:06d}"] = selected
            saved[f"prefix_selected_{budget:06d}"] = selected
            for iteration, matrix in ((0, initial), (budget, selected)):
                record = dict(iteration=iteration, objective=2. if iteration == 0 else 1.,
                              objective_change=None if iteration == 0 else -1., backtracks=0,
                              projected_gradient_norm=1., mapping_domain_reason=None)
                history.append(record)
                # Candidate 5 is unchanged, so the strict earliest-tie rule
                # selects initialization and its equivalent coefficient.
                winner = 0 if j == 5 else iteration
                checkpoints.append(dict(iteration=iteration, selected_iteration=winner,
                    coefficient_key=f"checkpoint_{iteration:06d}",
                    selected_coefficient_key=f"prefix_selected_{iteration:06d}",
                    eligible_for_this_budget=True, diagnostic_only=False,
                    chart_epoch=0, selected_chart_epoch=0, endpoint_record=record))
                validation.append(dict(iteration=iteration, loss=_metrics(matrix, data, reduced)["validation_mse"]))
        row = dict(candidate_index=j, spec=spec, coefficient_archive=f"candidate-{j:04d}.npz",
            coefficient_keys=sorted(saved), diagnostic_metrics={key: _metrics(value, data, reduced) for key, value in saved.items()},
            eligible=not failed, raw_initializer_eligible=True, raw_initializer_converged=True,
            retained_failed_arrays_are_diagnostic_only=failed, status="initialization_spectrum_failed" if failed else "completed",
            exception=None, fresh_fit=True, initialization_cache_used=False, fitting_call_seconds=.01,
            history=history, history_chart_epochs=[0]*len(history), accepted_update_backtracks=0,
            positive_reported_objective_change_count=0, max_reported_objective_change=None if failed else -1.,
            optimization_converged=False, selected_output_converged=False,
            termination_reason="initialization_spectrum_failed" if failed else "max_iterations",
            terminal_record=None if failed else history[-1], n_iter=0 if failed else budget,
            selected_iteration=None if failed else (0 if j == 5 else budget),
            checkpoints=checkpoints, validation_history=validation)
        np.savez(directory / row["coefficient_archive"], **saved)
        candidates.append(row)
        archives.append(saved)
    methods, selected_arrays = {}, {}
    for method, initial in (("v2_reduced_grid", False), ("raw_initializer_selected", True)):
        key = "raw_initializer" if initial else "selected_coefficient"
        winner = 3 if initial else (None if failed else 1)
        trace = []
        for j, row in enumerate(candidates):
            eligible = j in (0, 3) if initial else not failed
            entry = dict(candidate_index=j, eligible=eligible)
            if eligible:
                entry["validation_mse"] = _metrics(archives[j][key], data, reduced)["validation_mse"]
            else:
                entry["exclusion_reason"] = "duplicate_initializer_penalty" if initial else "full_budget_fit_ineligible"
            trace.append(entry)
        output = dict(success=winner is not None, selected_index=winner, coefficient_key=key,
                      selection_uses_truth=False, audit=trace, validation_mse=None)
        if winner is not None:
            selected_arrays[method] = archives[winner][key]
            output.update(metrics=_metrics(selected_arrays[method], data, truth),
                parameters=candidates[winner]["spec"], selected_candidate_full_fit_seconds=.01,
                validation_mse=trace[winner]["validation_mse"], selected_iteration=0 if initial else task["budget"])
            if not initial:
                output.update(terminal_metrics=output["metrics"], optimization_converged=False,
                              selected_output_converged=False)
        methods[method] = output
    np.savez(directory / "selected-coefficients.npz", **selected_arrays)
    result = dict(complete=True, task=task, case=case, configuration=plan["configurations"][str(task["budget"])] ,
        plan_sha256=hashlib.sha256((root / "budget-plan.json").read_bytes()).hexdigest(), code_sha256={},
        fit_data_fingerprint=audit.common._fingerprint(data), truth_fingerprint=audit.common._fingerprint(truth),
        candidates=candidates, methods=methods, expected_candidates=6, n_eligible=0 if failed else 6,
        n_raw_initializer_eligible=6, candidate_status_counts={"initialization_spectrum_failed" if failed else "completed": 6},
        failed_candidate_status_counts={"initialization_spectrum_failed": 6} if failed else {},
        fitting_call_seconds=.06, wall_seconds=.1, partial_fit_selection=False,
        later_budget_failures_censor_this_result=False,
        evidence_sha256={file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in directory.glob("*.npz")})
    _write_json(directory / "result.json", result)
    return directory, result


def test_audit_reconstructs_all_coefficients_and_selects_validation_not_truth(tmp_path):
    plan = _plan(tmp_path)
    _, _ = _write_task(tmp_path, plan)
    result = audit.audit(tmp_path, allow_local=True)
    assert result["audit"]["audit_passed"], result["audit"]["issues"]
    refined = next(row for row in result["per_replication"] if row["method"] == "v2_reduced_grid")
    assert refined["selected_index"] == 1 and refined["metrics"]["validation_mse"] == 0
    assert len(result["audit"]["tasks"][0]["candidates"][0]["metrics"]) == 7
    assert (tmp_path / "budget-audited-summary.json").exists()


def test_missing_tasks_keep_all_eighty_planned_denominators(tmp_path):
    _plan(tmp_path, full=True)
    result = audit.audit(tmp_path, allow_local=True, write=False)
    assert result["audit"]["planned_task_count"] == 80
    assert result["audit"]["planned_fit_count"] == 480
    assert result["summary"]["n_planned_method_outcomes"] == 160
    assert result["summary"]["method_status_counts"] == {"missing_task": 160}
    assert all(row["n_planned"] == 5 for row in result["summary"]["aggregate"])


def test_later_failure_preserves_earlier_success_and_raw_initializer(tmp_path):
    plan = _plan(tmp_path, budgets=(100, 500))
    _write_task(tmp_path, plan, 0)
    _write_task(tmp_path, plan, 1, failed=True)
    result = audit.audit(tmp_path, allow_local=True, write=False)
    assert result["audit"]["audit_passed"], result["audit"]["issues"]
    refined = [row for row in result["per_replication"] if row["method"] == "v2_reduced_grid"]
    assert [row["status"] for row in refined] == ["success", "failed"]
    pair = next(row for row in result["summary"]["paired_budget_comparisons"] if row["method"] == "v2_reduced_grid")
    assert pair["earlier_success_later_failure_kept"] == [2000]
    assert pair["higher_minus_lower_risk"]["n"] == 0
    assert all(row["status"] == "success" for row in result["per_replication"] if row["method"] == "raw_initializer_selected")


@pytest.mark.parametrize("mutation,code", [
    ("winner", "nonminimal_between_candidate_selection"),
    ("loss", "metric_mismatch"),
    ("candidate", "unreadable_task_evidence"),
    ("hash", "evidence_hash_mismatch"),
    ("prefix", "nonminimal_checkpoint_prefix"),
    ("hidden_eligible", "full_fit_eligibility_mismatch"),
])
def test_corrupt_evidence_is_rejected(tmp_path, mutation, code):
    plan = _plan(tmp_path)
    directory, result = _write_task(tmp_path, plan)
    if mutation == "winner":
        result["methods"]["v2_reduced_grid"]["selected_index"] = 2
    elif mutation == "loss":
        result["candidates"][0]["diagnostic_metrics"]["raw_initializer"]["population_prediction_excess"] = 100
    elif mutation == "candidate":
        result["candidates"].pop()
    elif mutation == "hash":
        result["evidence_sha256"]["data.npz"] = "0"*64
    elif mutation == "prefix":
        result["candidates"][0]["checkpoints"][-1]["selected_iteration"] = 0
    else:
        result["candidates"][1]["eligible"] = False
    _write_json(directory / "result.json", result)
    audited = audit.audit(tmp_path, allow_local=True, write=False)
    assert not audited["audit"]["audit_passed"]
    assert code in {issue["code"] for issue in audited["audit"]["issues"]}
    assert len(audited["per_replication"]) == 2


def test_objective_descent_never_compares_different_chart_epochs():
    candidate = dict(history_chart_epochs=[0, 0, 1, 1], history=[
        dict(iteration=0, objective=10., objective_change=None),
        dict(iteration=1, objective=9., objective_change=-1.),
        dict(iteration=1, objective=15., objective_change=None),
        dict(iteration=2, objective=14., objective_change=-1.)])
    output = audit.objective_diagnostics(candidate)
    assert [row["objective_increase_above_tolerance_count"] for row in output["epochs"]] == [0, 0]
    assert output["chart_transitions"][0]["objective_jump"] == 6
    assert output["chart_transitions"][0]["included_in_descent_check"] is False


def test_guard_cannot_be_bypassed_by_slurm_job_id_alone(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    with pytest.raises(RuntimeError, match="scratch2"):
        audit.audit("/not-scratch2/guard-only", write=False)


def test_duplicate_tasks_do_not_reduce_expected_denominator(tmp_path):
    plan = _plan(tmp_path, budgets=(100, 500))
    plan["tasks"] = [plan["tasks"][0]]
    _write_json(tmp_path / "budget-plan.json", plan)
    output = audit.audit(tmp_path, allow_local=True, write=False)
    assert output["audit"]["planned_task_count"] == 2
    assert "planned_task_inventory_mismatch" in {issue["code"] for issue in output["audit"]["issues"]}
