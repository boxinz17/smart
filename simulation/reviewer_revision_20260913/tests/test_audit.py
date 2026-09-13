"""Tiny saved-artifact corruption fixtures; no simulation fitting is run."""
import importlib.util
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("revision_audit", Path(__file__).parents[1] / "audit.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def metrics(C, data, truth):
    error = C-truth["C_star"]
    return dict(coefficient_rmse=float(np.linalg.norm(error)/np.sqrt(C.size)),
        coefficient_squared_error=float(np.linalg.norm(error)**2),
        population_prediction_excess=float(np.trace(error.T @ truth["Sigma_x"] @ error)/C.shape[1]),
        training_mse=float(np.linalg.norm(data["Y"]-data["X"]@C)**2/data["Y"].size),
        validation_mse=float(np.linalg.norm(data["Y_validation"]-data["X_validation"]@C)**2/data["Y_validation"].size),
        test_mse=float(np.linalg.norm(truth["Y_test"]-truth["X_test"]@C)**2/truth["Y_test"].size),
        numerical_rank=int(np.linalg.matrix_rank(C)))


def fixture(root, seeds=(1, 2), missing=()):
    methods = ["target_rrr", "target_ridge_rrr", "v2", "v2_transfer_only", "initializer_only"]
    case = dict(case_id="tiny", family="fixture", level=0, p=4, q=3,
                n_train=6, n_validation=5, n_test=7, target_rank=1, source_rank=2)
    config = dict(init_penalties=[.1, .2], penalties_u=[0., .1], penalties_v=[0.])
    tasks = [dict(case_index=0, seed=seed, variants=["full_caps"], expected_methods=methods) for seed in seeds]
    plan = dict(schema=1, cases=[case], tasks=tasks, configuration=config,
                n_cases=1, n_tasks=len(tasks), seed_ids=list(seeds))
    save_json(root / "plan.json", plan)
    plan_hash = audit.hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    for i, task in enumerate(tasks):
        if i in missing:
            continue
        directory = root / "tasks" / f"task-{i:06d}"
        directory.mkdir(parents=True)
        rng = np.random.default_rng(task["seed"])
        C = np.arange(1, 5)[:, None] @ np.array([[.2, .3, .1]])
        X, Xv, Xt = rng.normal(size=(6, 4)), rng.normal(size=(5, 4)), rng.normal(size=(7, 4))
        data = dict(X=X, Y=X@C, X_validation=Xv, Y_validation=Xv@C, C0=C.copy())
        truth = dict(C_star=C, Sigma_x=np.eye(4), X_test=Xt, Y_test=Xt@C)
        np.savez(directory / "data.npz", **data)
        np.savez(directory / "truth.npz", **truth)
        specs = audit._declared_specs(case, config, "full_caps")
        candidates = []
        variant = directory / "full_caps"
        variant.mkdir()
        for j, spec in enumerate(specs):
            if j == 2:
                candidates.append(dict(index=j, spec=spec, eligible=False,
                    exception=dict(type="NumericalFailure", message="Deliberate fixture"),
                    method=None, status="failed", selected_iteration=None, elapsed_seconds=1.))
            else:
                coefficient = C*(.8 if j == 0 else .9)
                np.savez(variant / f"candidate-{j:04d}.npz", coefficient=coefficient)
                candidates.append(dict(index=j, spec=spec, eligible=True, exception=None,
                    method="target_rrr" if j == 0 else "sparse_refinement", status="ok",
                    selected_iteration=0 if j == 0 else 5, elapsed_seconds=1.,
                    validation_mse=metrics(coefficient, data, truth)["validation_mse"]))
        save_json(variant / "candidates.json", dict(variant="full_caps", candidates=candidates,
            complete=True, expected_n_candidates=3, n_eligible=2, total_seconds=3.,
            specs_sha256=audit._specs_digest(specs)))
        initializers = []
        for j, spec in enumerate(specs[1:]):
            coefficient = C*(.6 + .05*j)
            initializers.append(dict(index=j, spec=spec, eligible=True, error=None, status="ok",
                validation_mse=metrics(coefficient, data, truth)["validation_mse"], elapsed_seconds=.1))
            np.savez(variant / f"initializer-{j:04d}.npz", coefficient=coefficient)
        save_json(variant / "initializers.json", dict(complete=True, candidates=initializers, total_seconds=.2))
        coefficients = dict(target_rrr=C*.8, target_ridge_rrr=C*.95, v2=C*.9,
                            v2_transfer_only=C*.9, initializer_only=C*.65)
        outcomes = {}
        for method, coefficient in coefficients.items():
            outcomes[method] = dict(success=True, metrics=metrics(coefficient, data, truth),
                total_seconds=1., parameters={})
            if method.startswith("v2"):
                outcomes[method].update(candidate_index=1, selected_method="sparse_refinement",
                    parameters=specs[1], selected_iteration=5, n_candidates=3, n_eligible=2)
            elif method == "initializer_only":
                outcomes[method].update(candidate_index=1, selected_method="lasso_svd_initializer",
                    parameters=specs[2], selected_iteration=0, n_candidates=3, n_eligible=2)
        np.savez(directory / "coefficients.npz", **coefficients)
        save_json(directory / "result.json", dict(schema=1, task=task, case=case,
            metadata=dict(seed=task["seed"], case_id=case["case_id"]), complete=True,
            plan_sha256=plan_hash, fit_data_fingerprint=audit._fingerprint(data),
            truth_fingerprint=audit._fingerprint(truth), methods=outcomes))
    return root


def inspect(root, **kwargs):
    return audit.audit(root, allow_tiny_local=True, **kwargs)


def codes(result):
    return {issue["code"] for issue in result["audit"]["issues"]}


def test_complete_audit_recomputes_and_writes_paired_statistics(tmp_path):
    result = inspect(fixture(tmp_path))
    assert result["audit"]["audit_passed"], result["audit"]["issues"]
    assert result["audit"]["all_planned_tasks_complete"]
    assert result["summary"]["status_counts"]["success"] == 10
    for filename in ("audit.json", "summary.json", "per_replication.csv", "aggregate.csv", "paired.csv"):
        assert (tmp_path / filename).stat().st_size > 0
    paired = next(row for row in result["paired"] if row["method"] == "v2" and
                  row["reference"] == "target_ridge_rrr" and row["metric"] == "population_prediction_excess")
    assert paired["n_paired"] == paired["n_planned"] == 2
    assert paired["mean"] > 0 and paired["harm_frequency"] == 1
    assert 0 < paired["harm_wilson95_low"] < paired["harm_wilson95_high"] <= 1


def test_missing_task_retains_all_planned_denominators(tmp_path):
    result = inspect(fixture(tmp_path, seeds=(1, 2, 3), missing=(2,)))
    assert "missing_task" in codes(result)
    assert result["summary"]["status_counts"]["missing_task"] == 5
    aggregate = next(row for row in result["aggregate"] if row["method"] == "v2" and row["metric"] == "coefficient_rmse")
    assert aggregate["n_planned"] == 3 and aggregate["n_success"] == 2 and aggregate["n_missing"] == 1


@pytest.mark.parametrize("field", ["fit_data_fingerprint", "truth_fingerprint", "plan_sha256"])
def test_corrupted_fingerprints_invalidate_all_task_metrics(tmp_path, field):
    fixture(tmp_path, seeds=(1,))
    path = tmp_path / "tasks/task-000000/result.json"
    value = json.loads(path.read_text()); value[field] = "wrong"; save_json(path, value)
    result = inspect(tmp_path)
    assert not result["audit"]["audit_passed"]
    assert all(row["status"] == "audit_invalid" for row in result["per_replication"])


def test_false_complete_and_metrics_corruption_are_detected(tmp_path):
    fixture(tmp_path, seeds=(1,))
    path = tmp_path / "tasks/task-000000/result.json"
    value = json.loads(path.read_text())
    value["complete"] = False
    value["methods"]["v2"]["metrics"]["coefficient_rmse"] = 0
    save_json(path, value)
    result = inspect(tmp_path)
    assert {"false_complete", "metric_mismatch"} <= codes(result)


def test_failed_candidate_cannot_be_dropped_from_declared_grid(tmp_path):
    fixture(tmp_path, seeds=(1,))
    path = tmp_path / "tasks/task-000000/full_caps/candidates.json"
    value = json.loads(path.read_text()); value["candidates"].pop(); save_json(path, value)
    result = inspect(tmp_path)
    assert {"candidate_count_mismatch", "candidate_specs_mismatch"} <= codes(result)


def test_unselected_candidate_loss_is_independently_checked(tmp_path):
    fixture(tmp_path, seeds=(1,))
    path = tmp_path / "tasks/task-000000/full_caps/candidate-0000.npz"
    np.savez(path, coefficient=np.zeros((4, 3)))
    result = inspect(tmp_path)
    assert "candidate_validation_mismatch" in codes(result)


def test_v2_must_choose_minimum_but_transfer_excludes_target_endpoint(tmp_path):
    fixture(tmp_path, seeds=(1,))
    directory = tmp_path / "tasks/task-000000"
    truth, data = audit._arrays(directory / "truth.npz"), audit._arrays(directory / "data.npz")
    coefficient = .99*truth["C_star"]
    np.savez(directory / "full_caps/candidate-0000.npz", coefficient=coefficient)
    path = directory / "full_caps/candidates.json"
    value = json.loads(path.read_text())
    value["candidates"][0]["validation_mse"] = metrics(coefficient, data, truth)["validation_mse"]
    save_json(path, value)
    result = inspect(tmp_path)
    assert "nonminimal_validation_selection" in codes(result)
    rows = {row["method"]: row for row in result["per_replication"]}
    assert rows["v2"]["status"] == "audit_invalid"
    assert rows["v2_transfer_only"]["status"] == "success"


def test_initializer_is_audited_even_when_corresponding_refinement_fails(tmp_path):
    fixture(tmp_path, seeds=(1,))
    result = inspect(tmp_path)
    initializer = next(row for row in result["per_replication"] if row["method"] == "initializer_only")
    assert initializer["status"] == "success"
    assert initializer["candidate_index"] == 1
    path = tmp_path / "tasks/task-000000/full_caps/initializer-0001.npz"
    np.savez(path, coefficient=np.zeros((4, 3)))
    assert "initializer_validation_mismatch" in codes(inspect(tmp_path))


def test_reference_failure_is_explicit_and_pairs_are_unavailable(tmp_path):
    fixture(tmp_path, seeds=(1,))
    directory = tmp_path / "tasks/task-000000"
    path = directory / "result.json"
    value = json.loads(path.read_text())
    value["methods"]["target_rrr"] = dict(success=False, error="deliberate numerical failure", total_seconds=.2)
    save_json(path, value)
    coefficients = audit._arrays(directory / "coefficients.npz")
    del coefficients["target_rrr"]
    np.savez(directory / "coefficients.npz", **coefficients)
    result = inspect(tmp_path)
    assert result["audit"]["audit_passed"], result["audit"]["issues"]
    paired = next(row for row in result["paired"] if row["method"] == "v2" and row["reference"] == "target_rrr")
    assert paired["n_planned"] == paired["n_unavailable_pairs"] == 1
    assert paired["n_paired"] == 0 and paired["mean"] is None


def test_missing_reference_is_an_error(tmp_path):
    fixture(tmp_path, seeds=(1,))
    path = tmp_path / "tasks/task-000000/result.json"
    value = json.loads(path.read_text()); del value["methods"]["target_rrr"]; save_json(path, value)
    result = inspect(tmp_path)
    assert {"missing_reference", "missing_method"} <= codes(result)


def test_monte_carlo_statistics_do_not_invent_uncertainty_at_n1():
    assert audit._moments([2])["mean"] == 2
    assert audit._moments([2])["mcse"] is None
    assert audit._moments([2])["ci95_low"] is None
    assert audit._moments([])["mean"] is None
    assert audit._moments([1, 3])["mcse"] == pytest.approx(1)
    low, high = audit._wilson(0, 30)
    assert low == 0 and 0 < high < .12


def test_cap_control_has_separate_variant_and_matched_grid(tmp_path):
    config = dict(init_penalties=[.1, .2], penalties_u=[0., .1], penalties_v=[0.])
    case = dict(p=20, q=15, target_rank=3, source_rank=10)
    active = audit._declared_specs(case, config, "active_caps")
    control = audit._declared_specs(case, config, "cap_control")
    assert len(active) == 3*len(control) == 24
    assert all(spec["free_directions"] == [3, 3] for spec in active+control)
    assert audit._method_variant("v2_transfer_only_cap_control") == "cap_control"


def test_production_audit_cannot_run_locally_without_explicit_tiny_opt_in(tmp_path):
    fixture(tmp_path, seeds=(1,))
    with pytest.raises(RuntimeError, match="Production auditing"):
        audit.audit(tmp_path)


def test_benchmark_loss_scaling_and_complete_tuning_inventory(tmp_path):
    fixture(tmp_path, seeds=(1,))
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["configuration"].update(benchmark_ridges=[0., .1], mixture_weights=[0.], nuclear_penalties=[.1])
    save_json(plan_path, plan)
    path = tmp_path / "tasks/task-000000/result.json"
    result = json.loads(path.read_text())
    result["plan_sha256"] = audit.hashlib.sha256(plan_path.read_bytes()).hexdigest()
    for method in ("target_rrr", "target_ridge_rrr"):
        outcome = result["methods"][method]
        grid = audit._benchmark_grid(method, plan["cases"][0], plan["configuration"])
        selected_loss = outcome["metrics"]["validation_mse"]*plan["cases"][0]["q"]/2
        outcome.update(selected_index=0, audit=[dict(candidate_index=i, candidate=candidate,
            status="ok", validation_loss=selected_loss+i) for i, candidate in enumerate(grid)])
    save_json(path, result)
    good = inspect(tmp_path)
    assert good["audit"]["audit_passed"], good["audit"]["issues"]
    for method in ("target_rrr", "target_ridge_rrr"):
        del result["methods"][method]["selected_index"]
    save_json(path, result)
    assert inspect(tmp_path)["audit"]["audit_passed"]
    result["methods"]["target_ridge_rrr"]["audit"][1]["status"] = "failed"
    result["methods"]["target_ridge_rrr"]["audit"].pop()
    save_json(path, result)
    bad = inspect(tmp_path)
    assert "benchmark_candidate_grid_mismatch" in codes(bad)


def test_published_method_intercept_is_preserved_in_prediction_risk(tmp_path):
    fixture(tmp_path, seeds=(1,))
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["tasks"][0]["expected_methods"].append("park_two_stage_nr")
    save_json(plan_path, plan)
    directory = tmp_path / "tasks/task-000000"
    data, truth = audit._arrays(directory / "data.npz"), audit._arrays(directory / "truth.npz")
    coefficient, intercept = .7*truth["C_star"], np.array([.2, -.1, .3])
    coefficients = audit._arrays(directory / "coefficients.npz")
    coefficients.update(park_two_stage_nr=coefficient, park_two_stage_nr__intercept=intercept)
    np.savez(directory / "coefficients.npz", **coefficients)
    checked = metrics(coefficient, data, truth)
    checked["population_prediction_excess"] += float(intercept @ intercept / 3)
    checked["training_mse"] = float(np.mean((data["Y"]-data["X"]@coefficient-intercept)**2))
    checked["validation_mse"] = float(np.mean((data["Y_validation"]-data["X_validation"]@coefficient-intercept)**2))
    checked["test_mse"] = float(np.mean((truth["Y_test"]-truth["X_test"]@coefficient-intercept)**2))
    path = directory / "result.json"
    result = json.loads(path.read_text())
    result.update(task=plan["tasks"][0], plan_sha256=audit.hashlib.sha256(plan_path.read_bytes()).hexdigest())
    result["methods"]["park_two_stage_nr"] = dict(success=True, metrics=checked,
        intercept=intercept.tolist(), total_seconds=1., diagnostics={})
    save_json(path, result)
    checked_result = inspect(tmp_path)
    assert checked_result["audit"]["audit_passed"], checked_result["audit"]["issues"]
    coefficients["park_two_stage_nr__intercept"][0] += 1
    np.savez(directory / "coefficients.npz", **coefficients)
    assert {"intercept_record_mismatch", "metric_mismatch"} <= codes(inspect(tmp_path))


def park_external_fixture():
    case = dict(p=4, q=3)
    data = dict(X=np.ones((6, 4)), Y=np.ones((6, 3)), X_validation=np.ones((5, 4)),
                Y_validation=np.ones((5, 3)), source_X=np.ones((7, 4)), source_Y=np.ones((7, 3)))
    config = dict(park_lambdas=[.1, 1.])
    admm = [dict(last_max_squared_update=1e-8, tolerance=1e-6, stopping_criterion_met=True) for _ in range(6)]
    admm[4].update(last_max_squared_update=1e-2, stopping_criterion_met=False)
    candidates = []
    for i, ((left, right), loss, pooled, correction) in enumerate(zip(
            ((.1, .1), (.1, 1.), (1., .1), (1., 1.)), (4., 1., .5, 3.), (1, 1, 4, 4), (2, 3, 5, 6))):
        candidates.append(dict(candidate_index=i, lambda_w=left, lambda_delta=right,
            validation_loss=loss, status="uncertified" if i == 2 else "ok",
            pooled_admm_index=pooled, correction_admm_index=correction,
            pooled_stopping_criterion_met=True, correction_stopping_criterion_met=i != 2))
    diagnostics = dict(options=dict(lambda_w=[.1, 1.], lambda_delta=[.1, 1.],
        require_convergence=True, mode="external_validation"), refit=False,
        tuning="revision_independent_target_holdout_pair_selection",
        n_train=6, n_source=7, n_predictors=4, n_responses=3, n_validation=5,
        candidate_results=deepcopy(candidates), admm_fits=admm,
        preprocessing_scope="target_training_and_raw_source_only_validation_rows_excluded",
        candidate_count=4, selected_index=1, validation_loss=1., selected_lambda_w=.1,
        selected_lambda_delta=1.)
    record = dict(success=True, diagnostics=diagnostics, candidates=candidates,
                  selected_index=1, validation_score=2/3, intercept=[0., 0., 0.])
    return record, config, data, case


def test_external_park_selection_excludes_unconverged_better_candidate():
    arguments = park_external_fixture()
    findings, score = audit._park_external_checks(*arguments)
    assert findings == []
    assert score == pytest.approx(2/3)


@pytest.mark.parametrize("mutation, expected", [
    ("bad_scale", "park_validation_score_scale_mismatch"),
    ("select_uncertified", "park_ineligible_selected_candidate"),
    ("drop_uncertified", "park_candidate_grid_mismatch"),
    ("nonminimal", "park_nonminimal_validation_selection"),
    ("stopping_flag", "park_stopping_flag_mismatch"),
    ("validation_in_preprocessing", "park_preprocessing_scope_mismatch")])
def test_external_park_candidate_audit_detects_corruption(mutation, expected):
    record, config, data, case = park_external_fixture()
    if mutation == "bad_scale":
        record["validation_score"] = 1.
    elif mutation == "select_uncertified":
        record["selected_index"] = 2
    elif mutation == "drop_uncertified":
        record["candidates"].pop(2)
        record["diagnostics"]["candidate_results"] = deepcopy(record["candidates"])
    elif mutation == "nonminimal":
        record["selected_index"] = 0
    elif mutation == "stopping_flag":
        record["diagnostics"]["admm_fits"][4]["last_max_squared_update"] = 1e-10
    else:
        record["diagnostics"]["preprocessing_scope"] = "all_rows"
    findings, _ = audit._park_external_checks(record, config, data, case)
    assert expected in {finding[0] for finding in findings}


def test_external_park_timed_out_method_keeps_failure_without_inventing_scores():
    record, config, data, case = park_external_fixture()
    record.update(success=False, error="declared timeout", candidates=[])
    record["diagnostics"]["candidate_results"] = []
    findings, score = audit._park_external_checks(record, config, data, case)
    assert score is None
    assert findings == [("park_failure_without_candidate_trace",
                         "Whole-method failure is retained; no completed candidate trace is available", "warning")]


def test_external_park_complete_saved_artifact_integration(tmp_path):
    fixture(tmp_path, seeds=(1,))
    record, park_config, _, _ = park_external_fixture()
    directory = tmp_path / "tasks/task-000000"
    data, truth = audit._arrays(directory / "data.npz"), audit._arrays(directory / "truth.npz")
    data.update(source_X=np.ones((7, 4)), source_Y=np.ones((7, 3)))
    np.savez(directory / "data.npz", **data)
    coefficients = audit._arrays(directory / "coefficients.npz")
    coefficient, intercept = truth["C_star"]*.7, np.array([.2, -.1, .3])
    coefficients.update({audit.PARK_EXTERNAL:coefficient, audit.PARK_EXTERNAL+"__intercept":intercept})
    np.savez(directory / "coefficients.npz", **coefficients)
    measured = audit._recompute(coefficient, data, truth, intercept)
    bridge_loss = measured["validation_mse"]*3/2
    for i, candidate in enumerate(record["candidates"]):
        candidate["validation_loss"] = bridge_loss + (1. if i in (0, 3) else -.1 if i == 2 else 0.)
    record["diagnostics"].update(candidate_results=deepcopy(record["candidates"]), validation_loss=bridge_loss)
    record.update(metrics=measured, validation_score=measured["validation_mse"],
                  intercept=intercept.tolist(), total_seconds=1.)
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["tasks"][0]["expected_methods"].append(audit.PARK_EXTERNAL)
    plan["configuration"].update(park_config)
    save_json(plan_path, plan)
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    result.update(task=plan["tasks"][0], plan_sha256=audit.hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                  fit_data_fingerprint=audit._fingerprint(data))
    result["methods"][audit.PARK_EXTERNAL] = record
    save_json(result_path, result)
    checked = inspect(tmp_path)
    assert checked["audit"]["audit_passed"], checked["audit"]["issues"]


def test_expanded_full_grid_count_is_configuration_driven():
    config = dict(init_penalties=[.003, .03, .1, .3, 1., 3.],
                  penalties_u=[0., .001, .0025, .01, .04, .16],
                  penalties_v=[0., .001, .0025, .01, .04, .16])
    case = dict(p=100, q=50, source_rank=10, target_rank=5)
    assert audit._expected_candidates(config, "full_caps") == 211
    assert len(audit._declared_specs(case, config, "full_caps")) == 211
