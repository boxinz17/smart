"""Population projection identities and tiny saved-artifact evaluation tests."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "simulation"))
from reviewer_revision_20260913 import population_floor as diagnostic


def risk(C, truth, sigma):
    error = C - truth
    return float(np.sum(error * (sigma @ error)) / truth.shape[1])


def test_correlated_predictors_require_weighted_not_euclidean_projection():
    sigma = np.array([[2., 1.], [1., 3.]])
    truth = np.array([[0., 0.], [1., 0.]])
    u = v = np.array([[1.], [0.]])
    result = diagnostic.population_floor(truth, sigma, u, v)
    np.testing.assert_allclose(result["coefficient"], [[.5, 0.], [0., 0.]])
    assert result["risk"] == pytest.approx(1.25)
    assert risk(u @ (u.T @ truth @ v) @ v.T, truth, sigma) == pytest.approx(1.5)
    assert result["normal_equation_residual"] < 1e-14
    # A general coefficient can beat this restricted-class floor.
    assert risk(truth, truth, sigma) == 0. < result["risk"]


def test_pythagorean_identity_rank_bound_and_frame_rotation_invariance():
    rng = np.random.default_rng(42)
    m = rng.normal(size=(6, 6)); sigma = m @ m.T + np.eye(6)
    truth = rng.normal(size=(6, 2)) @ rng.normal(size=(2, 5))
    u = np.linalg.qr(rng.normal(size=(6, 4)))[0]
    v = np.linalg.qr(rng.normal(size=(5, 3)))[0]
    result = diagnostic.population_floor(truth, sigma, u, v)
    assert result["floor_rank"] <= result["truth_rank"] == 2
    for _ in range(5):
        B = rng.normal(size=(4, 3)); delta = B-result["reduced_coefficient"]
        excess = np.sum(delta * ((u.T @ sigma @ u) @ delta))/5
        assert risk(u @ B @ v.T, truth, sigma) == pytest.approx(result["risk"]+excess)
    qu = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    qv = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    rotated = diagnostic.population_floor(truth, sigma, u @ qu, v @ qv)
    np.testing.assert_allclose(rotated["coefficient"], result["coefficient"], atol=1e-12)


def test_exact_containment_gives_zero_floor():
    u = np.eye(4)[:, :2]; v = np.eye(3)[:, :2]
    truth = u @ np.array([[1., 2.], [3., 4.]]) @ v.T
    sigma = .5 ** np.abs(np.arange(4)[:, None]-np.arange(4))
    assert diagnostic.population_floor(truth, sigma, u, v)["risk"] < 1e-28


@pytest.mark.parametrize("sigma,u", [(np.array([[1., 2.], [2., 1.]]), np.eye(2)),
    (np.array([[1., 1.], [0., 1.]]), np.eye(2)), (np.eye(2), np.ones((2, 2)))])
def test_invalid_covariance_or_frames_rejected(sigma, u):
    with pytest.raises(ValueError):
        diagnostic.population_floor(np.eye(2), sigma, u, np.eye(2))


def fixture(root):
    methods = ["source_subspace_rrr", "source_subspace_ridge_rrr", "initializer_only", "v2", "v2_transfer_only"]
    case = dict(case_id="tiny_covariance", family="fixture", level=0, p=2, q=2, n_train=4,
                n_validation=4, n_test=4, target_rank=2, source_rank=1)
    tasks = [dict(case_index=0, seed=seed, expected_methods=methods) for seed in (11, 12)]
    plan = dict(cases=[case], tasks=tasks, n_cases=1, n_tasks=2)
    (root / "plan.json").write_text(json.dumps(plan))
    digest = hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    receipt = dict(audit_passed=True, all_planned_tasks_complete=True, plan_sha256=digest,
        n_planned_tasks=2, n_validated_complete_tasks=2, n_errors=0,
        tasks=[dict(task_index=i, case_id=case["case_id"], state="complete") for i in range(2)])
    (root / "audit.json").write_text(json.dumps(receipt))
    rows = []
    sigma = np.array([[2., 1.], [1., 3.]])
    truth = np.array([[0., 1.], [1., 2.]])
    source = np.array([[1., 0.], [0., 0.]])
    floor_C = np.array([[.5, 0.], [0., 0.]])
    for index, task in enumerate(tasks):
        directory = root / "tasks" / f"task-{index:06d}"
        directory.mkdir(parents=True)
        # Accessing these poison response entries with allow_pickle=False would
        # raise: the diagnostic must load only the named evaluation inputs.
        poison = np.array([object()], dtype=object)
        np.savez(directory / "data.npz", C0=source, Y=poison, Y_validation=poison)
        np.savez(directory / "truth.npz", C_star=truth, Sigma_x=sigma, Y_test=poison)
        coefficients, outcomes = {}, {}
        for method in methods:
            status = "failed" if index == 1 and method == "source_subspace_ridge_rrr" else "success"
            C = truth if method.startswith("v2") else floor_C
            if status == "success":
                coefficients[method] = C
            outcomes[method] = dict(success=status == "success", parameters=dict(rank=2),
                                    error="fixture failure" if status == "failed" else None)
            rows.append(dict(task_index=index, case_id=case["case_id"], seed=task["seed"],
                method=method, status=status, population_prediction_excess=risk(C, truth, sigma) if status == "success" else "",
                fit_data_fingerprint="audited-data", truth_fingerprint="audited-truth"))
        np.savez(directory / "coefficients.npz", **coefficients)
        (directory / "result.json").write_text(json.dumps(dict(complete=True, plan_sha256=digest,
            task=task, case=case, methods=outcomes, fit_data_fingerprint="audited-data", truth_fingerprint="audited-truth")))
    with (root / "per_replication.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return root


def test_evaluation_never_loads_responses_and_keeps_failed_denominators(tmp_path):
    result = diagnostic.evaluate(fixture(tmp_path), allow_tiny_local=True)
    assert result["audit"]["diagnostic_passed"]
    assert result["audit"]["n_evaluated_tasks"] == 2
    v2 = next(row for row in result["summary"] if row["method"] == "v2")
    assert v2["n_below_floor"] == 2 and v2["outside_span_norm_mean"] > 0
    ridge = next(row for row in result["summary"] if row["method"] == "source_subspace_ridge_rrr")
    assert ridge["n_planned"] == 2 and ridge["n_success"] == ridge["n_failure"] == 1
    assert ridge["floor_n"] == 2 and ridge["risk_minus_floor_n"] == 1
    assert ridge["risk_minus_floor_mean"] == pytest.approx(0.)
    for name in ("population-floor-audit.json", "population-floor-per-replication.csv", "population-floor-summary.csv"):
        assert (tmp_path / name).exists()


@pytest.mark.parametrize("field,value", [("audit_passed", False), ("all_planned_tasks_complete", False),
    ("plan_sha256", "wrong"), ("n_validated_complete_tasks", 1)])
def test_passing_complete_main_audit_is_mandatory(tmp_path, field, value):
    fixture(tmp_path)
    path = tmp_path / "audit.json"; receipt = json.loads(path.read_text()); receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="complete passing audit"):
        diagnostic.evaluate(tmp_path, allow_tiny_local=True)
    assert not (tmp_path / "population-floor-audit.json").exists()


def test_selected_risk_is_recomputed_against_audited_csv(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "tasks/task-000000/coefficients.npz"
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["v2"] = np.zeros((2, 2)); np.savez(path, **arrays)
    with pytest.raises(RuntimeError, match="risk differs"):
        diagnostic.evaluate(tmp_path, allow_tiny_local=True)


def test_restricted_label_cannot_legitimately_beat_its_floor(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "tasks/task-000000/coefficients.npz"
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["source_subspace_rrr"] = arrays["v2"].copy()
    np.savez(path, **arrays)
    path = tmp_path / "per_replication.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
    for row in rows:
        if row["task_index"] == "0" and row["method"] == "source_subspace_rrr":
            row["population_prediction_excess"] = "0"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    result = diagnostic.evaluate(tmp_path, write=False, allow_tiny_local=True)
    assert not result["audit"]["diagnostic_passed"]
    assert {row["code"] for row in result["audit"]["issues"]} == {
        "restricted_estimator_outside_observed_spans", "restricted_estimator_below_population_floor"}


def test_duplicate_csv_rows_and_missing_audit_reports_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "per_replication.csv"
    text = path.read_text(); path.write_text(text+text.splitlines(keepends=True)[1])
    with pytest.raises(RuntimeError, match="exactly once"):
        diagnostic.evaluate(tmp_path, allow_tiny_local=True)
    path.write_text(text)
    path = tmp_path / "audit.json"; audit = json.loads(path.read_text()); audit["tasks"].pop()
    path.write_text(json.dumps(audit))
    with pytest.raises(RuntimeError, match="task coverage"):
        diagnostic.evaluate(tmp_path, allow_tiny_local=True)


def test_production_guard(tmp_path, monkeypatch):
    fixture(tmp_path); monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="requires Slurm"):
        diagnostic.evaluate(tmp_path)
