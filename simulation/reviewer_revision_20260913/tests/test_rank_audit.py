"""Tiny saved-artifact fixtures; no simulation fitting is performed."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation"))
from reviewer_revision_20260913 import audit as common, rank_audit


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def metric(coefficient, data, truth):
    error = coefficient - truth["C_star"]
    return dict(coefficient_rmse=float(np.sqrt(np.mean(error**2))),
        coefficient_squared_error=float(np.sum(error**2)),
        population_prediction_excess=float(np.sum(error * (truth["Sigma_x"] @ error)) / coefficient.shape[1]),
        validation_mse=float(np.mean((data["Y_validation"] - data["X_validation"] @ coefficient)**2)),
        training_mse=float(np.mean((data["Y"] - data["X"] @ coefficient)**2)),
        test_mse=float(np.mean((truth["Y_test"] - truth["X_test"] @ coefficient)**2)),
        numerical_rank=int(np.linalg.matrix_rank(coefficient)))


def fixture(root, seeds=(1, 2), missing=()):
    case = dict(case_id="tiny", family="fixture", level=0, p=4, q=3, n_train=4,
                n_validation=4, n_test=4, target_rank=1, source_rank=2)
    config = dict(rank_grid=[1, 2], source_truncation_grid=[2], init_penalties=[.1],
                  penalties_u=[0., .2], penalties_v=[0.], benchmark_ridges=[0.],
                  mixture_weights=[0.], nuclear_penalties=[.1], tuning_uses_truth=False, refit_on_validation=False)
    cells = [dict(cell_index=i, fitted_rank=r, source_rank=2, admissible=True, exclusion_reason=None)
             for i, r in enumerate((1, 2))]
    libraries = [dict(cell=cell, candidates=common._declared_specs(
        dict(case, fitted_rank=cell["fitted_rank"]), config, "full_caps")) for cell in cells]
    benchmarks = {method: [dict(method=method, rank=r, ridge=0., **({"source_rank": 2}
                  if method.startswith("source_subspace") else {})) for r in (1, 2)]
                  for method in ("target_rrr", "target_ridge_rrr", "source_subspace_rrr")}
    expected = list(benchmarks) + list(rank_audit.LABELS)
    tasks = [dict(case_index=0, seed=seed, expected_methods=expected) for seed in seeds]
    plan = dict(schema=1, study="operational_rank_selection", cases=[case], tasks=tasks,
                n_cases=1, n_tasks=len(tasks), configuration=config, code_sha256={"fixture": "fixed"},
                rank_cells={"tiny": cells}, v2_libraries={"tiny": libraries},
                competitor_libraries={"tiny": benchmarks})
    save(root / "rank-plan.json", plan)
    digest = hashlib.sha256((root / "rank-plan.json").read_bytes()).hexdigest()
    for index, task in enumerate(tasks):
        if index in missing:
            continue
        directory = root / "rank_tasks" / f"task-{index:06d}"
        directory.mkdir(parents=True)
        truth_C = np.zeros((4, 3)); truth_C[0, 0] = 2.
        source = np.zeros((4, 3)); source[0, 0] = 1.
        X = np.eye(4)
        data = dict(X=X, Y=X @ truth_C, X_validation=X, Y_validation=X @ truth_C, C0=source)
        truth = dict(C_star=truth_C, Sigma_x=np.eye(4), X_test=X, Y_test=X @ truth_C)
        frames = dict(left=np.eye(4), right=np.eye(3), singular_values=np.array([1., 0., 0.]))
        np.savez(directory / "data.npz", **data)
        np.savez(directory / "truth.npz", **truth)
        np.savez(directory / "observed-source-frames.npz", **frames)
        outcomes, coefficients = {}, {}
        for method, specs in benchmarks.items():
            audit = []
            for i, spec in enumerate(specs):
                C = truth_C * (.55 + .1 * i)
                diagnostics = dict(certified=True, source_frame_convention="supplied_common_observed_source_decomposition")
                score = metric(C, data, truth)["validation_mse"]
                audit.append(dict(candidate_index=i, candidate=spec, status="ok", coefficient_saved=True,
                    validation_mse=score, validation_loss=score * 3 / 2, elapsed_seconds=.02,
                    diagnostics=diagnostics))
                path = directory / "benchmarks" / method
                path.mkdir(parents=True, exist_ok=True)
                np.savez(path / f"candidate-{i:04d}.npz", coefficient=C)
            save(path / "candidates.json", dict(complete=True, candidates=audit, expected_n_candidates=2,
                n_eligible=2, specs_sha256=common._specs_digest(specs), total_seconds=.04, selected_index=1))
            coefficients[method] = C
            outcomes[method] = dict(success=True, selected_index=1, audit=audit, n_candidates=2, n_eligible=2,
                fitting_call_seconds=.04, wall_seconds=.06, selected_fitted_rank=2,
                selected_source_dimension=[2, 2] if method.startswith("source_subspace") else None,
                metrics=metric(C, data, truth))
        pools = {method: [] for method in rank_audit.LABELS}
        cell_records = []
        for cell, library in zip(cells, libraries):
            path = directory / "v2" / f"cell-{cell['cell_index']:03d}"
            path.mkdir(parents=True)
            specs, audit = library["candidates"], []
            for i, spec in enumerate(specs):
                C = truth_C * (.5 + .1 * cell["cell_index"] + .2 * i)
                if cell["cell_index"] == 1 and i == 1:
                    C = truth_C * .9
                row = dict(index=i, spec=spec, eligible=True, exception=None, status="ok", elapsed_seconds=.02,
                    validation_mse=metric(C, data, truth)["validation_mse"],
                    selected_iteration=0 if i == 0 else 2,
                    method="target_rrr" if i == 0 else "sparse_smart_v2")
                audit.append(row)
                np.savez(path / f"candidate-{i:04d}.npz", coefficient=C)
            save(path / "candidates.json", dict(complete=True, variant="full_caps", candidates=audit,
                expected_n_candidates=2, specs_sha256=common._specs_digest(specs), n_eligible=2, total_seconds=.05))
            initial = truth_C * (.65 + .1 * cell["cell_index"])
            init_score = metric(initial, data, truth)["validation_mse"]
            save(path / "initializers.json", dict(complete=True, total_seconds=.02, candidates=[dict(
                index=0, spec=specs[1], eligible=True, error=None, status="ok", elapsed_seconds=.01,
                validation_mse=init_score)]))
            np.savez(path / "initializer-0000.npz", coefficient=initial)
            for method in rank_audit.LABELS:
                pools[method].append(dict(cell_index=cell["cell_index"], eligible=True, status="ok",
                    candidate_index=0 if method == "initializer_only" else 1, spec=specs[1],
                    method="lasso_svd_initializer" if method == "initializer_only" else "sparse_smart_v2",
                    selected_iteration=0 if method == "initializer_only" else 2,
                    score=init_score if method == "initializer_only" else audit[1]["validation_mse"],
                    coefficient=initial if method == "initializer_only" else C))
            cell_records.append(dict(cell, status="completed", fitting_call_seconds=.05,
                                     wall_seconds=.09, n_candidates=2, n_eligible=2))
        runtime = dict(shared_computation_group="v2_rank_source_grid", fitting_call_seconds=.1,
            cell_wall_seconds=.18, initializer_pass_wall_seconds=.04, n_candidates=4, n_eligible=4,
            n_initializer_candidates=2, n_initializer_eligible=2, cold_selected_fit_seconds=None)
        save(directory / "v2" / "rank-cells.json", dict(complete=True, cells=cell_records, runtime=runtime))
        for method, pool in pools.items():
            winner = pool[1]
            coefficient = winner["coefficient"]
            coefficients[method] = coefficient
            outcomes[method] = dict(success=True, runtime=runtime, n_grid_cells=2, n_successful_cells=2,
                selected_index=1, selected_cell_index=1, selected_inner_candidate_index=winner["candidate_index"],
                parameters=winner["spec"], selected_fitted_rank=2, selected_source_dimension=2,
                selected_method=winner["method"], selected_iteration=winner["selected_iteration"],
                metrics=metric(coefficient, data, truth), selection_audit=[dict(record_index=i, cell_index=i,
                    eligible=True, status="ok", validation_mse=row["score"]) for i, row in enumerate(pool)])
            save(directory / f"{method}-rank-selection.json", dict(complete=True, selection=outcomes[method],
                candidates=[{k: v for k, v in row.items() if k != "coefficient"} for row in pool]))
        np.savez(directory / "coefficients.npz", **coefficients)
        save(directory / "result.json", dict(complete=True, rank_evidence_schema=2, plan_sha256=digest,
            task=task, case=case, executed_code_sha256=plan["code_sha256"], methods=outcomes,
            fit_data_fingerprint=common._fingerprint(data), truth_fingerprint=common._fingerprint(truth),
            observed_source_frames_fingerprint=common._fingerprint(frames),
            generation_and_source_fit_seconds=.03, shared_source_decomposition_seconds=.01, wall_seconds=2.))
    return root


def inspect(root):
    return rank_audit.audit(root, allow_tiny_local=True)


def codes(result):
    return {issue["code"] for issue in result["audit"]["issues"]}


def test_complete_joint_rank_audit_and_all_outputs(tmp_path):
    result = inspect(fixture(tmp_path))
    assert result["audit"]["audit_passed"], result["audit"]["issues"]
    assert result["summary"]["status_counts"] == {"success": 12}
    assert {row["selected_fitted_rank"] for row in result["rank_selection"]} == {2}
    for name in ("rank-audit.json", "rank-summary.json", "rank-per-replication.csv",
                 "rank-aggregate.csv", "rank-paired.csv", "rank-selection.csv", "rank-runtime.csv"):
        assert (tmp_path / name).exists()


def test_missing_task_stays_in_failure_denominators(tmp_path):
    result = inspect(fixture(tmp_path, seeds=(1, 2, 3), missing=(2,)))
    assert "missing_task" in codes(result)
    summary = next(row for row in result["aggregate"] if row["method"] == "v2")
    assert summary["n_planned"] == 3 and summary["n_success"] == 2 and summary["n_missing"] == 1
    paired = next(row for row in result["paired"] if row["method"] == "v2")
    assert paired["n_planned"] == 3 and paired["n_paired"] == 2


@pytest.mark.parametrize("path", ["v2/cell-000/candidate-0000.npz", "v2/cell-000/initializer-0000.npz",
                                  "benchmarks/target_rrr/candidate-0000.npz"])
def test_unselected_coefficients_are_independently_rescored(tmp_path, path):
    fixture(tmp_path, seeds=(1,))
    np.savez(tmp_path / "rank_tasks/task-000000" / path, coefficient=np.zeros((4, 3)))
    assert "candidate_validation_mismatch" in codes(inspect(tmp_path))


def test_joint_winner_cannot_be_replaced_by_another_grid_cell(tmp_path):
    fixture(tmp_path, seeds=(1,))
    directory = tmp_path / "rank_tasks/task-000000"
    result = json.loads((directory / "result.json").read_text())
    result["methods"]["v2"]["selected_cell_index"] = 0
    save(directory / "result.json", result)
    assert "nonminimal_joint_rank_selection" in codes(inspect(tmp_path))


def test_missing_candidate_or_cell_cannot_be_silently_dropped(tmp_path):
    fixture(tmp_path, seeds=(1,))
    path = tmp_path / "rank_tasks/task-000000/v2/cell-000/candidates.json"
    value = json.loads(path.read_text()); value["candidates"].pop(); save(path, value)
    assert {"candidate_index_or_count_mismatch", "candidate_specs_mismatch"} <= codes(inspect(tmp_path))


def test_full_tuning_time_is_reconciled(tmp_path):
    fixture(tmp_path, seeds=(1,))
    path = tmp_path / "rank_tasks/task-000000/v2/rank-cells.json"
    value = json.loads(path.read_text()); value["runtime"]["fitting_call_seconds"] = .001; save(path, value)
    assert "rank_tuning_total_mismatch" in codes(inspect(tmp_path))


def test_truth_isolation_detected_even_after_fingerprint_updated(tmp_path):
    fixture(tmp_path, seeds=(1,))
    directory = tmp_path / "rank_tasks/task-000000"
    data = common._arrays(directory / "data.npz")
    data["C_star"] = np.zeros((4, 3))
    np.savez(directory / "data.npz", **data)
    result = json.loads((directory / "result.json").read_text())
    result["fit_data_fingerprint"] = common._fingerprint(data)
    save(directory / "result.json", result)
    assert "truth_in_fit_archive" in codes(inspect(tmp_path))


def test_null_completion_must_match_v2_even_when_source_is_reconstructed(tmp_path):
    fixture(tmp_path, seeds=(1,))
    directory = tmp_path / "rank_tasks/task-000000"
    frames = common._arrays(directory / "observed-source-frames.npz")
    frames["left"][:, [1, 2]] = frames["left"][:, [2, 1]]
    np.savez(directory / "observed-source-frames.npz", **frames)
    result = json.loads((directory / "result.json").read_text())
    result["observed_source_frames_fingerprint"] = common._fingerprint(frames)
    save(directory / "result.json", result)
    assert "source_frames_differ_from_v2" in codes(inspect(tmp_path))


def test_explicit_method_failure_preserves_unavailable_pairs(tmp_path):
    fixture(tmp_path, seeds=(1,))
    directory = tmp_path / "rank_tasks/task-000000"
    path = directory / "benchmarks/target_rrr/candidates.json"
    archive = json.loads(path.read_text())
    for row in archive["candidates"]:
        row.update(status="failed", coefficient_saved=False, error="fixture_failure")
        (path.parent / f"candidate-{row['candidate_index']:04d}.npz").unlink()
    archive.update(n_eligible=0, selected_index=None)
    save(path, archive)
    result = json.loads((directory / "result.json").read_text())
    result["methods"]["target_rrr"].update(success=False, error="fixture_failure", n_eligible=0,
                                               audit=archive["candidates"])
    save(directory / "result.json", result)
    coefficients = common._arrays(directory / "coefficients.npz")
    del coefficients["target_rrr"]
    np.savez(directory / "coefficients.npz", **coefficients)
    checked = inspect(tmp_path)
    assert checked["audit"]["audit_passed"], checked["audit"]["issues"]
    row = next(row for row in checked["paired"] if row["method"] == "v2" and row["reference"] == "target_rrr")
    assert row["n_planned"] == row["n_unavailable_pairs"] == 1
    assert row["n_paired"] == 0
    runtime = next(row for row in checked["runtime"] if row["method"] == "target_rrr")
    assert runtime["n_failure"] == runtime["n"] == 1 and runtime["mean"] == .06


def test_local_production_guard_does_not_depend_on_tmpdir(tmp_path, monkeypatch):
    fixture(tmp_path, seeds=(1,))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Production rank auditing"):
        rank_audit.audit(tmp_path)
