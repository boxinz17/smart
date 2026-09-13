"""Bookkeeping and selection fixtures only; no local solver or simulation fits."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation"))
from reviewer_revision_20260913 import budget_study


def test_fixed_grid_and_schedule_are_declared_without_fitting():
    cases = budget_study.representative_cases()
    assert [case["case_id"] for case in cases] == list(budget_study.CASE_IDS)
    assert cases[-1]["n_train"] == 40
    for case in cases:
        specs = budget_study.candidate_specs(case)
        assert len(specs) == 6
        assert {spec["init_penalty"] for spec in specs} == {.1, .3}
        assert {spec["penalty"] for spec in specs} == {(.01, .01), (.04, .04), (.16, .16)}
    for budget in budget_study.DEFAULT_BUDGETS:
        config = budget_study.budget_configuration(budget)
        assert config["rrr_shortcut"] is False
        assert config["validation_patience"] is None
        assert config["initialization_cache"] is False
        assert config["stationarity_tol"] == 1e-6
        schedule = config["checkpoint_iterations"]
        assert schedule[0] == 0 and schedule[-1] == budget
        assert set(range(50, budget + 1, 50)).issubset(schedule)


def test_default_plan_has_eighty_independent_budget_tasks(tmp_path):
    plan = budget_study.build_plan(tmp_path)
    assert plan["n_tasks"] == 80 and plan["expected_fits"] == 480
    assert plan["seeds"] == [2000, 2001, 2002, 2003, 2004]
    assert len({(task["case_index"], task["seed"], task["budget"]) for task in plan["tasks"]}) == 80
    assert not (tmp_path / "budget_tasks").exists()
    assert (tmp_path / "budget-work-items.txt").read_text().splitlines()[-1] == "79"
    with pytest.raises(FileExistsError):
        budget_study.build_plan(tmp_path)


def test_later_failure_cannot_censor_earlier_valid_budget_selection():
    spec = {"init_penalty": .1}
    low = [dict(spec=spec, eligible=True, raw_initializer_eligible=True, population_error=999.)]
    high = [dict(spec=spec, eligible=False, raw_initializer_eligible=True, population_error=0.)]
    arrays = [{"selected_coefficient": np.array([[2.]]), "raw_initializer": np.array([[1.]])}]
    x, y = np.ones((1, 1)), np.array([[2.]])
    early = budget_study.choose_candidate(low, arrays, x, y)
    late = budget_study.choose_candidate(high, arrays, x, y)
    assert early["success"] and early["validation_mse"] == 0
    assert not late["success"]
    initial = budget_study.choose_candidate(high, arrays, x, y, initializer=True)
    assert initial["success"]
    assert late["audit"][0]["exclusion_reason"] == "full_budget_fit_ineligible"


def test_selection_ignores_supplied_scores_and_deduplicates_initializers():
    records = [dict(spec={"init_penalty": .1}, eligible=True, raw_initializer_eligible=True, validation_mse=-100.),
               dict(spec={"init_penalty": .1}, eligible=True, raw_initializer_eligible=True, validation_mse=100.),
               dict(spec={"init_penalty": .3}, eligible=False, raw_initializer_eligible=True)]
    arrays = [{"selected_coefficient": np.array([[0.]]), "raw_initializer": np.array([[0.]])},
              {"selected_coefficient": np.array([[2.]]), "raw_initializer": np.array([[0.]])},
              {"selected_coefficient": np.array([[2.]]), "raw_initializer": np.array([[2.]])}]
    x, y = np.ones((1, 1)), np.array([[2.]])
    selected = budget_study.choose_candidate(records, arrays, x, y)
    assert selected["selected_index"] == 1
    initial = budget_study.choose_candidate(records, arrays, x, y, initializer=True)
    assert initial["selected_index"] == 2
    assert initial["audit"][1]["exclusion_reason"] == "duplicate_initializer_penalty"


def test_failed_model_retains_finite_arrays_only_for_diagnosis():
    source = SimpleNamespace(leading_left=np.eye(2), leading_right=np.eye(2))
    initial = SimpleNamespace(P=np.eye(2), Q=np.eye(2), d=np.array([2., 1.]), converged=True,
                              kkt_residual=1e-10, n_iter=np.array([3, 4]))
    model = SimpleNamespace(success_=False, status_="line_search_failed", coefficient_=np.eye(2),
                            last_coefficient_=2 * np.eye(2), source_=source, initialization_=initial,
                            checkpoints_={}, history_=[], validation_history_=[],
                            termination_reason_="backtracking_exhausted", n_iter_=500, selected_iteration_=100)
    record, arrays = budget_study.collect_outputs(model, {"init_penalty": .1}, (2, 2))
    assert not record["eligible"] and record["retained_failed_arrays_are_diagnostic_only"]
    assert record["raw_initializer_eligible"]
    np.testing.assert_array_equal(arrays["raw_initializer"], np.diag([2., 1.]))
    assert set(arrays) == {"raw_initializer", "selected_coefficient", "terminal_coefficient"}


def test_nonfinite_success_is_excluded_without_promoting_partial_output():
    model = SimpleNamespace(success_=True, status_="completed", coefficient_=np.full((2, 2), np.nan),
                            last_coefficient_=np.eye(2), checkpoints_={}, history_=[], validation_history_=[]) 
    record, arrays = budget_study.collect_outputs(model, {"init_penalty": .1}, (2, 2))
    assert not record["eligible"] and record["status"] == "invalid_successful_output"
    assert "selected_coefficient" not in arrays and "terminal_coefficient" in arrays


def test_checkpoint_reconstruction_uses_its_own_and_selected_chart():
    class Chart:
        def __init__(self, swap):
            self.rotation = np.array([[0., 1.], [1., 0.]]) if swap else np.eye(2)

        def reconstruct(self, state):
            return self.rotation, state, np.eye(2)

    # Coordinates, source frames and selected chart all differ, so using the
    # terminal chart for an earlier selected state would change its coefficient.
    source = SimpleNamespace(left=np.diag([-1., 1.]), right=np.array([[0., 1.], [1., 0.]]))
    checkpoint = SimpleNamespace(iteration=50, state=np.array([3., 2.]),
        selected_state=np.array([1., 4.]), chart=Chart(True), selected_chart=Chart(False),
        selected_iteration=25, chart_epoch=1, selected_chart_epoch=0,
        status="completed", termination_reason="max_iterations", endpoint_record=None)
    model = SimpleNamespace(success_=False, status_="line_search_failed", source_=source,
                            checkpoints_={50: checkpoint}, history_=[], validation_history_=[])
    record, arrays = budget_study.collect_outputs(model, {"init_penalty": .1}, (2, 2))
    np.testing.assert_array_equal(arrays["checkpoint_000050"], np.diag([-2., 3.]))
    np.testing.assert_array_equal(arrays["prefix_selected_000050"], np.array([[0., -1.], [4., 0.]]))
    assert record["checkpoints"][0]["selected_iteration"] == 25
    assert record["checkpoints"][0]["diagnostic_only"]
    assert not record["eligible"]


def test_validation_overflow_excludes_one_candidate_and_continues():
    records = [dict(spec={"init_penalty": .1}, eligible=True),
               dict(spec={"init_penalty": .3}, eligible=True)]
    arrays = [{"selected_coefficient": np.array([[1e308]])},
              {"selected_coefficient": np.array([[1.]])}]
    selected = budget_study.choose_candidate(records, arrays, np.array([[2.]]), np.array([[2.]]))
    assert selected["success"] and selected["selected_index"] == 1
    assert selected["audit"][0]["exclusion_reason"] == "nonfinite_validation_arithmetic"


def test_each_candidate_uses_a_fresh_model_without_truth_or_cache():
    created, calls = [], []

    class FakeModel:
        def fit(self, X, Y, **kwargs):
            calls.append((X, Y, kwargs))
            self.success_, self.status_ = True, "completed"
            self.coefficient_ = self.last_coefficient_ = np.ones((1, 1))
            return self

    def factory(spec, config):
        model = FakeModel()
        created.append(model)
        return model

    data = dict(X=np.ones((2, 1)), Y=np.ones((2, 1)), C0=np.ones((1, 1)),
                X_validation=np.ones((1, 1)), Y_validation=np.ones((1, 1)))
    for budget in (100, 500):
        record, _ = budget_study.fit_candidate(data, {"init_penalty": .1},
                                               {"iterations": budget}, model_factory=factory)
        assert record["eligible"] and record["fresh_fit"]
        assert record["fitting_call_seconds"] >= 0
    assert len(created) == 2 and created[0] is not created[1]
    assert all(set(kwargs) == {"source", "validation_data"} for _, _, kwargs in calls)


def test_atomic_archive_is_numeric_and_hash_checked(tmp_path):
    path = tmp_path / "fit.npz"
    digest = budget_study._atomic_npz(path, {"coefficient": np.eye(2)})
    assert len(digest) == 64
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["coefficient"], np.eye(2))


def test_slurm_guard_runs_before_plan_reads_or_fitting(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        budget_study.run_task(tmp_path, 0)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    with pytest.raises(RuntimeError, match="scratch2"):
        budget_study.run_task(Path("/not-scratch2/guard-only"), 0)


@pytest.mark.parametrize("seeds,budgets", [([], [100]), ([True], [100]), ([0], [100]),
                                          ([2000], [0]), ([2000], [250]), ([2000], [])])
def test_bad_plan_controls_are_rejected(tmp_path, seeds, budgets):
    with pytest.raises(ValueError):
        budget_study.build_plan(tmp_path, seeds=seeds, budgets=budgets)
