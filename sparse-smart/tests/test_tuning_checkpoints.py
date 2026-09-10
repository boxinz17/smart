"""Budget checkpoint eligibility and isolation; no simulation fits are rerun."""

import numpy as np
import pytest

from sparse_smart import FitFailure
from .test_tuning import _tuner, data, fake_estimator


def test_failed_continuations_retain_completed_checkpoint_and_exclude_better_partials(data, fake_estimator):
    X, Y, source = data
    fake_estimator.outcomes = {
        (1, .01, .01): dict(coefficient=1., status="completed", termination_reason="max_iterations"),
        (1, .02, .01): dict(coefficient=2.),
        (3, .01, .01): dict(success=False, coefficient=0., status="numerical_stagnation", n_iter=2),
        (3, .02, .01): dict(success=False, coefficient=.5, status="numerical_stagnation", n_iter=2),
    }
    tuned = _tuner(iteration_budgets=(1, 3)).fit(X, Y, source=source)
    assert tuned.success_ and tuned.selected_budget_ == 1 and tuned.selected_candidate_id_ == 0
    assert tuned.iteration_budgets_ == (1, 3) and tuned.n_candidates_ == 4
    assert list(tuned.checkpoints_) == [1]
    assert tuned.model_ is tuned.checkpoints_[1] is fake_estimator.instances[0]
    assert tuned.n_iter_ == 1 and tuned.model_.history_ == [{"iteration": 1, "budget": 1}]
    assert tuned.model_.diagnostics_["budget_marker"] == 1
    assert tuned.validation_history_ == tuned.model_.validation_history_
    assert tuned.best_score_ == .5 and tuned.diagnostics_["selected_checkpoint_status"] == "completed"
    np.testing.assert_array_equal(tuned.predict(X), fake_estimator.instances[0].predict(X))
    for record in tuned.selection_history_[2:]:
        assert not record["success"] and record["validation_mse"] is None
        assert record["partial_validation_mse"] < tuned.best_score_
    assert tuned.diagnostics_["selected_checkpoint_retained_after_failure"]
    assert tuned.diagnostics_["retained_earlier_checkpoint"]
    assert tuned.diagnostics_["selected_checkpoint_continuations"] == [dict(
        candidate_id=2, iteration_budget=3, success=False, status="numerical_stagnation",
        termination_reason="numerical_stagnation")]
    assert tuned.diagnostics_["budget_statuses"][-1] == dict(iteration_budget=3, success=False,
        successful_candidates=0, failed_candidates=2, best_candidate_id=None, best_validation_mse=None,
        best_selection_score=None)
    assert tuned.diagnostics_["successful_candidate_fits"] == 2
    assert tuned.diagnostics_["failed_candidate_fits"] == 2


def test_successful_extension_selects_new_minimum_but_retains_one_model_per_budget(data, fake_estimator):
    X, Y, source = data
    fake_estimator.outcomes = {
        (1, .01, .01): dict(coefficient=2.), (1, .02, .01): dict(coefficient=3.),
        (3, .01, .01): dict(coefficient=1.), (3, .02, .01): dict(coefficient=.5, selected_iteration=2),
    }
    tuned = _tuner(iteration_budgets=(1, 3)).fit(X, Y, source=source)
    assert tuned.selected_budget_ == 3 and tuned.selected_candidate_id_ == 3
    assert tuned.selected_iteration_ == 2 and tuned.n_iter_ == 3
    assert tuned.best_score_ == .125 and tuned.best_params_["penalty_u"] == .02
    assert list(tuned.checkpoints_) == [1, 3]
    assert tuned.checkpoints_[1] is fake_estimator.instances[0]
    assert tuned.checkpoints_[3] is tuned.model_ is fake_estimator.instances[3]
    assert len(tuned.checkpoints_) == 2 < tuned.diagnostics_["successful_candidate_fits"] == 4
    assert tuned.diagnostics_["selected_checkpoint_continuations"] == []
    assert not tuned.diagnostics_["retained_earlier_checkpoint"]
    assert [r["candidate_id"] for r in tuned.selection_history_] == [0, 1, 2, 3]
    assert [r["grid_candidate_id"] for r in tuned.selection_history_] == [0, 1, 0, 1]
    assert [r["iteration_budget"] for r in tuned.selection_history_] == [1, 1, 3, 3]
    assert tuned.selection_history_[0]["params"] == tuned.selection_history_[2]["params"]
    assert "iteration_budget" not in tuned.best_params_


def test_equal_scores_prefer_earlier_budget_then_grid_order(data, fake_estimator):
    X, Y, source = data
    fake_estimator.outcomes = {(.01, .01): dict(coefficient=1.), (.02, .01): dict(coefficient=1.)}
    tuned = _tuner(iteration_budgets=(1, 2, 3)).fit(X, Y, source=source)
    assert tuned.selected_budget_ == 1 and tuned.selected_candidate_id_ == 0
    assert tuned.best_params_["penalty_u"] == .01
    assert [tuned.checkpoints_[b] for b in (1, 2, 3)] == fake_estimator.instances[::2]
    assert [r["candidate_id"] for r in tuned.diagnostics_["selected_checkpoint_continuations"]] == [2, 4]
    assert not tuned.diagnostics_["selected_checkpoint_retained_after_failure"]


@pytest.mark.parametrize("explicit", [False, True])
def test_every_budget_uses_identical_training_and_validation_data_without_refit(data, fake_estimator, explicit):
    X, Y, source = data
    validation = (X[:3] + 100., Y[:3] + 50.) if explicit else None
    tuned = _tuner(iteration_budgets=(1, 3)).fit(X, Y, source=source, validation_data=validation)
    if explicit:
        expected_X, expected_Y = X, Y
        expected_Xv, expected_Yv = validation
        assert tuned.train_indices_ is None and tuned.validation_indices_ is None
    else:
        expected_X, expected_Y = X[tuned.train_indices_], Y[tuned.train_indices_]
        expected_Xv, expected_Yv = X[tuned.validation_indices_], Y[tuned.validation_indices_]
        assert not set(tuned.train_indices_) & set(tuned.validation_indices_)
    assert len(fake_estimator.instances) == 4
    for model, budget in zip(fake_estimator.instances, (1, 1, 3, 3)):
        np.testing.assert_array_equal(model.training_X, expected_X)
        np.testing.assert_array_equal(model.training_Y, expected_Y)
        np.testing.assert_array_equal(model.validation_X, expected_Xv)
        np.testing.assert_array_equal(model.validation_Y, expected_Yv)
        assert model.source is source and model.options["iterations"] == budget
    assert tuned.diagnostics_["checkpoint_execution"] == "independent_fits"
    assert not tuned.diagnostics_["refit_on_all_data"]


def test_later_exception_does_not_discard_successful_budget(data, fake_estimator):
    X, Y, source = data
    fake_estimator.outcomes = {
        (3, .01, .01): dict(exception=FloatingPointError("numerical failure")),
        (3, .02, .01): dict(exception=FloatingPointError("numerical failure")),
    }
    tuned = _tuner(iteration_budgets=(1, 3)).fit(X, Y, source=source)
    assert tuned.success_ and tuned.selected_budget_ == 1
    assert all(record["status"] == "FloatingPointError" for record in tuned.selection_history_[2:])
    assert tuned.diagnostics_["selected_checkpoint_retained_after_failure"]


def test_fresh_fit_clears_checkpoints_and_all_failed_budgets_remain_ineligible(data, fake_estimator):
    X, Y, source = data
    tuned = _tuner(iteration_budgets=(1, 3)).fit(X, Y, source=source)
    assert tuned.checkpoints_
    fake_estimator.outcomes = {(.01, .01): dict(success=False), (.02, .01): dict(success=False)}
    tuned.fit(X, Y, source=source)
    assert not tuned.success_ and tuned.checkpoints_ == {}
    assert tuned.selected_budget_ is None and tuned.selected_candidate_id_ is None
    assert not hasattr(tuned, "coefficient_") and tuned.model_ is None
    assert tuned.diagnostics_["failed_candidate_fits"] == 4
    assert all(not row["success"] for row in tuned.diagnostics_["budget_statuses"])
    with pytest.raises(FitFailure):
        tuned.predict(X)


@pytest.mark.parametrize("budgets", [(), (1, 1, 3), (3, 1), (1, 2), (0, 3), (-1, 3),
    (1., 3), (True, 3), (np.nan, 3), (np.inf, 3), 3, True, "13"])
def test_invalid_budget_schedule_is_rejected_before_fitting(data, fake_estimator, budgets):
    X, Y, source = data
    with pytest.raises(ValueError, match="iteration"):
        _tuner(iteration_budgets=budgets).fit(X, Y, source=source)
    assert not fake_estimator.instances


@pytest.mark.parametrize("iterations", [0, 3, np.int64(3)])
def test_none_schedule_preserves_single_budget_including_zero_updates(data, fake_estimator, iterations):
    X, Y, source = data
    tuned = _tuner(iterations=iterations).fit(X, Y, source=source)
    assert tuned.iteration_budgets_ == (int(iterations),)
    assert tuned.selected_budget_ == iterations and tuned.n_candidates_ == 2
    assert list(tuned.checkpoints_) == [iterations]
    assert len(fake_estimator.instances) == 2
    assert tuned.diagnostics_["selected_checkpoint_continuations"] == []


def test_numpy_integer_budget_schedule_is_valid(data, fake_estimator):
    X, Y, source = data
    tuned = _tuner(iteration_budgets=np.array([1, 3])).fit(X, Y, source=source)
    assert tuned.iteration_budgets_ == (1, 3)
