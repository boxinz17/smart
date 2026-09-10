"""Extra validation points select states without creating budget prefixes."""

from copy import deepcopy
import pickle

import numpy as np
import pytest

from sparse_smart import FitFailure, SparseSMARTTuner
from sparse_smart import estimator
from .test_trajectory_checkpoints import install_solver, small_data, small_model


def _levels(count, changes):
    values = np.full(count, 1.5)
    values[0] = 2.
    for iteration, value in changes.items():
        values[iteration] = value
    return values


def _tuner(**options):
    prototype = small_model()
    defaults = dict(rank=prototype.rank, source_rank=prototype.source_rank,
        sparsity=prototype.sparsity, margins=prototype.margins,
        init_penalties=(.01,), penalties_u=(.002,), penalties_v=(.002,),
        support_limits=(0, 0), refinement_solver="chart", stationarity_tol=None,
        iterations=500, iteration_budgets=(250, 500), checkpoint_interval=250,
        checkpoint_execution="continuous", validation_iterations=(10, 25, 50))
    defaults.update(options)
    return SparseSMARTTuner(**defaults)


def test_extra_minima_are_retained_compactly_without_extra_checkpoints(small_data, monkeypatch):
    X, Y, source = small_data
    calls = install_solver(monkeypatch, levels=_levels(251, {10: .4, 25: .2, 50: .8}))
    fit = small_model(iterations=250, checkpoint_iterations=(0, 250), validation_interval=250,
        validation_iterations=(10, 25, 50, 500)).fit(
            X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert len(calls) == 1 and fit.success_
    assert fit.checkpoint_iterations_ == (0, 250)
    assert [row["iteration"] for row in fit.validation_history_] == [0, 10, 25, 50, 250]
    assert fit.diagnostics_["validation_iterations"] == [0, 10, 25, 50, 250]
    assert fit.diagnostics_["validation_iterations_requested"] == (10, 25, 50, 500)
    assert tuple(fit.best_validation_states_) == (10, 25)
    assert fit.selected_iteration_ == fit.checkpoints_[250].selected_iteration == 25
    assert fit.checkpoint_model(250).selected_iteration_ == 25
    np.testing.assert_array_equal(fit.checkpoints_[250].selected_state, fit.best_validation_states_[25])
    assert all(any(row.iteration == t for row in fit.history_) for t in fit.best_validation_states_)
    assert fit.checkpoint_model(0).best_validation_states_ == {}
    assert fit._checkpoint_summary(0)["diagnostics"]["validation_iterations"] == [0]
    with pytest.raises(TypeError):
        fit.best_validation_states_[50] = fit.last_state_
    with pytest.raises(ValueError):
        fit.best_validation_states_[10][0] = 99.
    with pytest.raises(FitFailure, match="checkpoint_unavailable"):
        fit.checkpoint_model(25)


def test_extra_validation_ties_keep_earliest_and_do_not_store_loser(small_data, monkeypatch):
    X, Y, source = small_data
    install_solver(monkeypatch, levels=_levels(251, {10: .2, 25: .2}))
    fit = small_model(iterations=250, checkpoint_iterations=(250,), validation_interval=250,
        validation_iterations=(10, 25)).fit(
            X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert fit.selected_iteration_ == 10
    assert tuple(fit.best_validation_states_) == (10,)
    assert fit.validation_history_[2]["selection_comparison"]["loss_difference"] == 0.


def test_budget_views_retain_early_winner_and_exclude_future_validation_states(small_data, monkeypatch):
    X, Y, source = small_data
    calls = install_solver(monkeypatch, levels=_levels(501, {10: .4, 25: .2, 300: .1}))
    fit = _tuner(validation_iterations=(10, 25, 300, 1000)).fit(
        X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert len(calls) == 1 and fit.success_
    path = fit.trajectory_models_[0]
    assert path.checkpoint_iterations_ == (0, 250, 500)
    assert fit.diagnostics_["validation_schedule"] == [0, 10, 25, 250, 300, 500]
    assert fit.checkpoints_[250].selected_iteration_ == 25
    assert fit.checkpoints_[500].selected_iteration_ == 300
    assert tuple(path.best_validation_states_) == (10, 25, 300)
    assert tuple(fit.checkpoints_[250].best_validation_states_) == (10, 25)
    assert fit.checkpoints_[250].diagnostics_["validation_iterations"] == [0, 10, 25, 250]
    assert not np.shares_memory(path.best_validation_states_[25],
                               fit.checkpoints_[250].best_validation_states_[25])


@pytest.mark.parametrize("field", ["objective", "smooth_loss", "penalty_value", "raw_gradient_norm"])
def test_failed_nonfinite_record_never_becomes_early_winner(small_data, monkeypatch, field):
    X, Y, source = small_data
    install_solver(monkeypatch, stop=25, status="numerical_failure", invalid_field=field,
                   levels=_levels(26, {10: .4, 25: .1}))
    fit = small_model(iterations=250, checkpoint_iterations=(0, 250), validation_interval=250,
        validation_iterations=(10, 25)).fit(
            X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert not fit.success_ and fit.selected_iteration_ == 10
    assert tuple(fit.best_validation_states_) == (10,)
    assert [row["iteration"] for row in fit.validation_history_] == [0, 10]
    assert fit.checkpoint_iterations_ == (0,)


def test_failure_after_positive_checkpoint_retains_early_minimum_without_cap_coverage(small_data, monkeypatch):
    X, Y, source = small_data
    install_solver(monkeypatch, stop=260, status="numerical_stagnation",
                   levels=_levels(261, {10: .2, 260: .1}))
    fit = _tuner().fit(X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert fit.success_ and fit.selected_iteration_ == 10
    assert [row["budget_reached"] for row in fit.selection_history_] == [True, False]
    assert all(row["trajectory_checkpoint_iteration"] == 250 for row in fit.selection_history_)
    assert fit.trajectory_models_[0].checkpoint_iterations_ == (0, 250)


def test_nonfinite_early_validation_does_not_rescue_failure_before_first_checkpoint(small_data, monkeypatch):
    X, Y, source = small_data
    install_solver(monkeypatch, levels=_levels(501, {10: .4, 25: .1}))
    original_mean, evaluations = np.mean, []

    def validation_mean(value, *args, **kwargs):
        if np.shape(value) == (4, 2):
            evaluations.append(1)
            if len(evaluations) == 3:
                return float("inf")
        return original_mean(value, *args, **kwargs)

    monkeypatch.setattr(estimator.np, "mean", validation_mean)
    fit = _tuner().fit(X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert not fit.success_ and not fit.checkpoints_
    path = fit.trajectory_models_[0]
    assert path.n_iter_ == 25 and path.checkpoint_iterations_ == (0,)
    assert tuple(path.best_validation_states_) == (10,)
    assert [row["iteration"] for row in path.validation_history_] == [0, 10]
    assert all(not row["success"] and not row["budget_reached"] for row in fit.selection_history_)


@pytest.mark.parametrize("schedule", [None, 10, (True,), (-1,), (1.,), (2, 1), (1, 1)])
def test_invalid_extra_validation_schedule_rejected_before_optimization(small_data, monkeypatch, schedule):
    X, Y, source = small_data
    calls = install_solver(monkeypatch)
    for model in (small_model(validation_iterations=schedule), _tuner(validation_iterations=schedule)):
        with pytest.raises(ValueError, match="validation_iterations"):
            model.fit(X, Y, source=source)
    assert calls == []


def test_independent_tuner_rejects_extra_schedule(small_data, monkeypatch):
    X, Y, source = small_data
    calls = install_solver(monkeypatch)
    with pytest.raises(ValueError, match="validation_iterations requires continuous"):
        _tuner(checkpoint_execution="independent", checkpoint_interval=None).fit(X, Y, source=source)
    assert calls == []


def test_successful_off_schedule_terminal_is_evaluated_without_creating_checkpoint(small_data, monkeypatch):
    X, Y, source = small_data
    install_solver(monkeypatch, levels=_levels(7, {2: .4, 6: .1}))
    fit = small_model(iterations=6, checkpoint_iterations=(0, 5), validation_interval=5,
        validation_iterations=(2,)).fit(
            X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert fit.selected_iteration_ == 6 and fit.checkpoint_iterations_ == (0, 5)
    assert [row["iteration"] for row in fit.validation_history_] == [0, 2, 5, 6]
    assert fit.checkpoint_model(5).selected_iteration_ == 2


@pytest.mark.parametrize("copy_method", [deepcopy, lambda model: pickle.loads(pickle.dumps(model))],
                         ids=["deepcopy", "pickle"])
@pytest.mark.parametrize("extra_schedule", [(), (10, 25)], ids=["default", "extra-winners"])
def test_fitted_models_and_prefixes_round_trip_with_readonly_independent_states(
        small_data, monkeypatch, copy_method, extra_schedule):
    X, Y, source = small_data
    calls = install_solver(monkeypatch, levels=_levels(251, {10: .4, 25: .2}))
    fit = small_model(iterations=250, checkpoint_iterations=(0, 250), validation_interval=250,
        validation_iterations=extra_schedule).fit(
            X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    expected = fit.predict(X).copy()
    for original in (fit, fit.checkpoint_model(250)):
        restored = copy_method(original)
        assert restored is not original and restored.success_
        assert restored.selected_iteration_ == original.selected_iteration_
        assert restored.history_ == original.history_
        assert restored.validation_history_ == original.validation_history_
        assert tuple(restored.best_validation_states_) == extra_schedule
        np.testing.assert_array_equal(restored.predict(X), expected)
        np.testing.assert_array_equal(restored.checkpoint_model(250).predict(X), expected)
        assert restored.checkpoint_model(0).best_validation_states_ == {}
        with pytest.raises(TypeError):
            restored.best_validation_states_[123] = restored.last_state_
        for t, state in restored.best_validation_states_.items():
            assert not state.flags.writeable
            assert not np.shares_memory(state, original.best_validation_states_[t])
            np.testing.assert_array_equal(state, original.best_validation_states_[t])
            with pytest.raises(ValueError):
                state[0] = 999.
        restored.coefficient_[:] = -999.
        np.testing.assert_array_equal(original.predict(X), expected)
    assert len(calls) == 1
