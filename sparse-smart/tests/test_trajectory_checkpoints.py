"""Continuous finite-prefix capture, sparse validation, and independent views."""
from dataclasses import replace

import numpy as np
import pytest

from sparse_smart import SparseSMART, ExactSource, FitFailure, Margins, PracticalCalibration
from sparse_smart import estimator
from sparse_smart.solver import _record, _result
from .test_estimator import example, model


@pytest.fixture
def small_data():
    rng = np.random.default_rng(26)
    X = rng.normal(size=(30, 3))
    Y = np.column_stack((2 * X[:, 0], np.zeros(len(X))))
    return X, Y, ExactSource(np.eye(3)[:, :2], np.eye(2))


def small_model(**kwargs):
    options = dict(rank=1, source_rank=2, sparsity=(1, 1), margins=Margins(.05, 6., .01),
        calibration=PracticalCalibration(.01, .002, 5., (0, 0)), iterations=6,
        refinement_solver="chart")
    options.update(kwargs)
    return SparseSMART(**options)


def install_solver(monkeypatch, *, stop=None, status="completed", invalid_field=None, move=.01, levels=None):
    calls = []

    def fake(chart, initial, design, response, *, iterations, iterate_callback, loss_offset=0., **kwargs):
        calls.append(dict(iterations=iterations, initial=initial.copy()))
        endpoint = iterations if stop is None else min(iterations, stop)
        history = []
        state = initial.copy()
        for iteration in range(endpoint + 1):
            state = initial.copy()
            state[chart.d_slice] -= move * iteration
            if levels is not None:
                state[chart.d_slice] = levels[iteration]
            smooth = chart.loss(state, design, response)
            residual = 0. if status == "converged" and iteration == endpoint else 1.
            record = _record(chart, state, iteration, smooth, 0., 5., move if iteration else 0.,
                             [], loss_offset, (residual, 1., 5., None))
            if invalid_field is not None and iteration == endpoint:
                record = replace(record, **{invalid_field: float("inf")})
            history.append(record)
            if iterate_callback is not None:
                iterate_callback(iteration, state.copy(), record)
        reason = "stationarity" if status == "converged" else "max_iterations" if status == "completed" else status
        return _result(state, status, "controlled solver result", endpoint, history, termination_reason=reason)

    monkeypatch.setattr(estimator, "refine", fake)
    return calls


@pytest.mark.parametrize("solver", ["chart", "anchor_projected"])
def test_real_continuous_prefix_equals_independent_short_fit_and_defaults(solver):
    X, Y, U, V, _ = example()
    source = ExactSource(U, V)
    options = dict(iterations=8, refinement_solver=solver,
                   calibration=PracticalCalibration(.003, .001, 10., (10, 10)))
    baseline = model(**options).fit(X, Y, source=source, validation_data=(X[80:], Y[80:]))
    trajectory = model(**options, checkpoint_iterations=(8, 0, 3)).fit(
        X, Y, source=source, validation_data=(X[80:], Y[80:]))
    shorter = model(**{**options, "iterations": 3}).fit(
        X, Y, source=source, validation_data=(X[80:], Y[80:]))
    assert baseline.success_ and trajectory.success_ and shorter.success_
    assert baseline.checkpoints_ == {} and baseline.checkpoint_iterations_ == ()
    np.testing.assert_array_equal(trajectory.coefficient_, baseline.coefficient_)
    assert trajectory.history_ == baseline.history_
    assert trajectory.validation_history_ == baseline.validation_history_
    assert trajectory.checkpoint_iterations_ == (0, 3, 8)
    checkpoint = trajectory.checkpoint_model(3)
    assert checkpoint.n_iter_ == checkpoint.iterations == 3 and checkpoint.success_
    assert checkpoint.status_ == shorter.status_ and checkpoint.termination_reason_ == shorter.termination_reason_
    assert checkpoint.history_ == shorter.history_
    assert checkpoint.validation_history_ == shorter.validation_history_
    assert trajectory.diagnostics_["line_search_strategy"] == "reset_initial_inverse"
    assert checkpoint.diagnostics_["line_search_strategy"] == "reset_initial_inverse"
    assert checkpoint.best_validation_loss_ == shorter.best_validation_loss_
    assert checkpoint.selected_iteration_ == shorter.selected_iteration_
    for attr in ("state_", "last_state_", "coefficient_", "last_coefficient_"):
        np.testing.assert_array_equal(getattr(checkpoint, attr), getattr(shorter, attr))
    assert checkpoint.diagnostics_["last_projected_gradient_norm"] == shorter.diagnostics_["last_projected_gradient_norm"]


def test_single_solver_call_captures_endpoint_and_sparse_validation_selection(small_data, monkeypatch):
    X, Y, source = small_data
    calls = install_solver(monkeypatch)
    fitted = small_model(checkpoint_iterations=(2, 5), validation_interval=3).fit(
        X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert len(calls) == 1 and calls[0]["iterations"] == 6
    assert len(fitted.history_) == 7
    assert [row["iteration"] for row in fitted.validation_history_] == [0, 2, 3, 5, 6]
    assert fitted.checkpoint_iterations_ == (2, 5)
    view = fitted.checkpoint_model(5)
    assert len(calls) == 1
    assert view.n_iter_ == view.selected_iteration_ == 5
    assert [row["iteration"] for row in view.validation_history_] == [0, 2, 3, 5]
    assert view.best_validation_loss_ == pytest.approx(np.mean((view.predict(X[:4])) ** 2))
    snapshot = fitted.checkpoints_[2]
    assert snapshot.history_length == 3 and snapshot.validation_history_length == 2
    assert not snapshot.state.flags.writeable and not snapshot.selected_state.flags.writeable
    assert not hasattr(snapshot, "coefficient_")


def test_sparse_validation_ties_keep_initializer_with_distinct_terminal_iteration(small_data, monkeypatch):
    X, Y, source = small_data
    install_solver(monkeypatch, move=0.)
    fitted = small_model(checkpoint_iterations=(2, 5), validation_interval=3).fit(
        X, Y, source=source, validation_data=(X[:4], Y[:4]))
    assert fitted.selected_iteration_ == 0
    for iteration in (2, 5):
        view = fitted.checkpoint_model(iteration)
        assert view.selected_iteration_ == 0 and view.n_iter_ == iteration
        assert view.best_validation_loss_ == fitted.validation_history_[0]["loss"]
        np.testing.assert_array_equal(view.state_, view.initial_state_)


def test_unsampled_better_iterates_do_not_enter_sparse_validation_selection(small_data, monkeypatch):
    X, Y, source = small_data
    install_solver(monkeypatch, levels=(2., .1, 1.5, 1.5, .05, 1.6, 1.7))
    fitted = small_model(checkpoint_iterations=(2, 5), validation_interval=3).fit(
        X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    # Iterations 1 and 4 predict much nearer the zero validation response, but
    # neither is eligible. The equal minima at eligible iterations 2/3 select 2.
    assert [row["iteration"] for row in fitted.validation_history_] == [0, 2, 3, 5, 6]
    assert fitted.selected_iteration_ == fitted.checkpoint_model(5).selected_iteration_ == 2
    assert fitted.validation_history_[1]["loss"] == fitted.validation_history_[2]["loss"]


@pytest.mark.parametrize("status", ["line_search_failed", "numerical_stagnation"])
def test_later_failure_preserves_successful_finite_prefix(small_data, monkeypatch, status):
    X, Y, source = small_data
    install_solver(monkeypatch, stop=4, status=status)
    fitted = small_model(checkpoint_iterations=(2, 6), validation_interval=3).fit(
        X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert not fitted.success_ and fitted.status_ == status
    assert fitted.checkpoint_iterations_ == (2,)
    view = fitted.checkpoint_model(2)
    assert view.success_ and view.status_ == "completed" and view.n_iter_ == 2
    assert view.termination_reason_ == "max_iterations"
    assert view.diagnostics_["status"] == "completed"
    assert not view.diagnostics_["precision_limited"]
    assert len(view.history_) == 3 and len(fitted.history_) == 5
    with pytest.raises(FitFailure, match="checkpoint_unavailable"):
        fitted.checkpoint_model(4)
    with pytest.raises(FitFailure):
        fitted.predict(X)
    assert np.isfinite(view.predict(X)).all()


@pytest.mark.parametrize("field", ["raw_gradient_norm", "objective", "smooth_loss"])
def test_accepted_numerical_failure_record_is_not_a_successful_checkpoint(small_data, monkeypatch, field):
    X, Y, source = small_data
    install_solver(monkeypatch, stop=2, status="numerical_failure", invalid_field=field)
    fitted = small_model(checkpoint_iterations=(1, 2, 6)).fit(X, Y, source=source)
    assert not fitted.success_ and fitted.checkpoint_iterations_ == (1,)
    assert fitted.checkpoint_model(1).success_
    with pytest.raises(FitFailure):
        fitted.checkpoint_model(2)


def test_successful_terminal_convergence_is_evaluated_and_captured_off_schedule(small_data, monkeypatch):
    X, Y, source = small_data
    calls = install_solver(monkeypatch, stop=2, status="converged")
    fitted = small_model(checkpoint_iterations=(4, 6), validation_interval=5, stationarity_tol=1e-6).fit(
        X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert len(calls) == 1 and fitted.checkpoint_iterations_ == (2,)
    assert [row["iteration"] for row in fitted.validation_history_] == [0, 2]
    assert fitted.selected_iteration_ == 2
    view = fitted.checkpoint_model(2)
    assert view.success_ and view.status_ == "converged"
    assert view.optimization_converged_ and view.converged_
    assert view.diagnostics_["selected_converged"]


def test_initial_convergence_and_checkpoint_without_validation(small_data, monkeypatch):
    X, Y, source = small_data
    install_solver(monkeypatch, stop=0, status="converged")
    fitted = small_model(checkpoint_iterations=(0, 6), stationarity_tol=1e-6).fit(X, Y, source=source)
    assert fitted.checkpoint_iterations_ == (0,) and fitted.checkpoints_[0].status == "converged"
    view = fitted.checkpoint_model(0)
    assert view.n_iter_ == 0 and view.best_validation_loss_ is None
    assert view.validation_history_ == [] and view.selected_iteration_ == 0
    np.testing.assert_array_equal(view.state_, view.last_state_)


def test_checkpoint_views_do_not_alias_parent_or_each_other(small_data, monkeypatch):
    X, Y, source = small_data
    install_solver(monkeypatch)
    parent = small_model(checkpoint_iterations=(2, 6)).fit(X, Y, source=source, validation_data=(X[:4], Y[:4]))
    a, b = parent.checkpoint_model(2), parent.checkpoint_model(2)
    assert not np.shares_memory(a.state_, b.state_) and not np.shares_memory(a.last_state_, parent.last_state_)
    assert not np.shares_memory(a.checkpoints_[2].state, parent.checkpoints_[2].state)
    saved = b.predict(X).copy()
    a.coefficient_[:] = -999
    a.source_.left[:] = 0
    a.chart_.center_u[:] = 0
    a.diagnostics_["calibration"]["extra_mutation"] = True
    a.validation_history_[0]["loss"] = -1
    a.history_.clear()
    a.checkpoints_.clear()
    np.testing.assert_array_equal(b.predict(X), saved)
    assert parent.checkpoints_ and parent.history_ and parent.validation_history_[0]["loss"] >= 0
    assert "extra_mutation" not in parent.diagnostics_["calibration"]


def test_fit_resets_checkpoints_and_failed_initialization_cannot_reuse_them(small_data, monkeypatch):
    X, Y, source = small_data
    install_solver(monkeypatch)
    fitted = small_model(checkpoint_iterations=(2,)).fit(X, Y, source=source)
    assert fitted.checkpoints_
    fitted.initialization_spectrum = "reject"
    fitted.fit(X, np.zeros_like(Y), source=source)
    assert not fitted.success_ and fitted.checkpoints_ == {} and fitted.checkpoint_iterations_ == ()
    with pytest.raises(FitFailure):
        fitted.checkpoint_model(2)


def test_late_validation_exception_preserves_progress_and_prior_checkpoint(small_data, monkeypatch):
    X, Y, source = small_data
    calls = install_solver(monkeypatch)
    original_mean = np.mean
    evaluations = []

    def validation_mean(value, *args, **kwargs):
        if np.shape(value) == (4, 2):
            evaluations.append(len(evaluations))
            if len(evaluations) == 3:
                return float("inf")
        return original_mean(value, *args, **kwargs)

    monkeypatch.setattr(estimator.np, "mean", validation_mean)
    fitted = small_model(checkpoint_iterations=(2, 4, 6), validation_interval=2)
    with pytest.raises(FloatingPointError, match="Nonfinite validation"):
        fitted.fit(X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert len(calls) == 1 and len(evaluations) == 3
    assert fitted.n_iter_ == fitted.history_[-1].iteration == 4
    assert len(fitted.history_) == 5 and fitted.checkpoint_iterations_ == (2,)
    assert not fitted.success_
    checkpoint = fitted.checkpoint_model(2)
    assert checkpoint.success_ and checkpoint.n_iter_ == 2
    assert len(checkpoint.history_) == 3 and checkpoint.selected_iteration_ == 2
    assert [row["iteration"] for row in checkpoint.validation_history_] == [0, 2]
    assert np.isfinite(checkpoint.predict(X)).all()


@pytest.mark.parametrize("options", [
    {"checkpoint_iterations": (2, 2)}, {"checkpoint_iterations": (-1,)},
    {"checkpoint_iterations": (7,)}, {"checkpoint_iterations": (True,)},
    {"checkpoint_iterations": (1.,)}, {"checkpoint_iterations": 2},
    {"validation_interval": 0}, {"validation_interval": True}, {"validation_interval": 2.5},
])
def test_invalid_capture_options_fail_before_solver(small_data, monkeypatch, options):
    X, Y, source = small_data
    calls = install_solver(monkeypatch)
    with pytest.raises(ValueError):
        small_model(**options).fit(X, Y, source=source)
    assert calls == []


@pytest.mark.parametrize("iteration", [True, -1, 1.5])
def test_invalid_checkpoint_lookup_is_rejected(iteration):
    with pytest.raises(ValueError):
        small_model().checkpoint_model(iteration)
