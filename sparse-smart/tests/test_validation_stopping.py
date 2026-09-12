"""Validation patience stops computation without asserting stationarity."""
import numpy as np
import pytest

from sparse_smart import ExactSource, PracticalCalibration
from sparse_smart import estimator
from sparse_smart.stopping import ValidationStopping, ValidationStopRequest
from sparse_smart.solver import _record, _result, refine
from sparse_smart.anchor_solver import refine_anchor_projected, _to_z
from .test_estimator import example, model
from .test_trajectory_checkpoints import small_data, small_model, install_solver
from .test_anchor_solver import calibration, interior_problem
from sparse_smart import Margins


def test_iteration_patience_minimum_and_late_improvement():
    policy = ValidationStopping(300, 500, .001)
    for t, loss in ((0, 1.), (250, 1.), (400, 1.), (450, .99), (500, .99), (700, .99)):
        assert not policy.observe(t, loss)["stop_requested"]
    check = policy.observe(750, .99)
    assert check["stop_requested"] and check["iterations_since_improvement"] == 300
    assert policy.snapshot()["stop_iteration"] == 750
    assert policy.snapshot()["last_significant_iteration"] == 450


def test_small_improvements_accumulate_and_actual_minimum_is_not_thresholded():
    policy = ValidationStopping(3, 0, .001)
    assert policy.observe(0, 1.)["significant_improvement"]
    assert not policy.observe(1, .9996)["significant_improvement"]
    assert not policy.observe(2, .9992)["significant_improvement"]
    assert policy.observe(3, .9988)["significant_improvement"]
    assert not policy.stopped
    policy.observe(4, .9985)
    policy.observe(5, .9985)
    assert policy.observe(6, .9985)["stop_requested"]
    assert policy.best_loss == .9985 and policy.best_iteration == 4
    assert policy.reference_loss == .9988 and policy.last_significant_iteration == 3


def test_zero_loss_ties_do_not_reset_patience_and_terminal_observation_cannot_stop():
    policy = ValidationStopping(1, 0, 0.)
    policy.observe(0, 0.)
    assert not policy.observe(1, 0., allow_stop=False)["stop_requested"]
    assert policy.last_significant_iteration == 0
    assert policy.observe(2, 0.)["stop_requested"]


@pytest.mark.parametrize("options", [
    {"validation_patience": 0}, {"validation_patience": -1},
    {"validation_patience": True}, {"validation_patience": 3.5},
    {"validation_min_iterations": -1}, {"validation_min_iterations": False},
    {"validation_min_iterations": 2.5},
    {"validation_min_relative_improvement": -1.},
    {"validation_min_relative_improvement": 1.},
    {"validation_min_relative_improvement": np.inf},
    {"validation_min_relative_improvement": np.nan},
    {"validation_min_relative_improvement": True},
])
def test_invalid_policy_rejected_before_initialization(small_data, options):
    X, Y, source = small_data
    with pytest.raises(ValueError, match="validation_"):
        small_model(**options).fit(X, Y, source=source, validation_data=(X, Y))


def test_enabled_policy_requires_explicit_validation_data(small_data):
    X, Y, source = small_data
    with pytest.raises(ValueError, match="validation_data is required"):
        small_model(validation_patience=3).fit(X, Y, source=source)


@pytest.mark.parametrize("solver", ["chart", "anchor_projected"])
def test_real_solver_stops_at_check_preserves_work_best_and_terminal(solver):
    X, Y, U, V, _ = example()
    source = ExactSource(U, V)
    XV, YV = np.zeros((7, X.shape[1])), np.ones((7, Y.shape[1]))
    options = dict(iterations=10, refinement_solver=solver, validation_interval=2,
                   calibration=PracticalCalibration(.003, .001, 10., (10, 10)))
    fitted = model(**options, checkpoint_iterations=(2, 10), validation_patience=3,
                   validation_min_iterations=4).fit(X, Y, source=source, validation_data=(XV, YV))
    shorter = model(**{**options, "iterations": 4}).fit(
        X, Y, source=source, validation_data=(XV, YV))
    assert fitted.success_ and fitted.status_ == "completed"
    assert fitted.termination_reason_ == "validation_stop"
    assert fitted.n_iter_ == 4 and len(fitted.history_) == 5
    assert fitted.selected_iteration_ == 0
    assert not fitted.converged_ and not fitted.optimization_converged_
    assert fitted.history_ == shorter.history_
    np.testing.assert_array_equal(fitted.last_state_, shorter.last_state_)
    np.testing.assert_array_equal(fitted.state_, fitted.initial_state_)
    assert not np.array_equal(fitted.state_, fitted.last_state_)
    assert [row["iteration"] for row in fitted.validation_history_] == [0, 2, 4]
    assert fitted.checkpoint_iterations_ == (2, 4)  # off-schedule terminal is retained
    assert fitted.validation_stopping_["stopped"]
    assert fitted.validation_stopping_["stop_iteration"] == 4
    earlier, terminal = fitted.checkpoint_model(2), fitted.checkpoint_model(4)
    assert earlier.termination_reason_ == "max_iterations"
    assert not earlier.validation_stopping_["stopped"]
    assert earlier.validation_stopping_["last_check_iteration"] == 2
    assert not fitted._checkpoint_summary(2)["diagnostics"]["validation_stopping"]["stopped"]
    assert terminal.termination_reason_ == "validation_stop" and not terminal.optimization_converged_
    assert terminal.validation_stopping_["stopped"]
    if solver == "anchor_projected":
        assert fitted.numerical_work_["totals"]["u"]["calls"] > 0
        assert fitted.numerical_work_["totals"]["u"]["calls"] == shorter.numerical_work_["totals"]["u"]["calls"]


@pytest.mark.parametrize("solver", ["chart", "anchor_projected"])
def test_disabled_policy_matches_existing_fixed_budget(solver):
    X, Y, U, V, _ = example()
    source = ExactSource(U, V)
    options = dict(iterations=5, refinement_solver=solver,
                   calibration=PracticalCalibration(.003, .001, 10., (10, 10)))
    default = model(**options).fit(X, Y, source=source, validation_data=(X, Y))
    disabled = model(**options, validation_patience=None, validation_min_iterations=0,
                     validation_min_relative_improvement=.9).fit(X, Y, source=source, validation_data=(X, Y))
    assert default.success_ and disabled.success_ and default.n_iter_ == disabled.n_iter_ == 5
    assert default.history_ == disabled.history_
    assert default.validation_history_ == disabled.validation_history_
    np.testing.assert_array_equal(default.coefficient_, disabled.coefficient_)
    assert not disabled.validation_stopping_["enabled"] and not disabled.validation_stopping_["stopped"]


def test_true_best_survives_subthreshold_gain_and_unscheduled_terminal(small_data, monkeypatch):
    X, Y, source = small_data
    levels = np.sqrt([1., .9996, .9992, .9988, .9985, .9985, .9985, .9985])
    visited = []

    def fake(chart, initial, design, response, *, iterations, iterate_callback, **kwargs):
        history = []
        for t in range(iterations + 1):
            state = initial.copy()
            state[chart.d_slice] = levels[t]
            record = _record(chart, state, t, chart.loss(state, design, response), 0.,
                             5., .01, [], 0., (1., 1., 5., None))
            history.append(record)
            visited.append(t)
            request = iterate_callback(t, state.copy(), record)
            if isinstance(request, ValidationStopRequest):
                return _result(state, "completed", request.message, t, history,
                               termination_reason="validation_stop")
        return _result(state, "completed", "budget", iterations, history,
                       termination_reason="max_iterations")

    monkeypatch.setattr(estimator, "refine", fake)
    fitted = small_model(iterations=7, validation_patience=3, validation_min_iterations=0).fit(
        X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert visited == list(range(7)) and fitted.n_iter_ == 6
    assert fitted.termination_reason_ == "validation_stop"
    assert fitted.selected_iteration_ == 4  # gain smaller than the reset threshold
    assert fitted.validation_stopping_["last_significant_iteration"] == 3
    assert fitted.checkpoint_iterations_ == (6,)


def test_periodic_incumbents_between_checkpoints_are_retained_for_replay(small_data, monkeypatch):
    X, Y, source = small_data
    install_solver(monkeypatch, levels=(2., 1.9, 1.8, 1.7, 1.6, 1.5, 1.4))
    fitted = small_model(validation_interval=2, checkpoint_iterations=(3, 6)).fit(
        X, Y, source=source, validation_data=(X[:4], np.zeros_like(Y[:4])))
    assert sorted(fitted.best_validation_states_) == [2, 4]
    assert sorted(fitted.checkpoint_model(3).best_validation_states_) == [2]
    assert all(not state.flags.writeable for state in fitted.best_validation_states_.values())


def test_numerical_failure_is_not_reclassified_as_validation_stop(small_data, monkeypatch):
    X, Y, source = small_data
    install_solver(monkeypatch, stop=2, status="numerical_failure", invalid_field="raw_gradient_norm")
    fitted = small_model(validation_patience=2, validation_min_iterations=0).fit(
        X, Y, source=source, validation_data=(np.zeros_like(X[:4]), np.ones_like(Y[:4])))
    assert not fitted.success_ and fitted.termination_reason_ == "numerical_failure"
    assert not fitted.validation_stopping_["stopped"]
    assert fitted.validation_stopping_["last_check_iteration"] == 1


@pytest.mark.parametrize("solver", [refine, refine_anchor_projected])
@pytest.mark.parametrize("ordinary_return", [True, False, {"stop": True}])
def test_solvers_ignore_ordinary_callback_returns(solver, ordinary_return):
    chart, h, X, Y = interior_problem()
    result = solver(chart, _to_z(chart, h), X, Y, calibration=calibration(chart),
                    margins=Margins(.05, 10., .01), iterations=3,
                    iterate_callback=lambda *args: ordinary_return)
    assert result.success and result.n_iter == 3 and result.termination_reason == "max_iterations"


@pytest.mark.parametrize("solver", [refine, refine_anchor_projected])
def test_explicit_stop_contract_and_callback_exceptions(solver):
    chart, h, X, Y = interior_problem()
    options = dict(calibration=calibration(chart), margins=Margins(.05, 10., .01), iterations=5)
    result = solver(chart, _to_z(chart, h), X, Y, **options,
                    iterate_callback=lambda t, *_: ValidationStopRequest() if t == 2 else None)
    assert result.success and result.n_iter == 2 and result.termination_reason == "validation_stop"
    assert len(result.history) == 3

    def broken(*args):
        raise RuntimeError("validation callback broke")

    with pytest.raises(RuntimeError, match="validation callback broke"):
        solver(chart, _to_z(chart, h), X, Y, **options, iterate_callback=broken)


@pytest.mark.parametrize("solver", [refine, refine_anchor_projected])
@pytest.mark.parametrize("stop_iteration, tolerance", [(0, 10.), (1, 4.)])
def test_certified_stationarity_precedes_explicit_callback_stop(solver, stop_iteration, tolerance):
    chart, h, X, Y = interior_problem()
    result = solver(chart, _to_z(chart, h), X, Y, calibration=calibration(chart),
                    margins=Margins(.05, 10., .01), iterations=5, stationarity_tol=tolerance,
                    iterate_callback=lambda t, *_: ValidationStopRequest() if t == stop_iteration else None)
    assert result.success and result.status == "converged"
    assert result.n_iter == stop_iteration and result.termination_reason == "stationarity"
    assert result.history[-1].projected_gradient_norm <= tolerance


@pytest.mark.parametrize("solver", ["chart", "anchor_projected"])
def test_estimator_coincident_stationarity_has_no_validation_stop_metadata(solver):
    X, Y, U, V, _ = example()
    source = ExactSource(U, V)
    validation = (np.zeros((7, X.shape[1])), np.ones((7, Y.shape[1])))
    options = dict(refinement_solver=solver,
                   calibration=PracticalCalibration(.003, .001, 10., (10, 10)))
    baseline = model(**options, iterations=1).fit(X, Y, source=source, validation_data=validation)
    first, last = [record.projected_gradient_norm for record in baseline.history_]
    assert first > last > 0
    tolerance = (first + last) / 2
    fitted = model(**options, iterations=5, stationarity_tol=tolerance, validation_patience=1,
                   validation_min_iterations=1, checkpoint_iterations=(0, 5)).fit(
        X, Y, source=source, validation_data=validation)
    assert fitted.success_ and fitted.n_iter_ == 1
    assert fitted.termination_reason_ == "stationarity" and fitted.optimization_converged_
    assert fitted.selected_iteration_ == 0 and not fitted.converged_
    assert not fitted.validation_stopping_["stopped"]
    assert fitted.validation_stopping_["stop_iteration"] is None
    assert not fitted.validation_history_[-1]["validation_stopping"]["stop_requested"]
    assert fitted.validation_history_[-1]["validation_stopping"]["iterations_since_improvement"] == 1
    assert fitted.checkpoint_model(1).optimization_converged_
    assert not fitted.checkpoint_model(1).validation_stopping_["stopped"]
