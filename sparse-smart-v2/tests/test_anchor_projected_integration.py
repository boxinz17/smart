"""Estimator integration of the opt-in constrained H-coordinate solver."""

from copy import deepcopy

import numpy as np
import pytest

from sparse_smart.anchor_solver import _to_h, _value_gradient_h
from sparse_smart_v2 import Margins, PracticalCalibration, SparseSMARTv2
from sparse_smart_v2.estimator import _sum_work
from sparse_smart_v2.solver import penalty_value
from sparse_smart_v2.tuning import SparseSMARTv2Tuner


OLD_SOLVER = "masked_chart_spectral_soft_hard"
NEW_SOLVER = "masked_anchor_projected"


def problem():
    """Rank-two data with nonzero free and penalized complementary rows."""
    rng = np.random.default_rng(982)
    left = np.array([[1., .1], [.1, 1.], [.3, .2], [.2, .4], [.1, .15], [.08, .07]])
    right = np.array([[1., .05], [.1, 1.], [.2, .3], [.25, .1], [.1, .13]])
    left, right = np.linalg.qr(left)[0], np.linalg.qr(right)[0]
    coefficient = (left * [2., 1.]) @ right.T
    X = rng.normal(size=(50, 6))
    Y = X @ coefficient + .03 * rng.normal(size=(50, 5))
    XV = rng.normal(size=(15, 6))
    source = np.zeros((6, 5))
    source[:5] = np.diag(np.arange(5, 0, -1))
    return X, Y, source, XV, XV @ coefficient


def model(**changes):
    options = dict(rank=2, source_rank=4,
        margins=Margins(.02, 8., .01, anchor_min=.04),
        calibration=PracticalCalibration(.03, (.02, .03), 20., (6, 4)),
        free_directions=(3, 3), iterations=6,
        checkpoint_iterations=(2, 4), validation_interval=1)
    options.update(changes)
    return SparseSMARTv2(**options)


def test_default_retains_explicit_original_solver_trajectory():
    X, Y, source, XV, YV = problem()
    default = model().fit(X, Y, source=source, validation_data=(XV, YV))
    explicit = model(refinement_solver=OLD_SOLVER).fit(
        X, Y, source=source, validation_data=(XV, YV))
    assert default.success_ and explicit.success_
    assert default.metadata_["refinement_solver"] == OLD_SOLVER
    assert default.metadata_["optimization_coordinates"] == "omega_d_Z"
    np.testing.assert_array_equal(default.last_state_, explicit.last_state_)
    np.testing.assert_array_equal(default.coefficient_, explicit.coefficient_)
    assert default.history_ == explicit.history_
    assert default.validation_history_ == explicit.validation_history_


def test_h_solver_checkpoints_preserve_masked_objective_states_and_validation():
    X, Y, source, XV, YV = problem()
    fit = model(refinement_solver=NEW_SOLVER).fit(
        X, Y, source=source, validation_data=(XV, YV))
    assert fit.success_, fit.message_
    assert fit.n_iter_ == 6 and fit.termination_reason_ == "max_iterations"
    assert not fit.optimization_converged_
    assert fit.metadata_["refinement_solver"] == NEW_SOLVER
    assert fit.metadata_["optimization_coordinates"] == "omega_d_H"
    assert fit.checkpoint_iterations_ == (0, 2, 4, 6)
    assert fit.numerical_work_["inner_proximal_solves"] > 0
    assert fit.numerical_work_["accepted_updates"] == 6
    assert len(fit.numerical_work_["segments"]) == 1
    design, response = X @ fit.source_.left, Y @ fit.source_.right
    excluded_free_entries = False
    for iteration in fit.checkpoint_iterations_:
        prefix = fit.checkpoint_model(iteration)
        assert prefix.refinement_solver == NEW_SOLVER
        assert prefix.metadata_["optimization_coordinates"] == "omega_d_H"
        assert prefix.n_iter_ == iteration
        assert max(row["iteration"] for row in prefix.validation_history_) <= iteration
        chart, state, free = prefix.chart_, prefix.last_state_, prefix.free_rows_
        P, d, Q = chart.reconstruct(state)
        np.testing.assert_allclose(P.T @ P, np.eye(2), atol=2e-13)
        np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=2e-13)
        np.testing.assert_allclose(prefix.last_coefficient_,
            ((prefix.source_.left @ P) * d) @ (prefix.source_.right @ Q).T, atol=2e-13)
        endpoint = prefix.terminal_record_
        for field in ("proximal_uncertainty", "mapping_refinements", "mapping_precision_limited"):
            assert getattr(prefix.result_, field) == getattr(endpoint, field)
        expected_penalty = penalty_value(chart, state, free, prefix.calibration_.penalties)
        smooth, gradient = _value_gradient_h(chart, _to_h(chart, state), design, response)
        assert endpoint.penalty_value == pytest.approx(expected_penalty, abs=2e-13)
        assert endpoint.objective == pytest.approx(smooth + expected_penalty, abs=2e-13)
        assert endpoint.raw_gradient_norm == pytest.approx(np.linalg.norm(gradient), rel=2e-12)
        for side in ("u", "v"):
            block = state[getattr(chart, f"z_{side}_slice")]
            mask = getattr(free, f"penalized_{side}")
            count = np.count_nonzero(block.reshape(mask.shape)[mask])
            assert getattr(endpoint, f"support_{side}") == count
            assert getattr(endpoint, f"raw_support_{side}") == count
            excluded_free_entries |= np.count_nonzero(block) > count
        actual_validation = np.mean((YV - prefix.predict(XV))**2)
        assert prefix.best_validation_loss_ == pytest.approx(actual_validation, rel=2e-12)
        assert prefix.selected_iteration_ <= iteration
    assert excluded_free_entries, "Fixture must distinguish free entries from counted support"
    untouched = fit.coefficient_.copy()
    prefix.coefficient_[:] = 0
    np.testing.assert_array_equal(fit.coefficient_, untouched)


def test_h_solver_normal_validation_stop_retains_selected_initializer():
    X, Y, source, XV, _ = problem()
    warm = model(refinement_solver=NEW_SOLVER, iterations=0,
                 checkpoint_iterations=()).fit(X, Y, source=source)
    assert warm.success_, warm.message_
    YV = warm.predict(XV)
    fit = model(refinement_solver=NEW_SOLVER, iterations=20,
        checkpoint_iterations=(2,), validation_patience=2,
        validation_min_iterations=0).fit(X, Y, source=source, validation_data=(XV, YV))
    assert fit.success_ and fit.termination_reason_ == "validation_stop"
    assert fit.n_iter_ == 2 and fit.selected_iteration_ == 0
    assert not fit.optimization_converged_
    np.testing.assert_allclose(fit.coefficient_, warm.coefficient_, atol=2e-13)
    assert np.linalg.norm(fit.last_coefficient_ - fit.coefficient_) > 1e-5
    assert fit.checkpoint_model(2).termination_reason_ == "validation_stop"


@pytest.mark.parametrize("limits", [(5, 4), (6, 3)])
def test_h_solver_rejects_restrictive_masked_caps_without_falling_back(limits):
    X, Y, source, _, _ = problem()
    fit = model(refinement_solver=NEW_SOLVER,
        calibration=PracticalCalibration(.03, (.02, .03), 20., limits))
    with pytest.raises(ValueError, match="restrictive hard caps are unsupported"):
        fit.fit(X, Y, source=source)
    assert not hasattr(fit, "coefficient_")
    assert fit.metadata_["refinement_solver"] == NEW_SOLVER


def test_tuner_records_solver_choice_without_mutating_templates():
    X, Y, source, XV, YV = problem()
    candidates = [model(refinement_solver=solver) for solver in (OLD_SOLVER, NEW_SOLVER)]
    tuner = SparseSMARTv2Tuner(candidates).fit(X, Y, source=source, validation_data=(XV, YV))
    assert tuner.success_
    assert all(row["eligible"] for row in tuner.results_)
    assert [row["params"]["refinement_solver"] for row in tuner.results_] == [OLD_SOLVER, NEW_SOLVER]
    assert tuner.best_params_["refinement_solver"] == tuner.best_estimator_.refinement_solver
    assert all(not hasattr(template, "coefficient_") for template in candidates)
    assert tuner.best_estimator_.metadata_["refinement_solver"] == tuner.best_params_["refinement_solver"]


def test_segment_work_adds_counters_but_preserves_search_configuration_and_last_diagnostic():
    first = dict(adapted_starts=2, recovery_probes=1, activations=1,
        failure_threshold=4, consecutive_updates=2, recovery_interval=25,
        u=dict(calls=3, iterations=20, reasons={"certified": 3},
               last={"n_iter": 8, "converged": True}))
    second = dict(adapted_starts=3, recovery_probes=0, activations=1,
        failure_threshold=4, consecutive_updates=2, recovery_interval=25,
        u=dict(calls=2, iterations=11, reasons={"certified": 2},
               last={"n_iter": 5, "converged": True}))
    original = deepcopy((first, second))
    total = _sum_work((first, second))
    assert (total["adapted_starts"], total["recovery_probes"], total["activations"]) == (5, 1, 2)
    assert (total["failure_threshold"], total["consecutive_updates"], total["recovery_interval"]) == (4, 2, 25)
    assert total["u"]["calls"] == 5 and total["u"]["iterations"] == 31
    assert total["u"]["reasons"] == {"certified": 5}
    assert total["u"]["last"] == second["u"]["last"]
    total["u"]["last"]["n_iter"] = 900
    assert (first, second) == original
