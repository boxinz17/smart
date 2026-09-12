"""Small numerical fixtures for measured proximal work and failure reuse."""
from dataclasses import replace
import json

import numpy as np
import pytest

from sparse_smart import AnchorChart, Margins, ResolvedCalibration
import sparse_smart.anchor_solver as solver


def scalar_problem():
    chart = AnchorChart(1, 1, [0], [0], np.eye(1), np.eye(1))
    state = chart.pack([], [], [1.], np.empty((0, 1)), np.empty((0, 1)))
    design = np.ones((4, 1))
    calibration = ResolvedCalibration(.03, (0., 0.), 1., (0, 0), "practical", {})
    return chart, state, design, 2. * design, calibration, Margins(.05, 5., .01, trial_radius=10.)


def exhausted():
    failed = solver._ProxResult(np.zeros((1, 1)), 2000, False, 1e-5, np.inf,
        iterations_run=2000, termination_reason="split_residual_not_met", residual_threshold=1e-11)
    passed = replace(failed, n_iter=1, iterations_run=1, converged=True, residual=0.,
        duality_gap=0., termination_reason="interior_soft_threshold")
    return solver._ProxConvergenceError(failed, passed)


def test_failed_reference_reuse_preserves_history_and_counts_actual_work(monkeypatch):
    original = solver._block_trial

    def difficult(*args, **kwargs):
        if args[3] < 16.:
            raise exhausted()
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_block_trial", difficult)
    chart, state, design, response, calibration, margins = scalar_problem()
    options = dict(calibration=calibration, margins=margins, iterations=3,
                   line_search_strategy="reset_initial_inverse")
    detailed = {}
    cached = solver.refine_anchor_projected(chart, state, design, response,
        numerical_work=detailed, **options)
    baseline = solver.refine_anchor_projected(chart, state, design, response,
        cache_failed_trials=False, **options)
    np.testing.assert_array_equal(cached.state, baseline.state)
    assert cached.history == baseline.history
    assert cached.status == baseline.status == "completed"
    assert cached.numerical_work is detailed
    a, b = detailed["totals"], baseline.numerical_work["totals"]
    assert a["cache_failure_hits"] == 3 and b["cache_failure_hits"] == 0
    assert b["block_calls"] - a["block_calls"] == 3
    assert b["u"]["iterations"] - a["u"]["iterations"] == 6000
    assert len(detailed["iterations"]) == 4
    assert "iterations" not in baseline.numerical_work
    assert all(row.mapping_domain_reason for row in cached.history)
    assert all(np.isinf(row.projected_gradient_norm) for row in cached.history)
    json.dumps(detailed, allow_nan=False)


def test_failed_reference_cache_is_exact_and_not_a_stationarity_certificate(monkeypatch):
    def fail(*args, **kwargs):
        raise exhausted()

    monkeypatch.setattr(solver, "_block_trial", fail)
    chart, state, _, _, calibration, margins = scalar_problem()
    gradient = np.ones_like(state)
    cache = {}
    args = (chart, state, gradient, 1., calibration.penalties, margins, 1e-11)
    result = solver._mapping(*args, stationarity_tol=1e-6, trial_cache=cache)
    assert np.isinf(result[0]) and result[3] and result[4:6] == (None, None)
    assert isinstance(cache["failure"], solver._ProxConvergenceError)
    assert "result" not in cache
    assert cache["key"] == solver._trial_key(*args)
    assert cache["key"] == solver._trial_key(chart, state.copy(), gradient.copy(),
                                            1., calibration.penalties, margins, 1e-11)
    for index, replacement in [(1, state + 1e-12), (2, gradient + 1e-12),
                               (3, 2.), (4, (.1, 0.)), (6, 1e-10)]:
        changed = list(args)
        changed[index] = replacement
        assert cache["key"] != solver._trial_key(*changed)
    solver._mapping(*args, stationarity_tol=1e-6, trial_cache=cache, cache_failed_trials=False)
    assert cache == {}


def test_prox_work_records_exhaustion_and_closed_form_cases():
    solution = np.array([[.5, 0.], [.4, .5], [0., .2]])
    solution *= .8 / np.linalg.norm(solution, 2)
    left, _, right = np.linalg.svd(solution, full_matrices=False)
    value = solution + np.array([.13, .37]) * np.sign(solution) + .6 * np.outer(left[:, 0], right[0])
    result = solver._weighted_l1_ball_prox(value, [.13, .37], .8, max_iterations=1)
    assert not result.converged and result.iterations_run == result.n_iter == 1
    assert result.termination_reason in {"split_residual_not_met", "duality_gap_not_certified"}
    assert result.residual_threshold > 0.
    result = solver._weighted_l1_ball_prox(np.zeros((2, 1)), [.1], .8)
    assert result.converged and result.iterations_run == 1
    assert result.termination_reason == "interior_soft_threshold"


def test_one_sweep_kkt_certificate_is_valid_despite_large_split_residual():
    value = np.array([[1.4, -.5], [.7, 1.2], [-.3, .8]])
    weights, radius, tolerance = np.array([.13, .37]), .8, 1e-11
    solved = solver._weighted_l1_ball_prox(value, weights, radius,
                                          tolerance=tolerance, max_iterations=1)
    assert solved.converged and solved.iterations_run == 1
    assert solved.residual > tolerance * max(1., np.linalg.norm(value))
    assert solved.duality_gap + solved.gap_roundoff <= tolerance * max(1., np.linalg.norm(value))**2
    # Check joint KKT conditions independently of the solver's gap formula.
    # Sign preservation makes the first clipped dual an L1 subgradient;
    # the remaining dual is in the spectral-ball normal cone.
    p = np.clip(value, -weights, weights)
    np.testing.assert_array_equal(p, weights * np.sign(solved.value))
    q = value - solved.value - p
    assert np.linalg.norm(solved.value, 2) <= radius + 2e-15
    support = radius * np.linalg.svd(q, compute_uv=False).sum()
    assert abs(support - np.sum(q * solved.value)) <= 2e-14


def test_relative_prox_certificate_cannot_hide_unresolved_mapping_uncertainty(monkeypatch):
    value = np.array([[1.4, -.5], [.7, 1.2], [-.3, .8]])
    prox = solver._weighted_l1_ball_prox(value, [.13, .37], .8, max_iterations=1)
    assert prox.converged and prox.error_bound > 0.
    # A relative certificate can still have too much error for a stricter
    # stationarity request, even if the approximate trial does not move.
    inverse = 20.
    allowance = inverse * np.hypot(prox.error_bound, prox.error_bound)
    trial = solver._BlockTrialResult(np.zeros(1), allowance, False, prox, prox)
    requested_gaps = []

    def unresolved(*args, **kwargs):
        requested_gaps.append(kwargs.get("absolute_gap_tol"))
        return trial

    monkeypatch.setattr(solver, "_block_trial", unresolved)
    tolerance = allowance / 100.
    result = solver._mapping(object(), np.zeros(1), np.ones(1), inverse,
        (.13, .37), Margins(.05, 5., .01), 1e-11, stationarity_tol=tolerance)
    assert requested_gaps[0] is None and len(requested_gaps) > 1
    assert requested_gaps[1] > 0.
    assert result[3] is None and result[4] == 0.
    assert result[5] == allowance and result[0] > tolerance
    assert result[7]  # The inner uncertainty remains visible and unresolved.


def test_hybrid_fallback_respects_total_budget_and_unmet_absolute_accuracy():
    weights, radius = np.array([.13, .37]), .8
    solution = np.array([[.5, 0.], [.4, .5], [0., .2]])
    solution *= radius / np.linalg.norm(solution, 2)
    left, _, right = np.linalg.svd(solution, full_matrices=False)
    value = solution + weights * np.sign(solution) + .6 * np.outer(left[:, 0], right[0])
    solved = solver._weighted_l1_ball_prox(value, weights, radius, tolerance=1e-12,
                                          absolute_gap_tol=1e-30, max_iterations=80)
    assert solved.converged  # The relative certificate remains usable.
    assert 0 < solved.dykstra_iterations <= 64 and solved.dual_iterations > 0
    assert solved.iterations_run == solved.dykstra_iterations + solved.dual_iterations == 80
    assert solved.n_iter <= solved.iterations_run
    assert solved.termination_reason == "absolute_target_unresolved"
    assert solved.duality_gap + solved.gap_roundoff > 1e-30
    assert solved.error_bound > np.sqrt(2e-30)
    assert np.linalg.norm(solved.value - solution) <= solved.error_bound


def test_short_prox_budget_never_launches_an_unbudgeted_fallback(monkeypatch):
    from sparse_smart import proximal_dual

    def forbidden(*args, **kwargs):
        raise AssertionError("No dual sweep is available in a one-sweep budget")

    monkeypatch.setattr(proximal_dual, "solve_weighted_l1_ball_dual", forbidden)
    weights, radius = np.array([.13, .37]), .8
    solution = np.array([[.5, 0.], [.4, .5], [0., .2]])
    solution *= radius / np.linalg.norm(solution, 2)
    left, _, right = np.linalg.svd(solution, full_matrices=False)
    value = solution + weights * np.sign(solution) + .6 * np.outer(left[:, 0], right[0])
    solved = solver._weighted_l1_ball_prox(value, weights, radius, max_iterations=1)
    assert not solved.converged
    assert solved.iterations_run == solved.dykstra_iterations == 1
    assert solved.dual_iterations == 0


def test_adaptive_search_avoids_repeated_failures_with_same_accepted_trajectory(monkeypatch):
    original = solver._block_trial

    def difficult(*args, **kwargs):
        if args[3] < 16.:
            raise exhausted()
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_block_trial", difficult)
    chart, state, design, response, calibration, margins = scalar_problem()
    options = dict(calibration=calibration, margins=margins, iterations=35)
    adaptive = solver.refine_anchor_projected(chart, state, design, response, **options)
    reset = solver.refine_anchor_projected(chart, state, design, response,
                                         line_search_strategy="reset_initial_inverse", **options)
    np.testing.assert_array_equal(adaptive.state, reset.state)
    assert adaptive.status == reset.status == "completed"
    assert adaptive.numerical_work["search"]["adapted_starts"] == 32
    assert adaptive.numerical_work["search"]["recovery_probes"] == 1
    assert adaptive.numerical_work["totals"]["block_calls"] < reset.numerical_work["totals"]["block_calls"]
    for a, b in zip(adaptive.history, reset.history):
        assert a.objective == b.objective
        assert a.step_size_inverse == b.step_size_inverse
        assert a.mapping_step_size_inverse == b.mapping_step_size_inverse == 1.
        if a.iteration:
            assert a.objective_change <= -.25 * a.step_size_inverse * a.step_norm**2
    assert all(np.isinf(row.projected_gradient_norm) for row in adaptive.history)
    assert adaptive.termination_reason == "max_iterations"


def test_adaptive_policy_recovers_and_never_activates_on_curvature_alone():
    policy = solver._FailureAwareSearch(1., 1., recovery_interval=2)
    assert policy.start(False) == 1.
    policy.accepted(65536., 0)
    assert policy.start(False) == 1.  # Curvature-induced backtracking is not enough.
    policy.accepted(65536., 4)
    assert policy.start(True) == 1.
    policy.accepted(65536., 4)
    assert policy.active
    assert policy.start(True) == 32768.
    policy.accepted(65536., 1)
    assert policy.start(True) == 32768.
    policy.accepted(65536., 1)
    assert policy.start(True) == 1. and policy.last_policy == "recovery_probe"
    policy.accepted(2., 1)
    assert not policy.active and policy.start(True) == 1.
    policy.accepted(65536., 4)
    policy.start(True)
    policy.accepted(65536., 4)
    assert policy.active
    assert policy.start(False) == 1. and not policy.active  # Reference mapping recovers.


@pytest.mark.parametrize("option", [dict(numerical_work={"old": 1}),
    dict(numerical_work=[]), dict(cache_failed_trials=1), dict(line_search_strategy="unknown")])
def test_invalid_work_or_search_options(option):
    chart, state, design, response, calibration, margins = scalar_problem()
    with pytest.raises(ValueError):
        solver.refine_anchor_projected(chart, state, design, response,
                                      calibration=calibration, margins=margins, **option)
