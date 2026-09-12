import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sparse_smart.chart import AnchorChart
from sparse_smart.stopping import ValidationStopRequest

from sparse_smart_v2.calibration import Margins, PracticalCalibration
from sparse_smart_v2.solver import _mapping, objective_change, penalty_value, refine
from sparse_smart_v2.support import choose_free_rows, threshold_state


def problem():
    rng = np.random.default_rng(21)
    chart = AnchorChart(4, 3, [2], [1], np.eye(1), np.eye(1))
    state = chart.pack([], [], [2.], [[.25], [0.], [.1]], [[.2], [.1]])
    X = rng.normal(size=(30, 4))
    Y = rng.normal(size=(30, 3))
    return chart, state, X, Y


def test_masked_prox_is_exact_quadratic_minimum():
    chart, state, _, _ = problem()
    free = choose_free_rows(chart, (2, 2))
    trial = state.copy()
    trial[chart.z_u_slice] = [5., -.7, .9]
    actual = threshold_state(chart, trial, free, (.3, .4), (1, 1), 2.)
    # One free nonanchor row is left exactly at its gradient step.
    assert actual[chart.z_u_slice][0] == 5.
    for sl, mask, lam, k in ((chart.z_u_slice, free.penalized_u, .3, 1),
                             (chart.z_v_slice, free.penalized_v, .4, 1)):
        v = trial[sl][mask.ravel()]
        scores = []
        for size in range(k + 1):
            for indices in itertools.combinations(range(v.size), size):
                candidate = np.zeros_like(v)
                idx = list(indices)
                candidate[idx] = np.sign(v[idx]) * np.maximum(np.abs(v[idx]) - lam / 2., 0.)
                scores.append(np.sum((candidate - v)**2) + lam * np.sum(np.abs(candidate)))
        selected = actual[sl][mask.ravel()]
        assert_allclose(np.sum((selected - v)**2) + lam * np.sum(np.abs(selected)), min(scores))


def test_refinement_preserves_chart_support_and_decreases_actual_objective():
    chart, state, X, Y = problem()
    free = choose_free_rows(chart, (2, 2))
    calibration = PracticalCalibration(.01, (.02, .03), 20., (1, 1))
    margins = Margins(.01, 10., .001, anchor_min=.01)
    states = []
    result = refine(chart, state, X, Y, calibration=calibration, margins=margins,
                    free_rows=free, iterations=8,
                    iterate_callback=lambda t, x, rec: states.append(x))
    assert result.success, result.message
    assert result.termination_reason == "max_iterations"
    assert result.n_iter == 8
    assert result.numerical_work["inner_proximal_solves"] == 0
    for x, record in zip(states, result.history):
        P, d, Q = chart.reconstruct(x)
        assert_allclose(P.T @ P, np.eye(1), atol=1e-12)
        assert_allclose(Q.T @ Q, np.eye(1), atol=1e-12)
        wu, wv = (P * d)[np.setdiff1d(np.arange(4), free.rows_u)], (Q * d)[np.setdiff1d(np.arange(3), free.rows_v)]
        assert np.count_nonzero(wu) <= 1 and np.count_nonzero(wv) <= 1
        loss = np.sum((Y - X @ (P * d) @ Q.T)**2) / (2 * len(X))
        penalty = .02 * np.abs(wu).sum() + .03 * np.abs(wv).sum()
        assert_allclose(record.objective, loss + penalty, atol=1e-12)
        assert record.line_search_start_inverse == 20.
    for before, after, record in zip(states, states[1:], result.history[1:]):
        required = -.25 * record.step_size_inverse * np.sum((after - before)**2)
        assert record.objective_change <= required + 1e-14


def test_stable_objective_change_matches_direct_masked_loss():
    chart, state, X, Y = problem()
    free = choose_free_rows(chart, (2, 2))
    trial = state.copy()
    trial[chart.z_u_slice] += [.05, .03, -.04]
    trial[chart.d_slice] += .01
    penalties = (.4, .7)
    expected = (chart.loss(trial, X, Y) + penalty_value(chart, trial, free, penalties)
                - chart.loss(state, X, Y) - penalty_value(chart, state, free, penalties))
    assert_allclose(objective_change(chart, state, trial, X, Y, free, penalties), expected, atol=1e-12)


@pytest.mark.parametrize("rank", [2, 3])
def test_rank_two_and_three_masked_chart_gradient_matches_ambient_objective(rank):
    rng = np.random.default_rng(811 + rank)
    p, q = 6, 5
    au, av = np.arange(1, rank + 1), np.arange(q - rank, q)
    chart = AnchorChart(p, q, au, av, np.eye(rank), np.eye(rank))
    rotations = rank * (rank - 1) // 2
    state = chart.pack(.02 * rng.normal(size=rotations), .02 * rng.normal(size=rotations),
        np.arange(rank, 0, -1, dtype=float), .02 * rng.normal(size=(p - rank, rank)),
        .02 * rng.normal(size=(q - rank, rank)))
    free = choose_free_rows(chart, (rank + 1, rank + 1))
    FL, _ = np.linalg.qr(rng.normal(size=(p, p)))
    FR, _ = np.linalg.qr(rng.normal(size=(q, q)))
    X, Y = rng.normal(size=(20, p)), rng.normal(size=(20, q))
    _, gradient = chart.value_gradient(state, X @ FL, Y @ FR)
    penalties = (.03, .07)
    for sl, mask, lam in ((chart.z_u_slice, free.penalized_u, penalties[0]),
                          (chart.z_v_slice, free.penalized_v, penalties[1])):
        gradient[sl][mask.ravel()] += lam * np.sign(state[sl][mask.ravel()])

    def ambient_objective(x):
        P, d, Q = chart.reconstruct(x)
        C = ((FL @ P) * d) @ (FR @ Q).T
        outside = (np.setdiff1d(np.arange(p), free.rows_u), np.setdiff1d(np.arange(q), free.rows_v))
        return (np.sum((Y - X @ C)**2) / (2 * len(X))
                + penalties[0] * np.abs((P * d)[outside[0]]).sum()
                + penalties[1] * np.abs((Q * d)[outside[1]]).sum())

    for _ in range(3):
        direction = rng.normal(size=state.size)
        epsilon = 1e-6
        finite_difference = (ambient_objective(state + epsilon * direction)
                             - ambient_objective(state - epsilon * direction)) / (2 * epsilon)
        assert_allclose(finite_difference, gradient @ direction, atol=2e-7, rtol=2e-7)


def test_zero_penalties_and_full_caps_use_same_procedure_for_any_free_rows():
    chart, state, X, Y = problem()
    margins = Margins(.01, 10., .001)
    results = []
    for counts in ((1, 1), (2, 2), (4, 3)):
        free = choose_free_rows(chart, counts)
        caps = (int(free.penalized_u.sum()), int(free.penalized_v.sum()))
        results.append(refine(chart, state, X, Y,
            calibration=PracticalCalibration(.01, 0., 20., caps), margins=margins,
            free_rows=free, iterations=4))
    assert all(r.success for r in results)
    for result in results[1:]:
        assert_array_equal(result.state, results[0].state)


def test_no_false_convergence_from_huge_inverse_step():
    chart, state, X, Y = problem()
    result = refine(chart, state, X, Y,
        calibration=PracticalCalibration(.01, 0., 1e25, (3, 2)),
        margins=Margins(.01, 10., .001), free_rows=choose_free_rows(chart, (1, 1)),
        iterations=10, stationarity_tol=1e-6)
    assert not result.success
    assert result.status == "numerical_stagnation"
    assert result.projected_gradient_norm > 1e-6
    assert result.n_iter == 0


def test_validation_stop_is_not_optimization_convergence():
    chart, state, X, Y = problem()
    result = refine(chart, state, X, Y,
        calibration=PracticalCalibration(.01, 0., 20., (3, 2)),
        margins=Margins(.01, 10., .001), free_rows=choose_free_rows(chart, (1, 1)),
        iterations=10, iterate_callback=lambda t, x, rec: ValidationStopRequest() if t == 2 else None)
    assert result.success and result.status == "completed"
    assert result.termination_reason == "validation_stop" and result.n_iter == 2


def test_exact_rank_two_fixed_point_is_successful_without_repeated_zero_updates():
    chart = AnchorChart(3, 3, [0, 1], [0, 1], np.eye(2), np.eye(2))
    state = chart.pack([0.], [0.], [3., 1.], [[0., 0.]], [[0., 0.]])
    result = refine(chart, state, np.eye(3), np.diag([3., 1., 0.]),
        calibration=PracticalCalibration(0., 0., 20., (2, 2)),
        margins=Margins(.01, 10., .1), free_rows=choose_free_rows(chart, (2, 2)),
        iterations=5)
    assert result.success and result.status == "converged"
    assert result.termination_reason == "stationarity"
    assert result.projected_gradient_norm == 0. and result.n_iter == 0
    assert_array_equal(result.state, state)


def test_masked_mapping_does_not_round_a_small_gradient_to_an_exact_fixed_point():
    chart = AnchorChart(2, 2, [0], [0], np.eye(1), np.eye(1))
    state = chart.pack([], [], [1e12], [[1e11]], [[0.]])
    gradient = np.zeros_like(state)
    gradient[chart.z_u_slice] = 1e-6
    calibration = PracticalCalibration(0., 0., 20., (1, 1))
    domain = dict(d_lower=.01, d_upper=1e14, gap=.01, anchor_min=.01)
    diagnostic = _mapping(chart, state, gradient, 20., calibration,
                          choose_free_rows(chart, (1, 1)), domain, 1.)
    assert diagnostic[0] == pytest.approx(1e-6)
    assert diagnostic[3] is None
    assert diagnostic[4] == 0.  # The float proposal rounded back to the state.


def test_line_search_failure_does_not_fallback_or_project():
    chart, state, X, Y = problem()
    result = refine(chart, state, X, Y,
        calibration=PracticalCalibration(.01, 0., 1e-10, (3, 2)),
        margins=Margins(.01, 10., .001), free_rows=choose_free_rows(chart, (1, 1)),
        iterations=3, max_backtracks=0)
    assert not result.success and result.status == "line_search_failed"
    assert result.n_iter == 0
    assert_array_equal(result.state, state)


def test_zero_penalty_still_rejects_initial_support_over_cap():
    chart, state, X, Y = problem()
    result = refine(chart, state, X, Y,
        calibration=PracticalCalibration(.01, 0., 20., (0, 0)),
        margins=Margins(.01, 10., .001), free_rows=choose_free_rows(chart, (1, 1)), iterations=2)
    assert result.status == "invalid_initial_state"


@pytest.mark.parametrize("kwargs", [{"iterations": True}, {"max_backtracks": -1}, {"stationarity_tol": 0}])
def test_invalid_controls_fail_explicitly(kwargs):
    chart, state, X, Y = problem()
    with pytest.raises(ValueError):
        refine(chart, state, X, Y,
            calibration=PracticalCalibration(.01, 0., 20., (3, 2)),
            margins=Margins(.01, 10., .001), free_rows=choose_free_rows(chart, (1, 1)), **kwargs)
