import numpy as np
import pytest

from sparse_smart import AnchorChart, Margins, ResolvedCalibration
from sparse_smart.anchor_solver import (
    _block_trial, _mapping, _project_operator_ball, _project_omega,
    _to_h, _to_z, _value_gradient_h, _weighted_l1_ball_prox,
    refine_anchor_projected,
)
from sparse_smart.chart import skew_matrix


def calibration(chart, L=20., penalties=(.0025, .01), supports=None):
    if supports is None:
        supports = ((chart.n_u - chart.rank) * chart.rank,
                    (chart.n_v - chart.rank) * chart.rank)
    return ResolvedCalibration(.03, penalties, L, supports, "practical", {})


def interior_problem():
    rng = np.random.default_rng(251)
    chart = AnchorChart(4, 3, [0, 2], [0, 1], np.eye(2), np.eye(2))
    h = chart.pack([.12], [-.09], [2.2, 1.2],
                   [[.08, -.2], [.11, .03]], [[-.07, .16]])
    X, Y = rng.normal(size=(8, 4)), rng.normal(size=(8, 3))
    return chart, h, X, Y


def test_h_coordinates_preserve_reconstruction_penalty_and_zeros():
    chart, h, X, Y = interior_problem()
    h[chart.z_u_slice][0] = 0.
    original = _to_z(chart, h)
    np.testing.assert_allclose(_to_h(chart, original), h, atol=2e-16)
    P, d, Q = chart.reconstruct(original)
    expected = np.sum((X @ ((P * d) @ Q.T) - Y)**2) / (2 * len(X))
    np.testing.assert_allclose(_value_gradient_h(chart, h, X, Y)[0], expected)
    _, _, d, hu, hv = chart.unpack(h)
    weights = .0025 * np.abs(hu).sum(axis=0) + .01 * np.abs(hv).sum(axis=0)
    expected_penalty = np.dot(d, weights)
    actual_penalty = .0025 * np.abs(original[chart.z_u_slice]).sum() + .01 * np.abs(original[chart.z_v_slice]).sum()
    np.testing.assert_allclose(expected_penalty, actual_penalty)
    assert np.array_equal(h[chart.z_u_slice] == 0, original[chart.z_u_slice] == 0)


def test_transformed_smooth_and_d_penalty_gradients_by_finite_differences():
    chart, h, X, Y = interior_problem()
    _, gradient = _value_gradient_h(chart, h, X, Y)
    numerical = np.empty_like(h)
    eps = 2e-6
    for j in range(h.size):
        plus, minus = h.copy(), h.copy()
        plus[j] += eps
        minus[j] -= eps
        numerical[j] = (chart.loss(_to_z(chart, plus), X, Y) -
                        chart.loss(_to_z(chart, minus), X, Y)) / (2 * eps)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-7, atol=2e-8)
    # The d derivative at fixed H differs from the existing fixed-Z derivative.
    _, old_gradient = chart.value_gradient(_to_z(chart, h), X, Y)
    assert np.linalg.norm(gradient[chart.d_slice] - old_gradient[chart.d_slice]) > .01
    _, _, _, hu, hv = chart.unpack(h)
    column_penalty = .0025 * np.abs(hu).sum(axis=0) + .01 * np.abs(hv).sum(axis=0)
    trial, _ = _block_trial(chart, h, np.zeros_like(h), 20., (.0025, .01),
                            Margins(.05, 10., .01), 1e-12)
    np.testing.assert_allclose(trial[chart.d_slice], h[chart.d_slice] - column_penalty / 20.)


def test_weighted_l1_ball_prox_matches_constructed_kkt_solution():
    weights = np.array([.13, .37])
    radius = .8
    solution = np.array([[.5, 0.], [.4, .5], [0., .2]])
    solution *= radius / np.linalg.norm(solution, 2)
    left, _, right = np.linalg.svd(solution, full_matrices=False)
    # Independent optimality certificate: p is an L1 subgradient at solution,
    # q is in the spectral-ball normal cone, and value - solution = p + q.
    p = weights * np.sign(solution)
    q = .6 * np.outer(left[:, 0], right[0])
    value = solution + p + q
    solved = _weighted_l1_ball_prox(value, weights, radius, tolerance=1e-12)
    assert solved.converged and solved.n_iter > 1
    assert np.linalg.norm(solved.value, 2) <= radius + 2e-15
    actual = .5 * np.sum((solved.value - value)**2) + np.sum(weights * np.abs(solved.value))
    reference = .5 * np.sum((solution - value)**2) + np.sum(weights * np.abs(solution))
    np.testing.assert_allclose(actual, reference, atol=2e-11)
    np.testing.assert_allclose(solved.value, solution, atol=2e-11)
    once = _project_operator_ball(np.sign(value) * np.maximum(np.abs(value) - weights, 0.), radius)
    once_objective = .5 * np.sum((once - value)**2) + np.sum(weights * np.abs(once))
    assert once_objective > actual + 1e-5


def test_active_anchor_iteration_reports_effective_and_literal_support_without_thresholding():
    chart = AnchorChart(5, 2, [0, 1], [0, 1], np.eye(2), np.eye(2))
    d, L, penalty = np.array([2., 1.]), 20., 4.
    margins = Margins(.05, 5., .01, trial_radius=5.)
    radius = np.sqrt((1. - margins.anchor_min) * (1. + margins.anchor_min))
    solution = np.array([[.5, 0.], [.4, .5], [0., .2]])
    solution *= radius / np.linalg.norm(solution, 2)
    left, _, right = np.linalg.svd(solution, full_matrices=False)
    # Independent KKT construction makes the first H proximal minimizer known.
    # Its exact two zeros acquire tiny projection residuals in the inner solve.
    incoming = (solution + (penalty * d / L) * np.sign(solution)
                + .6 * np.outer(left[:, 0], right[0]))
    state = chart.pack([0.], [0.], d, np.zeros((3, 2)), np.empty((0, 2)))
    design = np.sqrt(5.) * np.eye(5)
    response = design @ np.vstack([np.diag(d), L * incoming / d])
    result = refine_anchor_projected(chart, state, design, response,
        calibration=calibration(chart, L=L, penalties=(penalty, 0.)),
        margins=margins, iterations=1)
    assert result.success and result.n_iter == 1
    record = result.history[-1]
    assert record.backtracks == 0
    assert record.support_u == 4 and record.raw_support_u == 6
    assert record.support_v == record.raw_support_v == 0
    h = result.state[chart.z_u_slice].reshape(3, 2) / d
    np.testing.assert_allclose(h, solution, rtol=0., atol=2e-10)
    assert np.all(h[solution == 0.] != 0.)  # Reporting never mutates coefficients.
    assert np.max(np.abs(result.state[chart.z_u_slice].reshape(3, 2)[solution == 0.])) < record.support_tolerance_u
    assert chart.domain_reason(result.state, d_lower=margins.d_lower,
        d_upper=margins.d_upper, gap=margins.gap, anchor_min=margins.anchor_min) is None


def test_prox_reports_unfinished_inner_solve_and_handles_empty_blocks():
    weights, radius = np.array([.13, .37]), .8
    solution = np.array([[.5, 0.], [.4, .5], [0., .2]])
    solution *= radius / np.linalg.norm(solution, 2)
    left, _, right = np.linalg.svd(solution, full_matrices=False)
    value = solution + weights * np.sign(solution) + .6 * np.outer(left[:, 0], right[0])
    solved = _weighted_l1_ball_prox(value, weights, radius, max_iterations=1)
    assert not solved.converged
    # Unlike the former sign-preserving fixture, composing the two proxes
    # once does not satisfy this independently constructed joint KKT solution.
    assert np.linalg.norm(solved.value - solution) > 1e-4
    assert _weighted_l1_ball_prox(np.empty((0, 2)), [1., 2.], .8).converged
    np.testing.assert_array_equal(_weighted_l1_ball_prox(value, [0., 0.], 0.).value, 0.)
    with pytest.raises(FloatingPointError, match="norm"):
        _weighted_l1_ball_prox(np.full((2, 2), 1e200), [0., 0.], .8)


def test_cayley_projection_preserves_skew_metric_and_constraint():
    coordinates = np.array([2., -.3, 1.1, .2, 1.8, -.6])
    projected = _project_omega(coordinates, 4)
    matrix = skew_matrix(projected, 4)
    np.testing.assert_allclose(matrix + matrix.T, 0., atol=1e-15)
    assert np.linalg.norm(matrix, 2) <= .5 + 2e-16
    # Projection variational inequality against several independent feasible points.
    rng = np.random.default_rng(124)
    for _ in range(10):
        feasible = rng.normal(size=6)
        feasible *= .49 / np.linalg.norm(skew_matrix(feasible, 4), 2)
        assert np.dot(coordinates - projected, feasible - projected) <= 1e-12


def boundary_problem(direction=(.6, .8), target=(0., 2., 0.), d=1.):
    chart = AnchorChart(3, 1, [0], [0], np.eye(1), np.eye(1))
    margins = Margins(.05, 5., .01, anchor_min=.01, trial_radius=2.)
    radius = np.sqrt(1 - margins.anchor_min**2) * (1 - 2e-13)
    h = np.asarray(direction)[:, None] * radius
    state = chart.pack([], [], [d], h * d, np.empty((0, 1)))
    X = np.sqrt(3.) * np.eye(3)
    Y = X @ np.asarray(target)[:, None]
    return chart, state, X, Y, margins


def test_active_anchor_boundary_allows_tangential_descent_and_strict_feasibility():
    chart, state, X, Y, margins = boundary_problem()
    seen = []
    def observe(iteration, value, record):
        assert chart.domain_reason(value, d_lower=margins.d_lower, d_upper=margins.d_upper,
                                   gap=margins.gap, anchor_min=margins.anchor_min) is None
        seen.append(value.copy())
        value[:] = np.nan  # Callback arrays never alias solver state.
    result = refine_anchor_projected(chart, state, X, Y, calibration=calibration(chart, penalties=(0., 0.)),
        margins=margins, iterations=150, stationarity_tol=1e-7, iterate_callback=observe)
    assert result.success, (result.status, result.last_rejection)
    assert result.history[-1].objective < result.history[0].objective - .5
    assert result.history[-1].anchor_min_u >= margins.anchor_min
    assert result.history[-1].anchor_min_u < margins.anchor_min + 1e-7
    assert len(seen) == result.n_iter + 1
    np.testing.assert_array_equal(seen[-1], result.state)
    for previous, record in zip(result.history, result.history[1:]):
        assert record.objective <= previous.objective - .25 * record.step_size_inverse * record.step_norm**2 + 1e-14


def test_stationarity_includes_anchor_and_spectral_normal_cones():
    rho = np.sqrt(1 - .01**2)
    chart, state, X, Y, margins = boundary_problem(direction=(1., 0.), d=2 * rho)
    result = refine_anchor_projected(chart, state, X, Y, calibration=calibration(chart, penalties=(0., 0.)),
        margins=margins, iterations=10, stationarity_tol=1e-7)
    assert result.status == "converged" and result.n_iter == 0
    assert result.raw_gradient_norm > 3.
    assert result.projected_gradient_norm < 1e-7
    assert result.history[0].mapping_domain_reason is None


def test_complete_mapping_includes_cayley_normal_cone():
    chart = AnchorChart(2, 2, [0, 1], [0, 1], np.eye(2), np.eye(2))
    state = chart.pack([.5 * np.sqrt(2)], [0.], [2., 1.], np.empty((0, 2)), np.empty((0, 2)))
    gradient = np.zeros_like(state)
    gradient[0] = -3.
    residual = _mapping(chart, state, gradient, 20., (0., 0.), Margins(.05, 5., .01), 1e-12)
    assert residual[0] < 1e-12 and residual[1] == 3. and residual[3] is None


def test_reduced_supports_rejected_and_zero_iterations_preserve_original_start():
    chart, h, X, Y = interior_problem()
    state = _to_z(chart, h)
    margins = Margins(.05, 10., .01)
    with pytest.raises(ValueError, match="full complement"):
        refine_anchor_projected(chart, state, X, Y, calibration=calibration(chart, supports=(3, 2)), margins=margins)
    result = refine_anchor_projected(chart, state, X, Y, calibration=calibration(chart), margins=margins, iterations=0)
    assert result.success and result.n_iter == 0 and result.termination_reason == "max_iterations"
    np.testing.assert_array_equal(result.state, state)


def test_original_loss_offset_does_not_affect_acceptance():
    chart, h, X, Y = interior_problem()
    options = dict(calibration=calibration(chart), margins=Margins(.05, 10., .01), iterations=5)
    first = refine_anchor_projected(chart, _to_z(chart, h), X, Y, **options)
    shifted = refine_anchor_projected(chart, _to_z(chart, h), X, Y, loss_offset=1e30, **options)
    assert first.success and shifted.success
    np.testing.assert_array_equal(first.state, shifted.state)
    assert shifted.history[-1].smooth_loss >= 1e30


def test_numerically_zero_step_cannot_fake_progress_or_convergence():
    chart = AnchorChart(1, 1, [0], [0], np.eye(1), np.eye(1))
    state = chart.pack([], [], [1.], np.empty((0, 1)), np.empty((0, 1)))
    X, Y = np.ones((4, 1)), np.full((4, 1), 2.)
    for tolerance in (None, 1e-8):
        result = refine_anchor_projected(chart, state, X, Y,
            calibration=calibration(chart, L=1e300), margins=Margins(.05, 5., .01),
            iterations=3, stationarity_tol=tolerance)
        assert result.status == "numerical_stagnation" and not result.success
        assert result.n_iter == 0 and result.projected_gradient_norm > .9
    # A genuinely zero mapping remains valid under fixed-iteration semantics.
    result = refine_anchor_projected(chart, state, X, X,
        calibration=calibration(chart), margins=Margins(.05, 5., .01), iterations=3)
    assert result.success and result.n_iter == 3
    assert all(row.step_norm == 0. for row in result.history)


@pytest.mark.parametrize("penalties", [(np.nan, 0.), (0., -1.), (0., 1., 2.)])
def test_malformed_resolved_penalties_rejected(penalties):
    chart, h, X, Y = interior_problem()
    with pytest.raises(ValueError):
        refine_anchor_projected(chart, _to_z(chart, h), X, Y,
            calibration=calibration(chart, penalties=penalties), margins=Margins(.05, 10., .01))


@pytest.mark.parametrize("options", [{"stationarity_tol": True}, {"stationarity_tol": 0.},
                                    {"stationarity_tol": np.nan}, {"iterate_callback": 3}])
def test_invalid_options_rejected(options):
    chart, h, X, Y = interior_problem()
    with pytest.raises(ValueError):
        refine_anchor_projected(chart, _to_z(chart, h), X, Y, calibration=calibration(chart),
                                margins=Margins(.05, 10., .01), **options)
