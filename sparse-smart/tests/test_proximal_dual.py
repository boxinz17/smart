"""Independent KKT constructions for the convex inner proximal solver."""
import numpy as np
import pytest

from sparse_smart.proximal_dual import (
    _certificate, _spectral_bounds, solve_weighted_l1_ball_dual,
)


def boundary_kkt_problem():
    # At X=I every positive-semidefinite q is a spectral-ball normal.
    # Diagonal p=weights and off-diagonal p=0 form an L1 subgradient.
    solution = np.eye(3)
    weights = np.array([[.07, .5, .001], [.5, .03, .002], [.001, .002, .04]])
    p = np.diag(np.diag(weights))
    q = np.ones((3, 3))
    return solution + p + q, weights, 1., solution, p


def rectangular_kkt_problem():
    weights = np.array([.13, .37])
    radius = .8
    expected = np.array([[.5, 0.], [.4, .5], [0., .2]])
    expected *= radius / np.linalg.norm(expected, 2)
    left, _, right = np.linalg.svd(expected, full_matrices=False)
    p = weights * np.sign(expected)
    A = expected + p + .6 * np.outer(left[:, 0], right[0])
    return A, weights, radius, expected, p


def assert_kkt_accuracy(result, expected, radius):
    assert result.primal_operator_upper <= radius
    assert result.dual_box_violation == 0
    assert result.gap_upper >= result.gap_roundoff >= 0
    assert np.linalg.norm(result.value - expected) <= result.error_bound + 2e-12
    assert result.error_bound == pytest.approx(np.sqrt(2 * result.gap_upper), rel=1e-15)


def test_known_sparse_boundary_solution_and_error_bound():
    A, weights, radius, expected, _ = boundary_kkt_problem()
    result = solve_weighted_l1_ball_dual(A, weights, radius, tolerance=1e-12)
    assert result.relative_certified and result.termination_reason == "certified"
    assert result.absolute_certified is None
    assert 1 <= result.n_iter <= 2000
    assert 1 <= result.certificate_iteration <= result.n_iter
    assert_kkt_accuracy(result, expected, radius)
    np.testing.assert_allclose(result.value, expected, atol=5e-6, rtol=0)


def test_rectangular_rank_one_ball_normal_and_column_weights():
    A, weights, radius, expected, _ = rectangular_kkt_problem()
    result = solve_weighted_l1_ball_dual(A, weights, radius, tolerance=1e-12)
    assert result.relative_certified
    assert_kkt_accuracy(result, expected, radius)


def test_limited_solve_is_uncertified_but_retains_valid_error_bound():
    A, weights, radius, expected, _ = rectangular_kkt_problem()
    result = solve_weighted_l1_ball_dual(A, weights, radius, max_iterations=1)
    assert result.n_iter == result.certificate_iteration == 1
    assert not result.relative_certified
    assert result.termination_reason == "iteration_limit_uncertified"
    assert_kkt_accuracy(result, expected, radius)


def test_strict_absolute_request_is_never_relabeled_as_met():
    A, weights, radius, expected, _ = boundary_kkt_problem()
    result = solve_weighted_l1_ball_dual(
        A, weights, radius, absolute_gap_tol=1e-30, max_iterations=300)
    assert result.relative_certified
    assert not result.absolute_certified
    assert result.gap_upper > 1e-30
    assert result.termination_reason == "absolute_target_unresolved"
    assert result.n_iter == 300
    assert_kkt_accuracy(result, expected, radius)


def test_certified_warm_dual_and_cold_start_have_same_optimum():
    A, weights, radius, expected, exact_p = rectangular_kkt_problem()
    cold = solve_weighted_l1_ball_dual(A, weights, radius, tolerance=1e-12)
    warm = solve_weighted_l1_ball_dual(A, weights, radius, tolerance=1e-12, initial_p=exact_p)
    assert warm.relative_certified and warm.n_iter == 1
    assert warm.n_iter < cold.n_iter
    assert_kkt_accuracy(warm, expected, radius)
    assert np.linalg.norm(warm.value - cold.value) <= warm.error_bound + cold.error_bound + 2e-12


def test_initial_dual_is_clipped_for_new_weights_without_mutating_input():
    A, weights, radius, expected, _ = boundary_kkt_problem()
    initial_p = np.full_like(A, 100.)
    saved = initial_p.copy()
    result = solve_weighted_l1_ball_dual(A, weights, radius, initial_p=initial_p)
    assert result.relative_certified
    assert_kkt_accuracy(result, expected, radius)
    np.testing.assert_array_equal(initial_p, saved)


def test_zero_weights_and_inactive_ball_have_known_proximal_solutions():
    A = np.array([[3., .4], [.4, 2.]])
    left, singular, right = np.linalg.svd(A, full_matrices=False)
    expected = (left * np.minimum(singular, 1.)) @ right
    result = solve_weighted_l1_ball_dual(A, 0., 1.)
    assert result.relative_certified and result.n_iter == 1
    assert_kkt_accuracy(result, expected, 1.)
    weights = np.array([.02, .05])
    A = np.array([[.3, -.2], [.1, .5]])
    expected = np.sign(A) * np.maximum(np.abs(A) - weights, 0.)
    result = solve_weighted_l1_ball_dual(A, weights, 2.)
    assert result.relative_certified
    assert_kkt_accuracy(result, expected, 2.)
    np.testing.assert_allclose(result.value, expected, atol=2e-14, rtol=0)


@pytest.mark.parametrize("value,weights,radius", [
    (np.empty((0, 2)), [1., 2.], .8),
    (np.ones((3, 2)), [1., 2.], 0.),
])
def test_empty_and_zero_radius_are_exact(value, weights, radius):
    result = solve_weighted_l1_ball_dual(value, weights, radius, absolute_gap_tol=1e-30)
    assert result.relative_certified and result.absolute_certified
    assert result.n_iter == 0 and result.gap_upper == result.error_bound == 0
    np.testing.assert_array_equal(result.value, np.zeros_like(value))


def test_small_primal_step_alone_cannot_certify_dual_accuracy():
    # The projected primal is constant while the box dual is still moving.
    result = solve_weighted_l1_ball_dual([[100.]], [[5.]], 1., max_iterations=1)
    assert result.residual == 0
    assert not result.relative_certified
    assert result.gap_upper > 1.


def test_certificate_rejects_infeasible_primal_and_dual():
    A, weights, radius, expected, p = boundary_kkt_problem()
    q = A - expected - p
    with pytest.raises(ArithmeticError, match="box"):
        _certificate(A, weights, radius, expected, p + 2., q)
    with pytest.raises(ArithmeticError, match="feasible"):
        _certificate(A, weights, radius, expected * 1.1, p, q)


def test_svd_norm_bounds_cover_known_spectrum():
    value = np.array([[0., -3.], [.1, 0.], [0., 0.]])
    operator, nuclear = _spectral_bounds(value)
    assert 3. <= operator < 3. + 1e-12
    assert 3.1 <= nuclear < 3.1 + 1e-12


@pytest.mark.parametrize("kwargs", [
    {"max_iterations": 0}, {"max_iterations": True}, {"max_iterations": 1.5},
    {"tolerance": 0.}, {"absolute_gap_tol": 0.}, {"absolute_gap_tol": np.inf},
    {"initial_p": np.zeros((2, 3))}, {"initial_p": np.full((2, 2), np.nan)},
])
def test_invalid_accuracy_and_initial_dual_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_weighted_l1_ball_dual(np.eye(2), .1, 1., **kwargs)


def test_invalid_problem_and_overflow_are_rejected():
    for A, weights, radius in (([1., 2.], .1, 1.), (np.eye(2), -.1, 1.),
                               (np.eye(2), .1, -1.), (np.eye(2), [1., 2., 3.], 1.)):
        with pytest.raises(ValueError):
            solve_weighted_l1_ball_dual(A, weights, radius)
    with pytest.raises(FloatingPointError, match="norm"):
        solve_weighted_l1_ball_dual(np.full((2, 2), 1e200), .1, 1.)
