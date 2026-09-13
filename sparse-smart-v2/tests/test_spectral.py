"""Independent convex-reference and floating-boundary spectral prox checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import Bounds, LinearConstraint, minimize

from sparse_smart_v2.spectral import project_spectrum, spectral_proximal_step


@pytest.mark.parametrize("rank", [1, 2, 3, 5, 8])
def test_projection_matches_independent_convex_quadratic_reference(rank):
    rng = np.random.default_rng(480 + rank)
    lower, upper, gap = .2, 4., .15
    constraints = []
    if rank > 1:
        difference = np.eye(rank)[:-1] - np.eye(rank)[1:]
        constraints = [LinearConstraint(difference, gap, np.inf)]
    for _ in range(6):
        proposal = rng.normal(1.5, 3., rank)
        actual = project_spectrum(proposal, d_lower=lower, d_upper=upper, gap=gap)
        reference = minimize(lambda d: .5 * np.sum((d - proposal)**2),
            np.linspace(upper, lower, rank), jac=lambda d: d - proposal,
            bounds=Bounds(lower, upper), constraints=constraints, method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000})
        assert reference.success, reference.message
        assert_allclose(actual, reference.x, atol=2e-7, rtol=2e-7)
        assert np.all(actual >= lower) and np.all(actual <= upper)
        assert np.all(actual[:-1] - actual[1:] >= gap)


def test_projection_pools_without_reordering_singular_value_identities():
    result = project_spectrum([1., 4.], d_lower=.1, d_upper=10., gap=1.)
    assert_array_equal(result, [3., 2.])
    assert np.sum((result - [1., 4.])**2) < np.sum((np.array([4., 1.]) - [1., 4.])**2)


def test_already_feasible_projection_and_gradient_trial_are_byte_identical():
    values = np.array([3.11, 1.51, .401])
    projected = project_spectrum(values, d_lower=.1, d_upper=8., gap=.01)
    assert projected.tobytes() == values.tobytes()
    gradient = np.array([.3, -.07, .001])
    expected = values - gradient / 20.
    actual, mapped = spectral_proximal_step(values, gradient, 20., d_lower=.1, d_upper=8., gap=.01)
    assert actual.tobytes() == expected.tobytes()
    assert mapped.tobytes() == gradient.tobytes()


def test_roundoff_at_active_gap_and_bounds_stays_inside_strict_domain():
    actual = project_spectrum([1., 1.], d_lower=.05, d_upper=12., gap=.01)
    assert actual[0] - actual[1] >= .01
    assert_allclose(actual, [1.005, .995], atol=8 * np.finfo(float).eps, rtol=0)
    assert project_spectrum(actual, d_lower=.05, d_upper=12., gap=.01).tobytes() == actual.tobytes()
    for proposal in ([100., 100., 100.], [-100., -100., -100.], [100., -100., 100.]):
        point = project_spectrum(proposal, d_lower=.05, d_upper=12., gap=.01)
        assert np.all(point >= .05) and np.all(point <= 12.)
        assert np.all(point[:-1] - point[1:] >= .01)
    assert_array_equal(project_spectrum([9., -7., 4.], d_lower=1., d_upper=2., gap=.5), [2., 1.5, 1.])


def test_active_gap_residual_and_bound_residual_have_exact_fixed_points():
    d = np.array([2., 1.5])
    actual, mapped = spectral_proximal_step(d, [1., -1.], 20., d_lower=.1, d_upper=10., gap=.5)
    assert_array_equal(actual, d)
    assert_array_equal(mapped, [0., 0.])
    for point, gradient in ((1., 2.), (10., -2.)):
        actual, mapped = spectral_proximal_step([point], [gradient], 20., d_lower=1., d_upper=10., gap=.1)
        assert_array_equal(actual, [point])
        assert_array_equal(mapped, [0.])


def test_cancellation_safe_mapping_retains_tiny_interior_gradient():
    d, gradient = np.array([1e12]), np.array([1e-6])
    actual, mapped = spectral_proximal_step(d, gradient, 20., d_lower=.01, d_upper=1e14, gap=.01)
    assert_array_equal(actual, d)
    assert_array_equal(mapped, gradient)
    # The same scale at an active upper bound has a genuinely zero outward
    # mapping, although its unconstrained float64 trial also rounds away.
    actual, mapped = spectral_proximal_step(d, -gradient, 20., d_lower=.01, d_upper=1e12, gap=.01)
    assert_array_equal(actual, d)
    assert_array_equal(mapped, [0.])


def test_active_large_level_pooling_keeps_tiny_tangential_gradient():
    d = np.array([1e12 + .5, 1e12])
    gradient = np.array([2e-6, -1e-6])
    actual, mapped = spectral_proximal_step(d, gradient, 20., d_lower=.01, d_upper=1e14, gap=.5)
    assert_array_equal(actual, d)
    assert_allclose(mapped, [5e-7, 5e-7], atol=0, rtol=1e-15)


@pytest.mark.parametrize("values,kwargs", [
    ([], {}), ([np.nan], {}), ([1 + 1j], {}), ([[1., 2.]], {}),
    ([1., 2.], {"d_lower": 0}), ([1., 2.], {"gap": -1}),
    ([1., 2., 3.], {"d_lower": 1., "d_upper": 2., "gap": 1.}),
])
def test_invalid_or_empty_spectral_domains_fail_explicitly(values, kwargs):
    bounds = dict(d_lower=.1, d_upper=4., gap=.1)
    bounds.update(kwargs)
    with pytest.raises(ValueError):
        project_spectrum(values, **bounds)
