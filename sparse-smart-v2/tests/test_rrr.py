import numpy as np
import pytest

from sparse_smart_v2.rrr import shortcut_reason, target_rrr


def _loss(x, y, coefficient):
    return np.linalg.norm(y - x @ coefficient, "fro") ** 2 / (2 * len(x))


def test_design_weighted_truncation_is_not_coefficient_svd_truncation():
    x, y = np.diag([100., 1.]), np.diag([2., 1.])
    result = target_rrr(x, y, 1)
    np.testing.assert_allclose(result.coefficient, np.diag([.02, 0]), atol=1e-15)
    assert result.training_loss == pytest.approx(.25)
    ols = np.linalg.solve(x, y)
    u, d, vt = np.linalg.svd(ols)
    bad_coefficient = (u[:, :1] * d[:1]) @ vt[:1]
    assert _loss(x, y, bad_coefficient) > result.training_loss * 3


@pytest.mark.parametrize("shape,rank", [((15, 6, 4), 2), ((8, 12, 5), 3), ((9, 5, 8), 4)])
def test_independent_fitted_response_reference_and_factor_stationarity(shape, rank):
    n, p, q = shape
    rng = np.random.default_rng(734)
    x = rng.normal(size=(n, p)) * np.geomspace(.3, 5, p)
    y = rng.normal(size=(n, q))
    result = target_rrr(x, y, rank)
    # Independent reference first solves unrestricted least squares, then
    # truncates fitted responses. It never truncates the coefficient.
    ols = np.linalg.lstsq(x, y, rcond=None)[0]
    uf, sf, vtf = np.linalg.svd(x @ ols, full_matrices=False)
    expected_fit = (uf[:, :rank] * sf[:rank]) @ vtf[:rank]
    expected_coef = np.linalg.lstsq(x, expected_fit, rcond=None)[0]
    np.testing.assert_allclose(result.fitted_values, expected_fit, atol=5e-13)
    np.testing.assert_allclose(result.coefficient, expected_coef, atol=5e-13)
    np.testing.assert_allclose((result.left_factors * result.singular_values) @ result.right_factors.T,
                               result.coefficient, atol=5e-13)
    gradient = x.T @ (result.fitted_values - y)
    np.testing.assert_allclose(gradient @ result.right_factors, 0, atol=2e-11)
    np.testing.assert_allclose(result.left_factors.T @ gradient, 0, atol=2e-11)
    assert result.certificate["certified"]
    assert abs(result.objective_gap) <= result.certificate["loss_tolerance"]
    for _ in range(10):
        arbitrary = rng.normal(size=(p, rank)) @ rng.normal(size=(rank, q))
        assert result.training_loss <= _loss(x, y, arbitrary)


def test_duplicate_columns_minimum_norm_lift_and_design_nullspace():
    rng = np.random.default_rng(37)
    basis = rng.normal(size=(7, 3))
    x = np.column_stack((basis, basis[:, 0], np.zeros(7), basis[:, 2]))
    y = rng.normal(size=(7, 4))
    result = target_rrr(x, y, 2)
    assert result.design_rank == 3
    np.testing.assert_allclose(result.coefficient[0], result.coefficient[3], atol=1e-13)
    np.testing.assert_allclose(result.coefficient[2], result.coefficient[5], atol=1e-13)
    np.testing.assert_allclose(result.coefficient[4], 0, atol=1e-13)
    expected = np.linalg.lstsq(x, result.fitted_values, rcond=None)[0]
    np.testing.assert_allclose(result.coefficient, expected, atol=2e-13)
    null_shift = np.zeros_like(expected)
    null_shift[0] = 1
    null_shift[3] = -1
    np.testing.assert_allclose(x @ null_shift, 0, atol=1e-14)
    assert np.linalg.norm(expected + null_shift) > np.linalg.norm(expected)
    assert result.certificate["design_nullspace_residual_norm"] < 1e-12


@pytest.mark.parametrize("zero_design,zero_response,rank", [(True, False, 2), (False, True, 2), (False, False, 0)])
def test_zero_solutions_have_no_artificial_spectral_floor(zero_design, zero_response, rank):
    rng = np.random.default_rng(7)
    x = np.zeros((8, 5)) if zero_design else rng.normal(size=(8, 5))
    y = np.zeros((8, 3)) if zero_response else rng.normal(size=(8, 3))
    result = target_rrr(x, y, rank)
    assert result.effective_rank == 0
    assert result.left_factors.shape == (5, 0)
    assert result.singular_values.shape == (0,)
    assert result.right_factors.shape == (3, 0)
    np.testing.assert_array_equal(result.coefficient, np.zeros((5, 3)))
    assert result.training_loss == pytest.approx(np.linalg.norm(y) ** 2 / 16)
    assert result.certificate["certified"]


def test_rank_deficient_signal_returns_rank_at_most_requested():
    rng = np.random.default_rng(634)
    x = rng.normal(size=(20, 7))
    c = np.outer(rng.normal(size=7), rng.normal(size=5))
    result = target_rrr(x, x @ c, 4)
    assert result.requested_rank == 4
    assert result.effective_rank == 1
    np.testing.assert_allclose(result.coefficient, c, atol=3e-14)
    assert result.training_loss < 1e-26


def test_tied_cutoff_uses_canonical_response_coordinates():
    result = target_rrr(np.eye(3), np.eye(3), 1)
    np.testing.assert_allclose(result.coefficient, np.diag([1., 0., 0.]), atol=1e-14)
    assert result.certificate["rank_cutoff_tied"]
    assert result.training_loss == pytest.approx(1 / 3)
    # An arbitrary left-coordinate rotation must not rotate a tied response
    # selection. The same fit is recovered after undoing that rotation.
    rotation = np.linalg.qr(np.random.default_rng(31).normal(size=(3, 3)))[0]
    rotated = target_rrr(rotation, rotation, 1)
    np.testing.assert_allclose(rotated.coefficient, result.coefficient, atol=1e-13)


def test_near_but_distinct_response_singular_values_are_not_source_ties():
    # The source helper's 1e-12 tie convention is intentionally not used here.
    y = np.diag([1., 1. + 2e-13])
    result = target_rrr(np.eye(2), y, 1)
    np.testing.assert_allclose(result.coefficient, np.diag([0., y[1, 1]]), atol=1e-15)
    assert not result.certificate["rank_cutoff_tied"]


def test_numerical_design_rank_threshold_is_recorded_without_jitter():
    result = target_rrr(np.diag([1., 1e-18]), np.eye(2), 2)
    assert result.design_rank == 1
    assert result.certificate["design_rank_tolerance"] > 1e-18
    np.testing.assert_array_equal(result.coefficient, np.diag([1., 0.]))


def test_result_arrays_are_independent_and_read_only():
    x, y = np.eye(3), np.eye(3)
    result = target_rrr(x, y, 1)
    x[:] = 10
    y[:] = 10
    np.testing.assert_array_equal(result.coefficient, np.diag([1., 0., 0.]))
    for name in ("coefficient", "fitted_values", "singular_values", "left_factors", "right_factors"):
        assert not getattr(result, name).flags.writeable


@pytest.mark.parametrize("x,y,rank", [
    (np.eye(2), np.eye(2), -1), (np.eye(2), np.eye(2), True),
    (np.eye(2), np.eye(2), 1.2), (np.eye(2), np.eye(2), 3),
    (np.eye(2), np.eye(3), 1), (np.empty((0, 2)), np.empty((0, 2)), 1),
    (np.empty((2, 0)), np.eye(2), 0), (np.eye(2), np.empty((2, 0)), 0),
    (np.eye(2).astype(complex), np.eye(2), 1), (np.eye(2), [[1., np.nan], [0, 1]], 1),
    ([1., 2.], np.eye(2), 1), ([[1., np.inf], [0, 1]], np.eye(2), 1),
])
def test_rejects_invalid_inputs(x, y, rank):
    with pytest.raises(ValueError):
        target_rrr(x, y, rank)


@pytest.mark.parametrize("free,penalties,caps,expected", [
    ((2, 2), (0, 0), (6, 4), "zero_effective_penalties_full_support"),
    (None, (0, 0), (6, 4), "zero_effective_penalties_full_support"),
    ((5, 4), (1., 2.), (0, 0), "all_rows_unpenalized"),
    ((5, 2), (3., 0), (0, 4), "zero_effective_penalties_full_support"),
    ((2, 2), (0, 0), (5, 4), None),
    ((2, 2), (0, 1.), (6, 4), None),
    ((5, 2), (0, 1.), (0, 4), None),
])
def test_shortcut_requires_effectively_zero_penalties_and_full_masked_capacities(free, penalties, caps, expected):
    assert shortcut_reason(rank=2, p=5, q=4, free_directions=free,
                           penalties=penalties, support_limits=caps) == expected


@pytest.mark.parametrize("overrides", [
    {"rank": 6}, {"p": True}, {"free_directions": (1, 2)},
    {"free_directions": (6, 2)}, {"free_directions": (2, True)},
    {"penalties": (-1, 0)}, {"penalties": (0, np.nan)},
    {"support_limits": (7, 4)}, {"support_limits": (-1, 4)},
    {"support_limits": (6., 4)},
])
def test_shortcut_validates_inputs_even_if_penalties_would_not_trigger(overrides):
    options = dict(rank=2, p=5, q=4, free_directions=(2, 2), penalties=(1, 1), support_limits=(6, 4))
    options.update(overrides)
    with pytest.raises(ValueError):
        shortcut_reason(**options)
