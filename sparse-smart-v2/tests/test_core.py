from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sparse_smart.chart import AnchorChart
from sparse_smart.initialization import reduced_lasso as original_lasso
from sparse_smart.source import deterministic_svd, prepare_source as original_source

import sparse_smart_v2.source as source_module
import sparse_smart_v2.initialization as initialization_module
from sparse_smart_v2.calibration import PracticalCalibration
from sparse_smart_v2.initialization import InitializationFailure, reduced_lasso
from sparse_smart_v2.source import ExactSource, NoisySource, ObservedSource, prepare_source
from sparse_smart_v2.support import FreeRows, choose_free_rows, threshold_state, validate_support_limits


def test_exact_frames_are_full_and_preserve_supplied_orientation():
    left = np.eye(5)[:, [3, 1]] * [-1, 1]
    right = np.eye(4)[:, [2, 0]]
    result = prepare_source(ExactSource(left, right), p=5, q=4, source_rank=2)
    assert result.left.shape == (5, 5)
    assert result.right.shape == (4, 4)
    np.testing.assert_array_equal(result.leading_left, left)
    np.testing.assert_array_equal(result.leading_right, right)
    np.testing.assert_allclose(result.left.T @ result.left, np.eye(5), atol=1e-12)
    left[3, 0] = 7
    assert result.left[3, 0] == -1
    for array in (result.left, result.right, result.leading_left, result.leading_right):
        assert not array.flags.writeable


@pytest.mark.parametrize("wrapper", [ObservedSource, lambda x: NoisySource(x, 0.1, 0.5), lambda x: x])
def test_observed_trailing_svd_directions_are_kept(wrapper):
    rng = np.random.default_rng(12)
    left = np.linalg.qr(rng.normal(size=(6, 4)))[0]
    right = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    coefficient = (left * [5, 3, 2, 1]) @ right.T
    U, d, Vt = deterministic_svd(coefficient)
    result = prepare_source(wrapper(coefficient), p=6, q=4, source_rank=2)
    np.testing.assert_array_equal(result.left[:, :4], U)
    np.testing.assert_array_equal(result.right, Vt.T)
    np.testing.assert_allclose((result.left[:, :4] * d) @ result.right.T, coefficient, atol=1e-12)
    np.testing.assert_array_equal(result.leading_left, result.left[:, :2])
    if result.mode == "observed":
        assert result.noise_std is None and result.gap_lower is None


def test_prepared_source_can_be_reused_without_svd(monkeypatch):
    prepared = prepare_source(np.diag([3.0, 2.0, 0.0]), p=3, q=3, source_rank=2)
    def fail(*args, **kwargs):
        raise AssertionError("source SVD must not be repeated")
    monkeypatch.setattr(source_module, "deterministic_svd", fail)
    reused = prepare_source(prepared, p=3, q=3, source_rank=2)
    np.testing.assert_array_equal(reused.left, prepared.left)
    assert not reused.left.flags.writeable
    assert not np.shares_memory(reused.left, prepared.left)
    with pytest.raises(ValueError, match="prefixes"):
        prepare_source(prepared, p=3, q=3, source_rank=1)


def test_reject_old_thin_prepared_frames_and_inconsistent_prefixes():
    old = original_source(ExactSource(np.eye(4)[:, :2], np.eye(3)[:, :2]), 2, 4, 3)
    with pytest.raises(ValueError, match="shape"):
        prepare_source(old, p=4, q=3, source_rank=2)
    full = prepare_source(np.eye(3), p=3, q=3, source_rank=2)
    bad = replace(full, leading_left=full.leading_left[:, ::-1])
    with pytest.raises(ValueError, match="prefixes"):
        prepare_source(bad, p=3, q=3, source_rank=2)


@pytest.mark.parametrize("rank", [0, -1, True, 1.5, 4])
def test_source_rank_validation(rank):
    with pytest.raises(ValueError):
        prepare_source(np.eye(3), p=3, q=3, source_rank=rank)


def test_calibration_accepts_zero_initializer_and_distinct_penalties():
    calibration = PracticalCalibration(0, (0, 0.2), 20, [2, 3])
    assert calibration.init_penalty == 0
    assert calibration.penalties == (0.0, 0.2)
    assert calibration.support_limits == (2, 3)
    assert PracticalCalibration(0, 0.1, 2, (0, 0)).penalties == (0.1, 0.1)


@pytest.mark.parametrize("kwargs", [
    {"init_penalty": -1}, {"init_penalty": True}, {"penalty": [1, -1]},
    {"step_size_inverse": 0}, {"support_limits": [1.5, 1]}, {"support_limits": [True, 0]},
])
def test_calibration_rejects_invalid_inputs(kwargs):
    settings = dict(init_penalty=0.1, penalty=0.1, step_size_inverse=20, support_limits=(1, 1))
    settings.update(kwargs)
    with pytest.raises(ValueError):
        PracticalCalibration(**settings)


def test_zero_initializer_is_minimum_norm_reduced_least_squares_then_svd():
    design = np.array([[1., 1., 0.], [0., 0., 1.], [-1., -1., 0.], [0., 0., -1.]])
    response = design @ np.array([[1., 0.], [0., 1.], [2., 3.]])
    fitted = reduced_lasso(design, response, rank=2, penalty=0)
    expected = np.array([[0.5, 0.5], [0.5, 0.5], [2., 3.]])
    np.testing.assert_allclose(fitted.coefficient, expected, atol=1e-12)
    np.testing.assert_allclose((fitted.P * fitted.d) @ fitted.Q.T, expected, atol=1e-12)
    assert fitted.converged
    assert fitted.kkt_residual < 1e-12


def test_full_rank_positive_initializer_reuses_original_primitive():
    rng = np.random.default_rng(19)
    design, response = rng.normal(size=(20, 3)), rng.normal(size=(20, 2))
    expected = original_lasso(design, response, 2, 0.1)
    actual = reduced_lasso(design, response, 2, 0.1)
    np.testing.assert_array_equal(actual.coefficient, expected.coefficient)
    np.testing.assert_array_equal(actual.d, expected.d)
    assert actual.converged == expected.converged


@pytest.mark.parametrize("second_sign", [1, -1])
@pytest.mark.parametrize("scale", [1.0, 1e-8])
def test_positive_lasso_ties_use_minimum_norm_split(second_sign, scale):
    column = np.array([-1., -1., 1., 1.])
    design = np.column_stack((column, second_sign * column))
    response = scale * np.column_stack((2 * column, -3 * column))
    fitted = reduced_lasso(design, response, 1, scale * 0.2)
    expected = np.array([[0.9, -1.4], [second_sign * 0.9, second_sign * -1.4]])
    np.testing.assert_allclose(fitted.coefficient / scale, expected, atol=1e-10)
    assert fitted.converged and fitted.kkt_residual < 1e-10


def test_minimum_norm_tie_requires_more_than_unchanged_lasso_objective(monkeypatch):
    column = np.array([-1., -1., 1., 1.])
    design = np.column_stack((column, column))
    def unchanged(function, initial, **kwargs):
        return SimpleNamespace(success=True, x=initial.copy(), nit=0)
    monkeypatch.setattr(initialization_module, "minimize", unchanged)
    with pytest.raises(InitializationFailure, match="norm-optimality"):
        reduced_lasso(design, (2 * column)[:, None], 1, 0.2)


def test_multiple_nonunique_groups_share_each_unique_lasso_coefficient():
    rng = np.random.default_rng(32)
    reduced = rng.normal(size=(20, 3))
    response = reduced @ np.array([[2., -1.], [0.5, 1.5], [-3., 2.]])
    unique = original_lasso(reduced, response, 2, 0.1)
    design = np.column_stack((reduced[:, 0], reduced[:, 0], reduced[:, 1],
                              -reduced[:, 1], reduced[:, 2], reduced[:, 2]))
    tied = reduced_lasso(design, response, 2, 0.1)
    expected = unique.coefficient[[0, 0, 1, 1, 2, 2]] / 2
    expected[3] *= -1
    np.testing.assert_allclose(tied.coefficient, expected, atol=1e-7)
    assert tied.converged


def test_minimum_norm_certificate_allows_nonunique_dual_multipliers():
    design = np.array([[1., 1., 1.], [0., 1., 10.]])
    fitted = reduced_lasso(design, np.array([[1.2], [0.]]), 1, 0.1)
    np.testing.assert_allclose(fitted.coefficient[:, 0], [1., 0., 0.], atol=1e-9)
    assert fitted.converged


def _chart():
    return AnchorChart(6, 5, [2, 4], [1, 4], np.eye(2), np.eye(2))


def test_free_rows_expand_anchors_in_source_order_and_masks_count_entries():
    chart = _chart()
    minimal = choose_free_rows(chart)
    expanded = choose_free_rows(chart, (4, 3))
    np.testing.assert_array_equal(minimal.rows_u, [2, 4])
    np.testing.assert_array_equal(expanded.rows_u, [0, 1, 2, 4])
    np.testing.assert_array_equal(expanded.rows_v, [0, 1, 4])
    np.testing.assert_array_equal(expanded.penalized_u, [[False, False], [False, False], [True, True], [True, True]])
    assert expanded.capacities == (4, 4)
    assert not expanded.rows_u.flags.writeable and not expanded.penalized_u.flags.writeable


@pytest.mark.parametrize("counts", [(1, 3), (7, 3), (3, 6), (True, 3), (2.5, 3)])
def test_free_row_counts_are_validated(counts):
    with pytest.raises(ValueError):
        choose_free_rows(_chart(), counts)


def test_masked_threshold_preserves_free_coordinates_and_breaks_ties_by_index():
    chart = _chart()
    free = choose_free_rows(chart, (4, 3))
    state = chart.pack([0.1], [-0.2], [3., 2.],
                       [[8., 9.], [10., 11.], [3., -3.], [3., -3.]],
                       [[12., 13.], [4., -4.], [4., -4.]])
    before = state.copy()
    actual = threshold_state(chart, state, free, (2., 4.), (2, 1), 2.)
    _, _, _, zu, zv = chart.unpack(actual)
    np.testing.assert_array_equal(zu, [[8., 9.], [10., 11.], [2., -2.], [0., 0.]])
    np.testing.assert_array_equal(zv, [[12., 13.], [2., 0.], [0., 0.]])
    np.testing.assert_array_equal(actual[:chart.rank ** 2], state[:chart.rank ** 2])
    np.testing.assert_array_equal(state, before)
    assert not np.shares_memory(actual, state)


def test_zero_penalty_keeps_hard_caps_and_empty_masks_have_zero_capacity():
    chart = _chart()
    state = chart.pack([0.], [0.], [3., 2.], np.ones((4, 2)), np.ones((3, 2)))
    free = choose_free_rows(chart)
    actual = threshold_state(chart, state, free, penalties=0, support_limits=(1, 0), inverse_step=1)
    assert np.count_nonzero(chart.unpack(actual)[3]) == 1
    assert np.count_nonzero(chart.unpack(actual)[4]) == 0
    all_free = choose_free_rows(chart, (6, 5))
    assert all_free.capacities == (0, 0)
    np.testing.assert_array_equal(threshold_state(chart, state, all_free, 99, (0, 0), 1), state)
    with pytest.raises(ValueError, match="capacity"):
        validate_support_limits(all_free, (1, 0))


def test_inconsistent_user_constructed_mask_is_rejected():
    chart = _chart()
    free = choose_free_rows(chart)
    bad = FreeRows(free.rows_u, free.rows_v, np.zeros_like(free.penalized_u), free.penalized_v)
    with pytest.raises(ValueError, match="mask"):
        threshold_state(chart, np.zeros(chart.size), bad, 0, (0, 0), 1)
