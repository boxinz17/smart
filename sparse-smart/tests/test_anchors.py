import numpy as np
import pytest

from sparse_smart.anchors import AnchorFailure, select_anchor


def test_row_energy_screening_and_index_ties():
    factor = np.ones((4, 1)) / 2
    result = select_anchor(factor, anchor_min=0.05)
    np.testing.assert_array_equal(result.screened_indices, [0, 1, 2])
    np.testing.assert_array_equal(result.indices, [0])
    np.testing.assert_allclose(result.center, [[1.]])
    assert result.min_singular_value == 0.5


def test_anchor_check_uses_original_factor_not_whitened_factor():
    factor = np.ones((4, 1)) / 2
    # The original selected entry is .5; whitening would inflate it to .577.
    with pytest.raises(AnchorFailure, match="No anchor certified"):
        select_anchor(factor, anchor_min=0.54 / 8)


def test_whitening_exchange_bound_original_polar_and_screen_minimality():
    rng = np.random.default_rng(18)
    factor = np.linalg.qr(rng.normal(size=(30, 5)))[0]
    result = select_anchor(factor, anchor_min=0.001, qr_threshold=1.1)
    assert result.n_exchanges >= 1
    screened = result.screened_indices
    gram = factor[screened].T @ factor[screened]
    assert np.linalg.eigvalsh(gram)[0] >= 0.75 - 1e-13
    assert np.linalg.eigvalsh(factor[screened[:-1]].T @ factor[screened[:-1]])[0] < 0.75
    values, vectors = np.linalg.eigh(gram)
    white = factor[screened] @ ((vectors * (1 / np.sqrt(values))) @ vectors.T)
    selected = [int(np.flatnonzero(screened == index)[0]) for index in result.indices]
    remaining = [index for index in range(len(screened)) if index not in selected]
    transform = np.linalg.solve(white[selected].T, white[remaining].T)
    assert np.max(np.abs(transform)) <= 1.1 + 1e-12
    original = factor[result.indices]
    np.testing.assert_allclose(result.center.T @ result.center, np.eye(5), atol=1e-12)
    polar_positive = result.center.T @ original
    np.testing.assert_allclose(polar_positive, polar_positive.T, atol=1e-12)
    assert np.linalg.eigvalsh(polar_positive)[0] > 0
    assert np.all(np.diff(result.indices) > 0)


def test_square_factor_needs_no_complement_or_exchanges():
    factor = np.array([[0., -1.], [1., 0.]])
    result = select_anchor(factor, anchor_min=0.1)
    np.testing.assert_array_equal(result.indices, [0, 1])
    np.testing.assert_allclose(result.center, factor)
    assert result.n_exchanges == 0


def test_exchange_budget_failure_is_explicit():
    factor = np.linalg.qr(np.random.default_rng(3).normal(size=(50, 8)))[0]
    with pytest.raises(AnchorFailure, match="exchange limit"):
        select_anchor(factor, anchor_min=0.001, qr_threshold=1.01, max_exchanges=1)


def test_invalid_factor_or_qr_configuration():
    with pytest.raises(ValueError, match="orthonormal"):
        select_anchor(np.ones((3, 2)), anchor_min=0.01)
    with pytest.raises(ValueError, match="greater than one"):
        select_anchor(np.eye(2), anchor_min=0.01, qr_threshold=1)
