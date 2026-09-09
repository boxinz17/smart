import itertools
import numpy as np
import pytest
from sparse_smart.thresholding import hard_threshold, soft_threshold, threshold_step


def test_quadratic_update_matches_exhaustive_small_support_reference():
    # Exhaustive search is a tiny test oracle, never used by the estimator.
    u = np.array([1.7, -0.9, 0.4, -1.7, 0.1])
    lam, L, k = 0.3, 2.0, 2
    candidate = threshold_step(u, lam / L, k)
    objective = lambda w: L / 2 * np.sum((w - u)**2) + lam * np.abs(w).sum()
    possibilities = []
    for size in range(k + 1):
        for indices in itertools.combinations(range(u.size), size):
            w = np.zeros_like(u)
            w[list(indices)] = soft_threshold(u[list(indices)], lam / L)
            possibilities.append(objective(w))
    assert objective(candidate) == pytest.approx(min(possibilities))
    np.testing.assert_array_equal(hard_threshold([[2., -2.], [2., 1.]], 2), [[2., -2.], [0., 0.]])


def test_empty_and_saturated_supports():
    np.testing.assert_array_equal(hard_threshold([1, 2], 0), [0, 0])
    np.testing.assert_array_equal(hard_threshold([1, 2], 2), [1, 2])
    assert hard_threshold(np.empty((0, 2)), 0).shape == (0, 2)
    for bad in (-1, 3, 1.5, True):
        with pytest.raises(ValueError):
            hard_threshold([1, 2], bad)

