import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

import sparse_smart.initialization as initialization
from sparse_smart.initialization import reduced_lasso


def test_lasso_objective_normalization_has_soft_threshold_solution():
    n = 4
    design = np.sqrt(n) * np.eye(n)
    coefficient = np.array([[3., -1.], [0.1, 0.7], [0., -2.], [-0.3, 0.]])
    penalty = 0.4
    result = reduced_lasso(design, design @ coefficient, 2, penalty)
    expected = np.sign(coefficient) * np.maximum(np.abs(coefficient) - penalty, 0)
    np.testing.assert_allclose(result.coefficient, expected, atol=1e-12)
    np.testing.assert_allclose((result.P * result.d) @ result.Q.T, expected, atol=1e-12)
    assert result.converged
    assert result.kkt_residual < 1e-12
    assert np.max(np.abs(result.dual_gaps)) < 1e-12


def test_lasso_handles_more_source_coordinates_than_samples():
    rng = np.random.default_rng(11)
    design = rng.normal(size=(12, 20))
    coefficient = np.zeros((20, 20))
    coefficient[0, 0] = 3
    coefficient[3, 8] = -2
    response = design @ coefficient
    result = reduced_lasso(design, response, 2, 0.2)
    assert result.converged
    assert result.coefficient.shape == (20, 20)
    assert result.P.shape == result.Q.shape == (20, 2)
    residual = response - design @ result.coefficient
    gradient = -design.T @ residual / len(design)
    inactive = result.coefficient == 0
    assert np.max(np.abs(gradient[inactive])) <= 0.2 + 1e-7
    np.testing.assert_allclose(gradient[~inactive], -0.2 * np.sign(result.coefficient[~inactive]), atol=1e-7)
    zero_columns = ~np.any(response, axis=0)
    np.testing.assert_array_equal(result.n_iter[zero_columns], 0)
    np.testing.assert_array_equal(result.dual_gaps[zero_columns], 0.)
    nonzero = reduced_lasso(design, response[:, ~zero_columns], 2, 0.2)
    np.testing.assert_array_equal(result.coefficient[:, ~zero_columns], nonzero.coefficient)


def test_zero_response_is_solved_exactly_without_calling_lasso(monkeypatch):
    def unexpected_solver(**kwargs):
        raise AssertionError("zero responses must not call the numerical solver")
    monkeypatch.setattr(initialization, "Lasso", unexpected_solver)
    design = np.random.default_rng(19).normal(size=(3, 5))
    result = reduced_lasso(design, np.zeros((3, 4)), 2, .1, max_iter=1)
    assert result.converged and result.kkt_residual == 0.
    np.testing.assert_array_equal(result.coefficient, np.zeros((5, 4)))
    np.testing.assert_array_equal(result.dual_gaps, np.zeros(4))
    np.testing.assert_array_equal(result.n_iter, np.zeros(4, dtype=int))
    np.testing.assert_array_equal(result.d, np.zeros(2))


def test_tiny_nonzero_response_is_not_classified_as_zero(monkeypatch):
    def reached_solver(**kwargs):
        raise RuntimeError("nonzero response reached solver")
    monkeypatch.setattr(initialization, "Lasso", reached_solver)
    # Squaring these entries underflows; a norm or allclose check is unsafe.
    with pytest.raises(RuntimeError, match="nonzero response reached solver"):
        reduced_lasso(np.ones((3, 1)), np.full((3, 1), 1e-200), 1, .1)


def test_zero_columns_do_not_hide_nonconvergence_in_other_columns():
    rng = np.random.default_rng(45)
    design = rng.normal(size=(20, 40))
    response = np.column_stack((np.zeros(20), rng.normal(size=20), np.zeros(20)))
    result = reduced_lasso(design, response, 2, .001, max_iter=1, tol=1e-12)
    assert not result.converged
    np.testing.assert_array_equal(result.n_iter, [0, 1, 0])
    np.testing.assert_array_equal(result.coefficient[:, [0, 2]], np.zeros((40, 2)))


def test_lasso_nonconvergence_is_reported():
    rng = np.random.default_rng(45)
    design = rng.normal(size=(20, 40))
    response = rng.normal(size=(20, 10))
    result = reduced_lasso(design, response, 2, 0.001, max_iter=1, tol=1e-12)
    assert not result.converged
    assert np.all(result.n_iter == 1)


@pytest.mark.parametrize("nonzero_count,max_iter", [(1, 20000), (3, 20000), (3, 1)])
def test_batched_lasso_matches_independent_response_solves(nonzero_count, max_iter, monkeypatch):
    rng = np.random.default_rng(85)
    design = rng.normal(size=(30, 12))
    response = np.zeros((30, 6))
    nonzero_columns = np.array([1, 3, 5])[:nonzero_count]
    response[:, nonzero_columns] = rng.normal(size=(30, nonzero_count))
    coefficients = np.zeros((12, 6))
    dual_gaps = np.zeros(6)
    n_iter = np.zeros(6, dtype=int)
    warned = False
    for column in nonzero_columns:
        solver = Lasso(alpha=.03, fit_intercept=False, tol=1e-9,
                       max_iter=max_iter, selection="cyclic")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            solver.fit(design, response[:, column])
        warned |= any(issubclass(item.category, ConvergenceWarning) for item in caught)
        coefficients[:, column] = solver.coef_
        dual_gaps[column] = solver.dual_gap_
        n_iter[column] = solver.n_iter_

    fits = []

    class RecordingLasso(Lasso):
        def fit(self, X, y, **kwargs):
            fits.append(y.shape)
            return super().fit(X, y, **kwargs)

    monkeypatch.setattr(initialization, "Lasso", RecordingLasso)
    result = reduced_lasso(design, response, 2, .03, max_iter=max_iter)
    assert fits == [(30, nonzero_count)]
    np.testing.assert_allclose(result.coefficient, coefficients, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.dual_gaps, dual_gaps, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(result.n_iter, n_iter)
    assert result.converged == (not warned)
    gradient = design.T @ (design @ coefficients - response) / len(design)
    residual = np.maximum(np.abs(gradient) - .03, 0.)
    active = coefficients != 0
    residual[active] = np.abs(gradient[active] + .03 * np.sign(coefficients[active]))
    np.testing.assert_allclose(result.kkt_residual, residual.max(), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("kwargs", [{"rank": 0, "penalty": 0.1},
                                    {"rank": 3, "penalty": 0.1},
                                    {"rank": 1, "penalty": 0},
                                    {"rank": 1, "penalty": np.nan}])
def test_lasso_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        reduced_lasso(np.eye(2), np.eye(2), **kwargs)
