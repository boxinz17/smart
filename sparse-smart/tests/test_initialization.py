import numpy as np
import pytest

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


def test_lasso_nonconvergence_is_reported():
    rng = np.random.default_rng(45)
    design = rng.normal(size=(20, 40))
    response = rng.normal(size=(20, 10))
    result = reduced_lasso(design, response, 2, 0.001, max_iter=1, tol=1e-12)
    assert not result.converged
    assert np.all(result.n_iter == 1)


@pytest.mark.parametrize("kwargs", [{"rank": 0, "penalty": 0.1},
                                    {"rank": 3, "penalty": 0.1},
                                    {"rank": 1, "penalty": 0},
                                    {"rank": 1, "penalty": np.nan}])
def test_lasso_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        reduced_lasso(np.eye(2), np.eye(2), **kwargs)
