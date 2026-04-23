"""Tests for the helpers exported from smart.utils."""

import numpy as np

from smart import (
    evaluate_model_avg_err,
    extract_svd_subspaces,
    fit_baseline,
    generate_data,
)


def test_generate_data_shapes_and_determinism():
    n, p, q = 120, 30, 15
    data_a = generate_data(n=n, p=p, q=q, sigma0=0.01, random_seed=7)
    data_b = generate_data(n=n, p=p, q=q, sigma0=0.01, random_seed=7)

    assert data_a["X"].shape == (n, p)
    assert data_a["Y"].shape == (n, q)
    assert data_a["C_star"].shape == (p, q)
    assert data_a["C0"].shape == (p, q)

    # Reproducibility: the same seed must give bit-identical output.
    assert np.allclose(data_a["X"], data_b["X"])
    assert np.allclose(data_a["Y"], data_b["Y"])


def test_evaluate_model_avg_err_is_zero_on_perfect_match():
    C = np.random.default_rng(0).standard_normal((8, 5))
    assert evaluate_model_avg_err(C, C) == 0.0


def test_extract_svd_subspaces_returns_orthonormal(tiny_problem):
    C0 = tiny_problem["C0"]
    U_s, V_s = extract_svd_subspaces(C0, r_u=3, r_v=3)
    assert U_s.shape == (C0.shape[0], 3)
    assert V_s.shape == (C0.shape[1], 3)
    assert np.allclose(U_s.T @ U_s, np.eye(3), atol=1e-8)
    assert np.allclose(V_s.T @ V_s, np.eye(3), atol=1e-8)


def test_fit_baseline_ridge_recovers_reasonable_fit(tiny_problem):
    X, Y, C_true = tiny_problem["X"], tiny_problem["Y"], tiny_problem["C_star"]
    C_hat, alphas = fit_baseline(X, Y, model_type="ridge",
                                 alphas=np.logspace(-3, 2, 6))
    assert C_hat.shape == C_true.shape
    assert len(alphas) == C_true.shape[1]
    # Ridge on the training data should be much better than the zero predictor.
    baseline_err = evaluate_model_avg_err(np.zeros_like(C_true), C_true)
    ridge_err = evaluate_model_avg_err(C_hat, C_true)
    assert ridge_err < baseline_err
