"""Tests for the high-level SMART model-selection wrapper."""

import numpy as np

from smart import SMART, evaluate_model_avg_err, fit_baseline


def test_smart_run_full_selection(tiny_problem):
    X, Y = tiny_problem["X"], tiny_problem["Y"]
    C_true, C0 = tiny_problem["C_star"], tiny_problem["C0"]

    model = SMART(X, Y, C0=C0)
    model.run_full_selection(
        verbose=False,
        use_optuna_rank=False,
        fit_kwargs={"use_optuna": False,
                    "lambda_grid_u": np.logspace(-2, 1, 4),
                    "lambda_grid_v": np.logspace(-2, 1, 4),
                    "t_max": 60,
                    "tol": 1e-2},
    )

    estimates = model.get_estimates()
    assert estimates["C_hat"].shape == C_true.shape
    assert model.r_hat is not None and model.r_hat >= 1
    assert model.best_r_u is not None and model.best_r_v is not None

    # Sanity: SMART should not be catastrophically worse than ridge.
    C_ridge, _ = fit_baseline(X, Y, model_type="ridge",
                              alphas=np.logspace(-3, 2, 6))
    err_ridge = evaluate_model_avg_err(C_ridge, C_true)
    err_smart = evaluate_model_avg_err(estimates["C_hat"], C_true)
    assert err_smart <= err_ridge * 1.5
