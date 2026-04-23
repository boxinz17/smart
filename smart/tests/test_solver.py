"""End-to-end tests for the low-level SMARTSolver."""

import numpy as np

from smart import SMARTSolver, evaluate_model_avg_err, fit_baseline


def test_solver_default_gamma_matches_paper():
    """The public default of `gamma` must stay at the paper's main value."""
    solver = SMARTSolver(C0=np.eye(4), r_u=1, r_v=1)
    assert solver.gamma == 2.0


def test_solver_beats_ridge_on_clean_transfer(tiny_problem):
    """With a near-perfect source subspace, SMART should improve on ridge."""
    X, Y = tiny_problem["X"], tiny_problem["Y"]
    C_true, C0 = tiny_problem["C_star"], tiny_problem["C0"]
    r_star = 3
    r_s = 6

    C_ridge, _ = fit_baseline(X, Y, model_type="ridge",
                              alphas=np.logspace(-3, 2, 6))

    solver = SMARTSolver(C0=C0, r_u=r_s, r_v=r_s, t_max=100)
    solver.initialize(r=r_star, X=X, Y=Y, C=C_ridge)
    solution = solver.select_hyperparameters_via_bic(
        C_init=C_ridge, use_optuna=False,
        lambda_grid_u=np.logspace(-2, 1, 4),
        lambda_grid_v=np.logspace(-2, 1, 4),
    )
    C_smart = solution["C_hat"]
    assert C_smart.shape == C_true.shape

    err_ridge = evaluate_model_avg_err(C_ridge, C_true)
    err_smart = evaluate_model_avg_err(C_smart, C_true)
    assert err_smart < err_ridge
