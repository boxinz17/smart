"""Entrywise reduced Lasso and rank truncation for stage one."""

from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from .source import _positive_integer, _positive_real, _real_matrix, deterministic_svd


class InitializationFailure(ValueError):
    """The numerical initialization cannot be constructed."""


@dataclass(frozen=True)
class LassoInitialization:
    P: np.ndarray
    d: np.ndarray
    Q: np.ndarray
    coefficient: np.ndarray
    dual_gaps: np.ndarray
    kkt_residual: float
    converged: bool
    n_iter: np.ndarray


def reduced_lasso(design, response, rank, penalty, *, tol=1e-9, max_iter=20000, tie_tol=1e-12):
    """Minimize ``||response - design B||²/(2n) + penalty*|B|_1``.

    Each response is an ordinary Lasso with no intercept or rescaling.
    ``coefficient`` is the raw Lasso solution; ``P, d, Q`` give its
    rank-truncated approximation. The numerical minimizer is not claimed
    to be the canonical minimum-norm member of a nonunique solution set.
    Exactly zero response columns have the unique solution zero for positive
    penalty and are returned with zero dual gap and zero solver iterations.
    """
    design = _real_matrix(design, "design")
    response = _real_matrix(response, "response")
    rank = _positive_integer(rank, "rank")
    max_iter = _positive_integer(max_iter, "max_iter")
    penalty = _positive_real(penalty, "penalty")
    tol = _positive_real(tol, "tol")
    tie_tol = _positive_real(tie_tol, "tie_tol")
    if design.shape[0] < 1 or design.shape[1] < 1 or response.shape[1] < 1:
        raise ValueError("design and response must be nonempty")
    if response.shape[0] != design.shape[0]:
        raise ValueError("design and response must have the same number of rows")
    if rank > min(design.shape[1], response.shape[1]):
        raise ValueError("rank exceeds the reduced coefficient dimensions")
    if not np.isfinite(penalty) or penalty <= 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("penalty and tol must be positive")
    coefficient = np.empty((design.shape[1], response.shape[1]))
    dual_gaps = np.empty(response.shape[1])
    n_iter = np.empty(response.shape[1], dtype=int)
    warned = False
    for column in range(response.shape[1]):
        if not np.any(response[:, column]):
            # With positive penalty, zero is the unique minimizer for y=0.
            # Some sklearn versions warn after exhausting max_iter because
            # their response-scaled dual-gap tolerance is also exactly zero.
            # Use exact equality: tiny nonzero responses still need a solve.
            coefficient[:, column] = 0.
            dual_gaps[column] = 0.
            n_iter[column] = 0
            continue
        solver = Lasso(alpha=penalty, fit_intercept=False, tol=tol, max_iter=max_iter, selection="cyclic")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            solver.fit(design, response[:, column])
        warned |= any(issubclass(item.category, ConvergenceWarning) for item in caught)
        coefficient[:, column] = solver.coef_
        dual_gaps[column] = float(solver.dual_gap_)
        n_iter[column] = int(solver.n_iter_)
    if not np.isfinite(coefficient).all() or not np.isfinite(dual_gaps).all():
        raise InitializationFailure("Lasso returned nonfinite coefficients or dual gaps")
    gradient = design.T @ (design @ coefficient - response) / design.shape[0]
    residual = np.maximum(np.abs(gradient) - penalty, 0.0)
    active = coefficient != 0
    residual[active] = np.abs(gradient[active] + penalty * np.sign(coefficient[active]))
    kkt_residual = float(residual.max(initial=0.0))
    # The solver's relative dual-gap tolerance and an independently checked
    # first-order residual both must pass. Report the residual itself too.
    gradient_scale = max(1.0, penalty, float(np.max(np.abs(design.T @ response / design.shape[0]))))
    kkt_tolerance = max(100 * np.finfo(float).eps * gradient_scale, np.sqrt(tol) * gradient_scale)
    converged = not warned and kkt_residual <= kkt_tolerance
    P, d, Qt = deterministic_svd(coefficient, tie_tol=tie_tol)
    return LassoInitialization(P[:, :rank], d[:rank], Qt.T[:, :rank], coefficient, dual_gaps,
                               kkt_residual, bool(converged), n_iter)
