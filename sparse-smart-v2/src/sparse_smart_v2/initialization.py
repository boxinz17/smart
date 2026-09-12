"""Reduced Lasso--SVD initialization with a numerical minimum-norm tie rule."""

import numpy as np
from scipy.optimize import linprog, minimize

from sparse_smart.initialization import (
    InitializationFailure, LassoInitialization, reduced_lasso as _reduced_lasso,
)
from sparse_smart.source import deterministic_svd

from .calibration import _integer, _real
from .source import _matrix


def _kkt(design, response, coefficient, penalty):
    gradient = design.T @ (design @ coefficient - response) / design.shape[0]
    residual = np.maximum(np.abs(gradient) - penalty, 0)
    active = coefficient != 0
    residual[active] = np.abs(gradient[active] + penalty * np.sign(coefficient[active]))
    return gradient, float(residual.max(initial=0))


def _minimum_norm_stationarity(equality, values):
    """Check the convex norm subproblem's dual conditions at unit scale."""
    tolerance = 1e-7
    positive = values > tolerance
    dual = np.linalg.lstsq(equality[:, positive].T, -values[positive], rcond=None)[0]

    def valid(multiplier):
        residual = values + equality.T @ multiplier
        return (np.max(np.abs(residual[positive]), initial=0) <= tolerance
                and np.min(residual[~positive], initial=0) >= -tolerance
                and abs(float(values @ residual)) <= tolerance * max(1.0, float(values @ values)))

    if valid(dual):
        return True
    # The multiplier can be nonunique when the solution is on a boundary.
    # Search for any valid multiplier instead of rejecting the primal point
    # merely because the minimum-norm equality multiplier was unsuitable.
    A = equality.T
    inequalities = np.vstack((A[positive], -A[positive], -A[~positive]))
    # Leave numerical slack between the LP's requested feasibility and the
    # independent certificate, rather than testing a rounded boundary twice.
    bounds = np.concatenate((-values[positive] + tolerance / 4,
                             values[positive] + tolerance / 4,
                             values[~positive] + tolerance / 4))
    result = linprog(np.zeros(equality.shape[0]), A_ub=inequalities, b_ub=bounds,
                     bounds=[(None, None)] * equality.shape[0], method="highs",
                     options={"primal_feasibility_tolerance": 1e-9,
                              "dual_feasibility_tolerance": 1e-9})
    return bool(result.success and valid(result.x))


def _canonical_column(design, coefficient, gradient, penalty, *, tol, max_iter, kkt_residual):
    """Minimize Euclidean norm on the fitted-prediction/L1 Lasso face.

    All exact Lasso minimizers have the same fitted prediction and L1 norm.
    Their signs on the equicorrelation set are fixed by the loss gradient.
    The resulting nonnegative quadratic problem is convex. Numerical face
    membership and the solve are checked against the original fit; failure
    is explicit rather than silently accepting an arbitrary tied solution.
    """
    if not np.any(coefficient):
        return coefficient.copy(), 0
    scale = max(1.0, penalty, float(np.max(np.abs(gradient), initial=0)))
    face_tol = max(4 * kkt_residual, 10 * tol * scale, 128 * np.finfo(float).eps * scale)
    active = coefficient != 0
    face = np.flatnonzero(active | (np.abs(np.abs(gradient) - penalty) <= face_tol))
    signs = -np.sign(gradient[face])
    signs[signs == 0] = 1.0
    local_active = coefficient[face] != 0
    signs[local_active] = np.sign(coefficient[face][local_active])
    initial = np.abs(coefficient[face])
    # Keep both predictions and L1 norm: this remains faithful even if a
    # numerical equicorrelation test includes an almost-active coordinate.
    constraint = np.vstack((design[:, face] * signs, np.ones((1, face.size))))
    row_norms = np.linalg.norm(constraint, axis=1)
    constraint = constraint[row_norms > 0] / row_norms[row_norms > 0, None]
    _, values, Vt = np.linalg.svd(constraint, full_matrices=False)
    rank_tol = np.finfo(float).eps * max(constraint.shape) * values[0]
    equality_rank = int(np.count_nonzero(values > rank_tol))
    if equality_rank == face.size:
        return coefficient.copy(), 0
    equality = Vt[:equality_rank]
    # SLSQP's absolute objective tolerance must not treat a small-amplitude
    # noncanonical coefficient as already optimal. Work at unit coefficient
    # scale, preserving the same strictly convex norm minimization.
    magnitude = float(np.max(initial))
    normalized_initial = initial / magnitude
    target = equality @ normalized_initial
    result = minimize(
        lambda value: 0.5 * float(value @ value), normalized_initial,
        jac=lambda value: value, method="SLSQP", bounds=[(0.0, None)] * face.size,
        constraints={"type": "eq", "fun": lambda value: equality @ value - target,
                     "jac": lambda value: equality},
        options={"ftol": min(tol, 1e-12), "maxiter": max_iter},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise InitializationFailure(f"Minimum-norm Lasso tie solve failed: {result.message}")
    if not _minimum_norm_stationarity(equality, np.maximum(result.x, 0)):
        raise InitializationFailure("Minimum-norm Lasso tie solve failed its norm-optimality check")
    candidate = np.zeros_like(coefficient)
    candidate[face] = signs * np.maximum(result.x, 0) * magnitude
    certificate_tol = max(10 * tol, 256 * np.finfo(float).eps * max(design.shape))
    prediction = design @ coefficient
    prediction_error = np.linalg.norm(design @ candidate - prediction)
    l1 = float(np.sum(np.abs(coefficient)))
    if (prediction_error > certificate_tol * max(1.0, float(np.linalg.norm(prediction)))
            or abs(float(np.sum(np.abs(candidate))) - l1) > certificate_tol * max(1.0, l1)
            or np.linalg.norm(candidate) > np.linalg.norm(coefficient) + certificate_tol * max(1.0, np.linalg.norm(coefficient))):
        raise InitializationFailure("Minimum-norm Lasso tie solve failed prediction/L1/norm checks")
    return candidate, int(result.nit)


def reduced_lasso(design, response, rank, penalty, *, tol=1e-9, max_iter=20000, tie_tol=1e-12):
    """Fit reduced entrywise Lasso, then truncate its coefficient SVD.

    Positive penalties reuse the ``sparse_smart`` Lasso primitive. On a
    rank-deficient design, a checked convex tie solve selects the numerical
    minimum-Frobenius-norm solution. At zero penalty ``lstsq`` directly gives
    the minimum-norm reduced least-squares coefficient. This is still the
    same initializer, not a target-only or design-weighted RRR estimator.

    ``coefficient`` is the untruncated result; ``P, d, Q`` are its deterministic
    rank-r SVD. ``n_iter`` includes the numerical tie solve when needed.
    Failed convergence is reported; no alternate estimator is substituted.
    """
    design, response = _matrix(design, "design"), _matrix(response, "response")
    rank = _integer(rank, "rank", minimum=1)
    max_iter = _integer(max_iter, "max_iter", minimum=1)
    penalty = _real(penalty, "penalty")
    tol, tie_tol = _real(tol, "tol", positive=True), _real(tie_tol, "tie_tol", positive=True)
    if design.shape[0] < 1 or design.shape[1] < 1 or response.shape[1] < 1:
        raise ValueError("design and response must be nonempty")
    if response.shape[0] != design.shape[0]:
        raise ValueError("design and response must have the same number of rows")
    if rank > min(design.shape[1], response.shape[1]):
        raise ValueError("rank exceeds the reduced coefficient dimensions")
    if penalty > 0:
        initial = _reduced_lasso(design, response, rank, penalty, tol=tol,
                                 max_iter=max_iter, tie_tol=tie_tol)
        if not initial.converged or np.linalg.matrix_rank(design) == design.shape[1]:
            return initial
        coefficient, n_iter = initial.coefficient.copy(), initial.n_iter.copy()
        gradient, previous_kkt = _kkt(design, response, coefficient, penalty)
        for column in range(response.shape[1]):
            coefficient[:, column], extra = _canonical_column(
                design, coefficient[:, column], gradient[:, column], penalty,
                tol=tol, max_iter=max_iter, kkt_residual=previous_kkt,
            )
            n_iter[column] += extra
        dual_gaps = initial.dual_gaps.copy()
    else:
        try:
            coefficient = np.linalg.lstsq(design, response, rcond=None)[0]
        except np.linalg.LinAlgError as exc:
            raise InitializationFailure("Reduced least-squares initializer failed") from exc
        n_iter = np.zeros(response.shape[1], dtype=int)
        dual_gaps = np.zeros(response.shape[1])
    if not np.isfinite(coefficient).all():
        raise InitializationFailure("Initializer returned nonfinite coefficients")
    _, kkt_residual = _kkt(design, response, coefficient, penalty)
    scale = max(1.0, penalty, float(np.max(np.abs(design.T @ response / design.shape[0]))))
    kkt_tolerance = max(100 * np.finfo(float).eps * scale, np.sqrt(tol) * scale)
    converged = bool(np.isfinite(kkt_residual) and kkt_residual <= kkt_tolerance)
    if penalty > 0 and not converged:
        raise InitializationFailure("Minimum-norm Lasso tie solve failed the Lasso KKT check")
    P, d, Qt = deterministic_svd(coefficient, tie_tol=tie_tol)
    return LassoInitialization(P[:, :rank], d[:rank], Qt.T[:, :rank], coefficient,
                               dual_gaps, kkt_residual, converged, n_iter)


__all__ = ["InitializationFailure", "LassoInitialization", "reduced_lasso"]
