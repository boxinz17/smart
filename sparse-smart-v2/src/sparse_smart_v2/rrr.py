"""Target-only reduced-rank least squares, without chart restrictions.

The response is truncated in the metric induced by the design, rather than
truncating the ordinary least-squares coefficient. Numerical rank decisions
use the usual dimension-times-machine-epsilon SVD tolerances, recorded in the
certificate. A tied truncation boundary uses a canonical response-coordinate
subspace. The coefficient is the minimum-norm lift of that chosen fitted
response; it need not minimize coefficient norm over *different* tied optimal
fitted responses.
"""

from dataclasses import dataclass

import numpy as np

from sparse_smart.calibration import penalty_pair

from .calibration import _integer, _integer_pair
from .source import _frozen, _matrix


@dataclass(frozen=True)
class RRRResult:
    coefficient: np.ndarray
    fitted_values: np.ndarray
    singular_values: np.ndarray
    left_factors: np.ndarray
    right_factors: np.ndarray
    requested_rank: int
    effective_rank: int
    design_rank: int
    training_loss: float
    optimal_training_loss: float
    objective_gap: float
    certificate: dict


def shortcut_reason(*, rank, p, q, free_directions, penalties, support_limits):
    """Identify exactly when both masked penalties and support caps vanish.

    "Nonrestrictive" means a cap equals its entire masked capacity. Merely
    observing a sparse iterate under a smaller cap does not permit RRR.
    """
    p, q = _integer(p, "p", minimum=1), _integer(q, "q", minimum=1)
    rank = _integer(rank, "rank")
    if rank > min(p, q):
        raise ValueError("rank must not exceed min(p, q)")
    free = (rank, rank) if free_directions is None else _integer_pair(free_directions, "free_directions")
    for count, dimension in zip(free, (p, q)):
        if not rank <= count <= dimension:
            raise ValueError("free_directions must lie between rank and the corresponding dimension")
    lambdas = penalty_pair(penalties)
    limits = _integer_pair(support_limits, "support_limits")
    capacities = tuple((dimension - count) * rank for dimension, count in zip((p, q), free))
    if any(limit > capacity for limit, capacity in zip(limits, capacities)):
        raise ValueError("support limit exceeds masked capacity")
    if limits != capacities or any(value != 0 and capacity != 0 for value, capacity in zip(lambdas, capacities)):
        return None
    return "all_rows_unpenalized" if free == (p, q) else "zero_effective_penalties_full_support"


def _canonical_subspace(frame):
    """Coordinate-order basis, using the source-frame tie convention."""
    count = frame.shape[1]
    selected = np.empty_like(frame)
    found = 0
    tolerance = 32 * np.finfo(float).eps * max(frame.shape)
    for row in range(frame.shape[0]):
        candidate = frame @ frame[row]
        for _ in range(2):
            candidate -= selected[:, :found] @ (selected[:, :found].T @ candidate)
        norm = np.linalg.norm(candidate)
        if norm > tolerance:
            selected[:, found] = candidate / norm
            found += 1
        if found == count:
            return selected
    raise np.linalg.LinAlgError("Could not canonicalize the RRR tied response subspace")


def _right_subspace(values, right, count, tolerance):
    """Resolve only ties crossing the truncation boundary; keep distinct SVDs."""
    if count == 0 or count == len(values):
        return right[:, :count], False
    start, stop = count - 1, count
    if abs(values[start] - values[stop]) > tolerance:
        return right[:, :count], False
    while start > 0 and values[start - 1] - values[count] <= tolerance:
        start -= 1
    while stop + 1 < len(values) and values[count - 1] - values[stop + 1] <= tolerance:
        stop += 1
    canonical = _canonical_subspace(right[:, start:stop + 1])
    return np.column_stack((right[:, :start], canonical[:, :count - start])), True


def target_rrr(design, response, rank):
    """Solve ``min rank(C)<=rank ||response-design@C||_F^2/(2*n)``.

    Uses a thin SVD of the design, with no Gram inverse, spectral floor or
    jitter. Components below the recorded design SVD tolerance are treated as
    numerical null directions. Returned factors have ``effective_rank``
    columns, including zero columns only when that rank itself is zero.
    """
    x, y = _matrix(design, "design"), _matrix(response, "response")
    n, p = x.shape
    if n == 0 or p == 0 or y.shape[1] == 0 or y.shape[0] != n:
        raise ValueError("design and response must be nonempty matrices with the same row count")
    q = y.shape[1]
    rank = _integer(rank, "rank")
    if rank > min(p, q):
        raise ValueError("rank must not exceed min(p, q)")

    eps = np.finfo(float).eps
    u, sx, vt = np.linalg.svd(x, full_matrices=False)
    design_tolerance = float(eps * max(x.shape) * sx[0])
    design_rank = int(np.count_nonzero(sx > design_tolerance))
    u, sx, v = u[:, :design_rank], sx[:design_rank], vt[:design_rank].T
    projected = u.T @ y
    residual_outside = y - u @ projected
    if design_rank:
        _, sy, ryt = np.linalg.svd(projected, full_matrices=False)
        response_tolerance = float(eps * max(projected.shape) * sy[0])
        effective_rank = min(rank, int(np.count_nonzero(sy > response_tolerance)))
        tie_tolerance = float(8 * eps * max(projected.shape) * sy[0])
        right, cutoff_tied = _right_subspace(sy, ryt.T, effective_rank, tie_tolerance)
        reduced_fit = (projected @ right) @ right.T
        coefficient = (v / sx) @ reduced_fit
    else:
        sy = np.empty(0)
        response_tolerance = tie_tolerance = 0.
        effective_rank, cutoff_tied = 0, False
        coefficient = np.zeros((p, q))

    fitted = x @ coefficient
    training_loss = float(np.sum((y - fitted) ** 2) / (2 * n))
    optimal_loss = float((np.sum(residual_outside ** 2) + np.sum(sy[rank:] ** 2)) / (2 * n))
    gap = training_loss - optimal_loss
    loss_tolerance = float(256 * eps * max(n, p, q) * max(float(np.sum(y ** 2) / (2 * n)), np.finfo(float).tiny))
    if effective_rank:
        left, values, right_t = np.linalg.svd(coefficient, full_matrices=False)
        left, values, right = left[:, :effective_rank], values[:effective_rank], right_t[:effective_rank].T
        # Paired signs agree with the ordered source-frame convention.
        for column in range(effective_rank):
            pivot = np.flatnonzero(np.abs(left[:, column]) > 32 * eps)
            if pivot.size and left[pivot[0], column] < 0:
                left[:, column] *= -1
                right[:, column] *= -1
    else:
        left, values, right = np.empty((p, 0)), np.empty(0), np.empty((q, 0))

    if not (np.isfinite(coefficient).all() and np.isfinite(fitted).all()
            and np.isfinite([training_loss, optimal_loss, gap, loss_tolerance]).all()):
        raise np.linalg.LinAlgError("RRR numerical arithmetic produced nonfinite results")
    null_residual = coefficient - v @ (v.T @ coefficient)
    certificate = {
        "method": "design_svd_projected_response_truncation",
        "objective": "squared_frobenius_training_residual_over_2n",
        "design_rank_tolerance": design_tolerance,
        "response_rank_tolerance": response_tolerance,
        "tie_tolerance": tie_tolerance,
        "loss_tolerance": loss_tolerance,
        "certified": bool(abs(gap) <= loss_tolerance),
        "rank_cutoff_tied": bool(cutoff_tied),
        "tie_convention": "canonical_response_coordinates",
        "coefficient_convention": "minimum_norm_lift_of_selected_fitted_response",
        "design_nullspace_residual_norm": float(np.linalg.norm(null_residual)),
        "projected_response_singular_values": sy.tolist(),
        "scope": "numerical_design_range_at_recorded_svd_tolerance",
    }
    return RRRResult(*(_frozen(value) for value in (coefficient, fitted, values, left, right)),
                     rank, effective_rank, design_rank, training_loss, optimal_loss, gap, certificate)


__all__ = ["RRRResult", "shortcut_reason", "target_rrr"]
