"""Certified box-dual iteration for weighted L1 plus a spectral-ball constraint.

The primal problem is ``min_X .5*||X-A||_F**2 + <weights, abs(X)>`` with
``||X||_op <= radius``. Eliminating the ball dual gives a differentiable convex
objective on ``|p| <= weights`` whose gradient is ``-project_ball(A-p)`` and is
1-Lipschitz. Projected accelerated gradient with objective-based restarts solves
that *same* proximal problem.

Stopping uses a feasible primal/dual Fenchel gap, without a split-residual gate.
The numerical bound checks SVD reconstruction and orthogonality residuals and
includes floating-point allowances. It is not an interval-arithmetic proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import numpy as np


_EPS = np.finfo(float).eps
_LONG_EPS = np.finfo(np.longdouble).eps
_CHECK_EVERY = 10


@dataclass(frozen=True)
class DualProxResult:
    """A feasible proximal candidate and the accuracy actually established.

    ``gap_upper`` already includes ``gap_roundoff`` and all feasibility/SVD
    allowances. Strong convexity gives ``||value-X_opt||_F <= error_bound``.
    ``n_iter`` counts executed dual sweeps; ``certificate_iteration`` identifies
    the best returned candidate, which can precede the last executed sweep.
    An unmet absolute target never changes into a claim of absolute accuracy.
    """

    value: np.ndarray
    n_iter: int
    relative_certified: bool
    gap_upper: float
    error_bound: float
    residual: float
    termination_reason: str
    absolute_certified: bool | None
    gap_roundoff: float
    primal_operator_upper: float
    dual_box_violation: float
    feasibility_correction: float
    certificate_iteration: int


def _round_up(value):
    return float(np.nextafter(float(value), np.inf)) if value else 0.


def _frobenius_upper(value):
    value = np.asarray(value, dtype=np.longdouble)
    square = np.sum(value * value, dtype=np.longdouble)
    return _round_up(np.sqrt(square) * (1 + 16 * _LONG_EPS * max(1, value.size)))


def _spectral_bounds(value):
    """Bound the operator and nuclear norms from a checked thin SVD."""
    if not value.size:
        return 0., 0.
    left, singular, right = np.linalg.svd(value, full_matrices=False)
    left, singular, right = (np.asarray(item, dtype=np.longdouble)
                             for item in (left, singular, right))
    original = np.asarray(value, dtype=np.longdouble)
    reconstructed = (left * singular) @ right
    rank = len(singular)
    reconstruction_error = (
        _frobenius_upper(original - reconstructed)
        + 8 * _LONG_EPS * (rank + 2)
        * _frobenius_upper((np.abs(left) * np.abs(singular)) @ np.abs(right))
        + 8 * _LONG_EPS * _frobenius_upper(np.abs(original) + np.abs(reconstructed)))
    identity = np.eye(rank, dtype=np.longdouble)
    left_error = (
        _frobenius_upper(left.T @ left - identity)
        + 8 * _LONG_EPS * (value.shape[0] + 2)
        * _frobenius_upper(np.abs(left).T @ np.abs(left) + identity))
    right_error = (
        _frobenius_upper(right @ right.T - identity)
        + 8 * _LONG_EPS * (value.shape[1] + 2)
        * _frobenius_upper(np.abs(right) @ np.abs(right).T + identity))
    factor = np.sqrt((np.longdouble(1) + left_error) * (np.longdouble(1) + right_error))
    operator = _round_up(factor * singular[0] + reconstruction_error)
    nuclear = _round_up(factor * np.sum(singular, dtype=np.longdouble)
                        + np.sqrt(rank) * reconstruction_error)
    return operator, nuclear


def _project_ball(value, radius):
    if not value.size or radius == 0:
        return np.zeros_like(value)
    left, singular, right = np.linalg.svd(value, full_matrices=False)
    if singular[0] <= radius:
        return value.copy()
    return (left * np.minimum(singular, radius)) @ right


def _feasible_primal(value, radius):
    """Correct projection roundoff inward, never a substantive infeasibility."""
    original = value
    value = value.copy()
    if radius == 0:
        return np.zeros_like(value), float(np.linalg.norm(value)), 0.
    for _ in range(4):
        upper, _ = _spectral_bounds(value)
        if upper <= radius:
            return value, float(np.linalg.norm(value - original)), upper
        if upper - radius > 1e-9 * max(1., radius, float(np.linalg.norm(value))):
            raise ArithmeticError("Dual proximal candidate is not numerically ball feasible")
        value *= np.nextafter(radius / upper * (1 - 128 * _EPS * max(1, min(value.shape))), 0.)
    raise ArithmeticError("Could not establish dual proximal primal feasibility")


def _certificate(value, weights, radius, primal, p, q):
    """Fenchel gap: invariant error plus the L1 and ball support gaps."""
    if np.any(np.abs(p) > weights):
        raise ArithmeticError("Weighted L1 dual is outside its feasible box")
    primal, correction, operator_upper = _feasible_primal(primal, radius)
    a, w, x, p_long, q_long = (np.asarray(item, dtype=np.longdouble)
                              for item in (value, weights, primal, p, q))
    invariant = a - x - p_long - q_long
    invariant_term = np.longdouble(.5) * np.sum(invariant * invariant, dtype=np.longdouble)
    l1_terms = np.abs(x) * np.maximum(w - np.sign(x) * p_long, 0.)
    l1_gap = np.sum(l1_terms, dtype=np.longdouble)
    _, nuclear_upper = _spectral_bounds(q)
    support = np.longdouble(radius) * nuclear_upper
    pairing = np.sum(q_long * x, dtype=np.longdouble)
    ball_gap = support - pairing
    magnitude = (abs(invariant_term) + np.sum(np.abs(l1_terms), dtype=np.longdouble)
                 + abs(support) + np.sum(np.abs(q_long * x), dtype=np.longdouble))
    roundoff = float((64 * _LONG_EPS * max(1, x.size) + 8 * _EPS) * magnitude)
    if ball_gap < -roundoff:
        raise ArithmeticError("Dual proximal support-function certificate is invalid")
    gap_upper = _round_up(max(np.longdouble(0), invariant_term + l1_gap + ball_gap) + roundoff)
    if not np.isfinite(gap_upper):
        raise FloatingPointError("Dual proximal certificate exceeds floating-point range")
    return primal, gap_upper, roundoff, correction, operator_upper


def _dual_objective(value, dual, primal):
    # Equivalent to .5*||value-dual||^2 - .5*dist(value-dual, ball)^2,
    # without subtracting two potentially large squared norms.
    shifted = np.asarray(value - dual, dtype=np.longdouble)
    primal = np.asarray(primal, dtype=np.longdouble)
    return float(np.sum(shifted * primal - .5 * primal * primal, dtype=np.longdouble))


def solve_weighted_l1_ball_dual(
    A, weights, radius, *, tolerance=1e-11, absolute_gap_tol=None,
    max_iterations=2000, initial_p=None,
):
    """Solve the weighted-L1/spectral-ball proximal problem by its box dual.

    ``initial_p`` is optional within-problem dual initialization, for example
    from Dykstra. It is clipped into the current weighted box; the corresponding
    primal and ball dual are always recomputed for the supplied ``A``.
    No initializer, cache, or state is retained between calls.

    Relative accuracy means ``gap_upper / max(1, ||A||_F)**2 <= tolerance``.
    An optional absolute target is checked independently. The best certified
    candidate is returned even when the requested targets remain unresolved.
    """
    value = np.asarray(A, dtype=float)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Dual proximal input must be a finite matrix")
    value = value.copy()
    try:
        weights = np.broadcast_to(np.asarray(weights, dtype=float), value.shape).copy()
    except ValueError as error:
        raise ValueError("Dual proximal weights must broadcast to the input") from error
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Dual proximal weights must be finite and nonnegative")
    if not np.isfinite(radius) or radius < 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Dual proximal radius/tolerance are invalid")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, numbers.Integral) or max_iterations < 1:
        raise ValueError("Dual proximal iteration budget must be a positive integer")
    if absolute_gap_tol is not None and (not np.isfinite(absolute_gap_tol) or absolute_gap_tol <= 0):
        raise ValueError("Absolute gap tolerance must be positive and finite or None")
    radius = float(radius)
    with np.errstate(over="ignore", invalid="ignore"):
        scale = max(1., float(np.linalg.norm(value)))
    if not np.isfinite(scale):
        raise FloatingPointError("Dual proximal input norm exceeds floating-point range")
    p = np.zeros_like(value)
    if initial_p is not None:
        supplied = np.asarray(initial_p, dtype=float)
        if supplied.shape != value.shape or not np.isfinite(supplied).all():
            raise ValueError("Initial L1 dual must be a finite matrix matching the input")
        p = np.clip(supplied, -weights, weights).copy()
    if not value.size or radius == 0:
        return DualProxResult(np.zeros_like(value), 0, True, 0., 0., 0., "exact_zero",
                              None if absolute_gap_tol is None else True, 0., 0., 0., 0., 0)

    extrapolated = p.copy()
    primal = _project_ball(value - p, radius)
    momentum = 1.
    best = None
    for iteration in range(1, int(max_iterations) + 1):
        prediction = _project_ball(value - extrapolated, radius)
        next_p = np.clip(extrapolated + prediction, -weights, weights)
        next_primal = _project_ball(value - next_p, radius)
        if (iteration > 1 and _dual_objective(value, next_p, next_primal)
                > _dual_objective(value, p, primal)):
            # Restart at the last feasible dual and take a nonaccelerated step.
            momentum = 1.
            next_p = np.clip(p + primal, -weights, weights)
            next_primal = _project_ball(value - next_p, radius)
        residual = float(np.linalg.norm(next_primal - primal))
        next_momentum = (1 + math.sqrt(1 + 4 * momentum * momentum)) / 2
        extrapolated = next_p + (momentum - 1) / next_momentum * (next_p - p)
        momentum = next_momentum
        p, primal = next_p, next_primal
        if iteration == 1 or iteration % _CHECK_EVERY == 0 or iteration == max_iterations:
            q = value - p - primal
            candidate = _certificate(value, weights, radius, primal, p, q)
            if best is None or candidate[1] < best[0][1]:
                best = candidate, iteration, residual
            gap = candidate[1]
            if gap / scale / scale <= tolerance and (absolute_gap_tol is None or gap <= absolute_gap_tol):
                break
    candidate, certificate_iteration, residual = best
    primal, gap, roundoff, correction, operator_upper = candidate
    relative_certified = gap / scale / scale <= tolerance
    absolute_certified = None if absolute_gap_tol is None else gap <= absolute_gap_tol
    reason = ("certified" if relative_certified and absolute_certified is not False
              else "absolute_target_unresolved" if relative_certified else "iteration_limit_uncertified")
    return DualProxResult(primal, iteration, relative_certified, gap, _round_up(np.sqrt(np.longdouble(2) * gap)),
                          residual, reason, absolute_certified, roundoff, operator_upper,
                          0., correction, certificate_iteration)
