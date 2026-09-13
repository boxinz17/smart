"""Euclidean proximal steps on the bounded, ordered, gap-separated spectrum.

Shift ``d[i]`` by ``i * gap`` to obtain a nonincreasing isotonic problem
with common bounds ``[d_lower + (rank-1)*gap, d_upper]``. PAVA solves it
without permuting singular values or their associated factor columns.
Floating output is corrected inward by at most roundoff so the existing
strict float64 domain checks pass; the declared gap is never reduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np


def _vector(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be a nonempty finite real vector")
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite real vector")
    return value


def _bounds(rank, d_lower, d_upper, gap):
    for name, value in (("d_lower", d_lower), ("d_upper", d_upper), ("gap", gap)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
    if d_lower <= 0 or d_upper < d_lower or gap < 0:
        raise ValueError("require 0 < d_lower <= d_upper and gap >= 0")
    lower, upper, separation = map(np.longdouble, (d_lower, d_upper, gap))
    if upper - lower < (rank - 1) * separation:
        raise ValueError("the bounded gap-separated spectral domain is empty")
    return lower, upper, separation


def _feasible(values, lower, upper, gap):
    return bool(np.all(values >= lower) and np.all(values <= upper)
                and np.all(values[:-1] - values[1:] >= gap))


def _above(value, gap):
    result = float(np.longdouble(value) + np.longdouble(gap))
    while result - value < gap:
        result = np.nextafter(result, np.inf)
    return result


def _below(value, gap):
    result = float(np.longdouble(value) - np.longdouble(gap))
    while value - result < gap:
        result = np.nextafter(result, -np.inf)
    return result


def _round_feasible(projected, lower, upper, gap):
    """Move rounded boundaries inward, never relaxing the domain predicate."""
    result = np.array(projected, dtype=float, copy=True)
    if _feasible(result, lower, upper, gap):
        return result
    rank = result.size
    floor, ceiling = np.empty(rank), np.empty(rank)
    floor[-1], ceiling[0] = lower, upper
    for i in range(rank - 2, -1, -1):
        floor[i] = _above(floor[i + 1], gap)
    for i in range(1, rank):
        ceiling[i] = _below(ceiling[i - 1], gap)
    if np.any(floor > ceiling):
        raise FloatingPointError("no float64 spectrum represents the declared bounds and gap")
    result = np.clip(result, floor, ceiling)
    for i in range(rank - 2, -1, -1):
        result[i] = max(result[i], _above(result[i + 1], gap))
    scale = np.maximum(np.abs(np.asarray(projected, dtype=np.longdouble)), np.longdouble(lower))
    tolerance = (64 * rank * np.finfo(float).eps * scale
                 + 8 * rank * np.longdouble(np.nextafter(0., 1.)))
    if (not _feasible(result, lower, upper, gap)
            or np.any(np.abs(result.astype(np.longdouble) - projected) > tolerance)):
        raise FloatingPointError("spectral feasibility correction exceeded float64 roundoff")
    return result


@dataclass
class _Block:
    start: int
    stop: int
    offset: np.longdouble
    gradient: np.longdouble

    @property
    def size(self):
        return self.stop - self.start


def spectral_proximal_step(d, gradient, inverse_step, *, d_lower, d_upper, gap):
    """Return the projected gradient trial and a cancellation-safe residual.

    The residual is ``L * (d - projection(d-gradient/L))``. It is evaluated
    algebraically on each PAVA block, retaining the original gradient on
    unconstrained singleton blocks even if subtracting the step rounds away.
    Centered block comparisons similarly retain tiny gradients at active
    bounds. The residual describes the real-valued projection before the
    at-most-roundoff feasibility correction of its float64 representation.

    Already float64-feasible proposals are returned byte-for-byte unchanged.
    This routine is for refinement; it does not repair an initializer.
    """
    d, gradient = _vector(d, "d"), _vector(gradient, "gradient")
    if gradient.shape != d.shape:
        raise ValueError("d and gradient must have the same shape")
    if (isinstance(inverse_step, (bool, np.bool_)) or not isinstance(inverse_step, Real)
            or not np.isfinite(inverse_step) or inverse_step <= 0):
        raise ValueError("inverse_step must be finite and positive")
    rank = d.size
    lower, upper, separation = _bounds(rank, d_lower, d_upper, gap)
    L = np.longdouble(inverse_step)
    values, gradients = d.astype(np.longdouble), gradient.astype(np.longdouble)
    blocks = []
    for i in range(rank):
        blocks.append(_Block(i, i + 1, np.longdouble(0), gradients[i]))
        while len(blocks) > 1:
            left, right = blocks[-2:]
            # Compare means after cancelling the large common coefficient
            # level, rather than forming d - gradient/L at that level.
            anchor_difference = ((values[left.start] - values[right.start])
                                 + (left.start - right.start) * separation)
            difference = (anchor_difference + left.offset - right.offset
                          - (left.gradient - right.gradient) / L)
            if difference >= 0:
                break
            size = left.size + right.size
            offset = (left.size * left.offset
                      + right.size * (right.offset - anchor_difference)) / size
            mean_gradient = (left.size * left.gradient + right.size * right.gradient) / size
            blocks[-2:] = [_Block(left.start, right.stop, offset, mean_gradient)]

    projected, residual = np.empty(rank, dtype=np.longdouble), np.empty(rank, dtype=np.longdouble)
    for block in blocks:
        indices = np.arange(block.start, block.stop)
        relative = (indices - block.start).astype(np.longdouble)
        above = ((values[block.start] - upper) + block.start * separation
                 + block.offset - block.gradient / L)
        below = ((values[block.start] - lower) - (rank - 1 - block.start) * separation
                 + block.offset - block.gradient / L)
        if above > 0:
            projected[indices] = upper - indices * separation
            residual[indices] = L * ((values[indices] - upper) + indices * separation)
        elif below < 0:
            projected[indices] = lower + (rank - 1 - indices) * separation
            residual[indices] = L * ((values[indices] - lower) - (rank - 1 - indices) * separation)
        else:
            offset = (values[indices] - values[block.start]) + relative * separation
            residual[indices] = L * (offset - block.offset) + block.gradient
            projected[indices] = (values[block.start] + block.offset
                                  - block.gradient / L - relative * separation)
    with np.errstate(over="ignore", invalid="ignore"):
        original_trial = d - gradient / float(inverse_step)
    if np.isfinite(original_trial).all() and _feasible(original_trial, d_lower, d_upper, gap):
        output = original_trial.copy()
    else:
        output = _round_feasible(projected, float(lower), float(upper), float(separation))
    mapped = np.asarray(residual, dtype=float)
    if not np.isfinite(mapped).all():
        raise FloatingPointError("nonfinite spectral proximal residual")
    return output, mapped


def project_spectrum(values, *, d_lower, d_upper, gap):
    """Euclidean projection; singular-value identities are never permuted."""
    values = _vector(values, "values")
    return spectral_proximal_step(values, np.zeros_like(values), 1.,
                                  d_lower=d_lower, d_upper=d_upper, gap=gap)[0]


__all__ = ["project_spectrum", "spectral_proximal_step"]
