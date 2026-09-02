"""Small NumPy-only matrix-free least-squares primitives.

The BI-SMART quotient Jacobian can have millions of entries even when a
Jacobian-vector product is inexpensive.  This module implements the
Golub--Kahan LSQR recurrence using only ``matvec`` and ``rmatvec`` callbacks.
It deliberately contains no BI-SMART geometry: keeping the numerical kernel
generic makes its memory contract and stopping policy easier to audit.

LSQR started from zero converges to the minimum-Euclidean-norm coefficient
solution.  In the BI-SMART caller those coefficients belong to a Parseval
frame, so their lifted limit is the minimum-product-norm (horizontal)
Gauss--Newton direction.  The caller still performs the geometric tangency,
horizontality, normal-residual, and descent-identity certificates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
VectorAction = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True, slots=True)
class LSQRResult:
    """Result and recurrence diagnostics from a matrix-free LSQR solve."""

    solution: FloatArray
    iterations: int
    converged: bool
    stop_reason: str
    residual_norm: float
    normal_residual_norm: float
    operator_norm_estimate: float
    condition_estimate: float


def _finite_vector(value: object, *, size: int, name: str) -> FloatArray:
    """Normalize one callback vector and enforce the declared dimension."""

    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    vector = np.asarray(raw, dtype=float)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},); got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise FloatingPointError(f"{name} contains a nonfinite value")
    return vector


def lsqr(
    matvec: VectorAction,
    rmatvec: VectorAction,
    *,
    rows: int,
    columns: int,
    rhs: FloatArray,
    tolerance: float,
    max_iterations: int,
    condition_limit: float,
) -> LSQRResult:
    """Solve ``min_x ||A x - rhs||_2`` without materialising ``A``.

    Pseudocode
    ----------
    1. Start Golub--Kahan bidiagonalisation from ``rhs`` and ``A.T @ rhs``.
    2. Apply the next Givens rotation to the implicit bidiagonal matrix.
    3. Update the minimum-norm iterate from the zero initial vector.
    4. Stop on either the compatible-system residual test or the
       least-squares normal-residual test used by the original LSQR paper.
    5. Recompute both residuals through the callbacks before returning; this
       catches drift in the scalar recurrence and callback inconsistencies.

    ``condition_limit`` is a safety policy, not damping.  Crossing it returns
    an unsuccessful result so the BI-SMART caller can mark only the current
    refinement branch as numerically singular.
    """

    if isinstance(rows, (bool, np.bool_)) or int(rows) < 1:
        raise ValueError("rows must be a positive integer")
    if isinstance(columns, (bool, np.bool_)) or int(columns) < 1:
        raise ValueError("columns must be a positive integer")
    rows = int(rows)
    columns = int(columns)
    right_hand_side = _finite_vector(rhs, size=rows, name="rhs").copy()
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and strictly positive")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or int(max_iterations) < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    max_iterations = int(max_iterations)
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("condition_limit must be finite and larger than one")

    def apply(vector: FloatArray) -> FloatArray:
        return _finite_vector(matvec(vector), size=rows, name="matvec result")

    def apply_adjoint(vector: FloatArray) -> FloatArray:
        return _finite_vector(
            rmatvec(vector), size=columns, name="rmatvec result"
        )

    solution = np.zeros(columns, dtype=float)
    beta = float(np.linalg.norm(right_hand_side))
    rhs_norm = beta
    if beta == 0.0:
        # The zero vector is exactly the minimum-norm solution.  No Krylov
        # iteration can reveal spectral information from a zero starting
        # residual, so the caller records rank as un-certified.
        return LSQRResult(
            solution=solution,
            iterations=0,
            converged=True,
            stop_reason="zero_rhs",
            residual_norm=0.0,
            normal_residual_norm=0.0,
            operator_norm_estimate=0.0,
            condition_estimate=1.0,
        )

    left_vector = right_hand_side / beta
    right_vector = apply_adjoint(left_vector)
    alpha = float(np.linalg.norm(right_vector))
    if alpha == 0.0:
        # The residual is orthogonal to the range of A, so x=0 exactly solves
        # the least-squares normal equations.  This Krylov run supplies no
        # rank information; the BI-SMART caller must (and does) establish
        # injectivity independently before invoking LSQR.
        return LSQRResult(
            solution=solution,
            iterations=0,
            converged=True,
            stop_reason="stationary_rhs",
            residual_norm=rhs_norm,
            normal_residual_norm=0.0,
            operator_norm_estimate=0.0,
            condition_estimate=1.0,
        )
    right_vector /= alpha

    search_vector = right_vector.copy()
    rhobar = alpha
    phibar = beta
    # The first bidiagonal diagonal entry was computed before the loop.
    operator_norm_squared = alpha * alpha
    inverse_norm_squared = 0.0
    solution_norm = 0.0
    condition_estimate = 1.0
    stop_reason = "iteration_limit"
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        iterations = iteration

        # PSEUDOCODE 1: one Golub--Kahan bidiagonalisation step.  Keeping the
        # old vectors in these recurrences is what permits O(m+n) workspace.
        next_left = apply(right_vector) - alpha * left_vector
        beta = float(np.linalg.norm(next_left))
        if beta > 0.0:
            next_left /= beta
        left_vector = next_left

        next_right = apply_adjoint(left_vector) - beta * right_vector
        alpha = float(np.linalg.norm(next_right))
        if alpha > 0.0:
            next_right /= alpha
        right_vector = next_right

        operator_norm_squared += alpha * alpha + beta * beta
        operator_norm_estimate = float(np.sqrt(operator_norm_squared))

        # PSEUDOCODE 2: eliminate beta from the implicit bidiagonal system by
        # a stable Givens rotation (``hypot`` avoids avoidable overflow).
        rho = float(np.hypot(rhobar, beta))
        if rho == 0.0 or not np.isfinite(rho):
            stop_reason = "bidiagonal_breakdown"
            break
        cosine = rhobar / rho
        sine = beta / rho
        theta = sine * alpha
        rhobar = -cosine * alpha
        phi = cosine * phibar
        phibar = sine * phibar
        tau = sine * phi

        # PSEUDOCODE 3: update x and the short-recurrence search vector.  The
        # squared norms of ``search_vector / rho`` estimate ||A^dagger||.
        scaled_search = search_vector / rho
        solution += phi * scaled_search
        search_vector = right_vector - theta * scaled_search
        inverse_norm_squared += float(np.vdot(scaled_search, scaled_search).real)
        solution_norm = float(np.linalg.norm(solution))

        inverse_norm_estimate = float(np.sqrt(inverse_norm_squared))
        condition_estimate = operator_norm_estimate * inverse_norm_estimate
        recurrence_residual = abs(phibar)
        recurrence_normal = alpha * abs(tau)

        # PSEUDOCODE 4: standard scale-free LSQR stopping tests.  The first
        # handles an (approximately) compatible system; the second handles a
        # genuine least-squares residual by testing A.T r.
        compatible_bound = tolerance * (
            rhs_norm + operator_norm_estimate * solution_norm
        )
        normal_bound = tolerance * operator_norm_estimate * max(
            recurrence_residual, np.finfo(float).tiny
        )
        if condition_estimate >= condition_limit:
            stop_reason = "condition_limit"
            break
        if recurrence_residual <= compatible_bound:
            stop_reason = "compatible_residual"
            converged = True
            break
        if recurrence_normal <= normal_bound:
            stop_reason = "normal_residual"
            converged = True
            break
        if alpha == 0.0 and beta == 0.0:
            stop_reason = "exact_bidiagonal_termination"
            converged = True
            break

    # PSEUDOCODE 5: never trust only the scalar recurrence.  Explicit
    # callback checks use little memory and are also exercised by the
    # BI-SMART-level descent and geometry certificates.
    residual = apply(solution) - right_hand_side
    normal_residual = apply_adjoint(residual)
    residual_norm = float(np.linalg.norm(residual))
    normal_residual_norm = float(np.linalg.norm(normal_residual))
    operator_norm_estimate = float(np.sqrt(operator_norm_squared))
    if operator_norm_estimate == 0.0:
        condition_estimate = float("inf")
        converged = False
        stop_reason = "zero_operator_estimate"
    elif converged:
        explicit_compatible_bound = 8.0 * tolerance * (
            rhs_norm + operator_norm_estimate * solution_norm
        )
        explicit_normal_bound = (
            8.0
            * tolerance
            * operator_norm_estimate
            * max(residual_norm, np.finfo(float).tiny)
        )
        if (
            residual_norm > explicit_compatible_bound
            and normal_residual_norm > explicit_normal_bound
        ):
            converged = False
            stop_reason = "explicit_residual_check"

    return LSQRResult(
        solution=solution,
        iterations=iterations,
        converged=converged,
        stop_reason=stop_reason,
        residual_norm=residual_norm,
        normal_residual_norm=normal_residual_norm,
        operator_norm_estimate=operator_norm_estimate,
        condition_estimate=float(condition_estimate),
    )


__all__ = ["LSQRResult", "lsqr"]
