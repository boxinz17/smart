"""Strict linear-algebra primitives used by BI-SMART.

Several steps in the appendix are defined only when a matrix is positive
definite, a rank cutoff is strict, or a polar argument has full column rank.
Generic pseudoinverses and arbitrary basis completion would make those failed
branches appear successful and could destroy block-rotation invariance.  The
helpers below therefore fail loudly with :class:`NumericalFailure`.

The manuscript states exact inequalities.  Floating-point code interprets each
one using ``atol + rtol * scale``; callers can set both tolerances to zero to
recover the literal mathematical comparison.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .types import FailureReason, FloatArray, NumericalFailure, TruncatedSVD


DEFAULT_ATOL = 1e-12
DEFAULT_RTOL = 1e-10


def validate_matrix(matrix: Any, *, name: str = "matrix") -> FloatArray:
    """Return a finite, nonempty, two-dimensional floating-point array."""

    # PSEUDOCODE 1: Normalize array-like input without changing its dimensions.
    raw = np.asarray(matrix)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued; complex input is unsupported.")
    result = np.asarray(raw, dtype=float)

    # PSEUDOCODE 2: Reject objects for which BI-SMART matrix formulas are undefined.
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {result.shape}.")
    if 0 in result.shape:
        raise ValueError(f"{name} must have no empty dimension; got shape {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")

    # PSEUDOCODE 3: Return one normalized representation to downstream routines.
    return result


def comparison_tolerance(
    scale: float,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> float:
    """Compute the package-wide threshold ``atol + rtol * abs(scale)``."""

    raw_scale = np.asarray(scale)
    raw_atol = np.asarray(atol)
    raw_rtol = np.asarray(rtol)
    if any(np.iscomplexobj(value) for value in (raw_scale, raw_atol, raw_rtol)):
        raise ValueError(
            "scale, atol, and rtol must be real-valued; complex input is unsupported."
        )
    if any(value.ndim != 0 for value in (raw_scale, raw_atol, raw_rtol)):
        raise ValueError("scale, atol, and rtol must be scalars.")
    scale = float(raw_scale)
    atol = float(raw_atol)
    rtol = float(raw_rtol)
    if not np.isfinite(scale):
        raise ValueError("scale must be finite.")
    if not np.isfinite(atol) or atol < 0 or not np.isfinite(rtol) or rtol < 0:
        raise ValueError("atol and rtol must be finite and nonnegative.")
    return atol + rtol * abs(scale)


def _validate_rank(rank: int, shape: Tuple[int, int]) -> int:
    """Validate a positive truncation rank against a matrix shape."""

    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
        raise TypeError(f"rank must be an integer; got {type(rank).__name__}.")
    rank = int(rank)
    maximum = min(shape)
    if rank < 1 or rank > maximum:
        raise ValueError(f"rank must lie in [1, {maximum}]; got {rank}.")
    return rank


def truncated_svd(matrix: Any, rank: int) -> TruncatedSVD:
    """Compute a deterministic-shaped best rank-``rank`` approximation.

    This routine does *not* require a unique cutoff.  Use
    :func:`strict_truncated_svd` for the pilot and restricted-RRR checks in the
    appendix.  NumPy's singular-vector convention may vary inside a tied
    subspace, but the reconstructed approximation is invariant whenever the
    retained/excluded cutoff itself is strict.
    """

    # PSEUDOCODE 1: Validate dimensions and the requested retained rank.
    matrix = validate_matrix(matrix)
    rank = _validate_rank(rank, matrix.shape)

    # PSEUDOCODE 2: Compute the economy SVD in nonincreasing singular-value order.
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)

    # PSEUDOCODE 3: Retain r factors and record sigma_(r+1), using zero at full rank.
    next_value = float(singular_values[rank]) if rank < singular_values.size else 0.0
    return TruncatedSVD(
        u=u[:, :rank],
        singular_values=singular_values[:rank],
        vt=vt[:rank, :],
        next_singular_value=next_value,
    )


def strict_truncated_svd(
    matrix: Any,
    rank: int,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> TruncatedSVD:
    """Compute a rank truncation only when the appendix cutoff is strict.

    The accepted branch satisfies, up to the configured floating-point
    tolerance, ``sigma_r > sigma_(r+1)`` and ``sigma_r > 0``.  Failure maps to
    the explicit continuation rules for the hybrid pilot and restricted RRR.
    """

    # PSEUDOCODE 1: Compute the proposed rank-r truncation and excluded value.
    result = truncated_svd(matrix, rank)
    retained = float(result.singular_values[-1])
    excluded = float(result.next_singular_value)
    scale = float(result.singular_values[0])
    tolerance = comparison_tolerance(scale, atol=atol, rtol=rtol)

    # PSEUDOCODE 2: Reject a zero/numerically-zero r-th target direction.
    if retained <= tolerance:
        raise NumericalFailure(
            FailureReason.ZERO_RANK_COMPONENT,
            f"The retained sigma_{rank}={retained:.6g} is not larger than "
            f"the numerical zero threshold {tolerance:.6g}.",
        )

    # PSEUDOCODE 3: Reject a tied or unresolved retained/excluded boundary.
    if retained - excluded <= tolerance:
        raise NumericalFailure(
            FailureReason.NON_UNIQUE_RANK_CUTOFF,
            f"The rank-{rank} cutoff is not strict: sigma_r={retained:.6g}, "
            f"sigma_(r+1)={excluded:.6g}, tolerance={tolerance:.6g}.",
        )

    # PSEUDOCODE 4: Return factors only after both paper-defined checks pass.
    return result


def _coordinate_lexicographic_basis(
    projector: FloatArray,
    dimension: int,
    *,
    atol: float,
    rtol: float,
) -> FloatArray:
    """Choose a canonical subspace inside a tied left singular space.

    The rule depends only on the orthogonal projector, not on whichever basis
    an SVD backend returned for its range.  Scan ambient standard coordinates
    ``e_0, e_1, ...``; project each into the tied space; and retain the first
    ``dimension`` projections that add an independent direction.  Modified
    Gram--Schmidt turns those directions into an orthonormal basis.  Thus the
    selected *span* is the coordinate-lexicographically first admissible span.
    """

    # PSEUDOCODE 1: Validate the square tied-space projector and target dimension.
    projector = validate_matrix(projector, name="tied-space projector")
    if projector.shape[0] != projector.shape[1]:
        raise ValueError("tied-space projector must be square.")
    if isinstance(dimension, (bool, np.bool_)) or not isinstance(
        dimension, (int, np.integer)
    ):
        raise TypeError("dimension must be an integer.")
    dimension = int(dimension)
    if dimension < 1 or dimension > projector.shape[0]:
        raise ValueError("dimension is incompatible with the projector.")

    # PSEUDOCODE 2: Remove harmless last-bit asymmetry.  Projector scale is one;
    # retain a small machine floor even if callers request literal comparisons.
    projector = 0.5 * (projector + projector.T)
    independence_tolerance = max(
        comparison_tolerance(1.0, atol=atol, rtol=rtol),
        8.0 * np.finfo(float).eps * projector.shape[0],
    )

    # PSEUDOCODE 3: Traverse P e_i in coordinate order and keep the first
    # independent residuals.  A second orthogonalization pass controls drift.
    basis_vectors = []
    for coordinate in range(projector.shape[0]):
        candidate = projector[:, coordinate].copy()
        for _ in range(2):
            if basis_vectors:
                basis = np.column_stack(basis_vectors)
                candidate -= basis @ (basis.T @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm <= independence_tolerance:
            continue
        candidate /= norm

        # PSEUDOCODE 4: Fix each vector's otherwise irrelevant sign using its
        # first numerically nonzero ambient coordinate.
        nonzero = np.flatnonzero(np.abs(candidate) > independence_tolerance)
        pivot = int(nonzero[0]) if nonzero.size else int(np.argmax(np.abs(candidate)))
        if candidate[pivot] < 0:
            candidate = -candidate
        basis_vectors.append(candidate)
        if len(basis_vectors) == dimension:
            break

    # PSEUDOCODE 5: A rank-d projector must supply d directions.  Failure here
    # indicates numerical projector corruption, not a permissible arbitrary
    # coordinate completion.
    if len(basis_vectors) != dimension:
        raise np.linalg.LinAlgError(
            "Could not construct the requested coordinate-lexicographic "
            "basis from the tied singular-space projector."
        )
    return np.column_stack(basis_vectors)


def deterministic_rank_at_most_approximation(
    matrix: Any,
    rank: int,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> FloatArray:
    """Return the appendix's fixed best rank-at-most-``rank`` approximation.

    At an ordinary positive strict cutoff, this is the usual truncated-SVD
    reconstruction.  At a positive tied cutoff, the optimum is not unique.
    This function makes it deterministic without trusting an arbitrary SVD
    basis:

    1. form the invariant projector onto the full tied *left* singular space;
    2. select the coordinate-lexicographically first required subspace using
       :func:`_coordinate_lexicographic_basis`;
    3. obtain its paired right-side contribution by projecting ``matrix``
       itself, namely ``Q @ Q.T @ matrix``.

    If the requested cutoff is zero, the input has fewer than ``rank``
    positive singular directions.  The best rank-at-most-``rank`` matrix is
    then unique: reconstruct every numerically positive direction and add no
    arbitrary null-space vectors.

    Singular values within ``atol + rtol * sigma_1`` are treated as tied (or
    zero).  Setting both tolerances to zero requests literal comparisons.
    """

    # PSEUDOCODE 1: Validate the matrix/rank and compute one complete economy SVD.
    matrix = validate_matrix(matrix)
    rank = _validate_rank(rank, matrix.shape)
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)

    # PSEUDOCODE 2: If rank reaches the thin ambient dimension, the matrix
    # itself is the unique feasible exact reconstruction, including rank zero.
    if rank == singular_values.size:
        return matrix.copy()

    scale = float(singular_values[0])
    tolerance = comparison_tolerance(scale, atol=atol, rtol=rtol)
    cutoff = float(singular_values[rank - 1])
    excluded = float(singular_values[rank])

    # PSEUDOCODE 3: At a zero cutoff retain every positive direction and no
    # numerically arbitrary vectors from the left/right null spaces.
    if cutoff <= tolerance:
        positive = singular_values > tolerance
        if not np.any(positive):
            return np.zeros_like(matrix)
        return (u[:, positive] * singular_values[positive]) @ vt[positive, :]

    # PSEUDOCODE 4: Preserve ordinary truncated-SVD behavior when the retained
    # and excluded singular values are separated.
    if cutoff - excluded > tolerance:
        return (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]

    # PSEUDOCODE 5: Locate the complete positive tied cluster that straddles r.
    tied = np.abs(singular_values - cutoff) <= tolerance
    tied_indices = np.flatnonzero(tied)
    tied_start = int(tied_indices[0])
    tied_stop = int(tied_indices[-1]) + 1
    if tied_start >= rank or tied_stop <= rank:
        raise np.linalg.LinAlgError(
            "Numerical tie detection did not produce a cluster crossing the cutoff."
        )
    directions_needed = rank - tied_start

    # PSEUDOCODE 6: All strictly larger singular components are uniquely
    # retained, even if they contain complete tied groups of their own.
    if tied_start:
        approximation = (
            u[:, :tied_start] * singular_values[:tied_start]
        ) @ vt[:tied_start, :]
    else:
        approximation = np.zeros_like(matrix)

    # PSEUDOCODE 7: Build the invariant tied projector, select its canonical
    # coordinate subspace, and let the original matrix determine paired right
    # directions.  No basis vector returned inside the raw SVD tie survives.
    tied_left = u[:, tied_start:tied_stop]
    tied_projector = tied_left @ tied_left.T
    canonical_left = _coordinate_lexicographic_basis(
        tied_projector,
        directions_needed,
        atol=atol,
        rtol=rtol,
    )
    approximation += canonical_left @ (canonical_left.T @ matrix)
    return approximation


def strict_polar_factor(
    matrix: Any,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> FloatArray:
    """Return the column-orthonormal polar factor of a full-rank matrix.

    For ``M`` with shape ``(m, r)``, this computes ``Q = U V.T`` from a thin
    SVD.  If ``M`` lacks full column rank, the appendix declares the branch
    unsuccessful; completing ``Q`` with arbitrary coordinate vectors is
    forbidden because it is not block-rotation invariant.
    """

    # PSEUDOCODE 1: A Stiefel polar factor requires at least as many rows as columns.
    matrix = validate_matrix(matrix)
    n_rows, n_columns = matrix.shape
    if n_rows < n_columns:
        raise NumericalFailure(
            FailureReason.RANK_DEFICIENT_POLAR,
            f"A ({n_rows}, {n_columns}) matrix cannot have orthonormal columns.",
        )

    # PSEUDOCODE 2: Inspect all singular values before constructing the factor.
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    scale = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = comparison_tolerance(scale, atol=atol, rtol=rtol)

    # PSEUDOCODE 3: Refuse a deficient/numerically deficient polar argument.
    if singular_values.size != n_columns or float(singular_values[-1]) <= tolerance:
        smallest = float(singular_values[-1]) if singular_values.size else 0.0
        raise NumericalFailure(
            FailureReason.RANK_DEFICIENT_POLAR,
            f"Polar argument is not full column rank: sigma_min={smallest:.6g}, "
            f"threshold={tolerance:.6g}.",
        )

    # PSEUDOCODE 4: The unique full-column-rank polar factor is U V.T.
    factor = u @ vt
    if not np.all(np.isfinite(factor)):
        raise NumericalFailure(
            FailureReason.RANK_DEFICIENT_POLAR,
            "Polar factor contains non-finite values.",
        )
    return factor


def strict_spd_eigh(
    matrix: Any,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> Tuple[FloatArray, FloatArray]:
    """Return eigenpairs of a numerically symmetric positive-definite matrix.

    Returns eigenvalues in ascending order together with the corresponding
    orthonormal eigenvectors, matching :func:`numpy.linalg.eigh`.
    """

    # PSEUDOCODE 1: Validate a finite square Gram matrix.
    matrix = validate_matrix(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise NumericalFailure(
            FailureReason.NON_POSITIVE_DEFINITE,
            f"SPD matrix must be square; got shape {matrix.shape}.",
        )

    # PSEUDOCODE 2: Reject meaningful asymmetry instead of hiding an input bug.
    scale = float(np.linalg.norm(matrix, ord=2))
    tolerance = comparison_tolerance(scale, atol=atol, rtol=rtol)
    asymmetry = float(np.linalg.norm(matrix - matrix.T, ord=2))
    if asymmetry > tolerance:
        raise NumericalFailure(
            FailureReason.NON_POSITIVE_DEFINITE,
            f"Matrix is not symmetric: asymmetry={asymmetry:.6g}, "
            f"tolerance={tolerance:.6g}.",
        )

    # PSEUDOCODE 3: Remove harmless roundoff asymmetry and diagonalize.
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)

    # PSEUDOCODE 4: Require a strictly positive smallest eigenvalue.
    if float(eigenvalues[0]) <= tolerance:
        raise NumericalFailure(
            FailureReason.NON_POSITIVE_DEFINITE,
            f"Matrix is not numerically positive definite: lambda_min="
            f"{float(eigenvalues[0]):.6g}, threshold={tolerance:.6g}.",
        )
    return eigenvalues, eigenvectors


def spd_solve(
    matrix: Any,
    right_hand_side: Any,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> FloatArray:
    """Solve ``matrix @ solution = right_hand_side`` after a strict SPD check."""

    # PSEUDOCODE 1: Certify the Gram matrix and obtain a stable spectral basis.
    eigenvalues, eigenvectors = strict_spd_eigh(matrix, atol=atol, rtol=rtol)

    # PSEUDOCODE 2: Validate the right-hand side without requiring it to be 2-D.
    raw_rhs = np.asarray(right_hand_side)
    if np.iscomplexobj(raw_rhs):
        raise ValueError(
            "right_hand_side must be real-valued; complex input is unsupported."
        )
    rhs = np.asarray(raw_rhs, dtype=float)
    if rhs.ndim not in (1, 2) or rhs.shape[0] != eigenvalues.size:
        raise ValueError(
            "right_hand_side must be a vector or matrix whose first dimension "
            f"is {eigenvalues.size}; got shape {rhs.shape}."
        )
    if not np.all(np.isfinite(rhs)):
        raise ValueError("right_hand_side must contain only finite values.")

    # PSEUDOCODE 3: Apply Q diag(1/lambda) Q.T without forming an inverse.
    coordinates = eigenvectors.T @ rhs
    if rhs.ndim == 1:
        coordinates = coordinates / eigenvalues
    else:
        coordinates = coordinates / eigenvalues[:, None]
    return eigenvectors @ coordinates


def spd_inverse_sqrt(
    matrix: Any,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> FloatArray:
    """Return the symmetric inverse square root of a strictly SPD matrix."""

    # PSEUDOCODE 1: Enforce the same strict definiteness check used by screening/RRR.
    eigenvalues, eigenvectors = strict_spd_eigh(matrix, atol=atol, rtol=rtol)

    # PSEUDOCODE 2: Apply Q diag(lambda^(-1/2)) Q.T in the eigenbasis.
    weighted_vectors = eigenvectors * np.reciprocal(np.sqrt(eigenvalues))
    result = weighted_vectors @ eigenvectors.T

    # PSEUDOCODE 3: Symmetrize the last-bit floating-point discrepancy.
    return 0.5 * (result + result.T)


def spd_inverse(
    matrix: Any,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> FloatArray:
    """Return the symmetric inverse of a strictly SPD matrix.

    Screening should normally call :func:`spd_solve` directly.  This helper is
    provided for appendix formulas or diagnostics that explicitly require the
    inverse as a matrix.
    """

    # PSEUDOCODE 1: Validate the matrix once to determine its square dimension.
    matrix = validate_matrix(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise NumericalFailure(
            FailureReason.NON_POSITIVE_DEFINITE,
            f"SPD matrix must be square; got shape {matrix.shape}.",
        )

    # PSEUDOCODE 2: Solve against the identity rather than using a generic inverse.
    result = spd_solve(
        matrix,
        np.eye(matrix.shape[0], dtype=float),
        atol=atol,
        rtol=rtol,
    )

    # PSEUDOCODE 3: Restore exact symmetry lost only to final-bit arithmetic.
    return 0.5 * (result + result.T)


def spd_square_root(
    matrix: Any,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> FloatArray:
    """Return the symmetric positive square root of a strictly SPD matrix."""

    # PSEUDOCODE 1: Certify the matrix using the common SPD policy.
    eigenvalues, eigenvectors = strict_spd_eigh(matrix, atol=atol, rtol=rtol)

    # PSEUDOCODE 2: Apply Q diag(sqrt(lambda)) Q.T and restore exact symmetry.
    result = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    return 0.5 * (result + result.T)


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "comparison_tolerance",
    "deterministic_rank_at_most_approximation",
    "spd_inverse",
    "spd_inverse_sqrt",
    "spd_solve",
    "spd_square_root",
    "strict_polar_factor",
    "strict_spd_eigh",
    "strict_truncated_svd",
    "truncated_svd",
    "validate_matrix",
]
