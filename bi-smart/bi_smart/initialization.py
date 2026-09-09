"""Fitting-fold reduced-rank initializers and target-only safeguard.

The screen in :mod:`bi_smart.screening` uses only initialization responses.
This module consumes its selected whole-block unions on an independent fitting
fold.  It implements the exact restricted reduced-rank regression estimator in
Eq. ``(bismart-rrr-initializer)``, lifts that estimator into the redundant
block-invariant joint parameterization in Eq. ``(bismart-source-initializer)``,
and implements the target-only safeguard in Eq.
``(bismart-target-only-safeguard)``.

The returned ``BISMARTState`` is the fully specified starting point consumed
by the implemented quotient Gauss--Newton geometry and safeguarded line search
in :mod:`bi_smart.refinement`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .linalg import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    comparison_tolerance,
    deterministic_rank_at_most_approximation,
    spd_inverse_sqrt,
    strict_truncated_svd,
    truncated_svd,
)
from .refinement import BISMARTState, fitted_target
from .types import (
    Candidate,
    CandidateStatus,
    DEFAULT_PINV_RCOND,
    FailureReason,
    FloatArray,
    FoldData,
    NumericalFailure,
    ScreenResult,
    SourceDecomposition,
)


def _finite_matrix(value: NDArray[np.floating], *, name: str) -> FloatArray:
    """Normalize a finite, nonempty matrix for a local result object."""

    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued; complex input is unsupported.")
    result = np.asarray(raw, dtype=float)
    if result.ndim != 2 or 0 in result.shape:
        raise ValueError(f"{name} must be a nonempty matrix; got shape {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


# Backwards-readable mathematical name for the concrete state shared with the
# refinement module.  There is intentionally only one runtime state type.
JointParameterState = BISMARTState


@dataclass(frozen=True)
class InitializationResult:
    """Restricted-RRR candidate and, when successful, its joint start state."""

    candidate: Candidate
    state: Optional[JointParameterState] = None
    coordinate_matrix: Optional[FloatArray] = None

    def __post_init__(self) -> None:
        if self.candidate.status is CandidateStatus.SUCCESSFUL:
            if self.state is None or self.coordinate_matrix is None:
                raise ValueError("A successful initialization requires state and coordinate_matrix.")
            coordinate = _finite_matrix(self.coordinate_matrix, name="coordinate_matrix")
            assert self.candidate.matrix is not None
            if self.candidate.matrix.shape != fitted_target(self.state).shape:
                raise ValueError("Candidate and state target matrices have incompatible shapes.")
            object.__setattr__(self, "coordinate_matrix", coordinate)
        elif self.state is not None or self.coordinate_matrix is not None:
            raise ValueError("An unsuccessful initialization cannot contain a state or coordinates.")

    @property
    def successful(self) -> bool:
        """Whether all restricted-RRR and state-lifting checks passed."""

        return self.candidate.status is CandidateStatus.SUCCESSFUL

    @property
    def coefficient(self) -> Optional[FloatArray]:
        """Return the fitted coefficient, or ``None`` after a numerical failure.

        ``InitializationResult`` predates the direct full-sample API and stores
        its coefficient on ``candidate.matrix``.  This read-only convenience
        property gives simulation code the natural ``result.coefficient``
        spelling without creating a second result type or duplicating the
        candidate's success/failure bookkeeping.
        """

        return self.candidate.matrix


@dataclass(frozen=True)
class _RestrictedRRRSolution:
    """Private algebraic output shared by screened and full-span RRR calls."""

    coefficient: FloatArray
    coordinate_matrix: FloatArray


def _solve_restricted_rrr_coordinates(
    fitting_fold: FoldData,
    left_basis: FloatArray,
    right_basis: FloatArray,
    *,
    target_rank: int,
    atol: float,
    rtol: float,
    metadata: dict[str, Any],
) -> _RestrictedRRRSolution:
    """Solve restricted RRR in fixed left and right source coordinates.

    This is the numerical core of both public entry points in this module.
    The bases may describe a screened union of source blocks or the complete
    leading source spans.  The routine deliberately knows nothing about
    screening, source certificates, Wedin gates, or validation folds.

    ``metadata`` is populated as each algebraic object becomes available.  If
    a strict check raises :class:`NumericalFailure`, callers can therefore
    serialize useful diagnostics for the failed fit as well as successful
    ones.

    PSEUDOCODE
    ----------
    1. Form ``G = U.T @ (X.T @ X / n) @ U`` and require ``G`` to be
       numerically positive definite.
    2. Form ``M = G^(-1/2) @ U.T @ (X.T @ Y / n) @ V``.
    3. Require a positive, strict rank-``target_rank`` cutoff and replace
       ``M`` by its rank-``target_rank`` truncation ``M_r``.
    4. Unwhiten ``Gamma = G^(-1/2) @ M_r`` and reconstruct
       ``C_hat = U @ Gamma @ V.T``.
    """

    n = fitting_fold.n_samples

    # PSEUDOCODE 1: Record the entire reduced spectrum before applying the
    # common strict-SPD policy.  In particular, a singular design remains an
    # auditable unsuccessful fit instead of being silently pseudoinverted.
    fitting_gram = (fitting_fold.X.T @ fitting_fold.X) / n
    selected_gram = left_basis.T @ fitting_gram @ left_basis
    symmetric_gram = 0.5 * (selected_gram + selected_gram.T)
    gram_eigenvalues = np.linalg.eigvalsh(symmetric_gram)
    metadata["reduced_gram_eigenvalues"] = tuple(
        float(value) for value in gram_eigenvalues
    )
    smallest_gram_eigenvalue = float(gram_eigenvalues[0])
    largest_gram_eigenvalue = float(gram_eigenvalues[-1])
    metadata["reduced_gram_condition"] = (
        largest_gram_eigenvalue / smallest_gram_eigenvalue
        if smallest_gram_eigenvalue > 0.0
        else float("inf")
    )
    inverse_sqrt = spd_inverse_sqrt(
        selected_gram,
        atol=atol,
        rtol=rtol,
    )

    # PSEUDOCODE 2--3: Whiten the reduced cross-covariance.  Store the full
    # thin spectrum before enforcing the strict cutoff so zero/tied failures
    # still report the values that caused them.
    cross_covariance = (fitting_fold.X.T @ fitting_fold.Y) / n
    whitened = inverse_sqrt @ left_basis.T @ cross_covariance @ right_basis
    whitened_singular_values = np.linalg.svd(whitened, compute_uv=False)
    retained = float(whitened_singular_values[target_rank - 1])
    excluded = (
        float(whitened_singular_values[target_rank])
        if target_rank < whitened_singular_values.size
        else 0.0
    )
    metadata.update(
        {
            "whitened_singular_values": tuple(
                float(value) for value in whitened_singular_values
            ),
            "rrr_retained_singular_value": retained,
            "rrr_next_singular_value": excluded,
            "rrr_cutoff_gap": retained - excluded,
        }
    )
    whitened_svd = strict_truncated_svd(
        whitened,
        target_rank,
        atol=atol,
        rtol=rtol,
    )

    # PSEUDOCODE 4: Unwhiten only after the two strict checks pass.  Avoiding a
    # generic inverse here preserves the same SPD policy as screened BI-SMART.
    coordinate_matrix = inverse_sqrt @ whitened_svd.approximation
    coefficient = left_basis @ coordinate_matrix @ right_basis.T
    return _RestrictedRRRSolution(
        coefficient=coefficient,
        coordinate_matrix=coordinate_matrix,
    )


def _failed_initialization(
    *,
    label: str,
    reason: FailureReason,
    message: str,
    order_key: Sequence[Any],
    metadata: Mapping[str, Any],
) -> InitializationResult:
    """Create the branch record that Appendix Algorithm 2 will omit."""

    return InitializationResult(
        candidate=Candidate.unsuccessful(
            label=label,
            reason=reason,
            message=message,
            order_key=order_key,
            kind="restricted_rrr",
            metadata=metadata,
        )
    )


def initialize_restricted_rrr(
    fitting_fold: FoldData,
    source: SourceDecomposition,
    screen: ScreenResult,
    *,
    target_rank: int,
    label: str = "restricted_rrr",
    order_key: Sequence[Any] = (),
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> InitializationResult:
    """Recompute exact restricted RRR on the independent fitting fold.

    PSEUDOCODE
    ----------
    1. Read the exact block labels selected on the initialization fold and
       concatenate the associated columns of ``U_hat_0`` and ``V_hat_0``.
    2. Strictly certify the selected left reduced Gram matrix and compute its
       symmetric inverse square root.
    3. Whiten the fitting-fold cross-covariance on the left, take the unique
       nonzero rank-``r`` truncation, and unwhiten it
       [Eq. ``(bismart-rrr-initializer)``].
    4. Retain the resulting ambient coefficient as the exact fitting-fold RRR
       candidate; this is candidate ``t=0``, not merely solver scratch state.
    5. Factor its coordinate matrix, pad ``A`` and ``B`` with exact zero rows,
       and construct one full source core per block
       [Eq. ``(bismart-source-initializer)``].

    The caller should invoke this only after the observable Wedin gate passes.
    The gate is deliberately not repeated here because it depends on the
    source-error certificate and belongs to source-block orchestration.
    """

    if isinstance(target_rank, (bool, np.bool_)) or not isinstance(
        target_rank, (int, np.integer)
    ):
        raise TypeError("target_rank must be an integer.")
    target_rank = int(target_rank)
    if target_rank < 1 or target_rank > source.rank:
        raise ValueError("target_rank must lie in [1, source.rank].")
    if fitting_fold.n_features != source.source_shape[0]:
        raise ValueError("Fitting predictors and source rows are incompatible.")
    if fitting_fold.n_responses != source.source_shape[1]:
        raise ValueError("Fitting responses and source columns are incompatible.")
    if screen.partition.rank != source.rank:
        raise ValueError("Screen partition and source decomposition use different r0 values.")

    metadata = {
        "partition": screen.partition.blocks,
        "left_blocks": screen.left_blocks,
        "right_blocks": screen.right_blocks,
        "target_rank": target_rank,
    }
    if screen.status is not CandidateStatus.SUCCESSFUL:
        return _failed_initialization(
            label=label,
            reason=FailureReason.SCREEN_FAILED,
            message="Restricted RRR requires a successful screen with both block unions.",
            order_key=order_key,
            metadata=metadata,
        )

    partition = screen.partition
    left_indices = partition.union_indices(screen.left_blocks)
    right_indices = partition.union_indices(screen.right_blocks)
    if len(left_indices) < target_rank or len(right_indices) < target_rank:
        return _failed_initialization(
            label=label,
            reason=FailureReason.SCREEN_FAILED,
            message="A screened union has dimension smaller than target_rank.",
            order_key=order_key,
            metadata=metadata,
        )
    metadata = {
        **metadata,
        "left_dimension": len(left_indices),
        "right_dimension": len(right_indices),
    }

    U0 = source.left_vectors
    V0 = source.right_vectors

    # PSEUDOCODE 0: A generalized SourceDecomposition may express the same
    # source approximation in independently rotated left/right bases, giving a
    # full core inside each allowed block.  It must not mix *across* blocks or
    # move singular values between them: doing so would make the certified
    # partition inconsistent and the block-diagonal initializer would silently
    # discard source mass.  Treat either condition as a local invalid branch.
    assert source.coordinate_core is not None
    full_source_core = source.coordinate_core
    allowed_entries = np.zeros_like(full_source_core, dtype=bool)
    for block in partition.blocks:
        indices = np.asarray(block, dtype=int)
        allowed_entries[np.ix_(indices, indices)] = True
    off_block_core = np.where(allowed_entries, 0.0, full_source_core)
    core_scale = float(np.linalg.norm(full_source_core, ord=2))
    core_tolerance = comparison_tolerance(
        core_scale,
        atol=atol,
        rtol=rtol,
    )
    if float(np.linalg.norm(off_block_core, ord=2)) > core_tolerance:
        return _failed_initialization(
            label=label,
            reason=FailureReason.INVALID_INPUT,
            message=(
                "The source coordinate core has mass across the supplied "
                "partition blocks."
            ),
            order_key=order_key,
            metadata=metadata,
        )
    for block_label, block in enumerate(partition.blocks):
        indices = np.asarray(block, dtype=int)
        block_core = full_source_core[np.ix_(indices, indices)]
        actual_values = np.linalg.svd(block_core, compute_uv=False)
        expected_values = source.singular_values[indices]
        block_scale = max(
            float(expected_values[0]),
            float(actual_values[0]),
        )
        block_tolerance = comparison_tolerance(
            block_scale,
            atol=atol,
            rtol=rtol,
        )
        if not np.all(np.abs(actual_values - expected_values) <= block_tolerance):
            return _failed_initialization(
                label=label,
                reason=FailureReason.INVALID_INPUT,
                message=(
                    f"Source core block {block_label} does not carry the "
                    "singular values assigned to that certified block."
                ),
                order_key=order_key,
                metadata=metadata,
            )

    U_selected = U0[:, np.asarray(left_indices, dtype=int)]
    V_selected = V0[:, np.asarray(right_indices, dtype=int)]
    try:
        solution = _solve_restricted_rrr_coordinates(
            fitting_fold,
            U_selected,
            V_selected,
            target_rank=target_rank,
            atol=atol,
            rtol=rtol,
            metadata=metadata,
        )
    except NumericalFailure as failure:
        return _failed_initialization(
            label=label,
            reason=failure.reason,
            message=str(failure),
            order_key=order_key,
            metadata=metadata,
        )

    coordinate_matrix = solution.coordinate_matrix
    coefficient = solution.coefficient

    # PSEUDOCODE 4--5: Take a compact factorization of the *unwhitened*
    # coordinate solution, then embed its factors into r0 source coordinates.
    # Its rank is r after the strict whitened cutoff and invertible unwhitening.
    coordinate_svd = truncated_svd(coordinate_matrix, target_rank)
    A = np.zeros((source.rank, target_rank), dtype=float)
    B = np.zeros((source.rank, target_rank), dtype=float)
    A[np.asarray(left_indices, dtype=int), :] = coordinate_svd.u
    B[np.asarray(right_indices, dtype=int), :] = coordinate_svd.vt.T
    H = np.diag(coordinate_svd.singular_values)

    # PSEUDOCODE 6: Copy each full block of the already validated source core.
    # It is diagonal in the automatic SVD bases and generally full after
    # independent within-block left/right rotations.  Keeping the full blocks
    # is precisely what preserves the represented source under those rotations.
    G_blocks: list[FloatArray] = []
    for block in partition.blocks:
        indices = np.asarray(block, dtype=int)
        G_blocks.append(full_source_core[np.ix_(indices, indices)].copy())

    active_u = np.zeros(source.rank, dtype=bool)
    active_v = np.zeros(source.rank, dtype=bool)
    active_u[np.asarray(left_indices, dtype=int)] = True
    active_v[np.asarray(right_indices, dtype=int)] = True
    state = BISMARTState(
        U0=U0,
        V0=V0,
        A=A,
        B=B,
        H=H,
        G_blocks=tuple(G_blocks),
        active_u=active_u,
        active_v=active_v,
    )
    candidate = Candidate.successful(
        label=label,
        matrix=coefficient,
        order_key=order_key,
        kind="restricted_rrr",
        metadata=metadata,
    )
    return InitializationResult(
        candidate=candidate,
        state=state,
        coordinate_matrix=coordinate_matrix,
    )


def restricted_rrr(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    observed_source: NDArray[np.floating],
    target_rank: int,
    source_rank: int,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> InitializationResult:
    """Fit full-sample RRR inside the leading observed-source subspaces.

    This deliberately small API implements the first simulation pilot rather
    than the complete candidate-generating BI-SMART procedure.  Given a target
    sample ``(X, Y)`` and observed source coefficient ``observed_source``, it
    solves

    ``min ||Y - X U0 Gamma V0.T||_F^2  subject to rank(Gamma) <= target_rank``,

    where ``U0`` and ``V0`` contain the leading ``source_rank`` singular
    vectors of ``observed_source``.  All target observations are used once for
    this fit.  The routine performs no screen, source-error certification,
    Wedin gate, candidate selection, fallback, or Gauss--Newton refinement.

    Parameters
    ----------
    X, Y:
        Target design and response matrices with shapes ``(n, p)`` and
        ``(n, q)``.
    observed_source:
        Observed source coefficient matrix with shape ``(p, q)``.
    target_rank:
        Rank ``r`` imposed on the reduced coordinate coefficient.
    source_rank:
        Number ``r0`` of leading observed-source directions retained on both
        sides.  This pilot requires ``1 <= target_rank <= source_rank``.
    atol, rtol:
        Nonnegative numerical tolerances for the positive-definite reduced
        design check and strict rank-``r`` cutoff.

    Returns
    -------
    InitializationResult
        On success, ``result.coefficient`` is the fitted ``p``-by-``q``
        coefficient.  A singular reduced design, zero retained component, or
        unresolved rank cutoff produces ``result.successful == False`` and an
        explicit failure reason on ``result.candidate``.  Malformed inputs
        raise ``TypeError`` or ``ValueError`` immediately.

    PSEUDOCODE
    ----------
    1. Validate and pair every row of ``X`` and ``Y``.
    2. Compute the leading rank-``source_rank`` SVD of ``observed_source``;
       record, but do not gate on, its truncation gap.
    3. Call the common restricted-RRR core with *all* retained left and right
       source directions.
    4. Lift the fitted coordinate matrix into a one-block state solely so the
       result has the same representation as screened RRR initializations.
    """

    # PSEUDOCODE 1: Normalize the complete target sample through the package's
    # public fold value object.  Despite its historical name, no split occurs.
    fitting_fold = FoldData(X=X, Y=Y, name="full_sample")
    source_matrix = _finite_matrix(observed_source, name="observed_source")
    if source_matrix.shape != (
        fitting_fold.n_features,
        fitting_fold.n_responses,
    ):
        raise ValueError(
            "observed_source must have shape "
            f"({fitting_fold.n_features}, {fitting_fold.n_responses}); "
            f"got {source_matrix.shape}."
        )

    if isinstance(source_rank, (bool, np.bool_)) or not isinstance(
        source_rank, (int, np.integer)
    ):
        raise TypeError("source_rank must be an integer.")
    source_rank = int(source_rank)
    maximum_source_rank = min(source_matrix.shape)
    if source_rank < 1 or source_rank > maximum_source_rank:
        raise ValueError(
            f"source_rank must lie in [1, {maximum_source_rank}]."
        )
    if isinstance(target_rank, (bool, np.bool_)) or not isinstance(
        target_rank, (int, np.integer)
    ):
        raise TypeError("target_rank must be an integer.")
    target_rank = int(target_rank)
    if target_rank < 1 or target_rank > source_rank:
        raise ValueError("target_rank must lie in [1, source_rank].")

    # Validate tolerances before running an SVD so invalid public controls fail
    # deterministically even on data that would later trigger another branch.
    comparison_tolerance(0.0, atol=atol, rtol=rtol)

    # PSEUDOCODE 2: This is a plain truncated SVD.  In particular, the pilot
    # records a tied source boundary but intentionally does not invoke the
    # complete procedure's source-certificate boundary veto.
    source_svd = truncated_svd(source_matrix, source_rank)
    source = SourceDecomposition(
        u=source_svd.u,
        singular_values=source_svd.singular_values,
        vt=source_svd.vt,
        next_singular_value=source_svd.next_singular_value,
        source_shape=source_matrix.shape,
    )
    metadata: dict[str, Any] = {
        "method": "full_sample_restricted_rrr",
        "n_samples": fitting_fold.n_samples,
        "n_features": fitting_fold.n_features,
        "n_responses": fitting_fold.n_responses,
        "target_rank": target_rank,
        "source_rank": source_rank,
        "source_singular_values": tuple(
            float(value) for value in source.singular_values
        ),
        "source_next_singular_value": float(source.next_singular_value),
        "source_boundary_gap": float(
            source.singular_values[-1] - source.next_singular_value
        ),
        "atol": float(atol),
        "rtol": float(rtol),
    }

    # PSEUDOCODE 3: Use every retained source direction; there is no block
    # screen and hence no support-selection state to carry between samples.
    try:
        solution = _solve_restricted_rrr_coordinates(
            fitting_fold,
            source.left_vectors,
            source.right_vectors,
            target_rank=target_rank,
            atol=float(atol),
            rtol=float(rtol),
            metadata=metadata,
        )
    except NumericalFailure as failure:
        return _failed_initialization(
            label="full_sample/restricted_rrr",
            reason=failure.reason,
            message=str(failure),
            order_key=(),
            metadata=metadata,
        )

    # PSEUDOCODE 4: The direct estimator itself needs only ``coefficient``.
    # Build the established joint-state representation as well so downstream
    # diagnostics can reuse ``fitted_target``/``fitted_source`` without a
    # special case.  Its sole source block is the complete retained core.
    coordinate_svd = truncated_svd(solution.coordinate_matrix, target_rank)
    assert source.coordinate_core is not None
    state = BISMARTState(
        U0=source.left_vectors,
        V0=source.right_vectors,
        A=coordinate_svd.u,
        B=coordinate_svd.vt.T,
        H=np.diag(coordinate_svd.singular_values),
        G_blocks=(source.coordinate_core.copy(),),
        active_u=np.ones(source_rank, dtype=bool),
        active_v=np.ones(source_rank, dtype=bool),
    )
    candidate = Candidate.successful(
        label="full_sample/restricted_rrr",
        matrix=solution.coefficient,
        kind="restricted_rrr",
        metadata=metadata,
    )
    return InitializationResult(
        candidate=candidate,
        state=state,
        coordinate_matrix=solution.coordinate_matrix,
    )


def target_only_rrr(
    fitting_fold: FoldData,
    *,
    target_rank: int,
    label: str = "target_only_rrr",
    order_key: Sequence[Any] = (),
    pinv_rcond: float | None = DEFAULT_PINV_RCOND,
) -> Candidate:
    """Compute the target-only rank-at-most-``r`` safeguard.

    Implements

    ``X_ft^dagger { P_ft Y_ft }_r`` with
    ``P_ft = X_ft X_ft^dagger``

    from Eq. ``(bismart-target-only-safeguard)``.  The multiplication order
    avoids materializing the potentially large ``n_ft x n_ft`` projector.
    At a positive tied cutoff, the helper uses a fixed coordinate-lexicographic
    subspace of the invariant tied singular-space projector.  At a zero cutoff
    it reconstructs only the uniquely identified positive-rank component.
    Unlike source-guided RRR, the paper keeps this safeguard rather than
    declaring a tied cutoff unsuccessful.  ``pinv_rcond`` is always passed to
    NumPy explicitly; its package default is fixed rather than inherited from
    a NumPy-version-dependent pseudoinverse default.
    """

    if isinstance(target_rank, (bool, np.bool_)) or not isinstance(
        target_rank, (int, np.integer)
    ):
        raise TypeError("target_rank must be an integer.")
    target_rank = int(target_rank)
    if target_rank < 1:
        raise ValueError("target_rank must be positive.")
    if target_rank > min(fitting_fold.n_features, fitting_fold.n_responses):
        raise ValueError("target_rank cannot exceed min(number of predictors, responses).")
    # ``None`` was accepted by the initial scaffold.  Preserve that call shape
    # while mapping it to the package's named cutoff, never NumPy's implicit
    # and version-dependent default.
    if pinv_rcond is None:
        pinv_rcond = DEFAULT_PINV_RCOND
    raw_rcond = np.asarray(pinv_rcond)
    if np.iscomplexobj(raw_rcond):
        raise ValueError(
            "pinv_rcond must be real-valued; complex input is unsupported."
        )
    if raw_rcond.ndim != 0:
        raise ValueError("pinv_rcond must be a scalar.")
    pinv_rcond = float(raw_rcond)
    if not np.isfinite(pinv_rcond) or pinv_rcond < 0:
        raise ValueError("pinv_rcond must be finite and nonnegative.")

    # PSEUDOCODE 1: Compute the Moore--Penrose map.  It chooses the minimum-norm
    # coefficient representative when the fitting design is singular.  Pass
    # the named cutoff explicitly so behavior does not depend on NumPy's
    # version-specific default.
    design_pseudoinverse = np.linalg.pinv(
        fitting_fold.X,
        rcond=pinv_rcond,
    )

    # PSEUDOCODE 2: Form P_ft Y_ft without allocating P_ft itself.
    projected_response = fitting_fold.X @ (design_pseudoinverse @ fitting_fold.Y)

    # PSEUDOCODE 3: Apply a deterministic best rank-at-most-r approximation.
    # If r exceeds the matrix's possible rank, retaining its full thin rank is
    # exactly the same rank-at-most-r optimization.
    retained_rank = min(target_rank, min(projected_response.shape))
    truncated_response = deterministic_rank_at_most_approximation(
        projected_response,
        retained_rank,
    )

    # PSEUDOCODE 4: Map the fitted values back to the minimum-norm coefficient.
    coefficient = design_pseudoinverse @ truncated_response
    design_singular_values = np.linalg.svd(
        fitting_fold.X,
        compute_uv=False,
    )
    design_cutoff = (
        pinv_rcond * float(design_singular_values[0])
        if design_singular_values.size
        else 0.0
    )
    effective_design_rank = int(
        np.count_nonzero(design_singular_values > design_cutoff)
    )
    return Candidate.successful(
        label=label,
        matrix=coefficient,
        order_key=order_key,
        kind="target_only_rrr",
        metadata={
            "target_rank": target_rank,
            "design_rank": effective_design_rank,
            "pinv_rcond": pinv_rcond,
        },
    )


__all__ = [
    "InitializationResult",
    "JointParameterState",
    "initialize_restricted_rrr",
    "restricted_rrr",
    "target_only_rrr",
]
