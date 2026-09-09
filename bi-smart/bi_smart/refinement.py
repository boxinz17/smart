"""Fixed-support geometry for the BI-SMART refinement stage.

This module translates the parts of Appendix ``Block-invariant iterative
SMART`` that are completely specified into small, testable NumPy routines.
In the notation of the paper, a state is

``theta = (U0, V0, A, B, H, G)``

with fitted target and source matrices

``C(theta)  = U0 @ A @ H @ B.T @ V0.T`` and
``C0(theta) = U0 @ block_diag(G_blocks) @ V0.T``.

The source cores and target core are *full* matrices.  In particular, the
source cores must not be silently diagonalised: allowing full cores is what
makes the fitted matrices invariant to rotations inside a source spectral
block.

The appendix permits two equivalent implementations of the quotient
Gauss--Newton direction: a positive-definite solve in a horizontal basis, or
the product-metric Moore--Penrose solution of the singular full-tangent
equations.  This module uses the latter with two interchangeable backends.
The dense reference applies a rank-revealing SVD to the residual Jacobian;
the scalable backend applies zero-start LSQR through analytic Jacobian and
adjoint actions.  Both use projected canonical ambient coordinates as a
Parseval frame, so the lifted minimum-coefficient-norm solution is the
minimum-product-norm (hence horizontal) direction without choosing an
arbitrary orthogonal complement or squaring the condition number.

The code is intentionally verbose.  It is research scaffolding: comments name
the corresponding mathematical operation and make each remaining design
choice visible rather than hiding it behind a generic optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._matrix_free_linalg import lsqr
from ._quotient_geometry import (
    GeometryValidationError,
    ProductTangent,
    TangentParsevalFrame,
    geometry_dimensions,
    iter_vertical_gauge_generators,
    product_frobenius_inner_product,
    tangent_parseval_frame,
    validate_regular_state,
    vertical_gauge_generators,
)
from .types import FailureReason, NumericalFailure


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
DEFAULT_MAX_DENSE_WORK_BYTES = 512 * 1024**2
DEFAULT_MAX_COMPACT_WORK_BYTES = 256 * 1024**2
STATE_ORTHOGONALITY_TOLERANCE = 1e-10


class RefinementError(RuntimeError):
    """Raised when a fixed-support refinement operation is algebraically invalid."""


@dataclass(frozen=True, slots=True)
class GaussNewtonDiagnostics:
    """Numerical certificate for one quotient Gauss--Newton direction.

    The dimensions are mathematical invariants of the fixed-support quotient.
    ``gauss_newton_norm_squared`` is ``<xi, xi>_GN = ||J xi||^2``; half of it
    is the full-step quadratic-model reduction used by convergence-aware
    callers.  The remaining fields expose every tolerance-sensitive decision
    made by the selected solver, which is useful when a research run treats one
    calibrated refinement call as unsuccessful.  ``rank_certificate``
    distinguishes the dense residual-Jacobian SVD from the matrix-free
    structural and necessary numerical screens.  The three generic SVD fields
    are ``None`` for the matrix-free backend because its compact spectra are
    not spectra of the complete Jacobian.
    """

    tangent_dimension: int
    vertical_dimension: int
    quotient_dimension: int
    frame_dimension: int
    residual_dimension: int
    estimated_dense_work_bytes: int
    dense_work_limit_bytes: int | None
    jacobian_rank: int | None
    rank_tolerance: float | None
    largest_singular_value: float | None
    smallest_retained_singular_value: float | None
    residual_norm: float
    normal_residual_norm: float
    tangency_error: float
    horizontality_error: float
    descent_identity_error: float
    gauss_newton_norm_squared: float
    solver_backend: str = "dense"
    solver_iterations: int = 0
    solver_stop_reason: str = "dense_svd"
    condition_estimate: float | None = None
    rank_certificate: str = "dense_residual_jacobian_svd"
    structural_quotient_rank: int | None = None
    reduced_design_rank_tolerance: float | None = None
    reduced_design_largest_singular_value: float | None = None
    reduced_design_smallest_singular_value: float | None = None
    operator_scale_lower_bound: float | None = None
    compact_jacobian_rank: int | None = None
    compact_quotient_dimension: int | None = None
    compact_rank_tolerance: float | None = None
    compact_largest_singular_value: float | None = None
    compact_smallest_singular_value: float | None = None
    estimated_compact_work_bytes: int | None = None
    ambient_normal_sensitivity_upper_bound: float | None = None
    full_rank_tolerance_lower_bound: float | None = None

    def as_metadata(self) -> dict[str, int | float | str | None]:
        """Return JSON-friendly fields for candidate metadata."""

        return {
            "gn_tangent_dimension": self.tangent_dimension,
            "gn_vertical_dimension": self.vertical_dimension,
            "gn_quotient_dimension": self.quotient_dimension,
            "gn_frame_dimension": self.frame_dimension,
            "gn_residual_dimension": self.residual_dimension,
            "gn_estimated_dense_work_bytes": self.estimated_dense_work_bytes,
            "gn_dense_work_limit_bytes": self.dense_work_limit_bytes,
            "gn_jacobian_rank": self.jacobian_rank,
            "gn_rank_tolerance": self.rank_tolerance,
            "gn_largest_singular_value": self.largest_singular_value,
            "gn_smallest_retained_singular_value": (
                self.smallest_retained_singular_value
            ),
            "gn_residual_norm": self.residual_norm,
            "gn_normal_residual_norm": self.normal_residual_norm,
            "gn_tangency_error": self.tangency_error,
            "gn_horizontality_error": self.horizontality_error,
            "gn_descent_identity_error": self.descent_identity_error,
            "gn_norm_squared": self.gauss_newton_norm_squared,
            "gn_solver_backend": self.solver_backend,
            "gn_solver_iterations": self.solver_iterations,
            "gn_solver_stop_reason": self.solver_stop_reason,
            "gn_condition_estimate": self.condition_estimate,
            "gn_rank_certificate": self.rank_certificate,
            "gn_structural_quotient_rank": self.structural_quotient_rank,
            "gn_reduced_design_rank_tolerance": (
                self.reduced_design_rank_tolerance
            ),
            "gn_reduced_design_largest_singular_value": (
                self.reduced_design_largest_singular_value
            ),
            "gn_reduced_design_smallest_singular_value": (
                self.reduced_design_smallest_singular_value
            ),
            "gn_operator_scale_lower_bound": self.operator_scale_lower_bound,
            "gn_compact_jacobian_rank": self.compact_jacobian_rank,
            "gn_compact_quotient_dimension": self.compact_quotient_dimension,
            "gn_compact_rank_tolerance": self.compact_rank_tolerance,
            "gn_compact_largest_singular_value": (
                self.compact_largest_singular_value
            ),
            "gn_compact_smallest_singular_value": (
                self.compact_smallest_singular_value
            ),
            "gn_estimated_compact_work_bytes": (
                self.estimated_compact_work_bytes
            ),
            "gn_ambient_normal_sensitivity_upper_bound": (
                self.ambient_normal_sensitivity_upper_bound
            ),
            "gn_full_rank_tolerance_lower_bound": (
                self.full_rank_tolerance_lower_bound
            ),
        }


@dataclass(frozen=True, slots=True)
class GaussNewtonResult:
    """A certified horizontal direction and its numerical diagnostics."""

    direction: "BISMARTDirection"
    diagnostics: GaussNewtonDiagnostics


def _as_float_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    """Return ``value`` as a finite two-dimensional floating-point array."""

    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued; complex input is unsupported")
    array = np.asarray(raw, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array; got {array.ndim}D")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_active_mask(
    value: ArrayLike | None,
    *,
    size: int,
    rank: int,
    name: str,
) -> BoolArray:
    """Normalize an explicit selected-coordinate mask.

    ``None`` means that all source coordinates are active.  BI-SMART callers
    with screened supports should always pass the recorded selected block
    union.  Inferring a support from nonzero rows would be incorrect: a row in
    a selected block may legitimately have zero fitted energy.
    """

    if value is None:
        mask = np.ones(size, dtype=bool)
    else:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be a real boolean mask")
        if raw.ndim != 1 or raw.shape[0] != size:
            raise ValueError(f"{name} must have shape ({size},)")
        if raw.dtype != np.bool_:
            try:
                numeric = np.asarray(raw, dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{name} must contain only boolean or exact zero/one values"
                ) from error
            if not np.all(np.isfinite(numeric)) or not np.all(
                np.logical_or(numeric == 0.0, numeric == 1.0)
            ):
                raise ValueError(
                    f"{name} must contain only boolean or exact zero/one values"
                )
        mask = np.asarray(raw, dtype=bool).copy()
    if int(mask.sum()) < rank:
        raise ValueError(
            f"{name} selects {int(mask.sum())} coordinates, fewer than rank={rank}"
        )
    return mask


def _require_stiefel_factor(matrix: FloatArray, *, name: str) -> None:
    """Validate one public state factor at a fixed unit-scale tolerance."""

    rows, columns = matrix.shape
    if rows < columns:
        raise ValueError(f"{name} must have at least as many rows as columns")
    error = float(
        np.linalg.norm(matrix.T @ matrix - np.eye(columns), ord=2)
    )
    machine_floor = 64.0 * np.finfo(float).eps * max(rows, columns)
    tolerance = max(STATE_ORTHOGONALITY_TOLERANCE, machine_floor)
    if error > tolerance:
        raise ValueError(
            f"{name} must have orthonormal columns; "
            f"error={error:.3e}, tolerance={tolerance:.3e}"
        )


def _require_whole_block_mask(
    mask: BoolArray,
    block_sizes: tuple[int, ...],
    *,
    name: str,
) -> None:
    """Require a fixed support to contain either all or none of each block."""

    for index, coordinate_slice in enumerate(_block_slices(block_sizes)):
        selected = mask[coordinate_slice]
        if np.any(selected) and not np.all(selected):
            raise ValueError(f"{name} partially selects source block {index}")


@dataclass(frozen=True, slots=True)
class BISMARTState:
    """One fixed-support BI-SMART factor tuple.

    Parameters follow the appendix exactly:

    - ``U0`` and ``V0`` are the adjustable source bases.
    - ``A`` and ``B`` express target factors in those source coordinates.
    - ``H`` is the unrestricted target core.
    - ``G_blocks`` are unrestricted square source cores, one per partition
      block.  Their dimensions must sum to ``r0``.
    - ``active_u`` and ``active_v`` are boolean masks for the *recorded*
      selected block unions.  Inactive rows of ``A`` and ``B`` stay zero.

    The constructor validates dimensions, finiteness, whole-block support, and
    numerical column orthonormality.  This prevents invalid public states from
    reaching quotient geometry or a polar retraction.
    """

    U0: FloatArray
    V0: FloatArray
    A: FloatArray
    B: FloatArray
    H: FloatArray
    G_blocks: tuple[FloatArray, ...]
    active_u: BoolArray | None = None
    active_v: BoolArray | None = None

    def __post_init__(self) -> None:
        matrices = {
            "U0": _as_float_matrix(self.U0, name="U0"),
            "V0": _as_float_matrix(self.V0, name="V0"),
            "A": _as_float_matrix(self.A, name="A"),
            "B": _as_float_matrix(self.B, name="B"),
            "H": _as_float_matrix(self.H, name="H"),
        }
        for name, matrix in matrices.items():
            object.__setattr__(self, name, matrix)

        p, r0 = matrices["U0"].shape
        q, v_r0 = matrices["V0"].shape
        if v_r0 != r0:
            raise ValueError("U0 and V0 must have the same number of columns")
        if r0 < 1:
            raise ValueError("the working source rank r0 must be at least one")
        if p < r0 or q < r0:
            raise ValueError("U0 and V0 must have at least r0 rows")

        if matrices["A"].shape[0] != r0 or matrices["B"].shape[0] != r0:
            raise ValueError("A and B must each have r0 rows")
        rank = matrices["A"].shape[1]
        if rank < 1:
            raise ValueError("the target rank must be at least one")
        if matrices["B"].shape[1] != rank:
            raise ValueError("A and B must have the same number of columns")
        if matrices["H"].shape != (rank, rank):
            raise ValueError(f"H must have shape ({rank}, {rank})")

        blocks = tuple(
            _as_float_matrix(block, name=f"G_blocks[{index}]")
            for index, block in enumerate(self.G_blocks)
        )
        if not blocks:
            raise ValueError("G_blocks must contain at least one source block core")
        for index, block in enumerate(blocks):
            if block.shape[0] != block.shape[1]:
                raise ValueError(f"G_blocks[{index}] must be square")
        if sum(block.shape[0] for block in blocks) != r0:
            raise ValueError("the G block dimensions must sum to r0")
        object.__setattr__(self, "G_blocks", blocks)

        active_u = _as_active_mask(
            self.active_u, size=r0, rank=rank, name="active_u"
        )
        active_v = _as_active_mask(
            self.active_v, size=r0, rank=rank, name="active_v"
        )
        object.__setattr__(self, "active_u", active_u)
        object.__setattr__(self, "active_v", active_v)

        block_sizes = tuple(block.shape[0] for block in blocks)
        _require_whole_block_mask(active_u, block_sizes, name="active_u")
        _require_whole_block_mask(active_v, block_sizes, name="active_v")

        # The support is part of the state, not something inferred from its
        # numerical values.  Reject leakage into inactive rows early.
        if np.any(np.abs(matrices["A"][~active_u, :]) > 1e-12):
            raise ValueError("A must be zero outside active_u")
        if np.any(np.abs(matrices["B"][~active_v, :]) > 1e-12):
            raise ValueError("B must be zero outside active_v")

        _require_stiefel_factor(matrices["U0"], name="U0")
        _require_stiefel_factor(matrices["V0"], name="V0")
        _require_stiefel_factor(matrices["A"][active_u, :], name="A_active")
        _require_stiefel_factor(matrices["B"][active_v, :], name="B_active")

    @property
    def r0(self) -> int:
        """Working source rank."""

        return self.U0.shape[1]

    @property
    def rank(self) -> int:
        """Target rank."""

        return self.A.shape[1]

    @property
    def block_sizes(self) -> tuple[int, ...]:
        """Dimensions of the consecutive source blocks."""

        return tuple(block.shape[0] for block in self.G_blocks)


@dataclass(frozen=True, slots=True)
class BISMARTDirection:
    """A product-space increment or gradient with the shape of a state."""

    U0: FloatArray
    V0: FloatArray
    A: FloatArray
    B: FloatArray
    H: FloatArray
    G_blocks: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        for name in ("U0", "V0", "A", "B", "H"):
            object.__setattr__(
                self, name, _as_float_matrix(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "G_blocks",
            tuple(
                _as_float_matrix(block, name=f"G_blocks[{index}]")
                for index, block in enumerate(self.G_blocks)
            ),
        )


@dataclass(frozen=True, slots=True)
class TrustRegionThresholds:
    """Gauge-invariant finite-path thresholds from the BI-SMART appendix."""

    radius: float
    h_min: float
    g_min: float
    separation_min: float

    def __post_init__(self) -> None:
        values = (self.radius, self.h_min, self.g_min, self.separation_min)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("trust-region thresholds must be finite")
        if self.radius <= 0.0 or self.h_min <= 0.0 or self.g_min <= 0.0:
            raise ValueError("radius, h_min, and g_min must be strictly positive")
        if self.separation_min < 0.0:
            raise ValueError("separation_min must be nonnegative")


@dataclass(frozen=True, slots=True)
class PathCertificate:
    """Result of the five finite path-safety inequalities."""

    safe: bool
    failed_checks: tuple[str, ...]
    step_size: float
    direction_norm: float
    pair_distance: float
    target_core_lower_bound: float
    source_core_lower_bound: float
    separation_lower_bound: float


@dataclass(frozen=True, slots=True)
class ArmijoCheck:
    """Objective values used by one Armijo acceptance test."""

    accepted: bool
    current_objective: float
    trial_objective: float
    required_upper_bound: float
    gauss_newton_norm_squared: float


@dataclass(frozen=True, slots=True)
class BacktrackingResult:
    """Outcome of a finite safeguarded backtracking search."""

    accepted: bool
    state: BISMARTState | None
    step_size: float | None
    trials: int
    path_certificate: PathCertificate | None
    armijo_check: ArmijoCheck | None
    failure_reason: str | None = None


def _block_slices(block_sizes: Iterable[int]) -> tuple[slice, ...]:
    """Convert consecutive block sizes to coordinate slices."""

    slices: list[slice] = []
    start = 0
    for size in block_sizes:
        stop = start + int(size)
        slices.append(slice(start, stop))
        start = stop
    return tuple(slices)


def block_diagonal(blocks: tuple[FloatArray, ...]) -> FloatArray:
    """Materialize a block-diagonal matrix without changing its full cores."""

    normalized = tuple(
        _as_float_matrix(block, name=f"blocks[{index}]")
        for index, block in enumerate(blocks)
    )
    for index, block in enumerate(normalized):
        if block.shape[0] != block.shape[1]:
            raise ValueError(f"blocks[{index}] must be square")
    size = sum(block.shape[0] for block in normalized)
    result = np.zeros((size, size), dtype=float)
    for coordinate_slice, block in zip(
        _block_slices(block.shape[0] for block in normalized),
        normalized,
        strict=True,
    ):
        result[coordinate_slice, coordinate_slice] = block
    return result


def fitted_target(state: BISMARTState) -> FloatArray:
    """Return ``C(theta) = U0 A H B^T V0^T``."""

    return state.U0 @ state.A @ state.H @ state.B.T @ state.V0.T


def fitted_source(state: BISMARTState) -> FloatArray:
    """Return ``C0(theta) = U0 blockdiag(G_l) V0^T``.

    The blockwise sum avoids materialising the full ``r0 x r0`` core and
    emphasizes that each ``G_l`` is a full within-block matrix.
    """

    result = np.zeros((state.U0.shape[0], state.V0.shape[0]), dtype=float)
    for coordinate_slice, core in zip(
        _block_slices(state.block_sizes), state.G_blocks, strict=True
    ):
        result += (
            state.U0[:, coordinate_slice]
            @ core
            @ state.V0[:, coordinate_slice].T
        )
    return result


def target_differential(
    state: BISMARTState, direction: BISMARTDirection
) -> FloatArray:
    """Apply the differential ``D C_theta[direction]``.

    PSEUDOCODE:

    1. Differentiate each of the five factors in ``U0 A H B^T V0^T``.
    2. Hold the other four factors fixed in each term.
    3. Sum the five resulting matrices.
    """

    _validate_direction(state, direction)
    return (
        direction.U0 @ state.A @ state.H @ state.B.T @ state.V0.T
        + state.U0 @ direction.A @ state.H @ state.B.T @ state.V0.T
        + state.U0 @ state.A @ direction.H @ state.B.T @ state.V0.T
        + state.U0 @ state.A @ state.H @ direction.B.T @ state.V0.T
        + state.U0 @ state.A @ state.H @ state.B.T @ direction.V0.T
    )


def source_differential(
    state: BISMARTState, direction: BISMARTDirection
) -> FloatArray:
    """Apply the differential ``D C0_theta[direction]``."""

    _validate_direction(state, direction)
    source_core = block_diagonal(state.G_blocks)
    core_direction = block_diagonal(direction.G_blocks)
    return (
        direction.U0 @ source_core @ state.V0.T
        + state.U0 @ core_direction @ state.V0.T
        + state.U0 @ source_core @ direction.V0.T
    )


def joint_objective(
    state: BISMARTState,
    X: ArrayLike,
    Y: ArrayLike,
    observed_source: ArrayLike,
    omega: float,
) -> float:
    """Evaluate the fixed-support BI-SMART joint criterion.

    ``0.5/n * ||Y - X C(theta)||_F^2``
    ``+ 0.5*omega * ||observed_source - C0(theta)||_F^2``.
    """

    X_array, Y_array, source_array = _validate_problem_arrays(
        state, X, Y, observed_source, omega
    )
    target_residual = Y_array - X_array @ fitted_target(state)
    source_residual = source_array - fitted_source(state)
    return float(
        0.5 * np.sum(target_residual * target_residual) / X_array.shape[0]
        + 0.5 * omega * np.sum(source_residual * source_residual)
    )


# A concise alias is convenient inside optimisation code while the longer
# public name makes the joint nature of the criterion clear to API users.
objective = joint_objective


def euclidean_gradients(
    state: BISMARTState,
    X: ArrayLike,
    Y: ArrayLike,
    observed_source: ArrayLike,
    omega: float,
) -> BISMARTDirection:
    """Return the six explicit Euclidean gradient blocks.

    This is equation ``bismart-explicit-gradients`` in the appendix.  The
    returned ``A`` and ``B`` blocks are *Euclidean* gradients; call
    :func:`project_to_tangent` to zero inactive rows and project the active
    factors to their Stiefel tangent spaces.
    """

    X_array, Y_array, source_array = _validate_problem_arrays(
        state, X, Y, observed_source, omega
    )
    n_samples = X_array.shape[0]
    target_fit = fitted_target(state)
    source_fit = fitted_source(state)

    # R_tar is the gradient of the target loss with respect to C(theta).
    # Computing it from the residual avoids explicitly forming X.T @ X.
    R_target = X_array.T @ (X_array @ target_fit - Y_array) / n_samples
    # R_source is the gradient of the source fidelity term with respect to C0.
    R_source = omega * (source_fit - source_array)

    source_core = block_diagonal(state.G_blocks)
    grad_U0 = (
        R_target @ state.V0 @ state.B @ state.H.T @ state.A.T
        + R_source @ state.V0 @ source_core.T
    )
    grad_V0 = (
        R_target.T @ state.U0 @ state.A @ state.H @ state.B.T
        + R_source.T @ state.U0 @ source_core
    )
    grad_A = state.U0.T @ R_target @ state.V0 @ state.B @ state.H.T
    grad_B = state.V0.T @ R_target.T @ state.U0 @ state.A @ state.H
    grad_H = state.A.T @ state.U0.T @ R_target @ state.V0 @ state.B

    # The source core is constrained to be block diagonal, hence only the
    # diagonal block pieces of U0.T @ R_source @ V0 are valid gradients.
    full_grad_G = state.U0.T @ R_source @ state.V0
    grad_G_blocks = tuple(
        full_grad_G[coordinate_slice, coordinate_slice].copy()
        for coordinate_slice in _block_slices(state.block_sizes)
    )

    return BISMARTDirection(
        U0=grad_U0,
        V0=grad_V0,
        A=grad_A,
        B=grad_B,
        H=grad_H,
        G_blocks=grad_G_blocks,
    )


def stiefel_tangent_projection(Q: ArrayLike, Z: ArrayLike) -> FloatArray:
    """Project ``Z`` onto the tangent space of the Stiefel manifold at ``Q``.

    The formula is ``Z - Q sym(Q.T Z)``.  It assumes ``Q.T Q = I``; callers
    can use :func:`stiefel_error` when diagnosing an invalid state.
    """

    Q_array = _as_float_matrix(Q, name="Q")
    Z_array = _as_float_matrix(Z, name="Z")
    if Q_array.shape != Z_array.shape:
        raise ValueError("Q and Z must have identical shapes")
    qt_z = Q_array.T @ Z_array
    symmetric_part = 0.5 * (qt_z + qt_z.T)
    return Z_array - Q_array @ symmetric_part


def project_to_tangent(
    state: BISMARTState, euclidean: BISMARTDirection
) -> BISMARTDirection:
    """Project all constrained blocks onto the product tangent space.

    ``U0`` and ``V0`` use their full Stiefel tangent projections.  For ``A``
    and ``B`` only the explicitly recorded active rows are projected; inactive
    rows are set to zero.  ``H`` and every ``G_l`` are unconstrained linear
    blocks and therefore pass through unchanged.

    This is a product-manifold projection, not the quotient-horizontal
    projection.  Vertical rotation components are still present afterward.
    """

    # A Euclidean gradient generally has nonzero inactive A/B rows.  Those
    # rows are removed by this projection, so only shapes are checked here.
    _validate_direction(state, euclidean, require_fixed_support=False)
    projected_A = np.zeros_like(state.A)
    projected_A[state.active_u, :] = stiefel_tangent_projection(
        state.A[state.active_u, :], euclidean.A[state.active_u, :]
    )
    projected_B = np.zeros_like(state.B)
    projected_B[state.active_v, :] = stiefel_tangent_projection(
        state.B[state.active_v, :], euclidean.B[state.active_v, :]
    )
    return BISMARTDirection(
        U0=stiefel_tangent_projection(state.U0, euclidean.U0),
        V0=stiefel_tangent_projection(state.V0, euclidean.V0),
        A=projected_A,
        B=projected_B,
        H=euclidean.H.copy(),
        G_blocks=tuple(block.copy() for block in euclidean.G_blocks),
    )


# Alternate wording used in some mathematical codebases.
tangent_projection = project_to_tangent


def strict_polar_factor(
    matrix: ArrayLike,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    rank_tolerance: float | None = None,
) -> FloatArray:
    """Return the unique thin polar factor of a full-column-rank matrix.

    BI-SMART must reject a rank-deficient polar argument.  Completing a basis
    with coordinate vectors would break within-block rotation equivariance.

    Unless the legacy absolute ``rank_tolerance`` override is supplied, the
    threshold is computed *locally for this matrix* as
    ``max(atol + rtol*sigma_max, eps*dimension*sigma_max)``.  Consequently the
    four product-Stiefel retractions do not inherit a scale from one another,
    from a core matrix, or from the Gauss--Newton normal equations.
    """

    array = _as_float_matrix(matrix, name="matrix")
    rows, columns = array.shape
    if rows < columns:
        raise RefinementError("a Stiefel polar factor requires rows >= columns")
    left, singular_values, right_t = np.linalg.svd(array, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    absolute, relative = _solver_tolerances(atol, rtol)
    if rank_tolerance is None:
        tolerance = _rank_tolerance(
            largest,
            array.shape,
            atol=absolute,
            rtol=relative,
        )
    else:
        tolerance = float(rank_tolerance)
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise ValueError("rank_tolerance must be finite and nonnegative")
    if singular_values.size < columns or float(singular_values[-1]) <= tolerance:
        raise RefinementError("polar argument is rank deficient at the chosen tolerance")
    return left @ right_t


def polar_retraction(
    Q: ArrayLike,
    tangent: ArrayLike,
    *,
    step_size: float = 1.0,
    atol: float = 0.0,
    rtol: float = 0.0,
    rank_tolerance: float | None = None,
) -> FloatArray:
    """Apply ``Retr_Q(step*tangent) = polar(Q + step*tangent)``."""

    Q_array = _as_float_matrix(Q, name="Q")
    tangent_array = _as_float_matrix(tangent, name="tangent")
    if Q_array.shape != tangent_array.shape:
        raise ValueError("Q and tangent must have identical shapes")
    if not np.isfinite(step_size) or step_size < 0.0:
        raise ValueError("step_size must be finite and nonnegative")
    return strict_polar_factor(
        Q_array + step_size * tangent_array,
        atol=atol,
        rtol=rtol,
        rank_tolerance=rank_tolerance,
    )


def retract_state(
    state: BISMARTState,
    direction: BISMARTDirection,
    step_size: float,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    rank_tolerance: float | None = None,
) -> BISMARTState:
    """Apply the product polar/additive retraction from the appendix.

    Each Stiefel factor computes its own scale-aware polar rank threshold from
    ``atol`` and ``rtol``.  ``rank_tolerance`` remains as an explicit legacy
    absolute override for callers that require that historical behavior.
    """

    _validate_direction(state, direction)
    if not np.isfinite(step_size) or step_size < 0.0:
        raise ValueError("step_size must be finite and nonnegative")

    # Preserve the defining retraction identity exactly.  Sending an exact
    # zero direction through four independent SVD polar factors can perturb
    # already-orthonormal matrices by roundoff.  At an exact fit that turns a
    # zero trial objective into a tiny positive value and can falsely fail the
    # zero-direction Armijo equality.  This branch also avoids needless work.
    direction_arrays = (
        direction.U0,
        direction.V0,
        direction.A,
        direction.B,
        direction.H,
        *direction.G_blocks,
    )
    if step_size == 0.0 or all(
        not np.any(array) for array in direction_arrays
    ):
        return state

    # A and B live on Stiefel manifolds only within their selected rows.
    next_A = np.zeros_like(state.A)
    next_A[state.active_u, :] = polar_retraction(
        state.A[state.active_u, :],
        direction.A[state.active_u, :],
        step_size=step_size,
        atol=atol,
        rtol=rtol,
        rank_tolerance=rank_tolerance,
    )
    next_B = np.zeros_like(state.B)
    next_B[state.active_v, :] = polar_retraction(
        state.B[state.active_v, :],
        direction.B[state.active_v, :],
        step_size=step_size,
        atol=atol,
        rtol=rtol,
        rank_tolerance=rank_tolerance,
    )
    return BISMARTState(
        U0=polar_retraction(
            state.U0,
            direction.U0,
            step_size=step_size,
            atol=atol,
            rtol=rtol,
            rank_tolerance=rank_tolerance,
        ),
        V0=polar_retraction(
            state.V0,
            direction.V0,
            step_size=step_size,
            atol=atol,
            rtol=rtol,
            rank_tolerance=rank_tolerance,
        ),
        A=next_A,
        B=next_B,
        H=state.H + step_size * direction.H,
        G_blocks=tuple(
            core + step_size * core_direction
            for core, core_direction in zip(
                state.G_blocks, direction.G_blocks, strict=True
            )
        ),
        active_u=state.active_u,
        active_v=state.active_v,
    )


def product_inner_product(
    left: BISMARTDirection, right: BISMARTDirection
) -> float:
    """Frobenius product inner product across all six parameter blocks."""

    if len(left.G_blocks) != len(right.G_blocks):
        raise ValueError("directions have different numbers of G blocks")
    pairs = (
        (left.U0, right.U0),
        (left.V0, right.V0),
        (left.A, right.A),
        (left.B, right.B),
        (left.H, right.H),
    )
    total = sum(float(np.vdot(a, b).real) for a, b in pairs)
    total += sum(
        float(np.vdot(a, b).real)
        for a, b in zip(left.G_blocks, right.G_blocks, strict=True)
    )
    return total


def product_norm(direction: BISMARTDirection) -> float:
    """Product Frobenius norm of a state increment."""

    return float(np.sqrt(max(product_inner_product(direction, direction), 0.0)))


def gauss_newton_inner_product(
    state: BISMARTState,
    left: BISMARTDirection,
    right: BISMARTDirection,
    X: ArrayLike,
    omega: float,
) -> float:
    """Evaluate the joint Gauss--Newton metric on two tangent vectors."""

    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and strictly positive")
    X_array = _as_float_matrix(X, name="X")
    if X_array.shape[0] == 0:
        raise ValueError("X must contain at least one observation")
    if X_array.shape[1] != state.U0.shape[0]:
        raise ValueError("X has an incompatible number of predictor columns")
    left_target = X_array @ target_differential(state, left)
    right_target = X_array @ target_differential(state, right)
    left_source = source_differential(state, left)
    right_source = source_differential(state, right)
    return float(
        np.vdot(left_target, right_target).real / X_array.shape[0]
        + omega * np.vdot(left_source, right_source).real
    )


def stiefel_error(Q: ArrayLike) -> float:
    """Frobenius deviation of ``Q.T @ Q`` from identity."""

    array = _as_float_matrix(Q, name="Q")
    return float(np.linalg.norm(array.T @ array - np.eye(array.shape[1]), ord="fro"))


def pair_distance(
    state: BISMARTState, initial_state: BISMARTState, omega: float
) -> float:
    """Gauge-invariant fitted-pair distance from the initializer."""

    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and strictly positive")
    target_delta = fitted_target(state) - fitted_target(initial_state)
    source_delta = fitted_source(state) - fitted_source(initial_state)
    return float(
        np.sqrt(
            np.sum(target_delta * target_delta)
            + omega * np.sum(source_delta * source_delta)
        )
    )


def _smallest_singular_value(matrix: FloatArray) -> float:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return float(singular_values[-1]) if singular_values.size else 0.0


def source_block_separation(state: BISMARTState) -> float:
    """Minimum separation between spectra of ``G_l G_l.T``.

    The condition is vacuous for a one-block model, represented by infinity.
    """

    if len(state.G_blocks) == 1:
        return float("inf")
    spectra = tuple(
        np.linalg.eigvalsh(core @ core.T) for core in state.G_blocks
    )
    minimum = float("inf")
    for left_index, left_spectrum in enumerate(spectra[:-1]):
        for right_spectrum in spectra[left_index + 1 :]:
            distance = float(
                np.min(np.abs(left_spectrum[:, None] - right_spectrum[None, :]))
            )
            minimum = min(minimum, distance)
    return minimum


def _source_block_separation_records(
    state: BISMARTState,
) -> tuple[tuple[int, int, float, float], ...]:
    """Return pairwise squared-spectrum gaps and their own local scales."""

    spectra = tuple(
        np.linalg.eigvalsh(core @ core.T) for core in state.G_blocks
    )
    records: list[tuple[int, int, float, float]] = []
    for left, left_spectrum in enumerate(spectra[:-1]):
        for right in range(left + 1, len(spectra)):
            right_spectrum = spectra[right]
            gap = float(
                np.min(
                    np.abs(
                        left_spectrum[:, None] - right_spectrum[None, :]
                    )
                )
            )
            scale = max(
                float(np.max(np.abs(left_spectrum))),
                float(np.max(np.abs(right_spectrum))),
            )
            records.append((left, right, gap, scale))
    return tuple(records)


def default_core_thresholds(
    initial_state: BISMARTState,
    *,
    radius: float,
    zero_tolerance: float | None = None,
    separation_tolerance: float | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> TrustRegionThresholds:
    """Compute the appendix's data-dependent core/separation defaults.

    The defaults are half the corresponding initializer values.  The trust
    radius is not determined by this rule and remains an explicit caller
    input.  A zero required quantity makes the refinement branch unsuccessful,
    so this helper raises :class:`RefinementError` instead of manufacturing a
    threshold.  By default, ``H``, every ``G_l``, and every pair of squared
    source-core spectra use their own ``atol + rtol*local_scale`` comparison.
    The two legacy absolute-tolerance arguments remain available for callers
    that need to reproduce an older fixed threshold exactly.  An explicit
    ``zero_tolerance`` also controls separation when
    ``separation_tolerance`` is omitted, matching the original public API.
    """

    atol, rtol = _solver_tolerances(atol, rtol)
    if zero_tolerance is not None:
        zero_tolerance = float(zero_tolerance)
        if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
            raise ValueError("zero_tolerance must be finite and nonnegative")
    if separation_tolerance is not None:
        separation_tolerance = float(separation_tolerance)
        if not np.isfinite(separation_tolerance) or separation_tolerance < 0.0:
            raise ValueError("separation_tolerance must be finite and nonnegative")

    h_value = _smallest_singular_value(initial_state.H)
    h_tolerance = (
        zero_tolerance
        if zero_tolerance is not None
        else atol + rtol * float(np.linalg.norm(initial_state.H, ord=2))
    )
    g_values = tuple(
        _smallest_singular_value(core) for core in initial_state.G_blocks
    )
    g_tolerances = tuple(
        zero_tolerance
        if zero_tolerance is not None
        else atol + rtol * float(np.linalg.norm(core, ord=2))
        for core in initial_state.G_blocks
    )
    g_value = min(g_values)
    separation = source_block_separation(initial_state)
    if h_value <= h_tolerance:
        raise RefinementError("the initial target core is singular")
    for index, (value, tolerance) in enumerate(
        zip(g_values, g_tolerances, strict=True)
    ):
        if value <= tolerance:
            raise RefinementError(
                f"initial source block core {index} is singular"
            )
    for left, right, gap, scale in _source_block_separation_records(initial_state):
        if separation_tolerance is not None:
            tolerance = separation_tolerance
        elif zero_tolerance is not None:
            tolerance = zero_tolerance
        else:
            core_scale = float(np.sqrt(max(scale, 0.0)))
            core_tolerance = atol + rtol * core_scale
            tolerance = (
                2.0 * core_scale * core_tolerance + core_tolerance**2
            )
        if gap <= tolerance:
            raise RefinementError(
                f"initial source block-core spectra {left} and {right} "
                "are not separated"
            )
    return TrustRegionThresholds(
        radius=float(radius),
        h_min=0.5 * h_value,
        g_min=0.5 * g_value,
        separation_min=0.0 if len(initial_state.G_blocks) == 1 else 0.5 * separation,
    )


def check_path_certificate(
    state: BISMARTState,
    direction: BISMARTDirection,
    initial_state: BISMARTState,
    *,
    omega: float,
    step_size: float,
    initial_step_size: float,
    thresholds: TrustRegionThresholds,
) -> PathCertificate:
    """Evaluate the appendix's finite path-safety certificate.

    PSEUDOCODE corresponding to ``bismart-finite-path-certificate``:

    1. Bound the product retraction path by requiring ``eta*||xi|| <= 1/2``.
    2. Bound movement of the fitted target/source pair inside the trust radius.
    3. Keep the target core's smallest singular value above ``h_min``.
    4. Keep every source block core above ``g_min``.
    5. For multiple blocks, keep their squared-singular-value spectra separated.

    Passing this finite certificate avoids the impossible task of numerically
    testing an entire continuous retraction path.
    """

    _validate_compatible_states(state, initial_state)
    _validate_direction(state, direction)
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    if not np.isfinite(initial_step_size) or initial_step_size <= 0.0:
        raise ValueError("initial_step_size must be finite and positive")
    if step_size > initial_step_size:
        raise ValueError(
            "step_size cannot exceed initial_step_size: the finite-path "
            "certificate assumes eta <= bar_eta."
        )

    q_direction = product_norm(direction)
    current_pair_distance = pair_distance(state, initial_state, omega)
    h_lower = _smallest_singular_value(state.H) - step_size * np.linalg.norm(
        direction.H, ord=2
    )
    g_lowers = tuple(
        _smallest_singular_value(core)
        - step_size * np.linalg.norm(core_direction, ord=2)
        for core, core_direction in zip(
            state.G_blocks, direction.G_blocks, strict=True
        )
    )
    g_lower = min(g_lowers)

    # Operator norm of a block diagonal matrix is the maximum block norm.
    g_operator = max(np.linalg.norm(core, ord=2) for core in state.G_blocks)
    target_path_lipschitz = 8.0 * (
        np.linalg.norm(state.H, ord=2) + initial_step_size * q_direction
    ) + 1.0
    source_path_lipschitz = 4.0 * (
        g_operator + initial_step_size * q_direction
    ) + 1.0
    pair_path_lipschitz = float(
        np.sqrt(target_path_lipschitz**2 + omega * source_path_lipschitz**2)
    )
    pair_upper = (
        current_pair_distance
        + step_size * q_direction * pair_path_lipschitz
    )

    separation = source_block_separation(state)
    if len(state.G_blocks) == 1:
        separation_lower = float("inf")
    else:
        block_movements = tuple(
            2.0
            * step_size
            * np.linalg.norm(core, ord=2)
            * np.linalg.norm(core_direction, ord=2)
            + step_size**2 * np.linalg.norm(core_direction, ord=2) ** 2
            for core, core_direction in zip(
                state.G_blocks, direction.G_blocks, strict=True
            )
        )
        largest_pair_movement = max(
            block_movements[left] + block_movements[right]
            for left in range(len(block_movements))
            for right in range(left + 1, len(block_movements))
        )
        separation_lower = separation - largest_pair_movement

    failed: list[str] = []
    if step_size * q_direction > 0.5:
        failed.append("retraction_path_bound")
    if pair_upper > thresholds.radius:
        failed.append("pair_trust_radius")
    if h_lower < thresholds.h_min:
        failed.append("target_core_singularity")
    if g_lower < thresholds.g_min:
        failed.append("source_core_singularity")
    if (
        len(state.G_blocks) > 1
        and separation_lower < thresholds.separation_min
    ):
        failed.append("cross_block_separation")

    return PathCertificate(
        safe=not failed,
        failed_checks=tuple(failed),
        step_size=float(step_size),
        direction_norm=q_direction,
        pair_distance=pair_upper,
        target_core_lower_bound=float(h_lower),
        source_core_lower_bound=float(g_lower),
        separation_lower_bound=float(separation_lower),
    )


def check_armijo(
    state: BISMARTState,
    trial_state: BISMARTState,
    direction: BISMARTDirection,
    X: ArrayLike,
    Y: ArrayLike,
    observed_source: ArrayLike,
    *,
    omega: float,
    step_size: float,
    armijo_constant: float,
) -> ArmijoCheck:
    """Check the BI-SMART Armijo decrease inequality for a supplied trial."""

    if not (0.0 < armijo_constant < 1.0):
        raise ValueError("armijo_constant must lie strictly between zero and one")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    current_value = joint_objective(state, X, Y, observed_source, omega)
    trial_value = joint_objective(trial_state, X, Y, observed_source, omega)
    gn_norm_squared = gauss_newton_inner_product(
        state, direction, direction, X, omega
    )
    required_upper = current_value - (
        armijo_constant * step_size * gn_norm_squared
    )
    return ArmijoCheck(
        accepted=bool(trial_value <= required_upper),
        current_objective=current_value,
        trial_objective=trial_value,
        required_upper_bound=float(required_upper),
        gauss_newton_norm_squared=float(gn_norm_squared),
    )


def safeguarded_backtracking(
    state: BISMARTState,
    direction: BISMARTDirection,
    initial_state: BISMARTState,
    X: ArrayLike,
    Y: ArrayLike,
    observed_source: ArrayLike,
    *,
    omega: float,
    thresholds: TrustRegionThresholds,
    initial_step_size: float,
    contraction: float,
    armijo_constant: float,
    max_trials: int,
    polar_atol: float = 0.0,
    polar_rtol: float = 0.0,
    rank_tolerance: float | None = None,
) -> BacktrackingResult:
    """Try the finite safeguarded step sequence from the appendix.

    The caller supplies a valid horizontal direction, normally from
    :func:`quotient_gauss_newton_direction`.  Each safe trial uses local polar
    thresholds derived from ``polar_atol`` and ``polar_rtol``; this routine
    deliberately does not recompute the direction during a line search.
    """

    if not np.isfinite(initial_step_size) or initial_step_size <= 0.0:
        raise ValueError("initial_step_size must be finite and positive")
    if not (0.0 < contraction < 1.0):
        raise ValueError("contraction must lie strictly between zero and one")
    if (
        isinstance(max_trials, (bool, np.bool_))
        or not isinstance(max_trials, (int, np.integer))
        or max_trials <= 0
    ):
        raise ValueError("max_trials must be a positive integer")
    polar_atol, polar_rtol = _solver_tolerances(polar_atol, polar_rtol)

    last_certificate: PathCertificate | None = None
    last_armijo: ArmijoCheck | None = None
    for trial_index in range(int(max_trials)):
        step_size = initial_step_size * contraction**trial_index
        certificate = check_path_certificate(
            state,
            direction,
            initial_state,
            omega=omega,
            step_size=step_size,
            initial_step_size=initial_step_size,
            thresholds=thresholds,
        )
        last_certificate = certificate
        if not certificate.safe:
            continue
        try:
            trial_state = retract_state(
                state,
                direction,
                step_size,
                atol=polar_atol,
                rtol=polar_rtol,
                rank_tolerance=rank_tolerance,
            )
        except (np.linalg.LinAlgError, RefinementError):
            # The appendix treats a deficient polar trial as unsuccessful.
            continue
        armijo = check_armijo(
            state,
            trial_state,
            direction,
            X,
            Y,
            observed_source,
            omega=omega,
            step_size=step_size,
            armijo_constant=armijo_constant,
        )
        last_armijo = armijo
        if armijo.accepted:
            return BacktrackingResult(
                accepted=True,
                state=trial_state,
                step_size=float(step_size),
                trials=trial_index + 1,
                path_certificate=certificate,
                armijo_check=armijo,
            )

    return BacktrackingResult(
        accepted=False,
        state=None,
        step_size=None,
        trials=int(max_trials),
        path_certificate=last_certificate,
        armijo_check=last_armijo,
        failure_reason="no trial satisfied both path safety and Armijo decrease",
    )


def _as_bismart_direction(tangent: ProductTangent) -> BISMARTDirection:
    """Convert the geometry module's dependency-free tangent representation."""

    return BISMARTDirection(
        U0=tangent.U0,
        V0=tangent.V0,
        A=tangent.A,
        B=tangent.B,
        H=tangent.H,
        G_blocks=tangent.G_blocks,
    )


def _solver_tolerances(atol: float, rtol: float) -> tuple[float, float]:
    """Validate the two public floating-point interpretation constants."""

    raw_absolute = np.asarray(atol)
    raw_relative = np.asarray(rtol)
    if np.iscomplexobj(raw_absolute) or np.iscomplexobj(raw_relative):
        raise ValueError("atol and rtol must be real-valued; complex input is unsupported")
    if raw_absolute.ndim != 0 or raw_relative.ndim != 0:
        raise ValueError("atol and rtol must be scalars")
    absolute = float(raw_absolute)
    relative = float(raw_relative)
    if not np.isfinite(absolute) or absolute < 0.0:
        raise ValueError("atol must be finite and nonnegative")
    if not np.isfinite(relative) or relative < 0.0:
        raise ValueError("rtol must be finite and nonnegative")
    return absolute, relative


def _rank_tolerance(
    largest_singular_value: float,
    shape: tuple[int, int],
    *,
    atol: float,
    rtol: float,
) -> float:
    """Return a local absolute/relative/machine threshold for one SVD.

    The machine term prevents numerical gauge zeros from being counted as
    information when a caller deliberately sets both configured tolerances to
    zero.  Every caller supplies the largest singular value and shape of its
    *own* matrix, keeping polar arguments, gauge generators, and the residual
    Jacobian numerically independent even though they share this formula.
    """

    machine = (
        np.finfo(float).eps
        * max(shape)
        * max(float(largest_singular_value), 0.0)
    )
    return float(max(atol + rtol * largest_singular_value, machine))


def _dense_work_limit(value: int | None) -> int | None:
    """Validate an optional cap for the dense Jacobian/SVD working set."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("max_dense_work_bytes must be an integer or None")
    result = int(value)
    if result < 1:
        raise ValueError("max_dense_work_bytes must be positive")
    return result


def _estimated_dense_svd_bytes(rows: int, columns: int) -> int:
    """Conservatively estimate resident arrays for ``svd(J, full=False)``.

    NumPy/LAPACK implementations may use additional platform-dependent
    workspace.  Counting the Jacobian, one internal copy, thin U/V factors,
    and several singular-length work vectors is nevertheless enough to reject
    obviously unsafe multi-gigabyte dense calls before allocating ``J``.
    """

    thin = min(int(rows), int(columns))
    float_entries = (
        2 * int(rows) * int(columns)
        + int(rows) * thin
        + thin * int(columns)
        + 4 * thin
    )
    return int(np.dtype(float).itemsize * float_entries)


def _postcheck_tolerance(
    scale: float,
    dimension: int,
    *,
    atol: float,
    rtol: float,
) -> float:
    """Scale a verification tolerance without imposing an artificial unit floor."""

    machine_relative = 64.0 * np.finfo(float).eps * max(int(dimension), 1)
    return float(16.0 * atol + max(16.0 * rtol, machine_relative) * scale)


def _tangency_error(
    state: BISMARTState, direction: BISMARTDirection
) -> float:
    """Maximum fixed-support/Stiefel residual of a proposed direction."""

    errors = [
        np.linalg.norm(
            state.U0.T @ direction.U0 + direction.U0.T @ state.U0,
            ord="fro",
        ),
        np.linalg.norm(
            state.V0.T @ direction.V0 + direction.V0.T @ state.V0,
            ord="fro",
        ),
        np.linalg.norm(
            state.A[state.active_u, :].T @ direction.A[state.active_u, :]
            + direction.A[state.active_u, :].T @ state.A[state.active_u, :],
            ord="fro",
        ),
        np.linalg.norm(
            state.B[state.active_v, :].T @ direction.B[state.active_v, :]
            + direction.B[state.active_v, :].T @ state.B[state.active_v, :],
            ord="fro",
        ),
        np.linalg.norm(direction.A[~state.active_u, :], ord="fro"),
        np.linalg.norm(direction.B[~state.active_v, :], ord="fro"),
    ]
    return float(max(errors, default=0.0))


def _solver_backend(value: str) -> str:
    """Normalize the explicit dense/matrix-free backend policy."""

    if not isinstance(value, str):
        raise TypeError("solver_backend must be 'dense', 'matrix_free', or 'auto'")
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in {"dense", "matrix_free", "auto"}:
        raise ValueError(
            "solver_backend must be 'dense', 'matrix_free', or 'auto'"
        )
    return normalized


class _JointJacobianOperator:
    """Matrix-free action of ``J S`` and its exact Frobenius adjoint.

    ``S`` is the product-tangent Parseval synthesis map.  A forward action
    lifts one coefficient vector and applies the target/source differentials.
    The reverse action first applies their algebraic adjoints and then the
    Parseval analysis map ``S.T``.  At no point is an
    ``residual_dimension x frame_dimension`` array allocated.
    """

    __slots__ = (
        "state",
        "X",
        "omega",
        "frame",
        "sample_scale",
        "source_scale",
        "target_shape",
        "source_shape",
        "rows",
        "columns",
    )

    def __init__(
        self,
        state: BISMARTState,
        X: FloatArray,
        omega: float,
        frame: TangentParsevalFrame,
    ) -> None:
        self.state = state
        self.X = X
        self.omega = float(omega)
        self.frame = frame
        self.sample_scale = float(np.sqrt(X.shape[0]))
        self.source_scale = float(np.sqrt(omega))
        self.target_shape = (int(X.shape[0]), int(state.V0.shape[0]))
        self.source_shape = (int(state.U0.shape[0]), int(state.V0.shape[0]))
        self.rows = int(np.prod(self.target_shape) + np.prod(self.source_shape))
        self.columns = frame.coefficient_dimension

    def matvec(self, coefficients: FloatArray) -> FloatArray:
        """Return ``J S coefficients`` using the two analytic differentials."""

        tangent = self.frame.lift(coefficients)
        direction = _as_bismart_direction(tangent)
        target = (
            self.X @ target_differential(self.state, direction)
        ) / self.sample_scale
        source = self.source_scale * source_differential(self.state, direction)
        result = np.concatenate(
            (target.ravel(order="C"), source.ravel(order="C"))
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("matrix-free Jacobian action is not finite")
        return result

    def rmatvec(self, residual_coordinates: FloatArray) -> FloatArray:
        """Return ``S.T J.T residual_coordinates`` analytically.

        The formulas are the same factorwise adjoints used by
        :func:`euclidean_gradients`, but here the incoming target/source
        matrices are arbitrary dual vectors rather than statistical
        residuals.  This method is therefore a true adjoint for every vector,
        which focused tests verify independently.
        """

        raw = np.asarray(residual_coordinates)
        if np.iscomplexobj(raw):
            raise ValueError("residual_coordinates must be real-valued")
        vector = np.asarray(raw, dtype=float)
        if vector.shape != (self.rows,):
            raise ValueError(
                f"residual_coordinates must have shape ({self.rows},)"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("residual_coordinates must contain only finite values")

        target_size = int(np.prod(self.target_shape))
        target_dual = vector[:target_size].reshape(self.target_shape, order="C")
        source_dual = vector[target_size:].reshape(self.source_shape, order="C")
        R_target = self.X.T @ target_dual / self.sample_scale
        R_source = self.source_scale * source_dual

        state = self.state
        source_core = block_diagonal(state.G_blocks)
        grad_U0 = (
            R_target @ state.V0 @ state.B @ state.H.T @ state.A.T
            + R_source @ state.V0 @ source_core.T
        )
        grad_V0 = (
            R_target.T @ state.U0 @ state.A @ state.H @ state.B.T
            + R_source.T @ state.U0 @ source_core
        )
        grad_A = state.U0.T @ R_target @ state.V0 @ state.B @ state.H.T
        grad_B = state.V0.T @ R_target.T @ state.U0 @ state.A @ state.H
        grad_H = state.A.T @ state.U0.T @ R_target @ state.V0 @ state.B
        full_grad_G = state.U0.T @ R_source @ state.V0
        grad_G_blocks = tuple(
            full_grad_G[coordinate_slice, coordinate_slice].copy()
            for coordinate_slice in _block_slices(state.block_sizes)
        )
        ambient_adjoint = BISMARTDirection(
            U0=grad_U0,
            V0=grad_V0,
            A=grad_A,
            B=grad_B,
            H=grad_H,
            G_blocks=grad_G_blocks,
        )
        result = self.frame.analyze(ambient_adjoint)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("matrix-free Jacobian adjoint is not finite")
        return result


def _matrix_free_information_certificate(
    state: BISMARTState,
    X: FloatArray,
    *,
    quotient_dimension: int,
    atol: float,
    rtol: float,
) -> tuple[int, float, float, float]:
    """Certify quotient injectivity without probing rank from the LSQR RHS.

    The stacked derivative has a useful structure.  On a regular state, the
    fully observed source Frobenius term is injective on the rank-``r0``
    source-matrix quotient.  Once its differential is zero, only gauge
    directions and the target rank-``r`` matrix inside the selected source
    subspaces remain.  Because responses are fully observed and ``H`` is
    nonsingular, that latter derivative is injective exactly when

    ``X @ U0[:, active_u]``

    has full column rank.  A small SVD of this ``n x k_u`` reduced design is
    consequently a deterministic rank certificate for the complete quotient
    operator.  In particular it detects a singular design even when the
    current statistical residual is exactly zero--a case LSQR from zero could
    never diagnose on its own.
    """

    reduced_design = X @ state.U0[:, state.active_u]
    try:
        singular_values = np.linalg.svd(reduced_design, compute_uv=False)
    except np.linalg.LinAlgError as failure:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            f"the reduced fitting-design SVD did not converge: {failure}",
        ) from failure
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = _rank_tolerance(
        largest, reduced_design.shape, atol=atol, rtol=rtol
    )
    reduced_rank = int(np.count_nonzero(singular_values > tolerance))
    expected_reduced_rank = int(np.count_nonzero(state.active_u))
    if reduced_rank != expected_reduced_rank:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the reduced fitting design X @ U0[:, active_u] has numerical "
            f"rank {reduced_rank}, expected {expected_reduced_rank} at "
            f"tolerance {tolerance:.3e}; the quotient Gauss--Newton "
            "information is singular",
        )
    smallest = float(singular_values[-1])
    return quotient_dimension, largest, smallest, tolerance


def _matrix_free_compact_rank_screen(
    state: BISMARTState,
    X: FloatArray,
    omega: float,
    *,
    full_residual_dimension: int,
    full_frame_dimension: int,
    atol: float,
    rtol: float,
) -> tuple[int, int, float, float, float, int, float | None, float, float]:
    """SVD a rank-only restriction of the Jacobian independent of ``p,q,n``.

    Algebraic injectivity is insufficient for floating-point equivalence with
    the dense SVD.  For example, at an exact fit LSQR sees a zero right-hand
    side and cannot notice a weak source weight, core, block-separation, or
    target-core direction.

    Pseudocode
    ----------
    1. Embed the source span and, when available, one deterministic orthogonal
       predictor direction; compress its fitted design by reduced QR.  The
       ``R`` factor satisfies
       ``||Z M||_F = ||R M||_F`` for every compatible ``M``.
    2. Replace the large source bases by identity embeddings with at most one
       extra normal row on each side, while retaining ``A, B, H, G``,
       supports, and ``omega``.  This represents a true restriction of the
       full derivative and captures span/normal target cancellation.
    3. Materialize and SVD only this compact Jacobian.  Its size depends on
       ``r0``, selected dimensions, and ``min(n,r0+1)``, never on ambient
       ``p`` or ``q``.
    4. Require its numerical rank to equal
       ``r0(p_c + q_c - r0) + r(k_u + k_v - r)``, where ``p_c`` and ``q_c``
       are the compact embedding dimensions (each ``r0`` or ``r0+1``).

    This is a rigorous *necessary* screen for the full dense rank policy.  The
    compact operator is a restriction, so its smallest quotient singular
    value is an upper bound on the full operator's smallest one, while its
    largest singular value (and matrix dimensions) give no larger a cutoff.
    Consequently a compact rank failure implies a full dense rank failure.
    Passing does not claim the converse; full ``jacobian_rank`` remains unset
    in matrix-free diagnostics.
    """

    source_rank = state.r0
    left_normal_vector: FloatArray | None = None
    left_embedding = state.U0
    if state.U0.shape[0] > source_rank:
        row_residual_norms_squared = 1.0 - np.sum(state.U0 * state.U0, axis=1)
        coordinate_index = int(np.argmax(row_residual_norms_squared))
        left_normal_vector = -state.U0 @ state.U0[coordinate_index, :]
        left_normal_vector[coordinate_index] += 1.0
        left_normal_vector /= np.linalg.norm(left_normal_vector)
        left_embedding = np.column_stack((state.U0, left_normal_vector))

    reduced_source_design = X @ left_embedding
    try:
        _, compressed_design = np.linalg.qr(
            reduced_source_design, mode="reduced"
        )
    except np.linalg.LinAlgError as failure:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            f"the compact fitting-design QR did not converge: {failure}",
        ) from failure

    # The target differential in the original operator is divided by
    # sqrt(n_original), whereas _JointJacobianOperator will divide the compact
    # design by sqrt(n_compact).  Rescale R so these two maps are isometric:
    # (sqrt(k/n) R) / sqrt(k) = R / sqrt(n).
    compact_sample_size = int(compressed_design.shape[0])
    compressed_design = compressed_design * np.sqrt(
        compact_sample_size / X.shape[0]
    )

    compact_predictor_dimension = int(left_embedding.shape[1])
    compact_response_dimension = source_rank + int(
        state.V0.shape[0] > source_rank
    )
    compact_U0 = np.zeros(
        (compact_predictor_dimension, source_rank), dtype=float
    )
    compact_U0[:source_rank, :] = np.eye(source_rank, dtype=float)
    compact_V0 = np.zeros(
        (compact_response_dimension, source_rank), dtype=float
    )
    compact_V0[:source_rank, :] = np.eye(source_rank, dtype=float)
    compact_state = BISMARTState(
        U0=compact_U0,
        V0=compact_V0,
        A=state.A,
        B=state.B,
        H=state.H,
        G_blocks=state.G_blocks,
        active_u=state.active_u,
        active_v=state.active_v,
    )
    compact_frame = tangent_parseval_frame(
        compact_state, atol=atol, rtol=rtol, validate=False
    )
    compact_operator = _JointJacobianOperator(
        compact_state, compressed_design, omega, compact_frame
    )
    compact_work_bytes = _estimated_dense_svd_bytes(
        compact_operator.rows, compact_operator.columns
    )
    if compact_work_bytes > DEFAULT_MAX_COMPACT_WORK_BYTES:
        raise NumericalFailure(
            FailureReason.DENSE_SOLVER_LIMIT,
            "the compact quotient rank certificate would require "
            f"approximately {compact_work_bytes} bytes, exceeding its "
            f"internal safety cap {DEFAULT_MAX_COMPACT_WORK_BYTES}; this "
            "workspace is independent of p, q, and n but polynomial in r0",
        )
    compact_jacobian = np.empty(
        (compact_operator.rows, compact_operator.columns), dtype=float
    )
    basis_coefficients = np.zeros(compact_operator.columns, dtype=float)
    for column in range(compact_operator.columns):
        basis_coefficients[column] = 1.0
        compact_jacobian[:, column] = compact_operator.matvec(
            basis_coefficients
        )
        basis_coefficients[column] = 0.0
    try:
        compact_singular_values = np.linalg.svd(
            compact_jacobian, compute_uv=False
        )
    except np.linalg.LinAlgError as failure:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            f"the compact quotient rank SVD did not converge: {failure}",
        ) from failure
    compact_largest = (
        float(compact_singular_values[0])
        if compact_singular_values.size
        else 0.0
    )
    compact_tolerance = _rank_tolerance(
        compact_largest,
        compact_jacobian.shape,
        atol=atol,
        rtol=rtol,
    )
    compact_rank = int(
        np.count_nonzero(compact_singular_values > compact_tolerance)
    )
    compact_quotient_dimension = geometry_dimensions(
        compact_state
    ).quotient_dimension
    if compact_rank != compact_quotient_dimension:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the compact matrix-free Jacobian restriction has numerical rank "
            f"{compact_rank}, expected {compact_quotient_dimension} at "
            f"tolerance {compact_tolerance:.3e}; therefore the full dense "
            "Jacobian cannot satisfy its numerical-rank policy",
        )
    compact_smallest = float(
        compact_singular_values[compact_quotient_dimension - 1]
    )
    operator_norm_lower_bound = compact_largest
    full_source_core = block_diagonal(state.G_blocks)
    if state.U0.shape[0] > source_rank or state.V0.shape[0] > source_rank:
        operator_norm_lower_bound = max(
            operator_norm_lower_bound,
            float(
                np.sqrt(omega)
                * np.linalg.svd(full_source_core, compute_uv=False)[0]
            ),
        )

    # Within-span restriction does not contain U0/V0 normal directions.  Each
    # normal direction is automatically horizontal, so its smallest Rayleigh
    # ratio gives another rigorous upper bound on the full smallest quotient
    # singular value.  The V0-normal family has the exact r0-by-r0 Gram matrix
    # below.  For U0, the target scale depends on Xw; one deterministic unit
    # w perpendicular to U0 still supplies a valid (possibly non-sharp) bound
    # without constructing a p-by-p orthogonal complement.
    ambient_normal_upper_bounds: list[float] = []
    if state.U0.shape[0] > source_rank:
        assert left_normal_vector is not None
        target_normal_scale = float(
            np.vdot(
                X @ left_normal_vector,
                X @ left_normal_vector,
            ).real
            / X.shape[0]
        )
        left_normal_gram = (
            omega * full_source_core @ full_source_core.T
            + target_normal_scale
            * state.A
            @ state.H
            @ state.H.T
            @ state.A.T
        )
        left_normal_eigenvalues = np.linalg.eigvalsh(left_normal_gram)
        ambient_normal_upper_bounds.append(
            float(np.sqrt(max(float(left_normal_eigenvalues[0]), 0.0)))
        )
        operator_norm_lower_bound = max(
            operator_norm_lower_bound,
            float(np.sqrt(max(float(left_normal_eigenvalues[-1]), 0.0))),
        )
    if state.V0.shape[0] > source_rank:
        target_design = X @ state.U0 @ state.A @ state.H @ state.B.T
        right_normal_gram = (
            omega * full_source_core.T @ full_source_core
            + target_design.T @ target_design / X.shape[0]
        )
        right_normal_eigenvalues = np.linalg.eigvalsh(right_normal_gram)
        ambient_normal_upper_bounds.append(
            float(np.sqrt(max(float(right_normal_eigenvalues[0]), 0.0)))
        )
        operator_norm_lower_bound = max(
            operator_norm_lower_bound,
            float(np.sqrt(max(float(right_normal_eigenvalues[-1]), 0.0))),
        )
    ambient_normal_upper_bound = (
        min(ambient_normal_upper_bounds) if ambient_normal_upper_bounds else None
    )
    full_rank_tolerance_lower_bound = _rank_tolerance(
        operator_norm_lower_bound,
        (full_residual_dimension, full_frame_dimension),
        atol=atol,
        rtol=rtol,
    )
    if compact_smallest <= full_rank_tolerance_lower_bound:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the compact quotient singular value "
            f"{compact_smallest:.3e} is not above the full-rank threshold "
            f"lower bound {full_rank_tolerance_lower_bound:.3e}; the full "
            "dense Jacobian cannot satisfy its numerical-rank policy",
        )
    if (
        ambient_normal_upper_bound is not None
        and ambient_normal_upper_bound <= full_rank_tolerance_lower_bound
    ):
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "an ambient-normal horizontal direction has Rayleigh upper bound "
            f"{ambient_normal_upper_bound:.3e}, not above the full-rank "
            "threshold lower bound "
            f"{full_rank_tolerance_lower_bound:.3e}; the full dense Jacobian "
            "cannot satisfy its numerical-rank policy",
        )
    return (
        compact_rank,
        compact_quotient_dimension,
        compact_tolerance,
        compact_largest,
        compact_smallest,
        compact_work_bytes,
        ambient_normal_upper_bound,
        full_rank_tolerance_lower_bound,
        operator_norm_lower_bound,
    )


def _solve_dense_quotient_gauss_newton(
    state: BISMARTState,
    X: ArrayLike,
    Y: ArrayLike,
    observed_source: ArrayLike,
    omega: float,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    max_dense_work_bytes: int | None = DEFAULT_MAX_DENSE_WORK_BYTES,
) -> GaussNewtonResult:
    """Solve and certify the fixed-support quotient Gauss--Newton system.

    PSEUDOCODE implementing Appendix equations ``bismart-horizontal-space``,
    ``bismart-gn-metric``, and ``bismart-gn-direction``:

    1. Validate that ``state`` lies in the fixed-support quotient regular set.
    2. Project canonical ambient matrix coordinates onto the four Stiefel
       tangents and append canonical coordinates for the unrestricted cores.
       The result is a redundant Parseval frame ``S`` for the product tangent.
    3. Stack the target and source residuals, and apply the two existing
       differential maps to every frame element to form ``J S``.
    4. Compute ``z = -(J S)^dagger residual`` by a rank-revealing SVD.  The
       Parseval identity makes ``S z`` the product-Frobenius minimum-norm full
       tangent solution, which is horizontal when the only derivative kernel
       is the explicitly checked gauge space.
    5. Reject excess nullity instead of adding damping, lift ``z``, and verify
       tangency, horizontality, the normal residual, and the descent identity.

    This dense implementation is intentionally the correctness reference for
    the matrix-free backend and its focused equivalence tests.
    """

    atol, rtol = _solver_tolerances(atol, rtol)
    dense_work_limit = _dense_work_limit(max_dense_work_bytes)
    X_array, Y_array, source_array = _validate_problem_arrays(
        state, X, Y, observed_source, omega
    )

    # Algebraically irregular iterates are local branch failures in Algorithm
    # 2.  They are distinct from a regular state whose fitting design supplies
    # too little information for the quotient normal equations.
    try:
        validate_regular_state(state, atol=atol, rtol=rtol)
    except (GeometryValidationError, np.linalg.LinAlgError) as failure:
        raise NumericalFailure(
            FailureReason.IRREGULAR_ITERATE,
            f"quotient Gauss--Newton state is irregular: {failure}",
        ) from failure

    dimensions = geometry_dimensions(state)
    try:
        frame = tangent_parseval_frame(
            state, atol=atol, rtol=rtol, validate=False
        )
    except (GeometryValidationError, ValueError) as failure:
        raise NumericalFailure(
            FailureReason.IRREGULAR_ITERATE,
            f"could not construct the product tangent frame: {failure}",
        ) from failure

    # Certify the dimension removed by the four orthogonal gauge actions.  The
    # frame analysis is an isometry on tangent vectors, so the singular values
    # of these columns are their product-Frobenius singular values.
    try:
        vertical = tuple(vertical_gauge_generators(state))
    except (FloatingPointError, ValueError) as failure:
        raise NumericalFailure(
            FailureReason.IRREGULAR_ITERATE,
            f"could not construct finite gauge generators: {failure}",
        ) from failure
    if vertical:
        raw_vertical_coordinates = np.column_stack(
            [frame.analyze(generator) for generator in vertical]
        )
        # Gauge generators belonging to different source blocks can inherit
        # radically different core scales.  Rank-testing those raw columns
        # with one relative cutoff lets a large G block erase an independent
        # small-block rotation.  Parseval coordinates preserve the product
        # norm, so normalizing each generator tests independence of gauge
        # *directions* without allowing one block's units to set another's
        # zero threshold.
        vertical_column_norms = np.linalg.norm(
            raw_vertical_coordinates,
            axis=0,
        )
        if np.any(~np.isfinite(vertical_column_norms)) or np.any(
            vertical_column_norms == 0.0
        ):
            raise NumericalFailure(
                FailureReason.IRREGULAR_ITERATE,
                "the infinitesimal gauge action contains a zero or nonfinite "
                "generator",
            )
        vertical_coordinates = (
            raw_vertical_coordinates / vertical_column_norms[np.newaxis, :]
        )
        try:
            vertical_singular_values = np.linalg.svd(
                vertical_coordinates, compute_uv=False
            )
        except np.linalg.LinAlgError as failure:
            raise NumericalFailure(
                FailureReason.IRREGULAR_ITERATE,
                f"the vertical-space rank SVD did not converge: {failure}",
            ) from failure
        vertical_largest = float(vertical_singular_values[0])
        vertical_tolerance = _rank_tolerance(
            vertical_largest,
            vertical_coordinates.shape,
            atol=atol,
            rtol=rtol,
        )
        vertical_rank = int(
            np.count_nonzero(vertical_singular_values > vertical_tolerance)
        )
    else:
        vertical_rank = 0
    if vertical_rank != dimensions.vertical_dimension:
        raise NumericalFailure(
            FailureReason.IRREGULAR_ITERATE,
            "the infinitesimal gauge action has numerical rank "
            f"{vertical_rank}, expected {dimensions.vertical_dimension}",
        )

    residual_dimension = int(
        X_array.shape[0] * Y_array.shape[1] + source_array.size
    )
    estimated_dense_work_bytes = _estimated_dense_svd_bytes(
        residual_dimension, frame.coefficient_dimension
    )
    if (
        dense_work_limit is not None
        and estimated_dense_work_bytes > dense_work_limit
    ):
        raise NumericalFailure(
            FailureReason.DENSE_SOLVER_LIMIT,
            "the dense quotient reference solve would require approximately "
            f"{estimated_dense_work_bytes} bytes, exceeding the configured "
            f"max_dense_work_bytes={dense_work_limit}; use a smaller problem "
            "or deliberately raise the resource cap",
        )

    # The signs below use fitted minus observed residuals.  With this choice,
    # the first-order optimality equation is J.T @ (J z + residual) = 0 and the
    # required direction is the negative pseudoinverse action.
    sample_scale = float(np.sqrt(X_array.shape[0]))
    target_residual = (
        X_array @ fitted_target(state) - Y_array
    ) / sample_scale
    source_residual = np.sqrt(omega) * (
        fitted_source(state) - source_array
    )
    residual = np.concatenate(
        (target_residual.ravel(order="C"), source_residual.ravel(order="C"))
    )

    jacobian = np.empty(
        (residual.size, frame.coefficient_dimension), dtype=float
    )
    for column, tangent in enumerate(frame.iter_elements()):
        basis_direction = _as_bismart_direction(tangent)
        target_column = (
            X_array @ target_differential(state, basis_direction)
        ) / sample_scale
        source_column = np.sqrt(omega) * source_differential(
            state, basis_direction
        )
        jacobian[:, column] = np.concatenate(
            (target_column.ravel(order="C"), source_column.ravel(order="C"))
        )
    if not np.all(np.isfinite(jacobian)):
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the quotient Gauss--Newton residual Jacobian is not finite",
        )

    try:
        left, singular_values, right_t = np.linalg.svd(
            jacobian, full_matrices=False
        )
    except np.linalg.LinAlgError as failure:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            f"the residual-Jacobian SVD did not converge: {failure}",
        ) from failure

    largest = float(singular_values[0]) if singular_values.size else 0.0
    rank_tolerance = _rank_tolerance(
        largest, jacobian.shape, atol=atol, rtol=rtol
    )
    numerical_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    if numerical_rank != dimensions.quotient_dimension:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the quotient residual Jacobian has numerical rank "
            f"{numerical_rank}, expected {dimensions.quotient_dimension} "
            f"at tolerance {rank_tolerance:.3e}",
        )

    # Only the retained singular triplets enter the Moore--Penrose action.
    # Columns below the numerical rank threshold are exactly the redundant
    # frame and vertical gauge directions under the regularity checks above.
    retained_left = left[:, :numerical_rank]
    retained_right = right_t[:numerical_rank, :].T
    retained_values = singular_values[:numerical_rank]
    coefficients = -retained_right @ (
        (retained_left.T @ residual) / retained_values
    )
    if not np.all(np.isfinite(coefficients)):
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the Moore--Penrose coefficient vector is not finite",
        )
    try:
        tangent = frame.lift(coefficients)
    except ValueError as failure:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            f"could not lift a finite Gauss--Newton direction: {failure}",
        ) from failure
    direction = _as_bismart_direction(tangent)
    _validate_direction(state, direction)

    direction_norm = product_norm(direction)
    tangency_error = _tangency_error(state, direction)
    tangency_tolerance = _postcheck_tolerance(
        direction_norm,
        dimensions.tangent_dimension,
        atol=atol,
        rtol=rtol,
    )
    if tangency_error > tangency_tolerance:
        raise NumericalFailure(
            FailureReason.IRREGULAR_ITERATE,
            "the lifted Gauss--Newton direction failed the product-tangency "
            f"check ({tangency_error:.3e} > {tangency_tolerance:.3e})",
        )

    horizontality_error = 0.0
    for generator in vertical:
        generator_norm = float(
            np.sqrt(max(product_frobenius_inner_product(generator, generator), 0.0))
        )
        if generator_norm > 0.0:
            horizontality_error = max(
                horizontality_error,
                abs(product_frobenius_inner_product(tangent, generator))
                / generator_norm,
            )
    horizontality_tolerance = _postcheck_tolerance(
        direction_norm,
        dimensions.tangent_dimension,
        atol=atol,
        rtol=rtol,
    )
    if horizontality_error > horizontality_tolerance:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the minimum-norm solution is not horizontal to numerical "
            f"tolerance ({horizontality_error:.3e} > "
            f"{horizontality_tolerance:.3e})",
        )

    linearized_residual = jacobian @ coefficients + residual
    normal_residual = jacobian.T @ linearized_residual
    normal_residual_norm = float(np.linalg.norm(normal_residual))
    normal_scale = largest * (
        float(np.linalg.norm(jacobian @ coefficients))
        + float(np.linalg.norm(residual))
    )
    normal_tolerance = _postcheck_tolerance(
        normal_scale,
        max(jacobian.shape),
        atol=atol,
        rtol=rtol,
    )
    if normal_residual_norm > normal_tolerance:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the Moore--Penrose direction failed the normal-residual check "
            f"({normal_residual_norm:.3e} > {normal_tolerance:.3e})",
        )

    gradients = euclidean_gradients(
        state, X_array, Y_array, source_array, omega
    )
    directional_derivative = product_inner_product(gradients, direction)
    gn_norm_squared = gauss_newton_inner_product(
        state, direction, direction, X_array, omega
    )
    descent_error = float(abs(directional_derivative + gn_norm_squared))
    descent_scale = abs(directional_derivative) + abs(gn_norm_squared)
    descent_tolerance = _postcheck_tolerance(
        descent_scale,
        max(jacobian.shape),
        atol=atol,
        rtol=rtol,
    )
    if descent_error > descent_tolerance:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the quotient direction failed the Gauss--Newton descent identity "
            f"({descent_error:.3e} > {descent_tolerance:.3e})",
        )

    diagnostics = GaussNewtonDiagnostics(
        tangent_dimension=dimensions.tangent_dimension,
        vertical_dimension=dimensions.vertical_dimension,
        quotient_dimension=dimensions.quotient_dimension,
        frame_dimension=frame.coefficient_dimension,
        residual_dimension=int(residual.size),
        estimated_dense_work_bytes=estimated_dense_work_bytes,
        dense_work_limit_bytes=dense_work_limit,
        jacobian_rank=numerical_rank,
        rank_tolerance=rank_tolerance,
        largest_singular_value=largest,
        smallest_retained_singular_value=float(retained_values[-1]),
        residual_norm=float(np.linalg.norm(residual)),
        normal_residual_norm=normal_residual_norm,
        tangency_error=tangency_error,
        horizontality_error=float(horizontality_error),
        descent_identity_error=descent_error,
        gauss_newton_norm_squared=float(gn_norm_squared),
    )
    return GaussNewtonResult(direction=direction, diagnostics=diagnostics)


def _solve_matrix_free_quotient_gauss_newton(
    state: BISMARTState,
    X: ArrayLike,
    Y: ArrayLike,
    observed_source: ArrayLike,
    omega: float,
    *,
    atol: float,
    rtol: float,
    max_dense_work_bytes: int | None,
    matrix_free_max_iterations: int | None,
) -> GaussNewtonResult:
    """Solve the quotient system with streamed Jacobian/adjoint products.

    This backend preserves the dense solver's minimum-product-norm semantics
    on regular systems but replaces the residual-Jacobian SVD by zero-start
    LSQR.  Rank is *not* inferred from LSQR convergence:
    :func:`_matrix_free_information_certificate` establishes algebraic
    injectivity before the right-hand side is inspected, and the compact and
    external screens reject several important numerical degeneracies.

    Those inexpensive screens are necessary rather than complete tests of the
    dense SVD policy: a coupled near-null direction can mix compact and ambient
    normal components.  Accordingly this explicit scalable backend reports
    ``jacobian_rank=None`` and certifies the returned direction using its
    normal-residual, horizontality, tangency, and descent postconditions.
    """

    atol, rtol = _solver_tolerances(atol, rtol)
    dense_work_limit = _dense_work_limit(max_dense_work_bytes)
    X_array, Y_array, source_array = _validate_problem_arrays(
        state, X, Y, observed_source, omega
    )
    try:
        validate_regular_state(state, atol=atol, rtol=rtol)
    except (GeometryValidationError, np.linalg.LinAlgError) as failure:
        raise NumericalFailure(
            FailureReason.IRREGULAR_ITERATE,
            f"quotient Gauss--Newton state is irregular: {failure}",
        ) from failure

    dimensions = geometry_dimensions(state)
    try:
        frame = tangent_parseval_frame(
            state, atol=atol, rtol=rtol, validate=False
        )
    except (GeometryValidationError, ValueError) as failure:
        raise NumericalFailure(
            FailureReason.IRREGULAR_ITERATE,
            f"could not construct the product tangent frame: {failure}",
        ) from failure

    # This deterministic structural check precedes residual construction on
    # purpose.  A zero/exact-fit residual must not allow a singular fitting
    # design to masquerade as a successful zero Gauss--Newton direction.
    (
        certified_rank,
        reduced_largest,
        reduced_smallest,
        reduced_rank_tolerance,
    ) = _matrix_free_information_certificate(
        state,
        X_array,
        quotient_dimension=dimensions.quotient_dimension,
        atol=atol,
        rtol=rtol,
    )

    sample_scale = float(np.sqrt(X_array.shape[0]))
    target_residual = (
        X_array @ fitted_target(state) - Y_array
    ) / sample_scale
    source_residual = np.sqrt(omega) * (
        fitted_source(state) - source_array
    )
    residual = np.concatenate(
        (target_residual.ravel(order="C"), source_residual.ravel(order="C"))
    )
    operator = _JointJacobianOperator(state, X_array, omega, frame)
    if operator.rows != residual.size:
        raise RuntimeError("internal matrix-free residual dimension mismatch")

    (
        compact_rank,
        compact_quotient_dimension,
        compact_rank_tolerance,
        compact_largest,
        compact_smallest,
        compact_work_bytes,
        ambient_normal_upper_bound,
        full_rank_tolerance_lower_bound,
        operator_norm_lower_bound,
    ) = _matrix_free_compact_rank_screen(
        state,
        X_array,
        omega,
        full_residual_dimension=operator.rows,
        full_frame_dimension=operator.columns,
        atol=atol,
        rtol=rtol,
    )

    estimated_dense_work_bytes = _estimated_dense_svd_bytes(
        operator.rows, operator.columns
    )
    if matrix_free_max_iterations is None:
        # In exact arithmetic Golub--Kahan needs at most the nonzero quotient
        # rank.  A factor of two permits finite-precision loss of orthogonality
        # without silently creating an unbounded loop.
        iteration_limit = max(20, 2 * dimensions.quotient_dimension)
    else:
        if isinstance(matrix_free_max_iterations, (bool, np.bool_)) or not isinstance(
            matrix_free_max_iterations, (int, np.integer)
        ):
            raise TypeError("matrix_free_max_iterations must be an integer or None")
        iteration_limit = int(matrix_free_max_iterations)
        if iteration_limit < 1:
            raise ValueError("matrix_free_max_iterations must be positive")

    residual_norm = float(np.linalg.norm(residual))
    machine_relative = np.finfo(float).eps * max(operator.rows, operator.columns)
    scaled_absolute = (
        atol / max(residual_norm, np.finfo(float).tiny)
        if residual_norm > 0.0
        else 0.0
    )
    iterative_tolerance = float(
        min(0.25, max(rtol, scaled_absolute, machine_relative))
    )
    reduced_relative_cutoff = max(
        rtol + atol / max(reduced_largest, np.finfo(float).tiny),
        machine_relative,
    )
    condition_limit = float(
        max(1.0 + np.sqrt(np.finfo(float).eps), 1.0 / reduced_relative_cutoff)
    )

    try:
        iterative = lsqr(
            operator.matvec,
            operator.rmatvec,
            rows=operator.rows,
            columns=operator.columns,
            rhs=-residual,
            tolerance=iterative_tolerance,
            max_iterations=iteration_limit,
            condition_limit=condition_limit,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as failure:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            f"the matrix-free quotient iteration failed: {failure}",
        ) from failure
    if not iterative.converged:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the matrix-free quotient iteration did not meet its residual "
            f"certificate in {iterative.iterations} iterations "
            f"(reason={iterative.stop_reason}, "
            f"condition_estimate={iterative.condition_estimate:.3e})",
        )

    coefficients = iterative.solution
    if not np.all(np.isfinite(coefficients)):
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the matrix-free Moore--Penrose coefficient vector is not finite",
        )
    try:
        tangent = frame.lift(coefficients)
    except ValueError as failure:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            f"could not lift a finite matrix-free direction: {failure}",
        ) from failure
    direction = _as_bismart_direction(tangent)
    _validate_direction(state, direction)

    direction_norm = product_norm(direction)
    tangency_error = _tangency_error(state, direction)
    tangency_tolerance = _postcheck_tolerance(
        direction_norm,
        dimensions.tangent_dimension,
        atol=atol,
        rtol=rtol,
    )
    if tangency_error > tangency_tolerance:
        raise NumericalFailure(
            FailureReason.IRREGULAR_ITERATE,
            "the matrix-free direction failed the product-tangency check "
            f"({tangency_error:.3e} > {tangency_tolerance:.3e})",
        )

    # Stream the gauge basis one element at a time.  In contrast to the dense
    # reference, the matrix-free path never allocates either a tuple of all
    # full product tangents or a frame_dimension x vertical_dimension matrix.
    horizontality_error = 0.0
    try:
        for generator in iter_vertical_gauge_generators(state):
            generator_norm = float(
                np.sqrt(
                    max(
                        product_frobenius_inner_product(generator, generator),
                        0.0,
                    )
                )
            )
            if not np.isfinite(generator_norm) or generator_norm == 0.0:
                raise NumericalFailure(
                    FailureReason.IRREGULAR_ITERATE,
                    "the streamed gauge action contains a zero or nonfinite "
                    "generator",
                )
            horizontality_error = max(
                horizontality_error,
                abs(product_frobenius_inner_product(tangent, generator))
                / generator_norm,
            )
    except (FloatingPointError, ValueError) as failure:
        raise NumericalFailure(
            FailureReason.IRREGULAR_ITERATE,
            f"could not stream finite gauge generators: {failure}",
        ) from failure
    horizontality_tolerance = _postcheck_tolerance(
        direction_norm,
        dimensions.tangent_dimension,
        atol=atol,
        rtol=rtol,
    )
    if horizontality_error > horizontality_tolerance:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the matrix-free minimum-norm solution is not horizontal to "
            f"numerical tolerance ({horizontality_error:.3e} > "
            f"{horizontality_tolerance:.3e})",
        )

    applied_direction = operator.matvec(coefficients)
    linearized_residual = applied_direction + residual
    normal_residual = operator.rmatvec(linearized_residual)
    normal_residual_norm = float(np.linalg.norm(normal_residual))
    operator_scale = max(
        iterative.operator_norm_estimate,
        reduced_largest,
        np.finfo(float).tiny,
    )
    normal_scale = operator_scale * (
        float(np.linalg.norm(applied_direction)) + residual_norm
    )
    normal_tolerance = _postcheck_tolerance(
        normal_scale,
        max(operator.rows, operator.columns),
        atol=atol,
        rtol=rtol,
    )
    if normal_residual_norm > normal_tolerance:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the matrix-free Moore--Penrose direction failed the explicit "
            f"normal-residual check ({normal_residual_norm:.3e} > "
            f"{normal_tolerance:.3e})",
        )

    gradients = euclidean_gradients(
        state, X_array, Y_array, source_array, omega
    )
    directional_derivative = product_inner_product(gradients, direction)
    gn_norm_squared = gauss_newton_inner_product(
        state, direction, direction, X_array, omega
    )
    descent_error = float(abs(directional_derivative + gn_norm_squared))
    descent_scale = abs(directional_derivative) + abs(gn_norm_squared)
    descent_tolerance = _postcheck_tolerance(
        descent_scale,
        max(operator.rows, operator.columns),
        atol=atol,
        rtol=rtol,
    )
    if descent_error > descent_tolerance:
        raise NumericalFailure(
            FailureReason.SINGULAR_NORMAL_EQUATIONS,
            "the matrix-free quotient direction failed the Gauss--Newton "
            f"descent identity ({descent_error:.3e} > "
            f"{descent_tolerance:.3e})",
        )

    diagnostics = GaussNewtonDiagnostics(
        tangent_dimension=dimensions.tangent_dimension,
        vertical_dimension=dimensions.vertical_dimension,
        quotient_dimension=dimensions.quotient_dimension,
        frame_dimension=frame.coefficient_dimension,
        residual_dimension=int(residual.size),
        estimated_dense_work_bytes=estimated_dense_work_bytes,
        dense_work_limit_bytes=dense_work_limit,
        # LSQR cannot report a complete numerical rank without effectively
        # rebuilding the dense SVD.  Keep that field honest; the separately
        # reported algebraic, compact, ambient-normal, and solve-specific
        # checks are the matrix-free certificate.
        jacobian_rank=None,
        rank_tolerance=None,
        largest_singular_value=None,
        smallest_retained_singular_value=None,
        residual_norm=residual_norm,
        normal_residual_norm=normal_residual_norm,
        tangency_error=tangency_error,
        horizontality_error=float(horizontality_error),
        descent_identity_error=descent_error,
        gauss_newton_norm_squared=float(gn_norm_squared),
        solver_backend="matrix_free",
        solver_iterations=iterative.iterations,
        solver_stop_reason=iterative.stop_reason,
        condition_estimate=iterative.condition_estimate,
        rank_certificate=(
            "algebraic_injectivity_plus_necessary_numerical_screens"
            "_and_rhs_postchecks"
        ),
        structural_quotient_rank=certified_rank,
        reduced_design_rank_tolerance=reduced_rank_tolerance,
        reduced_design_largest_singular_value=reduced_largest,
        reduced_design_smallest_singular_value=reduced_smallest,
        compact_jacobian_rank=compact_rank,
        compact_quotient_dimension=compact_quotient_dimension,
        compact_rank_tolerance=compact_rank_tolerance,
        compact_largest_singular_value=compact_largest,
        compact_smallest_singular_value=compact_smallest,
        estimated_compact_work_bytes=compact_work_bytes,
        ambient_normal_sensitivity_upper_bound=ambient_normal_upper_bound,
        full_rank_tolerance_lower_bound=full_rank_tolerance_lower_bound,
        operator_scale_lower_bound=operator_norm_lower_bound,
    )
    return GaussNewtonResult(direction=direction, diagnostics=diagnostics)


def solve_quotient_gauss_newton(
    state: BISMARTState,
    X: ArrayLike,
    Y: ArrayLike,
    observed_source: ArrayLike,
    omega: float,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    max_dense_work_bytes: int | None = DEFAULT_MAX_DENSE_WORK_BYTES,
    solver_backend: str = "dense",
    matrix_free_max_iterations: int | None = None,
) -> GaussNewtonResult:
    """Solve and certify the quotient Gauss--Newton system.

    ``solver_backend='dense'`` preserves the rank-revealing SVD reference and
    enforces ``max_dense_work_bytes``.  ``'matrix_free'`` uses analytic
    Jacobian/adjoint products plus LSQR and never constructs the full residual
    Jacobian or vertical-coordinate matrix.  It checks algebraic injectivity,
    necessary compact numerical screens, and solve-specific postconditions,
    but it does not reproduce every near-singular decision of the complete
    dense SVD.  ``'auto'`` explicitly accepts that policy: it uses the dense
    reference when its conservative workspace estimate fits the cap and
    otherwise selects the matrix-free backend.
    """

    backend = _solver_backend(solver_backend)
    if backend == "auto":
        dense_work_limit = _dense_work_limit(max_dense_work_bytes)
        X_array, Y_array, source_array = _validate_problem_arrays(
            state, X, Y, observed_source, omega
        )
        dimensions = geometry_dimensions(state)
        residual_dimension = int(
            X_array.shape[0] * Y_array.shape[1] + source_array.size
        )
        estimate = _estimated_dense_svd_bytes(
            residual_dimension, dimensions.coefficient_dimension
        )
        backend = (
            "dense"
            if dense_work_limit is None or estimate <= dense_work_limit
            else "matrix_free"
        )

    if backend == "dense":
        return _solve_dense_quotient_gauss_newton(
            state,
            X,
            Y,
            observed_source,
            omega,
            atol=atol,
            rtol=rtol,
            max_dense_work_bytes=max_dense_work_bytes,
        )
    return _solve_matrix_free_quotient_gauss_newton(
        state,
        X,
        Y,
        observed_source,
        omega,
        atol=atol,
        rtol=rtol,
        max_dense_work_bytes=max_dense_work_bytes,
        matrix_free_max_iterations=matrix_free_max_iterations,
    )


def quotient_gauss_newton_direction(
    state: BISMARTState,
    X: ArrayLike,
    Y: ArrayLike,
    observed_source: ArrayLike,
    omega: float,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    max_dense_work_bytes: int | None = DEFAULT_MAX_DENSE_WORK_BYTES,
    solver_backend: str = "dense",
    matrix_free_max_iterations: int | None = None,
) -> BISMARTDirection:
    """Return only the certified horizontal direction.

    :func:`solve_quotient_gauss_newton` exposes the associated rank and
    postcondition diagnostics.  This wrapper preserves the original public
    return type used by callers that only need the search direction.
    """

    return solve_quotient_gauss_newton(
        state,
        X,
        Y,
        observed_source,
        omega,
        atol=atol,
        rtol=rtol,
        max_dense_work_bytes=max_dense_work_bytes,
        solver_backend=solver_backend,
        matrix_free_max_iterations=matrix_free_max_iterations,
    ).direction


# The paper also uses the shorter phrase "horizontal GN direction".
horizontal_gauss_newton_direction = quotient_gauss_newton_direction


def _validate_problem_arrays(
    state: BISMARTState,
    X: ArrayLike,
    Y: ArrayLike,
    observed_source: ArrayLike,
    omega: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    X_array = _as_float_matrix(X, name="X")
    Y_array = _as_float_matrix(Y, name="Y")
    source_array = _as_float_matrix(observed_source, name="observed_source")
    if X_array.shape[0] == 0:
        raise ValueError("X and Y must contain at least one observation")
    if X_array.shape[0] != Y_array.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    if X_array.shape[1] != state.U0.shape[0]:
        raise ValueError("X has an incompatible number of predictor columns")
    if Y_array.shape[1] != state.V0.shape[0]:
        raise ValueError("Y has an incompatible number of response columns")
    expected_source_shape = (state.U0.shape[0], state.V0.shape[0])
    if source_array.shape != expected_source_shape:
        raise ValueError(
            f"observed_source must have shape {expected_source_shape}; "
            f"got {source_array.shape}"
        )
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and strictly positive")
    return X_array, Y_array, source_array


def _validate_direction(
    state: BISMARTState,
    direction: BISMARTDirection,
    *,
    require_fixed_support: bool = True,
) -> None:
    expected = {
        "U0": state.U0.shape,
        "V0": state.V0.shape,
        "A": state.A.shape,
        "B": state.B.shape,
        "H": state.H.shape,
    }
    for name, shape in expected.items():
        if getattr(direction, name).shape != shape:
            raise ValueError(
                f"direction.{name} must have shape {shape}; "
                f"got {getattr(direction, name).shape}"
            )
    if len(direction.G_blocks) != len(state.G_blocks):
        raise ValueError("direction has an incompatible number of G blocks")
    for index, (core_direction, core) in enumerate(
        zip(direction.G_blocks, state.G_blocks, strict=True)
    ):
        if core_direction.shape != core.shape:
            raise ValueError(
                f"direction.G_blocks[{index}] must have shape {core.shape}"
            )
    if require_fixed_support:
        if np.any(np.abs(direction.A[~state.active_u, :]) > 1e-12):
            raise ValueError("direction.A must be zero outside active_u")
        if np.any(np.abs(direction.B[~state.active_v, :]) > 1e-12):
            raise ValueError("direction.B must be zero outside active_v")


def _validate_compatible_states(
    state: BISMARTState, initial_state: BISMARTState
) -> None:
    if state.U0.shape != initial_state.U0.shape:
        raise ValueError("state and initial_state have incompatible U0 shapes")
    if state.V0.shape != initial_state.V0.shape:
        raise ValueError("state and initial_state have incompatible V0 shapes")
    if state.A.shape != initial_state.A.shape or state.B.shape != initial_state.B.shape:
        raise ValueError("state and initial_state have incompatible target factors")
    if state.H.shape != initial_state.H.shape:
        raise ValueError("state and initial_state have incompatible target cores")
    if state.block_sizes != initial_state.block_sizes:
        raise ValueError("state and initial_state have incompatible source partitions")
    if not np.array_equal(state.active_u, initial_state.active_u):
        raise ValueError("state and initial_state have different active_u supports")
    if not np.array_equal(state.active_v, initial_state.active_v):
        raise ValueError("state and initial_state have different active_v supports")


__all__ = [
    "ArmijoCheck",
    "BISMARTDirection",
    "BISMARTState",
    "BacktrackingResult",
    "GaussNewtonDiagnostics",
    "GaussNewtonResult",
    "PathCertificate",
    "RefinementError",
    "TrustRegionThresholds",
    "block_diagonal",
    "check_armijo",
    "check_path_certificate",
    "default_core_thresholds",
    "euclidean_gradients",
    "fitted_source",
    "fitted_target",
    "gauss_newton_inner_product",
    "horizontal_gauss_newton_direction",
    "joint_objective",
    "objective",
    "pair_distance",
    "polar_retraction",
    "product_inner_product",
    "product_norm",
    "project_to_tangent",
    "quotient_gauss_newton_direction",
    "retract_state",
    "safeguarded_backtracking",
    "solve_quotient_gauss_newton",
    "source_block_separation",
    "source_differential",
    "stiefel_error",
    "stiefel_tangent_projection",
    "strict_polar_factor",
    "tangent_projection",
    "target_differential",
]
