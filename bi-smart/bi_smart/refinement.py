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

The appendix defines the horizontal quotient Gauss--Newton direction through
an intrinsic variational equation, but it does not prescribe a concrete,
numerically stable construction of a horizontal basis.  Consequently this
module deliberately stops at that boundary.  It implements the fitted maps,
their differentials, the objective, explicit Euclidean gradients, product
Stiefel tangent projection, polar retraction, and the finite trust/Armijo
safeguards.  :func:`quotient_gauss_newton_direction` contains detailed
pseudocode and raises :class:`NotImplementedError` until the horizontal-basis
and normal-equation assembly have been designed and validated.

The code is intentionally verbose.  It is research scaffolding: comments name
the corresponding mathematical operation and make each remaining design
choice visible rather than hiding it behind a generic optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class RefinementError(RuntimeError):
    """Raised when a fixed-support refinement operation is algebraically invalid."""


def _as_float_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    """Return ``value`` as a finite two-dimensional floating-point array."""

    array = np.asarray(value, dtype=float)
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
        mask = np.asarray(value, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != size:
            raise ValueError(f"{name} must have shape ({size},)")
        mask = mask.copy()
    if int(mask.sum()) < rank:
        raise ValueError(
            f"{name} selects {int(mask.sum())} coordinates, fewer than rank={rank}"
        )
    return mask


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

    This class checks dimensions and finiteness but does not force exact
    orthonormality.  The latter is a numerical property checked separately by
    :func:`stiefel_error`; allowing small floating-point drift makes diagnostic
    use more convenient.
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
        del p, q  # Dimensions are named here only to make the checks readable.
        if v_r0 != r0:
            raise ValueError("U0 and V0 must have the same number of columns")
        if r0 < 1:
            raise ValueError("the working source rank r0 must be at least one")

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

        # The support is part of the state, not something inferred from its
        # numerical values.  Reject leakage into inactive rows early.
        if np.any(np.abs(matrices["A"][~active_u, :]) > 1e-12):
            raise ValueError("A must be zero outside active_u")
        if np.any(np.abs(matrices["B"][~active_v, :]) > 1e-12):
            raise ValueError("B must be zero outside active_v")

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

    size = sum(block.shape[0] for block in blocks)
    result = np.zeros((size, size), dtype=float)
    for coordinate_slice, block in zip(
        _block_slices(block.shape[0] for block in blocks), blocks, strict=True
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
    matrix: ArrayLike, *, rank_tolerance: float | None = None
) -> FloatArray:
    """Return the unique thin polar factor of a full-column-rank matrix.

    BI-SMART must reject a rank-deficient polar argument.  Completing a basis
    with coordinate vectors would break within-block rotation equivariance.
    """

    array = _as_float_matrix(matrix, name="matrix")
    rows, columns = array.shape
    if rows < columns:
        raise RefinementError("a Stiefel polar factor requires rows >= columns")
    left, singular_values, right_t = np.linalg.svd(array, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = (
        float(rank_tolerance)
        if rank_tolerance is not None
        else np.finfo(float).eps * max(rows, columns) * max(largest, 1.0)
    )
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
        Q_array + step_size * tangent_array, rank_tolerance=rank_tolerance
    )


def retract_state(
    state: BISMARTState,
    direction: BISMARTDirection,
    step_size: float,
    *,
    rank_tolerance: float | None = None,
) -> BISMARTState:
    """Apply the product polar/additive retraction from the appendix."""

    _validate_direction(state, direction)
    if not np.isfinite(step_size) or step_size < 0.0:
        raise ValueError("step_size must be finite and nonnegative")

    # A and B live on Stiefel manifolds only within their selected rows.
    next_A = np.zeros_like(state.A)
    next_A[state.active_u, :] = polar_retraction(
        state.A[state.active_u, :],
        direction.A[state.active_u, :],
        step_size=step_size,
        rank_tolerance=rank_tolerance,
    )
    next_B = np.zeros_like(state.B)
    next_B[state.active_v, :] = polar_retraction(
        state.B[state.active_v, :],
        direction.B[state.active_v, :],
        step_size=step_size,
        rank_tolerance=rank_tolerance,
    )
    return BISMARTState(
        U0=polar_retraction(
            state.U0,
            direction.U0,
            step_size=step_size,
            rank_tolerance=rank_tolerance,
        ),
        V0=polar_retraction(
            state.V0,
            direction.V0,
            step_size=step_size,
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


def default_core_thresholds(
    initial_state: BISMARTState,
    *,
    radius: float,
    zero_tolerance: float = 0.0,
    separation_tolerance: float | None = None,
) -> TrustRegionThresholds:
    """Compute the appendix's data-dependent core/separation defaults.

    The defaults are half the corresponding initializer values.  The trust
    radius is not determined by this rule and remains an explicit caller
    input.  A zero required quantity makes the refinement branch unsuccessful,
    so this helper raises :class:`RefinementError` instead of manufacturing a
    threshold.  ``separation_tolerance`` is separate because block separation
    compares eigenvalues of ``G_l G_l.T`` and therefore lives on a squared
    core scale; when omitted it retains the legacy ``zero_tolerance`` policy.
    """

    if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    if separation_tolerance is None:
        separation_tolerance = zero_tolerance
    if not np.isfinite(separation_tolerance) or separation_tolerance < 0.0:
        raise ValueError("separation_tolerance must be finite and nonnegative")
    h_value = _smallest_singular_value(initial_state.H)
    g_value = min(_smallest_singular_value(core) for core in initial_state.G_blocks)
    separation = source_block_separation(initial_state)
    if h_value <= zero_tolerance:
        raise RefinementError("the initial target core is singular")
    if g_value <= zero_tolerance:
        raise RefinementError("an initial source block core is singular")
    if (
        len(initial_state.G_blocks) > 1
        and separation <= separation_tolerance
    ):
        raise RefinementError("initial source block-core spectra are not separated")
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
    rank_tolerance: float | None = None,
) -> BacktrackingResult:
    """Try the finite safeguarded step sequence from the appendix.

    This routine is usable once a valid horizontal direction has been supplied
    by a future quotient backend.  It does *not* call
    :func:`quotient_gauss_newton_direction` itself.
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


_QUOTIENT_DIRECTION_PSEUDOCODE = """\
The horizontal quotient Gauss--Newton direction is not implemented yet.

PSEUDOCODE for the missing backend:
1. Construct deterministic orthonormal coordinate bases for the product
   tangent spaces at U0, V0, active(A), and active(B), and ordinary bases for
   H and every full G block.  Inactive A/B rows must never enter this basis.
2. Assemble the infinitesimal gauge (vertical-space) operator for block
   rotations Omega_u/Omega_v and rank-r rotations Omega_A/Omega_B:
       dU0 = U0 Omega_u;      dV0 = V0 Omega_v
       dA  = -Omega_u A + A Omega_A
       dB  = -Omega_v B + B Omega_B
       dG  = -Omega_u G + G Omega_v
       dH  = -Omega_A H + H Omega_B.
3. Compute, with a documented rank tolerance and deterministic signs/order, an
   orthonormal basis for the product-Frobenius orthogonal complement of that
   vertical range.  This is the horizontal basis required by the paper.
4. Apply target_differential and source_differential to each horizontal basis
   vector.  Stack vec(X @ dC)/sqrt(n) and sqrt(omega)*vec(dC0) to form the
   Gauss--Newton Jacobian J_H.  Stack the corresponding target and source
   residuals to form r.
5. Assemble J_H.T @ J_H and -J_H.T @ r.  Check symmetry, finite values, and
   positive definiteness using an explicit numerical tolerance.  Do not hide a
   singular system behind arbitrary damping: singularity is an unsuccessful
   BI-SMART refinement branch under the appendix's rules.
6. Solve the positive-definite horizontal normal equations and lift the
   coordinate vector back to BISMARTDirection.
7. Verify product tangency, horizontal orthogonality to every gauge generator,
   and the descent identity DF[xi] = -||xi||_GN^2 to numerical tolerance.
8. Return the direction together with diagnostics (horizontal dimension,
   smallest normal-equation eigenvalue, residual, and tolerances) so failures
   can be represented explicitly in the candidate library.

The paper specifies the variational equation but not the deterministic basis,
rank tolerance, sparse/dense assembly strategy, or numerical failure policy
needed for Steps 1--6.  Those choices require a dedicated implementation and
rotation-equivariance tests before this function can safely return a direction.
"""


def quotient_gauss_newton_direction(
    state: BISMARTState,
    X: ArrayLike,
    Y: ArrayLike,
    observed_source: ArrayLike,
    omega: float,
) -> BISMARTDirection:
    """Return the fixed-support horizontal quotient Gauss--Newton direction.

    This is intentionally a hard implementation boundary.  See the numbered
    pseudocode in the raised exception; returning a product tangent gradient or
    an unconstrained least-squares direction here would not implement the
    algorithm analysed in the appendix.
    """

    # Validate cheap caller errors before reporting the known missing backend.
    _validate_problem_arrays(state, X, Y, observed_source, omega)
    raise NotImplementedError(_QUOTIENT_DIRECTION_PSEUDOCODE)


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
    "source_block_separation",
    "source_differential",
    "stiefel_error",
    "stiefel_tangent_projection",
    "strict_polar_factor",
    "tangent_projection",
    "target_differential",
]
