"""Internal product/quotient geometry used by BI-SMART refinement.

The public refinement module owns the user-facing state and direction value
objects.  This module deliberately does *not* import those classes: doing so
would create a circular import as soon as :mod:`bi_smart.refinement` uses the
geometry routines below.  Instead, the routines accept ``StateLike`` and
``DirectionLike`` protocols and return the small internal
:class:`ProductTangent` value object.  A caller can turn that object into its
public direction class by passing the six identically named fields.

Two geometric choices deserve explicit documentation.

1. The tangent-space representation is the projected canonical *Parseval
   frame*.  If ``P_Q`` is the orthogonal projection onto the Stiefel tangent
   space at ``Q`` and ``E_ij`` are the ambient canonical matrices, then
   ``{P_Q(E_ij)}`` is a Parseval frame.  It is deterministic and requires no
   arbitrary completion of ``Q`` to a square orthogonal matrix.  It is
   generally redundant; that redundancy is intentional.
2. The BI-SMART appendix permits the product-metric Moore--Penrose solution
   of the singular full-tangent normal equations.  Synthesis by a Parseval
   frame is a coisometry, so a minimum-Euclidean-norm coefficient solution
   synthesizes to the minimum-product-Frobenius-norm tangent solution.  The
   latter is horizontal, because the kernel of the fitted-pair differential
   is precisely the vertical gauge space at a regular separated state.

All vectorization in this module uses NumPy's row-major (``order="C"``)
convention.  The convention has no mathematical effect as long as residuals
and Jacobian columns use it consistently, but fixing it here removes a common
source of hard-to-find implementation discrepancies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _real_array(value: ArrayLike, *, name: str) -> np.ndarray:
    """Normalize array-like input while rejecting lossy complex conversion."""

    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued; complex input is unsupported")
    return np.asarray(raw, dtype=float)


class StateLike(Protocol):
    """Structural interface required from a BI-SMART parameter state."""

    U0: ArrayLike
    V0: ArrayLike
    A: ArrayLike
    B: ArrayLike
    H: ArrayLike
    G_blocks: Sequence[ArrayLike]
    active_u: ArrayLike | None
    active_v: ArrayLike | None


class DirectionLike(Protocol):
    """Structural interface required from a product-space direction."""

    U0: ArrayLike
    V0: ArrayLike
    A: ArrayLike
    B: ArrayLike
    H: ArrayLike
    G_blocks: Sequence[ArrayLike]


class GeometryValidationError(ValueError):
    """A state violates a regular fixed-support quotient invariant.

    ``check`` is a stable, machine-readable local code.  The refinement layer
    can map every instance to its public ``IRREGULAR_ITERATE`` branch failure
    while retaining a precise diagnostic message for researchers.
    """

    def __init__(
        self,
        check: str,
        message: str,
        *,
        value: float | None = None,
        tolerance: float | None = None,
    ) -> None:
        self.check = check
        self.value = value
        self.tolerance = tolerance
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ProductTangent:
    """One tangent-shaped six-block array tuple.

    Only array shape and finiteness are checked here.  Manifold tangency and
    compatibility with a particular state remain solver-boundary checks;
    :class:`TangentParsevalFrame` constructs them correctly by design.
    """

    U0: FloatArray
    V0: FloatArray
    A: FloatArray
    B: FloatArray
    H: FloatArray
    G_blocks: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        for name in ("U0", "V0", "A", "B", "H"):
            array = _real_array(getattr(self, name), name=name)
            if array.ndim != 2 or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite two-dimensional array")
            object.__setattr__(self, name, array)
        blocks = tuple(
            _real_array(block, name=f"G_blocks[{index}]")
            for index, block in enumerate(self.G_blocks)
        )
        if any(block.ndim != 2 or not np.all(np.isfinite(block)) for block in blocks):
            raise ValueError("G_blocks must contain finite two-dimensional arrays")
        object.__setattr__(self, "G_blocks", blocks)


@dataclass(frozen=True, slots=True)
class GeometryDimensions:
    """Ambient-frame, product-tangent, gauge, and quotient dimensions."""

    p: int
    q: int
    source_rank: int
    target_rank: int
    active_u_dimension: int
    active_v_dimension: int
    block_sizes: tuple[int, ...]
    coefficient_dimension: int
    tangent_dimension: int
    vertical_dimension: int
    quotient_dimension: int

    @property
    def frame_dimension(self) -> int:
        """Backward-friendly synonym for ``coefficient_dimension``."""

        return self.coefficient_dimension


@dataclass(frozen=True, slots=True)
class RegularStateDiagnostics:
    """Scale-aware numerical quantities certified for a regular state."""

    dimensions: GeometryDimensions
    orthonormality_errors: tuple[tuple[str, float], ...]
    target_core_smallest_singular_value: float
    source_core_smallest_singular_values: tuple[float, ...]
    minimum_cross_block_separation: float
    cross_block_separations: tuple[tuple[int, int, float, float], ...]
    absolute_tolerance: float
    relative_tolerance: float


@dataclass(frozen=True, slots=True)
class FrameBlock:
    """One consecutive coefficient segment in a tangent Parseval frame."""

    name: str
    coefficient_slice: slice
    shape: tuple[int, int]
    constrained: bool

    @property
    def size(self) -> int:
        """Number of projected canonical elements in this segment."""

        return int(self.shape[0] * self.shape[1])


@dataclass(frozen=True, slots=True)
class _StateArrays:
    """Normalized views used internally after structural shape checks."""

    U0: FloatArray
    V0: FloatArray
    A: FloatArray
    B: FloatArray
    H: FloatArray
    G_blocks: tuple[FloatArray, ...]
    active_u: BoolArray
    active_v: BoolArray

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return tuple(int(block.shape[0]) for block in self.G_blocks)


def _finite_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    """Normalize one finite nonempty two-dimensional array."""

    array = _real_array(value, name=name)
    if array.ndim != 2 or 0 in array.shape:
        raise GeometryValidationError(
            "invalid_shape",
            f"{name} must be a nonempty two-dimensional matrix; got {array.shape}",
        )
    if not np.all(np.isfinite(array)):
        raise GeometryValidationError(
            "nonfinite_state", f"{name} must contain only finite values"
        )
    return array


def _active_mask(value: ArrayLike | None, *, size: int, name: str) -> BoolArray:
    """Normalize an explicit support mask without inferring it from values."""

    if value is None:
        return np.ones(size, dtype=bool)
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape != (size,):
        raise GeometryValidationError(
            "invalid_support_shape", f"{name} must have shape ({size},)"
        )
    # Accept boolean masks and exact zero/one masks, but reject values whose
    # truth conversion would silently change the selected block union.
    if raw.dtype != np.bool_:
        if np.iscomplexobj(raw):
            raise GeometryValidationError(
                "invalid_support_values", f"{name} must be a real boolean mask"
            )
        try:
            numeric = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as error:
            raise GeometryValidationError(
                "invalid_support_values", f"{name} must be a boolean mask"
            ) from error
        if not np.all(np.isfinite(numeric)) or not np.all(
            np.logical_or(numeric == 0.0, numeric == 1.0)
        ):
            raise GeometryValidationError(
                "invalid_support_values",
                f"{name} must contain only boolean (or exact zero/one) values",
            )
    return np.asarray(raw, dtype=bool).copy()


def _state_arrays(state: StateLike) -> _StateArrays:
    """Read and structurally validate a state-like object.

    This helper checks dimensions and finiteness only.  Manifold regularity,
    fixed-support block unions, and numerical ranks are handled by
    :func:`validate_regular_state` so callers receive specific failure codes.
    """

    try:
        U0 = _finite_matrix(state.U0, name="U0")
        V0 = _finite_matrix(state.V0, name="V0")
        A = _finite_matrix(state.A, name="A")
        B = _finite_matrix(state.B, name="B")
        H = _finite_matrix(state.H, name="H")
        G_blocks = tuple(
            _finite_matrix(block, name=f"G_blocks[{index}]")
            for index, block in enumerate(state.G_blocks)
        )
    except AttributeError as error:
        raise GeometryValidationError(
            "missing_state_attribute",
            "state must define U0, V0, A, B, H, G_blocks, active_u, and active_v",
        ) from error

    p, source_rank = U0.shape
    q, right_source_rank = V0.shape
    if p < source_rank or q < source_rank:
        raise GeometryValidationError(
            "invalid_stiefel_shape",
            "U0 and V0 must have at least as many rows as columns",
        )
    if right_source_rank != source_rank:
        raise GeometryValidationError(
            "incompatible_source_rank",
            "U0 and V0 must have the same number of columns",
        )
    if A.shape[0] != source_rank or B.shape[0] != source_rank:
        raise GeometryValidationError(
            "incompatible_target_factor_shape",
            "A and B must each have one row per source coordinate",
        )
    target_rank = A.shape[1]
    if B.shape[1] != target_rank or H.shape != (target_rank, target_rank):
        raise GeometryValidationError(
            "incompatible_target_rank",
            "A, B, and H must share one positive target rank",
        )
    if target_rank < 1:
        raise GeometryValidationError(
            "invalid_target_rank", "the target rank must be at least one"
        )

    if not G_blocks:
        raise GeometryValidationError(
            "empty_source_partition", "G_blocks must contain at least one block"
        )
    for index, block in enumerate(G_blocks):
        if block.shape[0] != block.shape[1]:
            raise GeometryValidationError(
                "nonsquare_source_core", f"G_blocks[{index}] must be square"
            )
    if sum(block.shape[0] for block in G_blocks) != source_rank:
        raise GeometryValidationError(
            "incompatible_source_partition",
            "the G block dimensions must sum to the source rank",
        )

    try:
        active_u_value = state.active_u
        active_v_value = state.active_v
    except AttributeError as error:
        raise GeometryValidationError(
            "missing_state_attribute", "state must define active_u and active_v"
        ) from error
    active_u = _active_mask(active_u_value, size=source_rank, name="active_u")
    active_v = _active_mask(active_v_value, size=source_rank, name="active_v")

    return _StateArrays(
        U0=U0,
        V0=V0,
        A=A,
        B=B,
        H=H,
        G_blocks=G_blocks,
        active_u=active_u,
        active_v=active_v,
    )


def _block_slices(block_sizes: Sequence[int]) -> tuple[slice, ...]:
    """Convert consecutive source-block sizes to half-open slices."""

    result: list[slice] = []
    start = 0
    for size in block_sizes:
        stop = start + int(size)
        result.append(slice(start, stop))
        start = stop
    return tuple(result)


def _stiefel_dimension(rows: int, columns: int) -> int:
    """Dimension of ``St(rows, columns)``."""

    return int(rows * columns - columns * (columns + 1) // 2)


def _dimensions(arrays: _StateArrays) -> GeometryDimensions:
    """Evaluate the appendix's product and quotient dimension formulas."""

    p, source_rank = arrays.U0.shape
    q = arrays.V0.shape[0]
    target_rank = arrays.A.shape[1]
    active_u_dimension = int(np.count_nonzero(arrays.active_u))
    active_v_dimension = int(np.count_nonzero(arrays.active_v))
    block_sizes = arrays.block_sizes

    # The projected canonical frame has one member per ambient coordinate of
    # every constrained factor and one ordinary member per linear-core entry.
    coefficient_dimension = int(
        p * source_rank
        + q * source_rank
        + active_u_dimension * target_rank
        + active_v_dimension * target_rank
        + target_rank**2
        + sum(size**2 for size in block_sizes)
    )
    tangent_dimension = int(
        _stiefel_dimension(p, source_rank)
        + _stiefel_dimension(q, source_rank)
        + _stiefel_dimension(active_u_dimension, target_rank)
        + _stiefel_dimension(active_v_dimension, target_rank)
        + target_rank**2
        + sum(size**2 for size in block_sizes)
    )
    vertical_dimension = int(
        sum(size * (size - 1) for size in block_sizes)
        + target_rank * (target_rank - 1)
    )
    quotient_dimension = int(
        source_rank * (p + q - source_rank)
        + target_rank
        * (active_u_dimension + active_v_dimension - target_rank)
    )
    # This equality is the dimension calculation in Proposition
    # ``bismart-quotient``.  A mismatch would indicate an implementation bug,
    # not a data-dependent numerical failure.
    if tangent_dimension - vertical_dimension != quotient_dimension:
        raise RuntimeError("internal BI-SMART quotient dimension mismatch")

    return GeometryDimensions(
        p=int(p),
        q=int(q),
        source_rank=int(source_rank),
        target_rank=int(target_rank),
        active_u_dimension=active_u_dimension,
        active_v_dimension=active_v_dimension,
        block_sizes=block_sizes,
        coefficient_dimension=coefficient_dimension,
        tangent_dimension=tangent_dimension,
        vertical_dimension=vertical_dimension,
        quotient_dimension=quotient_dimension,
    )


def geometry_dimensions(state: StateLike) -> GeometryDimensions:
    """Return all dimensions needed by the dense quotient solver."""

    return _dimensions(_state_arrays(state))


def expected_tangent_dimension(state: StateLike) -> int:
    """Return the product-manifold tangent dimension."""

    return geometry_dimensions(state).tangent_dimension


def expected_vertical_dimension(state: StateLike) -> int:
    """Return the dimension of the free block/target rotation action."""

    return geometry_dimensions(state).vertical_dimension


def expected_quotient_dimension(state: StateLike) -> int:
    """Return ``r0(p+q-r0) + r(ku+kv-r)`` from the appendix."""

    return geometry_dimensions(state).quotient_dimension


def _validate_tolerances(atol: float, rtol: float) -> tuple[float, float]:
    absolute = float(atol)
    relative = float(rtol)
    if not np.isfinite(absolute) or absolute < 0.0:
        raise ValueError("atol must be finite and nonnegative")
    if not np.isfinite(relative) or relative < 0.0:
        raise ValueError("rtol must be finite and nonnegative")
    return absolute, relative


def _comparison_tolerance(scale: float, atol: float, rtol: float) -> float:
    """Return a configured tolerance with an unavoidable machine floor.

    The floor remains relative to the invariant's own scale.  In particular,
    it does not introduce a unit-size absolute threshold that would reject a
    perfectly regular small-core state.
    """

    nonnegative_scale = max(float(scale), 0.0)
    return float(
        max(
            atol + rtol * nonnegative_scale,
            64.0 * np.finfo(float).eps * nonnegative_scale,
        )
    )


def _check_whole_blocks(
    mask: BoolArray,
    block_sizes: tuple[int, ...],
    *,
    name: str,
) -> None:
    """Require a selected support to be a union of entire source blocks."""

    for block_index, coordinate_slice in enumerate(_block_slices(block_sizes)):
        block_mask = mask[coordinate_slice]
        if np.any(block_mask) and not np.all(block_mask):
            raise GeometryValidationError(
                f"{name}_not_block_union",
                f"{name} partially selects source block {block_index}; "
                "fixed BI-SMART supports must be unions of whole blocks",
            )


def _check_stiefel(
    matrix: FloatArray,
    *,
    name: str,
    atol: float,
    rtol: float,
) -> float:
    """Certify column orthonormality in the operator norm."""

    columns = matrix.shape[1]
    gram_error = float(
        np.linalg.norm(matrix.T @ matrix - np.eye(columns), ord=2)
    )
    tolerance = _comparison_tolerance(1.0, atol, rtol)
    if gram_error > tolerance:
        raise GeometryValidationError(
            f"{name}_not_orthonormal",
            f"{name} is not column-orthonormal: error={gram_error:.6g}, "
            f"tolerance={tolerance:.6g}",
            value=gram_error,
            tolerance=tolerance,
        )
    return gram_error


def _smallest_singular_value(
    matrix: FloatArray,
    *,
    name: str,
    atol: float,
    rtol: float,
) -> float:
    """Certify full numerical rank with a scale-aware SVD threshold."""

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    tolerance = _comparison_tolerance(largest, atol, rtol)
    if smallest <= tolerance:
        raise GeometryValidationError(
            f"{name}_singular",
            f"{name} is numerically singular: sigma_min={smallest:.6g}, "
            f"tolerance={tolerance:.6g}",
            value=smallest,
            tolerance=tolerance,
        )
    return smallest


def validate_regular_state(
    state: StateLike,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> RegularStateDiagnostics:
    """Validate the regular, separated, fixed-support quotient state.

    PSEUDOCODE
    ----------
    1. Check all six parameter blocks, the source partition, and support-mask
       dimensions for finite, compatible shapes.
    2. Require each recorded support to select complete source blocks, with at
       least ``r`` active rows, and reject leakage into inactive rows.
    3. Certify ``U0``, ``V0``, and the active pieces of ``A`` and ``B`` as
       Stiefel factors using ``atol + rtol * scale``.
    4. Certify the target core and every source-block core as nonsingular.
    5. For every distinct source-block pair, compare the spectra of
       ``G_l G_l.T`` and ``G_k G_k.T``.  Their minimum distance must exceed a
       scale-aware threshold.  The check is vacuous for a single block.

    The function raises :class:`GeometryValidationError` at the first failed
    invariant and otherwise returns all numerical margins needed for solver
    diagnostics.  It never modifies the supplied state.
    """

    absolute, relative = _validate_tolerances(atol, rtol)
    arrays = _state_arrays(state)
    dimensions = _dimensions(arrays)
    block_sizes = arrays.block_sizes

    _check_whole_blocks(arrays.active_u, block_sizes, name="active_u")
    _check_whole_blocks(arrays.active_v, block_sizes, name="active_v")
    if dimensions.active_u_dimension < dimensions.target_rank:
        raise GeometryValidationError(
            "active_u_too_small",
            "active_u selects fewer rows than the target rank",
        )
    if dimensions.active_v_dimension < dimensions.target_rank:
        raise GeometryValidationError(
            "active_v_too_small",
            "active_v selects fewer rows than the target rank",
        )

    # Inactive coordinates are absent from the fixed-support manifold.  A
    # scale-aware comparison tolerates harmless roundoff but never infers the
    # support from which rows happen to be numerically nonzero.
    a_scale = float(np.max(np.abs(arrays.A)))
    b_scale = float(np.max(np.abs(arrays.B)))
    a_leakage = (
        float(np.max(np.abs(arrays.A[~arrays.active_u, :])))
        if np.any(~arrays.active_u)
        else 0.0
    )
    b_leakage = (
        float(np.max(np.abs(arrays.B[~arrays.active_v, :])))
        if np.any(~arrays.active_v)
        else 0.0
    )
    a_tolerance = _comparison_tolerance(a_scale, absolute, relative)
    b_tolerance = _comparison_tolerance(b_scale, absolute, relative)
    if a_leakage > a_tolerance:
        raise GeometryValidationError(
            "A_support_leakage",
            f"A is nonzero outside active_u: leakage={a_leakage:.6g}, "
            f"tolerance={a_tolerance:.6g}",
            value=a_leakage,
            tolerance=a_tolerance,
        )
    if b_leakage > b_tolerance:
        raise GeometryValidationError(
            "B_support_leakage",
            f"B is nonzero outside active_v: leakage={b_leakage:.6g}, "
            f"tolerance={b_tolerance:.6g}",
            value=b_leakage,
            tolerance=b_tolerance,
        )

    orthonormality_errors = (
        ("U0", _check_stiefel(arrays.U0, name="U0", atol=absolute, rtol=relative)),
        ("V0", _check_stiefel(arrays.V0, name="V0", atol=absolute, rtol=relative)),
        (
            "A_active",
            _check_stiefel(
                arrays.A[arrays.active_u, :],
                name="A_active",
                atol=absolute,
                rtol=relative,
            ),
        ),
        (
            "B_active",
            _check_stiefel(
                arrays.B[arrays.active_v, :],
                name="B_active",
                atol=absolute,
                rtol=relative,
            ),
        ),
    )

    h_smallest = _smallest_singular_value(
        arrays.H, name="H", atol=absolute, rtol=relative
    )
    g_smallest = tuple(
        _smallest_singular_value(
            block,
            name=f"G_blocks_{index}",
            atol=absolute,
            rtol=relative,
        )
        for index, block in enumerate(arrays.G_blocks)
    )

    spectra = tuple(
        np.linalg.eigvalsh(block @ block.T) for block in arrays.G_blocks
    )
    separation_records: list[tuple[int, int, float, float]] = []
    minimum_separation = float("inf")
    for left in range(len(spectra)):
        for right in range(left + 1, len(spectra)):
            left_spectrum = spectra[left]
            right_spectrum = spectra[right]
            separation = float(
                np.min(
                    np.abs(left_spectrum[:, np.newaxis] - right_spectrum[np.newaxis, :])
                )
            )
            spectral_scale = max(
                float(np.max(np.abs(left_spectrum))),
                float(np.max(np.abs(right_spectrum))),
            )
            core_scale = float(np.sqrt(max(spectral_scale, 0.0)))
            core_tolerance = _comparison_tolerance(
                core_scale, absolute, relative
            )
            tolerance = (
                2.0 * core_scale * core_tolerance + core_tolerance**2
            )
            separation_records.append((left, right, separation, tolerance))
            minimum_separation = min(minimum_separation, separation)
            if separation <= tolerance:
                raise GeometryValidationError(
                    "cross_block_spectrum_collision",
                    f"source blocks {left} and {right} are not spectrally "
                    f"separated: gap={separation:.6g}, "
                    f"tolerance={tolerance:.6g}",
                    value=separation,
                    tolerance=tolerance,
                )

    return RegularStateDiagnostics(
        dimensions=dimensions,
        orthonormality_errors=orthonormality_errors,
        target_core_smallest_singular_value=h_smallest,
        source_core_smallest_singular_values=g_smallest,
        minimum_cross_block_separation=minimum_separation,
        cross_block_separations=tuple(separation_records),
        absolute_tolerance=absolute,
        relative_tolerance=relative,
    )


def _stiefel_projection(Q: FloatArray, ambient: FloatArray) -> FloatArray:
    """Apply the product-Frobenius orthogonal Stiefel tangent projection."""

    qt_ambient = Q.T @ ambient
    symmetric = 0.5 * (qt_ambient + qt_ambient.T)
    return ambient - Q @ symmetric


class TangentParsevalFrame:
    """Projected-canonical Parseval frame for one product tangent space.

    A coefficient vector contains ambient coordinates for ``U0``, ``V0``,
    active ``A``, active ``B``, ``H``, and each full ``G`` block, in that
    order.  :meth:`lift` projects the four constrained coordinate matrices
    and embeds active target-factor rows into their fixed full shapes.

    ``coefficient_dimension`` is generally larger than ``tangent_dimension``.
    Callers must therefore use a Moore--Penrose solve (or an equivalent
    rank-revealing method), not demand full column rank from a Jacobian built
    by :meth:`iter_elements`.
    """

    __slots__ = (
        "_U0",
        "_V0",
        "_A",
        "_B",
        "_H_shape",
        "_G_shapes",
        "_active_u",
        "_active_v",
        "_blocks",
        "_dimensions",
    )

    def __init__(self, arrays: _StateArrays) -> None:
        self._U0 = arrays.U0
        self._V0 = arrays.V0
        self._A = arrays.A
        self._B = arrays.B
        self._H_shape = arrays.H.shape
        self._G_shapes = tuple(block.shape for block in arrays.G_blocks)
        self._active_u = arrays.active_u
        self._active_v = arrays.active_v
        self._dimensions = _dimensions(arrays)

        # PSEUDOCODE: assign one deterministic consecutive slice to every
        # projected-canonical/ordinary matrix frame.
        specifications = [
            ("U0", arrays.U0.shape, True),
            ("V0", arrays.V0.shape, True),
            ("A_active", arrays.A[arrays.active_u, :].shape, True),
            ("B_active", arrays.B[arrays.active_v, :].shape, True),
            ("H", arrays.H.shape, False),
        ]
        specifications.extend(
            (f"G_blocks[{index}]", block.shape, False)
            for index, block in enumerate(arrays.G_blocks)
        )
        blocks: list[FrameBlock] = []
        start = 0
        for name, shape, constrained in specifications:
            stop = start + int(shape[0] * shape[1])
            blocks.append(
                FrameBlock(
                    name=name,
                    coefficient_slice=slice(start, stop),
                    shape=(int(shape[0]), int(shape[1])),
                    constrained=constrained,
                )
            )
            start = stop
        if start != self._dimensions.frame_dimension:
            raise RuntimeError("internal tangent-frame dimension mismatch")
        self._blocks = tuple(blocks)

    @property
    def blocks(self) -> tuple[FrameBlock, ...]:
        """Ordered coefficient layout used by :meth:`lift` and :meth:`analyze`."""

        return self._blocks

    @property
    def dimensions(self) -> GeometryDimensions:
        """All product and quotient dimensions for the represented state."""

        return self._dimensions

    @property
    def coefficient_dimension(self) -> int:
        """Number of (possibly redundant) projected canonical frame members."""

        return self._dimensions.frame_dimension

    @property
    def tangent_dimension(self) -> int:
        """Mathematical dimension of the product tangent space."""

        return self._dimensions.tangent_dimension

    def _coefficient_matrices(self, coefficients: ArrayLike) -> tuple[FloatArray, ...]:
        vector = _real_array(coefficients, name="coefficients")
        if vector.ndim != 1 or vector.shape != (self.coefficient_dimension,):
            raise ValueError(
                "coefficients must have shape "
                f"({self.coefficient_dimension},); got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("coefficients must contain only finite values")
        return tuple(
            vector[block.coefficient_slice].reshape(block.shape, order="C")
            for block in self._blocks
        )

    def lift(self, coefficients: ArrayLike) -> ProductTangent:
        """Synthesize a product tangent from Parseval-frame coefficients.

        PSEUDOCODE
        ----------
        1. Reshape the consecutive coefficient segments in the documented
           row-major order.
        2. Orthogonally project the ambient ``U0`` and ``V0`` segments onto
           their Stiefel tangent spaces.
        3. Project the active ``A`` and ``B`` segments and embed them in zero
           full-size arrays, preserving fixed support exactly.
        4. Pass the unrestricted ``H`` and full block-core segments through.
        """

        matrices = self._coefficient_matrices(coefficients)
        U0_coordinates, V0_coordinates, A_coordinates, B_coordinates, H = matrices[:5]
        G_blocks = matrices[5:]

        tangent_A = np.zeros_like(self._A)
        tangent_A[self._active_u, :] = _stiefel_projection(
            self._A[self._active_u, :], A_coordinates
        )
        tangent_B = np.zeros_like(self._B)
        tangent_B[self._active_v, :] = _stiefel_projection(
            self._B[self._active_v, :], B_coordinates
        )
        return ProductTangent(
            U0=_stiefel_projection(self._U0, U0_coordinates),
            V0=_stiefel_projection(self._V0, V0_coordinates),
            A=tangent_A,
            B=tangent_B,
            H=H.copy(),
            G_blocks=tuple(block.copy() for block in G_blocks),
        )

    # ``synthesize`` is standard frame terminology and makes the linear-map
    # role clearer in numerical code; ``lift`` matches the quotient language.
    synthesize = lift

    def analyze(self, direction: DirectionLike) -> FloatArray:
        """Return Parseval analysis coefficients of a direction-shaped tuple.

        For an actual tangent vector this is simply the concatenation of its
        ambient entries.  Applying the tangent projections here also makes the
        operation the true adjoint of :meth:`lift` for arbitrary ambient input.
        """

        U0 = _finite_matrix(direction.U0, name="direction.U0")
        V0 = _finite_matrix(direction.V0, name="direction.V0")
        A = _finite_matrix(direction.A, name="direction.A")
        B = _finite_matrix(direction.B, name="direction.B")
        H = _finite_matrix(direction.H, name="direction.H")
        G_blocks = tuple(
            _finite_matrix(block, name=f"direction.G_blocks[{index}]")
            for index, block in enumerate(direction.G_blocks)
        )
        expected_shapes = (
            self._U0.shape,
            self._V0.shape,
            self._A.shape,
            self._B.shape,
            self._H_shape,
        )
        actual_shapes = (U0.shape, V0.shape, A.shape, B.shape, H.shape)
        if actual_shapes != expected_shapes or tuple(
            block.shape for block in G_blocks
        ) != self._G_shapes:
            raise ValueError("direction shapes do not match this tangent frame")

        projected = (
            _stiefel_projection(self._U0, U0),
            _stiefel_projection(self._V0, V0),
            _stiefel_projection(
                self._A[self._active_u, :], A[self._active_u, :]
            ),
            _stiefel_projection(
                self._B[self._active_v, :], B[self._active_v, :]
            ),
            H,
            *G_blocks,
        )
        return np.concatenate(
            tuple(matrix.reshape(-1, order="C") for matrix in projected)
        )

    def element(self, index: int) -> ProductTangent:
        """Return one projected canonical frame member by deterministic index."""

        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("frame index must be an integer")
        normalized = int(index)
        if normalized < 0 or normalized >= self.coefficient_dimension:
            raise IndexError("tangent frame index is out of range")
        coefficients = np.zeros(self.coefficient_dimension, dtype=float)
        coefficients[normalized] = 1.0
        return self.lift(coefficients)

    def iter_elements(self) -> Iterator[ProductTangent]:
        """Yield projected canonical members in coefficient-layout order."""

        for index in range(self.coefficient_dimension):
            yield self.element(index)


def tangent_parseval_frame(
    state: StateLike,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    validate: bool = True,
) -> TangentParsevalFrame:
    """Construct the deterministic fixed-support product tangent frame."""

    if validate:
        validate_regular_state(state, atol=atol, rtol=rtol)
    else:
        # Even an unchecked frame needs a structurally coherent state.  The
        # lower-level parser supplies that guarantee without numerical gates.
        _validate_tolerances(atol, rtol)
    return TangentParsevalFrame(_state_arrays(state))


def _zeros(arrays: _StateArrays) -> ProductTangent:
    """Allocate one all-zero product tangent with the state's block shapes."""

    return ProductTangent(
        U0=np.zeros_like(arrays.U0),
        V0=np.zeros_like(arrays.V0),
        A=np.zeros_like(arrays.A),
        B=np.zeros_like(arrays.B),
        H=np.zeros_like(arrays.H),
        G_blocks=tuple(np.zeros_like(block) for block in arrays.G_blocks),
    )


def _skew_basis(size: int) -> Iterator[tuple[int, int, FloatArray]]:
    """Yield the normalized canonical basis of ``so(size)``."""

    normalization = 1.0 / np.sqrt(2.0)
    for row in range(size):
        for column in range(row + 1, size):
            generator = np.zeros((size, size), dtype=float)
            generator[row, column] = normalization
            generator[column, row] = -normalization
            yield row, column, generator


def iter_vertical_gauge_generators(state: StateLike) -> Iterator[ProductTangent]:
    """Yield deterministic infinitesimal generators of the gauge action.

    Generator order is:

    1. left source-block rotations, block by block and lexicographic ``i<j``;
    2. right source-block rotations in the same order;
    3. left target-rank rotations; and
    4. right target-rank rotations.

    Differentiating the appendix action gives

    ``dU0=U0 Omega_u``, ``dA=-Omega_u A + A Omega_A``,
    ``dV0=V0 Omega_v``, ``dB=-Omega_v B + B Omega_B``,
    ``dG=-Omega_u G + G Omega_v``, and
    ``dH=-Omega_A H + H Omega_B``.

    Whole-block support makes every generated ``dA``/``dB`` remain on the
    recorded support, including when a source block is inactive.
    """

    arrays = _state_arrays(state)
    block_slices = _block_slices(arrays.block_sizes)
    generated = 0

    # PSEUDOCODE 1: left source rotations R_l.
    for block_index, (coordinate_slice, block) in enumerate(
        zip(block_slices, arrays.G_blocks, strict=True)
    ):
        for _, _, omega in _skew_basis(block.shape[0]):
            full_omega = np.zeros((arrays.U0.shape[1], arrays.U0.shape[1]))
            full_omega[coordinate_slice, coordinate_slice] = omega
            zero = _zeros(arrays)
            core_directions = list(zero.G_blocks)
            core_directions[block_index] = -omega @ block
            yield ProductTangent(
                U0=arrays.U0 @ full_omega,
                V0=zero.V0,
                A=-full_omega @ arrays.A,
                B=zero.B,
                H=zero.H,
                G_blocks=tuple(core_directions),
            )
            generated += 1

    # PSEUDOCODE 2: right source rotations S_l.
    for block_index, (coordinate_slice, block) in enumerate(
        zip(block_slices, arrays.G_blocks, strict=True)
    ):
        for _, _, omega in _skew_basis(block.shape[0]):
            full_omega = np.zeros((arrays.V0.shape[1], arrays.V0.shape[1]))
            full_omega[coordinate_slice, coordinate_slice] = omega
            zero = _zeros(arrays)
            core_directions = list(zero.G_blocks)
            core_directions[block_index] = block @ omega
            yield ProductTangent(
                U0=zero.U0,
                V0=arrays.V0 @ full_omega,
                A=zero.A,
                B=-full_omega @ arrays.B,
                H=zero.H,
                G_blocks=tuple(core_directions),
            )
            generated += 1

    # PSEUDOCODE 3: target left rotations Q_u.
    for _, _, omega in _skew_basis(arrays.A.shape[1]):
        zero = _zeros(arrays)
        yield ProductTangent(
            U0=zero.U0,
            V0=zero.V0,
            A=arrays.A @ omega,
            B=zero.B,
            H=-omega @ arrays.H,
            G_blocks=zero.G_blocks,
        )
        generated += 1

    # PSEUDOCODE 4: target right rotations Q_v.
    for _, _, omega in _skew_basis(arrays.B.shape[1]):
        zero = _zeros(arrays)
        yield ProductTangent(
            U0=zero.U0,
            V0=zero.V0,
            A=zero.A,
            B=arrays.B @ omega,
            H=arrays.H @ omega,
            G_blocks=zero.G_blocks,
        )
        generated += 1

    expected = _dimensions(arrays).vertical_dimension
    if generated != expected:
        raise RuntimeError("internal vertical-generator dimension mismatch")


def vertical_gauge_generators(state: StateLike) -> tuple[ProductTangent, ...]:
    """Materialize the gauge generators for small dense/reference callers.

    Matrix-free refinement uses :func:`iter_vertical_gauge_generators`
    directly so its memory does not grow as product dimension times gauge
    dimension.  This tuple-returning compatibility API remains convenient for
    tests and explicit dense calculations.
    """

    return tuple(iter_vertical_gauge_generators(state))


def vertical_coordinate_matrix(
    state: StateLike,
    frame: TangentParsevalFrame | None = None,
) -> FloatArray:
    """Return analysis-coordinate columns for all vertical generators."""

    tangent_frame = frame if frame is not None else tangent_parseval_frame(state)
    generators = vertical_gauge_generators(state)
    if not generators:
        return np.zeros((tangent_frame.coefficient_dimension, 0), dtype=float)
    return np.column_stack(
        tuple(tangent_frame.analyze(generator) for generator in generators)
    )


def product_frobenius_inner_product(
    left: DirectionLike, right: DirectionLike
) -> float:
    """Return the sum of Frobenius products across all six parameter blocks."""

    left_blocks = tuple(
        _real_array(block, name=f"left.G_blocks[{index}]")
        for index, block in enumerate(left.G_blocks)
    )
    right_blocks = tuple(
        _real_array(block, name=f"right.G_blocks[{index}]")
        for index, block in enumerate(right.G_blocks)
    )
    if len(left_blocks) != len(right_blocks):
        raise ValueError("directions have different numbers of source-core blocks")
    pairs = (
        (_real_array(left.U0, name="left.U0"), _real_array(right.U0, name="right.U0")),
        (_real_array(left.V0, name="left.V0"), _real_array(right.V0, name="right.V0")),
        (_real_array(left.A, name="left.A"), _real_array(right.A, name="right.A")),
        (_real_array(left.B, name="left.B"), _real_array(right.B, name="right.B")),
        (_real_array(left.H, name="left.H"), _real_array(right.H, name="right.H")),
        *zip(left_blocks, right_blocks, strict=True),
    )
    total = 0.0
    for left_matrix, right_matrix in pairs:
        if left_matrix.shape != right_matrix.shape:
            raise ValueError("direction block shapes do not match")
        total += float(np.vdot(left_matrix, right_matrix).real)
    return total


def product_frobenius_norm(direction: DirectionLike) -> float:
    """Return the product-Frobenius norm of one direction."""

    squared = product_frobenius_inner_product(direction, direction)
    return float(np.sqrt(max(squared, 0.0)))


__all__ = [
    "DirectionLike",
    "FrameBlock",
    "GeometryDimensions",
    "GeometryValidationError",
    "ProductTangent",
    "RegularStateDiagnostics",
    "StateLike",
    "TangentParsevalFrame",
    "expected_quotient_dimension",
    "expected_tangent_dimension",
    "expected_vertical_dimension",
    "geometry_dimensions",
    "iter_vertical_gauge_generators",
    "product_frobenius_inner_product",
    "product_frobenius_norm",
    "tangent_parseval_frame",
    "validate_regular_state",
    "vertical_coordinate_matrix",
    "vertical_gauge_generators",
]
