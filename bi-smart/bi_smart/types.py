"""Shared value objects for the BI-SMART pseudocode package.

The paper describes BI-SMART as a *candidate-generating procedure*: an
algebraic check can make one branch unsuccessful without invalidating the
remaining branches.  The types in this module make that distinction explicit.
In particular, a failed numerical branch is data, represented by
``CandidateStatus.UNSUCCESSFUL`` and a ``FailureReason``; it is not silently
represented by ``None`` or by a matrix containing NaNs.

Indexing convention
-------------------
The paper labels singular directions and blocks from one.  Python code in this
package stores singular-direction indices and block labels from zero.  A
*cut position*, however, is the number of leading directions before a cut, so
the integers ``1, ..., r0 - 1`` agree numerically with the paper's cut labels.
For example, cut positions ``(1, 3)`` at ``r0=5`` produce blocks
``((0,), (1, 2), (3, 4))``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

# NumPy has changed the implicit default cutoff used by ``pinv`` across API
# generations.  BI-SMART therefore names and stores its target-only cutoff so
# the same configuration has the same numerical meaning on every supported
# NumPy release.
DEFAULT_PINV_RCOND = 1e-12


class CandidateStatus(str, Enum):
    """Whether a paper-defined candidate passed all checks in its branch."""

    SUCCESSFUL = "successful"
    UNSUCCESSFUL = "unsuccessful"


class FailureReason(str, Enum):
    """Machine-readable reasons for omitting a BI-SMART branch.

    The values follow the explicit continuation rules in Complete automatic
    BI-SMART (Appendix Algorithm ``alg:bismart-complete``).  Keeping the reason
    separate from a human-readable message lets callers summarize failures
    without parsing exception strings.
    """

    INVALID_INPUT = "invalid_input"
    SOURCE_RANK_BOUNDARY = "source_rank_boundary_failed"
    NON_POSITIVE_DEFINITE = "non_positive_definite"
    NON_UNIQUE_RANK_CUTOFF = "non_unique_rank_cutoff"
    ZERO_RANK_COMPONENT = "zero_rank_component"
    RANK_DEFICIENT_POLAR = "rank_deficient_polar"
    SCREEN_FAILED = "screen_failed"
    WEDIN_GATE = "wedin_gate_failed"
    INVALID_INITIAL_THRESHOLDS = "invalid_initial_thresholds"
    IRREGULAR_ITERATE = "irregular_iterate"
    SINGULAR_NORMAL_EQUATIONS = "singular_normal_equations"
    DENSE_SOLVER_LIMIT = "dense_solver_limit_exceeded"
    LINE_SEARCH = "line_search_failed"
    # Compatibility value for diagnostics serialized by the earlier
    # pseudocode-only scaffold.  The implemented backend no longer emits it.
    NOT_IMPLEMENTED = "not_implemented"


class NumericalFailure(RuntimeError):
    """A recoverable algebraic failure associated with one candidate branch."""

    def __init__(self, reason: FailureReason, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def _float_matrix(value: Any, *, name: str) -> FloatArray:
    """Convert a value to a finite, nonempty two-dimensional float array."""

    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued; complex input is unsupported.")
    array = np.asarray(raw, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix; got shape {array.shape}.")
    if 0 in array.shape:
        raise ValueError(f"{name} must have no empty dimension; got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _real_float(value: Any, *, name: str) -> float:
    """Convert one public scalar without silently dropping an imaginary part."""

    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued; complex input is unsupported.")
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    return float(raw)


def _require_orthonormal_columns(value: FloatArray, *, name: str) -> None:
    """Reject public basis matrices that are not orthonormal to roundoff.

    Source decompositions may be constructed directly by users, rather than
    only through NumPy's SVD.  A dimension-scaled machine-precision guard is
    deliberately used here instead of an application-level modeling
    tolerance: these factors encode an exact Stiefel constraint.
    """

    rank = value.shape[1]
    gram_error = float(
        np.linalg.norm(value.T @ value - np.eye(rank, dtype=float), ord=np.inf)
    )
    tolerance = 64.0 * np.finfo(float).eps * max(value.shape)
    if gram_error > tolerance:
        raise ValueError(
            f"{name} must have orthonormal columns; "
            f"Gram error {gram_error:.6g} exceeds {tolerance:.6g}."
        )


def _nonnegative_int(value: Any, *, name: str) -> int:
    """Validate an integer while rejecting booleans, which are ``int`` subclasses."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative; got {result}.")
    return result


@dataclass(frozen=True)
class FoldData:
    """One target-data fold used by the three-fold appendix procedure.

    ``X`` has shape ``(n, p)`` and ``Y`` has shape ``(n, q)``.  The
    initialization, fitting, and validation folds should be constructed before
    this object reaches the algorithm; this class checks matrix compatibility,
    not statistical independence.
    """

    X: FloatArray
    Y: FloatArray
    name: str = "fold"

    def __post_init__(self) -> None:
        # PSEUDOCODE 1: Convert both inputs to finite numeric matrices.
        X = _float_matrix(self.X, name=f"{self.name}.X")
        Y = _float_matrix(self.Y, name=f"{self.name}.Y")

        # PSEUDOCODE 2: A fold pairs rows of X and Y observation by observation.
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"{self.name}.X and {self.name}.Y must have the same number of rows; "
                f"got {X.shape[0]} and {Y.shape[0]}."
            )

        # PSEUDOCODE 3: Store normalized arrays for all downstream calculations.
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "Y", Y)

    @property
    def n_samples(self) -> int:
        """Number of observations in this fold."""

        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Number of target predictors."""

        return int(self.X.shape[1])

    @property
    def n_responses(self) -> int:
        """Number of target responses."""

        return int(self.Y.shape[1])


@dataclass(frozen=True)
class BlockPartition:
    """A partition of ``range(r0)`` into nonempty consecutive blocks.

    The tuple representation is deliberately immutable.  BI-SMART must retain
    the selected *block labels*, rather than reconstructing them later from
    numerically nonzero rows of a factor.
    """

    blocks: Tuple[Tuple[int, ...], ...]

    def __post_init__(self) -> None:
        # PSEUDOCODE 1: Normalize every supplied block to a tuple of Python ints.
        normalized = tuple(
            tuple(_nonnegative_int(index, name="block index") for index in block)
            for block in self.blocks
        )
        if not normalized:
            raise ValueError("A block partition must contain at least one block.")

        # PSEUDOCODE 2: Verify exact consecutive coverage starting at direction 0.
        expected_start = 0
        for label, block in enumerate(normalized):
            if not block:
                raise ValueError(f"Block {label} is empty.")
            expected = tuple(range(expected_start, expected_start + len(block)))
            if block != expected:
                raise ValueError(
                    "Blocks must be ordered, consecutive, disjoint, and start at zero; "
                    f"block {label} is {block}, expected {expected}."
                )
            expected_start += len(block)

        # PSEUDOCODE 3: Freeze the normalized indexing convention.
        object.__setattr__(self, "blocks", normalized)

    @classmethod
    def from_cut_positions(cls, rank: int, cuts: Sequence[int]) -> "BlockPartition":
        """Construct consecutive blocks from boundary positions.

        A cut ``j`` lies between zero-based directions ``j-1`` and ``j`` and
        therefore corresponds numerically to index ``j`` in
        Eq. ``(bismart-certified-cuts)``.
        """

        # PSEUDOCODE 1: Validate r0 and canonicalize the set of internal cuts.
        rank = _nonnegative_int(rank, name="rank")
        if rank < 1:
            raise ValueError("rank must be at least one.")
        normalized_cuts = tuple(
            sorted(_nonnegative_int(cut, name="cut position") for cut in cuts)
        )
        if len(set(normalized_cuts)) != len(normalized_cuts):
            raise ValueError("Cut positions must be unique.")
        if any(cut <= 0 or cut >= rank for cut in normalized_cuts):
            raise ValueError(f"Cut positions must lie in [1, {rank - 1}].")

        # PSEUDOCODE 2: Convert successive boundaries into half-open Python ranges.
        boundaries = (0,) + normalized_cuts + (rank,)
        blocks = tuple(
            tuple(range(left, right))
            for left, right in zip(boundaries[:-1], boundaries[1:])
        )

        # PSEUDOCODE 3: Delegate the coverage invariant to the value object's validator.
        return cls(blocks=blocks)

    @property
    def rank(self) -> int:
        """Number ``r0`` of source directions covered by the partition."""

        return sum(len(block) for block in self.blocks)

    @property
    def n_blocks(self) -> int:
        """Number of blocks in the partition."""

        return len(self.blocks)

    @property
    def block_sizes(self) -> Tuple[int, ...]:
        """Dimensions ``m_l`` of the blocks."""

        return tuple(len(block) for block in self.blocks)

    @property
    def cut_positions(self) -> Tuple[int, ...]:
        """Internal boundary positions in the paper-compatible convention."""

        cumulative = np.cumsum(self.block_sizes, dtype=int)
        return tuple(int(value) for value in cumulative[:-1])

    def union_indices(self, block_labels: Sequence[int]) -> Tuple[int, ...]:
        """Return source-direction indices in a selected union of blocks."""

        # PSEUDOCODE 1: Treat a selected support as a mathematical set of labels.
        labels = tuple(
            sorted(_nonnegative_int(label, name="block label") for label in block_labels)
        )
        if len(labels) != len(set(labels)):
            raise ValueError("Selected block labels must be unique.")
        if any(label < 0 or label >= self.n_blocks for label in labels):
            raise ValueError(f"Block labels must lie in [0, {self.n_blocks - 1}].")

        # PSEUDOCODE 2: Preserve partition order when concatenating directions.
        return tuple(index for label in labels for index in self.blocks[label])

    def union_dimension(self, block_labels: Sequence[int]) -> int:
        """Dimension ``k(J)`` of a selected block union."""

        return len(self.union_indices(block_labels))


@dataclass(frozen=True)
class TruncatedSVD:
    """Economy rank truncation together with the first excluded singular value."""

    u: FloatArray
    singular_values: FloatArray
    vt: FloatArray
    next_singular_value: float

    def __post_init__(self) -> None:
        u = _float_matrix(self.u, name="u")
        raw_values = np.asarray(self.singular_values)
        if np.iscomplexobj(raw_values):
            raise ValueError(
                "singular_values must be real-valued; complex input is unsupported."
            )
        values = np.asarray(raw_values, dtype=float)
        vt = _float_matrix(self.vt, name="vt")
        if values.ndim != 1 or values.size < 1:
            raise ValueError("singular_values must be a nonempty vector.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("singular_values must be finite and nonnegative.")
        if np.any(values[:-1] < values[1:]):
            raise ValueError("singular_values must be in nonincreasing order.")
        rank = values.size
        if u.shape[1] != rank or vt.shape[0] != rank:
            raise ValueError("u, singular_values, and vt have incompatible ranks.")
        next_value = _real_float(
            self.next_singular_value,
            name="next_singular_value",
        )
        if not np.isfinite(next_value) or next_value < 0:
            raise ValueError("next_singular_value must be finite and nonnegative.")
        if next_value > float(values[-1]):
            raise ValueError(
                "next_singular_value cannot exceed the smallest retained singular value."
            )
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "singular_values", values)
        object.__setattr__(self, "vt", vt)
        object.__setattr__(self, "next_singular_value", next_value)

    @property
    def rank(self) -> int:
        """Retained rank."""

        return int(self.singular_values.size)

    @property
    def approximation(self) -> FloatArray:
        """Reconstruct ``U diag(s) V.T`` without forming a dense diagonal matrix."""

        return (self.u * self.singular_values) @ self.vt


@dataclass(frozen=True)
class SourceDecomposition:
    """Leading observed-source geometry and its current coordinate core.

    The automatic constructor stores the diagonal SVD core.  ``coordinate_core``
    also permits the same source approximation to be expressed after
    independent left/right orthogonal basis changes inside tied blocks; in
    those coordinates the core is full within each block, as required by the
    block-invariant appendix parameterization.
    """

    u: FloatArray
    singular_values: FloatArray
    vt: FloatArray
    next_singular_value: float
    source_shape: Tuple[int, int]
    coordinate_core: Optional[FloatArray] = None

    def __post_init__(self) -> None:
        truncated = TruncatedSVD(
            u=self.u,
            singular_values=self.singular_values,
            vt=self.vt,
            next_singular_value=self.next_singular_value,
        )
        shape = tuple(int(value) for value in self.source_shape)
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError(f"source_shape must contain two positive dimensions; got {shape}.")
        if truncated.u.shape[0] != shape[0] or truncated.vt.shape[1] != shape[1]:
            raise ValueError("Source singular vectors are incompatible with source_shape.")
        if truncated.rank > min(shape):
            raise ValueError("The retained source rank exceeds the matrix dimensions.")
        _require_orthonormal_columns(truncated.u, name="u")
        _require_orthonormal_columns(truncated.vt.T, name="vt.T")
        if self.coordinate_core is None:
            core = np.diag(truncated.singular_values)
        else:
            core = _float_matrix(self.coordinate_core, name="coordinate_core")
            if core.shape != (truncated.rank, truncated.rank):
                raise ValueError(
                    "coordinate_core must have shape "
                    f"({truncated.rank}, {truncated.rank})."
                )
            # Independent orthogonal basis changes preserve the singular values
            # of the core.  Enforce that invariant so the certified gaps remain
            # meaningful even when the core is no longer diagonal.
            core_singular_values = np.linalg.svd(core, compute_uv=False)
            scale = max(float(truncated.singular_values[0]), 1.0)
            if not np.allclose(
                core_singular_values,
                truncated.singular_values,
                atol=1e-12 * scale,
                rtol=1e-10,
            ):
                raise ValueError(
                    "coordinate_core singular values must equal singular_values."
                )
        object.__setattr__(self, "u", truncated.u)
        object.__setattr__(self, "singular_values", truncated.singular_values)
        object.__setattr__(self, "vt", truncated.vt)
        object.__setattr__(self, "next_singular_value", truncated.next_singular_value)
        object.__setattr__(self, "source_shape", shape)
        object.__setattr__(self, "coordinate_core", core)

    @property
    def rank(self) -> int:
        """Working source rank ``r0``."""

        return int(self.singular_values.size)

    @property
    def left_vectors(self) -> FloatArray:
        """Alias for the paper's ``U_hat_0``."""

        return self.u

    @property
    def right_vectors(self) -> FloatArray:
        """Alias for the paper's ``V_hat_0``."""

        return self.vt.T

    @property
    def diagonal(self) -> FloatArray:
        """Diagonal matrix ``D_hat_0``."""

        return np.diag(self.singular_values)

    @property
    def approximation(self) -> FloatArray:
        """Leading source approximation in the stored left/right bases."""

        assert self.coordinate_core is not None
        return self.u @ self.coordinate_core @ self.vt


@dataclass(frozen=True)
class SourceBlockLibrary:
    """Certified source decomposition and nested automatic partitions."""

    decomposition: SourceDecomposition
    source_error_bound: float
    c_gap: float
    boundary_gap: float
    boundary_passes: bool
    certified_cuts: Tuple[int, ...]
    observed_internal_gaps: FloatArray
    partitions: Tuple[BlockPartition, ...]
    atol: float = 1e-12
    rtol: float = 1e-10

    def __post_init__(self) -> None:
        epsilon = _real_float(self.source_error_bound, name="source_error_bound")
        c_gap = _real_float(self.c_gap, name="c_gap")
        boundary_gap = _real_float(self.boundary_gap, name="boundary_gap")
        raw_gaps = np.asarray(self.observed_internal_gaps)
        if np.iscomplexobj(raw_gaps):
            raise ValueError(
                "observed_internal_gaps must be real-valued; "
                "complex input is unsupported."
            )
        gaps = np.asarray(raw_gaps, dtype=float)
        cuts = tuple(
            _nonnegative_int(cut, name="certified cut") for cut in self.certified_cuts
        )
        partitions = tuple(self.partitions)
        atol = _real_float(self.atol, name="atol")
        rtol = _real_float(self.rtol, name="rtol")

        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("source_error_bound must be finite and nonnegative.")
        if not np.isfinite(c_gap) or c_gap <= 2:
            raise ValueError("c_gap must be finite and strictly larger than 2.")
        if not np.isfinite(atol) or atol < 0 or not np.isfinite(rtol) or rtol < 0:
            raise ValueError("atol and rtol must be finite and nonnegative.")
        if not np.isfinite(boundary_gap):
            raise ValueError("boundary_gap must be finite.")
        expected_gaps = max(self.decomposition.rank - 1, 0)
        if gaps.ndim != 1 or gaps.size != expected_gaps:
            raise ValueError(
                f"observed_internal_gaps must have length {expected_gaps}; got {gaps.shape}."
            )
        if np.any(~np.isfinite(gaps)) or np.any(gaps < 0):
            raise ValueError("observed_internal_gaps must be finite and nonnegative.")
        if any(cut <= 0 or cut >= self.decomposition.rank for cut in cuts):
            raise ValueError("certified_cuts contain an invalid boundary position.")
        if len(set(cuts)) != len(cuts):
            raise ValueError("certified_cuts must be unique.")
        if not partitions:
            raise ValueError("The partition library must contain at least one partition.")
        if any(partition.rank != self.decomposition.rank for partition in partitions):
            raise ValueError("Every partition must cover the retained source rank.")
        if partitions[0].cut_positions != tuple(sorted(cuts)):
            raise ValueError("The first partition must be the finest certified partition.")
        if partitions[-1].n_blocks != 1:
            raise ValueError("The last partition must be the one-block partition.")
        if len(partitions) != len(cuts) + 1:
            raise ValueError("The nested library must contain one partition per removed cut.")
        for finer, coarser in zip(partitions[:-1], partitions[1:]):
            finer_cuts = set(finer.cut_positions)
            coarser_cuts = set(coarser.cut_positions)
            if not coarser_cuts < finer_cuts or len(finer_cuts - coarser_cuts) != 1:
                raise ValueError("Each successive partition must remove exactly one cut.")

        threshold = c_gap * epsilon
        retained = float(self.decomposition.singular_values[-1])
        excluded = float(self.decomposition.next_singular_value)
        boundary_scale = max(
            abs(retained),
            abs(excluded),
            abs(boundary_gap),
            abs(threshold),
        )
        expected_boundary = boundary_gap - threshold > atol + rtol * boundary_scale
        if bool(self.boundary_passes) != expected_boundary:
            raise ValueError(
                "boundary_passes is inconsistent with the tolerance-aware "
                "strict boundary check."
            )
        values = self.decomposition.singular_values
        expected_cuts = tuple(
            position
            for position, gap in enumerate(gaps, start=1)
            if float(gap) - threshold
            > atol
            + rtol
            * max(
                abs(float(values[position - 1])),
                abs(float(values[position])),
                abs(float(gap)),
                abs(threshold),
            )
        )
        if cuts != expected_cuts:
            raise ValueError("certified_cuts are inconsistent with the observed-gap rule.")

        object.__setattr__(self, "source_error_bound", epsilon)
        object.__setattr__(self, "c_gap", c_gap)
        object.__setattr__(self, "boundary_gap", boundary_gap)
        object.__setattr__(self, "boundary_passes", bool(self.boundary_passes))
        object.__setattr__(self, "certified_cuts", cuts)
        object.__setattr__(self, "observed_internal_gaps", gaps)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "rtol", rtol)


@dataclass(frozen=True)
class ScreenResult:
    """Output of one appendix block-screen branch.

    Array fields are optional because a failed branch may stop before producing
    them.  A successful result must provide all fitted objects and both selected
    supports.
    """

    status: CandidateStatus
    partition: BlockPartition
    left_blocks: Tuple[int, ...] = ()
    right_blocks: Tuple[int, ...] = ()
    statistic: Optional[FloatArray] = None
    left_factor: Optional[FloatArray] = None
    right_factor: Optional[FloatArray] = None
    pilot_matrix: Optional[FloatArray] = None
    frozen_matrix: Optional[FloatArray] = None
    failure_reason: Optional[FailureReason] = None
    message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, CandidateStatus):
            raise TypeError("status must be a CandidateStatus.")
        left = tuple(
            _nonnegative_int(label, name="left block label") for label in self.left_blocks
        )
        right = tuple(
            _nonnegative_int(label, name="right block label") for label in self.right_blocks
        )
        self.partition.union_indices(left)
        self.partition.union_indices(right)

        if self.status is CandidateStatus.SUCCESSFUL:
            required = {
                "statistic": self.statistic,
                "left_factor": self.left_factor,
                "right_factor": self.right_factor,
                "pilot_matrix": self.pilot_matrix,
                "frozen_matrix": self.frozen_matrix,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"Successful screen result is missing: {', '.join(missing)}.")
            if self.failure_reason is not None:
                raise ValueError("A successful screen result cannot have a failure_reason.")
        elif self.failure_reason is None:
            raise ValueError("An unsuccessful screen result must provide a failure_reason.")

        # Publicly constructed diagnostics must obey the same real-valued
        # contract as matrices entering the numerical routines.  Normalize
        # every populated field, including optional arrays on failed branches.
        for field_name in (
            "statistic",
            "left_factor",
            "right_factor",
            "pilot_matrix",
            "frozen_matrix",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _float_matrix(value, name=f"screen.{field_name}"),
                )

        object.__setattr__(self, "left_blocks", left)
        object.__setattr__(self, "right_blocks", right)


@dataclass(frozen=True)
class Candidate:
    """One deterministically labelled member of the validation library."""

    label: str
    status: CandidateStatus
    matrix: Optional[FloatArray] = None
    order_key: Tuple[Any, ...] = ()
    kind: str = "unspecified"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    failure_reason: Optional[FailureReason] = None
    message: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Candidate label must be nonempty.")
        if not isinstance(self.status, CandidateStatus):
            raise TypeError("status must be a CandidateStatus.")
        matrix = None if self.matrix is None else _float_matrix(self.matrix, name="candidate.matrix")
        if self.status is CandidateStatus.SUCCESSFUL:
            if matrix is None:
                raise ValueError("A successful candidate must contain a coefficient matrix.")
            if self.failure_reason is not None:
                raise ValueError("A successful candidate cannot have a failure_reason.")
        else:
            if matrix is not None:
                raise ValueError("An unsuccessful candidate cannot contain a coefficient matrix.")
            if self.failure_reason is None:
                raise ValueError("An unsuccessful candidate must provide a failure_reason.")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "order_key", tuple(self.order_key))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def successful(
        cls,
        *,
        label: str,
        matrix: FloatArray,
        order_key: Sequence[Any] = (),
        kind: str = "unspecified",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "Candidate":
        """Construct a successful candidate without repeating status bookkeeping."""

        return cls(
            label=label,
            status=CandidateStatus.SUCCESSFUL,
            matrix=matrix,
            order_key=tuple(order_key),
            kind=kind,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def unsuccessful(
        cls,
        *,
        label: str,
        reason: FailureReason,
        message: str,
        order_key: Sequence[Any] = (),
        kind: str = "unspecified",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "Candidate":
        """Construct a failed branch record that validation will omit."""

        return cls(
            label=label,
            status=CandidateStatus.UNSUCCESSFUL,
            matrix=None,
            order_key=tuple(order_key),
            kind=kind,
            metadata={} if metadata is None else metadata,
            failure_reason=reason,
            message=message,
        )


@dataclass(frozen=True)
class RefinementControls:
    """Fixed numerical controls for Appendix Algorithms 1 and 2.

    The manuscript intentionally leaves these values to the caller.  Keeping
    them in a separate value object lets the exact screen/RRR scaffold run
    without inventing optimization defaults, while still making the complete
    calibration grid representable when refinement is requested.

    The field names correspond to the paper as follows:

    ``armijo_constant = c_A``, ``contraction = beta``,
    ``initial_step_size = bar_eta``, ``radius_ratio = q_rho``,
    ``radius_half_width = J_rho``,
    ``max_backtracking_cap = B_cal``, and ``iteration_cap = T``.

    ``max_dense_work_bytes``, ``gauss_newton_backend``, and
    ``matrix_free_max_iterations`` are numerical process controls rather than
    statistical or manuscript tuning constants.  The default ``"dense"``
    backend retains the appendix's strict full-Jacobian numerical-rank check.
    ``"matrix_free"`` is an explicit scalable LSQR mode, and ``"auto"`` opts
    into that mode when the dense workspace would exceed the cap.  The
    matrix-free mode checks algebraic injectivity, necessary compact numerical
    screens, and solve-specific postconditions; it cannot claim every
    near-rank-deficiency decision of the full dense SVD.

    Core and block-separation thresholds use the computable rule in
    ``(bismart-default-thresholds)``.  A future alternative threshold rule
    should be represented explicitly rather than hidden in this object.
    """

    armijo_constant: float
    contraction: float
    initial_step_size: float
    radius_ratio: float
    radius_half_width: int
    max_backtracking_cap: int
    iteration_cap: int
    max_dense_work_bytes: int = 512 * 1024**2
    gauss_newton_backend: str = "dense"
    matrix_free_max_iterations: Optional[int] = None

    def __post_init__(self) -> None:
        # PSEUDOCODE 1: Validate the Armijo and geometric-backtracking
        # constants before any branch-specific state has been constructed.
        armijo = _real_float(self.armijo_constant, name="armijo_constant")
        contraction = _real_float(self.contraction, name="contraction")
        initial_step = _real_float(self.initial_step_size, name="initial_step_size")
        radius_ratio = _real_float(self.radius_ratio, name="radius_ratio")
        if not np.isfinite(armijo) or not 0.0 < armijo < 1.0:
            raise ValueError("armijo_constant must lie strictly between zero and one.")
        if not np.isfinite(contraction) or not 0.0 < contraction < 1.0:
            raise ValueError("contraction must lie strictly between zero and one.")
        if not np.isfinite(initial_step) or initial_step <= 0.0:
            raise ValueError("initial_step_size must be finite and strictly positive.")
        if not np.isfinite(radius_ratio) or radius_ratio <= 1.0:
            raise ValueError("radius_ratio must be finite and strictly larger than one.")

        # PSEUDOCODE 2: These integer caps make every appendix loop finite.
        radius_half_width = _nonnegative_int(
            self.radius_half_width,
            name="radius_half_width",
        )
        backtracking_cap = _nonnegative_int(
            self.max_backtracking_cap,
            name="max_backtracking_cap",
        )
        iteration_cap = _nonnegative_int(
            self.iteration_cap,
            name="iteration_cap",
        )
        max_dense_work_bytes = _nonnegative_int(
            self.max_dense_work_bytes,
            name="max_dense_work_bytes",
        )
        if backtracking_cap < 1 or iteration_cap < 1:
            raise ValueError(
                "max_backtracking_cap and iteration_cap must both be positive."
            )
        if max_dense_work_bytes < 1:
            raise ValueError("max_dense_work_bytes must be positive.")

        # PSEUDOCODE 3: Select the linear-algebra implementation explicitly.
        # This does not alter the appendix objective or candidate ordering.
        if not isinstance(self.gauss_newton_backend, str):
            raise TypeError("gauss_newton_backend must be a string.")
        backend = self.gauss_newton_backend.strip().lower().replace("-", "_")
        if backend not in {"auto", "dense", "matrix_free"}:
            raise ValueError(
                "gauss_newton_backend must be 'auto', 'dense', or 'matrix_free'."
            )
        iterative_cap = self.matrix_free_max_iterations
        if iterative_cap is not None:
            iterative_cap = _nonnegative_int(
                iterative_cap,
                name="matrix_free_max_iterations",
            )
            if iterative_cap < 1:
                raise ValueError("matrix_free_max_iterations must be positive.")

        object.__setattr__(self, "armijo_constant", armijo)
        object.__setattr__(self, "contraction", contraction)
        object.__setattr__(self, "initial_step_size", initial_step)
        object.__setattr__(self, "radius_ratio", radius_ratio)
        object.__setattr__(self, "radius_half_width", radius_half_width)
        object.__setattr__(self, "max_backtracking_cap", backtracking_cap)
        object.__setattr__(self, "iteration_cap", iteration_cap)
        object.__setattr__(self, "max_dense_work_bytes", max_dense_work_bytes)
        object.__setattr__(self, "gauss_newton_backend", backend)
        object.__setattr__(self, "matrix_free_max_iterations", iterative_cap)


@dataclass(frozen=True)
class BISMARTConfig:
    """Validated structural inputs shared by the complete appendix procedure.

    The manuscript does not prescribe numerical defaults for the quotient
    Gauss--Newton line search or trust-grid calibration.  ``refinement_controls``
    is therefore optional for exact-stage runs and must be supplied explicitly
    when ``BISMART.fit_folds(..., enable_refinement=True)`` is requested.
    ``pinv_rcond`` is the reproducible cutoff used only by the mandatory
    target-only pseudoinverse safeguard.
    """

    target_rank: int
    source_rank: int
    source_error_bound: float
    budget_path: Tuple[Tuple[int, int], ...]
    source_weights: Tuple[float, ...]
    refinement_controls: Optional[RefinementControls] = None
    c_gap: float = 3.0
    atol: float = 1e-12
    rtol: float = 1e-10
    pinv_rcond: float = DEFAULT_PINV_RCOND

    def __post_init__(self) -> None:
        # PSEUDOCODE 1: Validate ranks and the source-error calibration.
        r = _nonnegative_int(self.target_rank, name="target_rank")
        r0 = _nonnegative_int(self.source_rank, name="source_rank")
        if r < 1 or r0 < 1 or r > r0:
            raise ValueError("Ranks must satisfy 1 <= target_rank <= source_rank.")
        epsilon = _real_float(self.source_error_bound, name="source_error_bound")
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("source_error_bound must be finite and nonnegative.")

        # PSEUDOCODE 2: Canonicalize the finite, deterministically ordered grids.
        budgets = tuple(
            (
                _nonnegative_int(ku, name="left budget"),
                _nonnegative_int(kv, name="right budget"),
            )
            for ku, kv in self.budget_path
        )
        if not budgets:
            raise ValueError("budget_path must contain at least one pair.")
        if any(ku < r or kv < r or ku > r0 or kv > r0 for ku, kv in budgets):
            raise ValueError("Every budget pair must lie in [target_rank, source_rank]^2.")
        if len(set(budgets)) != len(budgets):
            raise ValueError("budget_path must not contain duplicate budget pairs.")
        weights = tuple(
            _real_float(weight, name="source weight")
            for weight in self.source_weights
        )
        if not weights or any(not np.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("source_weights must be a nonempty sequence of positive finite values.")
        if len(set(weights)) != len(weights):
            raise ValueError("source_weights must not contain duplicate values.")
        controls = self.refinement_controls
        if controls is not None and not isinstance(controls, RefinementControls):
            raise TypeError("refinement_controls must be RefinementControls or None.")

        # PSEUDOCODE 3: Validate the fixed gap constant and numerical tolerances.
        pinv_rcond = _real_float(self.pinv_rcond, name="pinv_rcond")
        c_gap = _real_float(self.c_gap, name="c_gap")
        atol = _real_float(self.atol, name="atol")
        rtol = _real_float(self.rtol, name="rtol")
        if not np.isfinite(pinv_rcond) or pinv_rcond < 0:
            raise ValueError("pinv_rcond must be finite and nonnegative.")
        if not np.isfinite(c_gap) or c_gap <= 2:
            raise ValueError("c_gap must be finite and strictly larger than 2.")
        if not np.isfinite(atol) or atol < 0 or not np.isfinite(rtol) or rtol < 0:
            raise ValueError("atol and rtol must be finite and nonnegative.")

        object.__setattr__(self, "target_rank", r)
        object.__setattr__(self, "source_rank", r0)
        object.__setattr__(self, "source_error_bound", epsilon)
        object.__setattr__(self, "budget_path", budgets)
        object.__setattr__(self, "source_weights", weights)
        object.__setattr__(self, "refinement_controls", controls)
        object.__setattr__(self, "pinv_rcond", pinv_rcond)
        object.__setattr__(self, "c_gap", c_gap)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "rtol", rtol)


@dataclass(frozen=True)
class BISMARTResult:
    """Selected target estimate plus the complete auditable candidate library."""

    selected_candidate: Candidate
    candidates: Tuple[Candidate, ...]
    source_library: Optional[SourceBlockLibrary] = None

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if self.selected_candidate.status is not CandidateStatus.SUCCESSFUL:
            raise ValueError("selected_candidate must be successful.")
        if not candidates:
            raise ValueError("candidates must be nonempty.")
        selected_key = (
            self.selected_candidate.label,
            self.selected_candidate.order_key,
        )
        if not any(
            candidate is self.selected_candidate
            or (candidate.label, candidate.order_key) == selected_key
            for candidate in candidates
        ):
            raise ValueError("selected_candidate must be retained in candidates.")
        object.__setattr__(self, "candidates", candidates)

    @property
    def coefficient(self) -> FloatArray:
        """Selected target coefficient matrix ``C_hat``."""

        # The post-init invariant guarantees that matrix is present.
        assert self.selected_candidate.matrix is not None
        return self.selected_candidate.matrix


__all__ = [
    "BISMARTConfig",
    "BISMARTResult",
    "BlockPartition",
    "Candidate",
    "CandidateStatus",
    "DEFAULT_PINV_RCOND",
    "FailureReason",
    "FloatArray",
    "FoldData",
    "NumericalFailure",
    "RefinementControls",
    "ScreenResult",
    "SourceBlockLibrary",
    "SourceDecomposition",
    "TruncatedSVD",
]
