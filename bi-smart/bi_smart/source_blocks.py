"""Observed-source decomposition and automatic BI-SMART block partitions.

This module implements the executable source-only part of the appendix:

1. Eq. ``(bismart-observed-source-svd)`` computes the leading source SVD.
2. Eq. ``(bismart-rank-boundary-check)`` certifies that truncation at ``r0``
   does not cut an unresolved source block.
3. Eqs. ``(bismart-certified-cuts)``--``(bismart-partition-library)`` build a
   deterministic nested library from finest to coarsest.
4. Eqs. ``(bismart-observable-block-gap)``--``(bismart-wedin-gate)`` decide
   whether RRR/refinement is allowed for one partition.

The boundary and Wedin checks intentionally return diagnostics instead of
raising by default.  The complete algorithm uses a failed check as a local
continuation rule: it omits source-guided candidates, or retains pilot/frozen
candidates while skipping RRR and refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np

from .linalg import truncated_svd, validate_matrix
from .types import (
    BlockPartition,
    FailureReason,
    NumericalFailure,
    SourceBlockLibrary,
    SourceDecomposition,
)


_DEFAULT_ATOL = 1e-12
_DEFAULT_RTOL = 1e-10


def _real_scalar(value: Any, *, name: str) -> float:
    """Normalize a public scalar without dropping an imaginary component."""

    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued; complex input is unsupported.")
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    return float(raw)


def _validate_tolerances(atol: float, rtol: float) -> Tuple[float, float]:
    """Normalize nonnegative tolerances used by source spectral decisions."""

    absolute = _real_scalar(atol, name="atol")
    relative = _real_scalar(rtol, name="rtol")
    if (
        not np.isfinite(absolute)
        or absolute < 0
        or not np.isfinite(relative)
        or relative < 0
    ):
        raise ValueError("atol and rtol must be finite and nonnegative.")
    return absolute, relative


def _strict_gap_passes(
    gap: float,
    threshold: float,
    *,
    spectral_scale: float,
    atol: float,
    rtol: float,
) -> bool:
    """Return a conservative numerical version of ``gap > threshold``.

    A gap is a subtraction of two singular values.  Scaling its roundoff guard
    only by the (possibly almost-zero) gap would therefore miss cancellation.
    The guard is instead relative to the singular values that produced it.
    Equality and values within the guard fail, preserving the manuscript's
    strict branch semantics without certifying a cut created by roundoff.
    """

    scale = max(
        abs(float(spectral_scale)),
        abs(float(gap)),
        abs(float(threshold)),
    )
    margin = float(atol) + float(rtol) * scale
    return float(gap) - float(threshold) > margin


def _nonstrict_gap_passes(
    gap: float,
    threshold: float,
    *,
    spectral_scale: float,
    atol: float,
    rtol: float,
) -> bool:
    """Return a tolerance-aware version of ``gap >= threshold``.

    Unlike the strict source-boundary rules above, equality belongs to the
    Wedin gate's acceptance region.  A computed gap that lies below the
    threshold only by its absolute/relative roundoff guard is therefore
    treated as equality.  ``spectral_scale`` is the scale of the singular
    values whose subtraction produced the active observable lower bound.
    """

    scale = max(
        abs(float(spectral_scale)),
        abs(float(gap)),
        abs(float(threshold)),
    )
    margin = float(atol) + float(rtol) * scale
    return float(gap) + margin >= float(threshold)


def _nonnegative_integer(value: Any, *, name: str) -> int:
    """Normalize an integer index while rejecting booleans and fractional values."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative; got {result}.")
    return result


def _validate_error_bound(source_error_bound: float) -> float:
    """Validate the externally calibrated source operator-error bound."""

    epsilon = _real_scalar(source_error_bound, name="source_error_bound")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("source_error_bound must be finite and nonnegative.")
    return epsilon


def _source_calibration(
    source_error_bound: float,
    c_gap: float,
    *,
    atol: float = _DEFAULT_ATOL,
    rtol: float = _DEFAULT_RTOL,
) -> Tuple[float, float, float, float]:
    """Validate the two scalars used by every source certification rule."""

    epsilon = _validate_error_bound(source_error_bound)
    gap_constant = _real_scalar(c_gap, name="c_gap")
    if not np.isfinite(gap_constant) or gap_constant <= 2:
        raise ValueError("c_gap must be finite and strictly larger than 2.")
    absolute, relative = _validate_tolerances(atol, rtol)
    return epsilon, gap_constant, absolute, relative


def gaussian_source_error_bound(
    noise_standard_deviation: float,
    n_rows: int,
    n_columns: int,
    tail_parameter: float,
) -> float:
    """Return the Gaussian source-error certificate from the appendix.

    For independent ``N(0, tau0**2)`` entries and ``x0 > 0``, Eq.
    ``(bismart-source-certificate-gaussian)`` uses

    ``tau0 * (sqrt(p) + sqrt(q) + sqrt(2*x0))``.

    The function performs the arithmetic only.  Supplying this certificate is
    a modeling decision: it must not be estimated from target responses.
    """

    # PSEUDOCODE 1: Validate the externally supplied noise scale and tail level.
    tau0 = _real_scalar(
        noise_standard_deviation,
        name="noise_standard_deviation",
    )
    x0 = _real_scalar(tail_parameter, name="tail_parameter")
    if not np.isfinite(tau0) or tau0 < 0:
        raise ValueError("noise_standard_deviation must be finite and nonnegative.")
    if not np.isfinite(x0) or x0 <= 0:
        raise ValueError("tail_parameter must be finite and strictly positive.")
    if isinstance(n_rows, (bool, np.bool_)) or not isinstance(n_rows, (int, np.integer)):
        raise TypeError("n_rows must be an integer.")
    if isinstance(n_columns, (bool, np.bool_)) or not isinstance(
        n_columns, (int, np.integer)
    ):
        raise TypeError("n_columns must be an integer.")
    p, q = int(n_rows), int(n_columns)
    if p < 1 or q < 1:
        raise ValueError("n_rows and n_columns must be positive.")

    # PSEUDOCODE 2: Evaluate the unitarily invariant, source-only certificate.
    return tau0 * (np.sqrt(p) + np.sqrt(q) + np.sqrt(2.0 * x0))


def compute_source_decomposition(
    observed_source: Any,
    source_rank: int,
) -> SourceDecomposition:
    """Compute Eq. ``(bismart-observed-source-svd)`` at a supplied ``r0``.

    ``r0`` remains an explicit input because the appendix's complete algorithm
    takes it as input.  Automatic selection of ``r0`` is not specified well
    enough in the manuscript to invent here.
    """

    # PSEUDOCODE 1: Validate the observed p-by-q source coefficient matrix.
    source = validate_matrix(observed_source, name="observed_source")

    # PSEUDOCODE 2: Compute the leading r0 factors and sigma_(r0+1).
    truncated = truncated_svd(source, source_rank)

    # PSEUDOCODE 3: Package the source geometry without assigning meaning to
    # individual singular vectors inside tied retained blocks.
    return SourceDecomposition(
        u=truncated.u,
        singular_values=truncated.singular_values,
        vt=truncated.vt,
        next_singular_value=truncated.next_singular_value,
        source_shape=source.shape,
    )


# A short alias is useful in formulas and interactive work.
decompose_source = compute_source_decomposition


def source_rank_boundary_gap(decomposition: SourceDecomposition) -> float:
    """Return ``d_hat[r0] - d_hat[r0+1]`` in paper notation."""

    return float(decomposition.singular_values[-1] - decomposition.next_singular_value)


def source_rank_boundary_passes(
    decomposition: SourceDecomposition,
    source_error_bound: float,
    *,
    c_gap: float = 3.0,
    atol: float = _DEFAULT_ATOL,
    rtol: float = _DEFAULT_RTOL,
) -> bool:
    """Evaluate Eq. ``(bismart-rank-boundary-check)`` conservatively.

    The inequality is strict.  Equality and a tolerance-sized neighborhood of
    equality therefore fail.  The relative tolerance is scaled by the two
    singular values forming the boundary, so an ``epsilon0=0`` tie cannot pass
    merely because its numerical SVD representatives differ by roundoff.
    """

    # PSEUDOCODE 1: Validate the source-only calibration constants.
    epsilon, gap_constant, absolute, relative = _source_calibration(
        source_error_bound,
        c_gap,
        atol=atol,
        rtol=rtol,
    )

    # PSEUDOCODE 2: Compare the observed truncation gap with c_gap*epsilon_hat,
    # requiring clearance beyond numerical uncertainty in the two boundary
    # singular values.
    threshold = gap_constant * epsilon
    retained = float(decomposition.singular_values[-1])
    excluded = float(decomposition.next_singular_value)
    return _strict_gap_passes(
        retained - excluded,
        threshold,
        spectral_scale=max(abs(retained), abs(excluded)),
        atol=absolute,
        rtol=relative,
    )


def require_source_rank_boundary(
    decomposition: SourceDecomposition,
    source_error_bound: float,
    *,
    c_gap: float = 3.0,
    atol: float = _DEFAULT_ATOL,
    rtol: float = _DEFAULT_RTOL,
) -> None:
    """Raise a recoverable branch failure unless the rank boundary passes."""

    if not source_rank_boundary_passes(
        decomposition,
        source_error_bound,
        c_gap=c_gap,
        atol=atol,
        rtol=rtol,
    ):
        gap = source_rank_boundary_gap(decomposition)
        threshold = float(c_gap) * float(source_error_bound)
        raise NumericalFailure(
            FailureReason.SOURCE_RANK_BOUNDARY,
            f"Source rank boundary is not certified: gap={gap:.6g}, "
            f"required strict threshold={threshold:.6g}.",
        )


def observed_internal_gaps(decomposition: SourceDecomposition) -> np.ndarray:
    """Return consecutive retained-source gaps ``d_hat[j] - d_hat[j+1]``."""

    # PSEUDOCODE 1: Singular values are already sorted by the SVD.
    values = decomposition.singular_values

    # PSEUDOCODE 2: Difference adjacent values; r0=1 produces an empty vector.
    return values[:-1] - values[1:]


def certified_cut_positions(
    decomposition: SourceDecomposition,
    source_error_bound: float,
    *,
    c_gap: float = 3.0,
    atol: float = _DEFAULT_ATOL,
    rtol: float = _DEFAULT_RTOL,
) -> Tuple[int, ...]:
    """Return the certified internal cuts in Eq. ``(bismart-certified-cuts)``.

    Returned positions are sorted from left to right.  Position ``j`` lies
    after the first ``j`` singular directions, so it has the same numerical
    value as the paper's one-based boundary index.
    """

    # PSEUDOCODE 1: Calibrate the strict observed-gap threshold.
    epsilon, gap_constant, absolute, relative = _source_calibration(
        source_error_bound,
        c_gap,
        atol=atol,
        rtol=rtol,
    )
    threshold = gap_constant * epsilon

    # PSEUDOCODE 2: Compute all r0-1 adjacent gaps in source order.
    gaps = observed_internal_gaps(decomposition)

    # PSEUDOCODE 3: Keep only boundaries separated from the strict threshold by
    # more than the roundoff scale of their adjacent singular values.
    values = decomposition.singular_values
    return tuple(
        position
        for position, gap in enumerate(gaps, start=1)
        if _strict_gap_passes(
            float(gap),
            threshold,
            spectral_scale=max(
                abs(float(values[position - 1])),
                abs(float(values[position])),
            ),
            atol=absolute,
            rtol=relative,
        )
    )


def build_nested_partitions(
    source_rank: int,
    cuts: Sequence[int],
    internal_gaps: Sequence[float],
    *,
    atol: float = _DEFAULT_ATOL,
    rtol: float = _DEFAULT_RTOL,
    gap_scales: Sequence[float] | None = None,
) -> Tuple[BlockPartition, ...]:
    """Build Eq. ``(bismart-partition-library)`` from finest to coarsest.

    Certified cuts are deleted by increasing observed gap, with the smaller cut
    position breaking numerical ties.  ``gap_scales`` may provide the local
    singular-value scale for each subtracted gap; otherwise each gap supplies
    its own scale.  This ordering is part of the estimator and later
    contributes to deterministic validation tie-breaking.
    """

    # PSEUDOCODE 1: Validate source rank and the complete adjacent-gap vector.
    if isinstance(source_rank, (bool, np.bool_)) or not isinstance(
        source_rank, (int, np.integer)
    ):
        raise TypeError("source_rank must be an integer.")
    r0 = int(source_rank)
    if r0 < 1:
        raise ValueError("source_rank must be at least one.")
    raw_gaps = np.asarray(internal_gaps)
    if np.iscomplexobj(raw_gaps):
        raise ValueError(
            "internal_gaps must be real-valued; complex input is unsupported."
        )
    gaps = np.asarray(raw_gaps, dtype=float)
    if gaps.ndim != 1 or gaps.size != max(r0 - 1, 0):
        raise ValueError(f"internal_gaps must have length {max(r0 - 1, 0)}.")
    if np.any(~np.isfinite(gaps)) or np.any(gaps < 0):
        raise ValueError("internal_gaps must be finite and nonnegative.")
    absolute, relative = _validate_tolerances(atol, rtol)
    if gap_scales is None:
        scales = np.abs(gaps)
    else:
        raw_scales = np.asarray(gap_scales)
        if np.iscomplexobj(raw_scales):
            raise ValueError(
                "gap_scales must be real-valued; complex input is unsupported."
            )
        scales = np.asarray(raw_scales, dtype=float)
        if scales.ndim != 1 or scales.size != gaps.size:
            raise ValueError(f"gap_scales must have length {gaps.size}.")
        if np.any(~np.isfinite(scales)) or np.any(scales < 0):
            raise ValueError("gap_scales must be finite and nonnegative.")

    # PSEUDOCODE 2: Canonicalize the certified set and construct its finest partition.
    current_cuts = tuple(
        sorted(_nonnegative_integer(cut, name="cut position") for cut in cuts)
    )
    finest = BlockPartition.from_cut_positions(r0, current_cuts)
    library = [finest]

    # PSEUDOCODE 3: Rank cuts by increasing gap.  First form contiguous
    # tolerance-sized groups in exact gap order; then use smaller position
    # inside each numerical-tie group.  Grouping avoids a non-transitive fuzzy
    # comparator while retaining deterministic behavior.
    exact_order = sorted(current_cuts, key=lambda cut: (float(gaps[cut - 1]), cut))
    removal_order: list[int] = []
    tie_group: list[int] = []
    group_anchor: int | None = None
    for cut in exact_order:
        if group_anchor is None:
            tie_group = [cut]
            group_anchor = cut
            continue
        anchor_gap = float(gaps[group_anchor - 1])
        cut_gap = float(gaps[cut - 1])
        comparison_scale = max(
            float(scales[group_anchor - 1]),
            float(scales[cut - 1]),
            abs(anchor_gap),
            abs(cut_gap),
        )
        tie_tolerance = absolute + relative * comparison_scale
        if cut_gap - anchor_gap <= tie_tolerance:
            tie_group.append(cut)
        else:
            removal_order.extend(sorted(tie_group))
            tie_group = [cut]
            group_anchor = cut
    removal_order.extend(sorted(tie_group))

    # PSEUDOCODE 4: Delete one cut at a time, retaining every deterministic coarsening.
    remaining = set(current_cuts)
    for cut in removal_order:
        remaining.remove(cut)
        library.append(BlockPartition.from_cut_positions(r0, sorted(remaining)))

    # PSEUDOCODE 5: The final library member is necessarily the one-block partition.
    return tuple(library)


def build_source_block_library(
    observed_source: Any,
    source_rank: int,
    source_error_bound: float,
    *,
    c_gap: float = 3.0,
    atol: float = _DEFAULT_ATOL,
    rtol: float = _DEFAULT_RTOL,
) -> SourceBlockLibrary:
    """Construct all source-only objects needed by complete BI-SMART.

    A failed rank-boundary check is recorded in ``boundary_passes``.  The caller
    must then follow Appendix Algorithm ``alg:bismart-complete`` and omit every
    source-guided candidate; the returned decomposition remains useful for an
    auditable diagnostic.
    """

    # PSEUDOCODE 1: Compute the observed leading-r0 source SVD.
    decomposition = compute_source_decomposition(observed_source, source_rank)

    # PSEUDOCODE 2: Evaluate the strict outer boundary check before target use.
    epsilon, gap_constant, absolute, relative = _source_calibration(
        source_error_bound,
        c_gap,
        atol=atol,
        rtol=rtol,
    )
    boundary_gap = source_rank_boundary_gap(decomposition)
    boundary_passes = source_rank_boundary_passes(
        decomposition,
        epsilon,
        c_gap=gap_constant,
        atol=absolute,
        rtol=relative,
    )

    # PSEUDOCODE 3: Certify internal boundaries using the same source-only scale.
    gaps = observed_internal_gaps(decomposition)
    cuts = certified_cut_positions(
        decomposition,
        epsilon,
        c_gap=gap_constant,
        atol=absolute,
        rtol=relative,
    )

    # PSEUDOCODE 4: Coarsen deterministically until the single-block safeguard.
    values = decomposition.singular_values
    partitions = build_nested_partitions(
        decomposition.rank,
        cuts,
        gaps,
        atol=absolute,
        rtol=relative,
        gap_scales=np.maximum(np.abs(values[:-1]), np.abs(values[1:])),
    )

    # PSEUDOCODE 5: Return both usable objects and the failed/passed boundary diagnostic.
    return SourceBlockLibrary(
        decomposition=decomposition,
        source_error_bound=epsilon,
        c_gap=gap_constant,
        boundary_gap=boundary_gap,
        boundary_passes=boundary_passes,
        certified_cuts=cuts,
        observed_internal_gaps=gaps,
        partitions=partitions,
        atol=absolute,
        rtol=relative,
    )


# Compatibility-style descriptive alias; both names return the same value object.
build_source_library = build_source_block_library


def _validate_block(
    block: Sequence[int],
    *,
    source_rank: int,
) -> Tuple[int, ...]:
    """Normalize one nonempty consecutive block inside ``range(r0)``."""

    normalized = tuple(
        _nonnegative_integer(index, name="block index") for index in block
    )
    if not normalized:
        raise ValueError("block must be nonempty.")
    if normalized != tuple(range(normalized[0], normalized[-1] + 1)):
        raise ValueError("block must contain consecutive indices in increasing order.")
    if normalized[0] < 0 or normalized[-1] >= source_rank:
        raise ValueError(f"block must lie inside range({source_rank}).")
    return normalized


def _observable_block_gap_details(
    decomposition: SourceDecomposition,
    block: Sequence[int],
    source_error_bound: float,
) -> Tuple[float, float]:
    """Return one observable block gap and its active spectral scale.

    The scale follows the boundary term attaining the minimum rather than the
    largest singular value in the complete source matrix.  This keeps the
    relative guard local when the source spectrum spans several magnitudes.
    """

    epsilon = _validate_error_bound(source_error_bound)
    indices = _validate_block(block, source_rank=decomposition.rank)
    a, b = indices[0], indices[-1]
    values = decomposition.singular_values

    # Each pair is (observable lower-bound term, scale of the singular-value
    # subtraction that produced it).  The first term compares d_hat_b to zero.
    terms = [
        (
            float(values[b] - epsilon),
            abs(float(values[b])),
        )
    ]
    if a > 0:
        terms.append(
            (
                float(values[a - 1] - values[a] - 2.0 * epsilon),
                max(abs(float(values[a - 1])), abs(float(values[a]))),
            )
        )
    if b < decomposition.rank - 1:
        terms.append(
            (
                float(values[b] - values[b + 1] - 2.0 * epsilon),
                max(abs(float(values[b])), abs(float(values[b + 1]))),
            )
        )

    gap = min(term for term, _ in terms)
    # Exact ties can arise in idealized spectra.  The larger active scale gives
    # a representation-independent guard without importing unrelated blocks.
    spectral_scale = max(scale for term, scale in terms if term == gap)
    return gap, spectral_scale


def observable_block_gap(
    decomposition: SourceDecomposition,
    block: Sequence[int],
    source_error_bound: float,
) -> float:
    """Compute ``gamma_lower(B)`` from Eq. ``(bismart-observable-block-gap)``.

    Nonexistent left/right boundary terms are omitted exactly as in the paper.
    The first term always separates the block's smallest retained singular
    value from the source null space.
    """

    # PSEUDOCODE 1: Evaluate every existing block-boundary term and return the
    # weakest one.  The internal helper also records its local spectral scale
    # for the tolerance-aware Wedin comparison below.
    gap, _ = _observable_block_gap_details(
        decomposition,
        block,
        source_error_bound,
    )
    return gap


def observable_partition_gaps(
    decomposition: SourceDecomposition,
    partition: BlockPartition,
    source_error_bound: float,
) -> Tuple[float, ...]:
    """Compute the observable lower gap for every block in a partition."""

    if partition.rank != decomposition.rank:
        raise ValueError(
            "partition must cover exactly the retained source singular directions."
        )
    return tuple(
        observable_block_gap(decomposition, block, source_error_bound)
        for block in partition.blocks
    )


@dataclass(frozen=True)
class WedinGateResult:
    """Auditable result of Eq. ``(bismart-wedin-gate)``."""

    passes: bool
    block_gaps: Tuple[float, ...]
    threshold: float
    failing_blocks: Tuple[int, ...]

    @property
    def minimum_gap(self) -> float:
        """Smallest observable block gap in the partition."""

        return min(self.block_gaps)


def evaluate_wedin_gate(
    decomposition: SourceDecomposition,
    partition: BlockPartition,
    source_error_bound: float,
    *,
    atol: float = _DEFAULT_ATOL,
    rtol: float = _DEFAULT_RTOL,
) -> WedinGateResult:
    """Evaluate the observable whole-partition Wedin gate.

    Passing means ``min_l gamma_lower(B_l) >= 4*epsilon_hat``.  Equality and a
    configured roundoff neighborhood below equality pass because the displayed
    appendix gate is non-strict.  Setting both tolerances to zero recovers the
    literal floating-point comparison.  Failure does not erase successful
    pilot/frozen candidates; it only skips RRR and refinement.
    """

    # PSEUDOCODE 1: Validate epsilon_hat and compute every block-specific lower gap.
    epsilon = _validate_error_bound(source_error_bound)
    absolute, relative = _validate_tolerances(atol, rtol)
    if partition.rank != decomposition.rank:
        raise ValueError(
            "partition must cover exactly the retained source singular directions."
        )
    gap_details = tuple(
        _observable_block_gap_details(decomposition, block, epsilon)
        for block in partition.blocks
    )
    gaps = tuple(gap for gap, _ in gap_details)

    # PSEUDOCODE 2: Apply the common appendix threshold 4*epsilon_hat.
    threshold = 4.0 * epsilon
    failing = tuple(
        label
        for label, (gap, spectral_scale) in enumerate(gap_details)
        if not _nonstrict_gap_passes(
            gap,
            threshold,
            spectral_scale=spectral_scale,
            atol=absolute,
            rtol=relative,
        )
    )

    # PSEUDOCODE 3: Preserve all diagnostics so the branch decision is inspectable.
    return WedinGateResult(
        passes=not failing,
        block_gaps=gaps,
        threshold=threshold,
        failing_blocks=failing,
    )


def passes_wedin_gate(
    decomposition: SourceDecomposition,
    partition: BlockPartition,
    source_error_bound: float,
    *,
    atol: float = _DEFAULT_ATOL,
    rtol: float = _DEFAULT_RTOL,
) -> bool:
    """Boolean convenience wrapper around :func:`evaluate_wedin_gate`."""

    return evaluate_wedin_gate(
        decomposition,
        partition,
        source_error_bound,
        atol=atol,
        rtol=rtol,
    ).passes


# Paper-style short alias.
wedin_gate = passes_wedin_gate


__all__ = [
    "WedinGateResult",
    "build_nested_partitions",
    "build_source_block_library",
    "build_source_library",
    "certified_cut_positions",
    "compute_source_decomposition",
    "decompose_source",
    "evaluate_wedin_gate",
    "gaussian_source_error_bound",
    "observable_block_gap",
    "observable_partition_gaps",
    "observed_internal_gaps",
    "passes_wedin_gate",
    "require_source_rank_boundary",
    "source_rank_boundary_gap",
    "source_rank_boundary_passes",
    "wedin_gate",
]
