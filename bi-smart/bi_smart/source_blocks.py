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

    epsilon = float(source_error_bound)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("source_error_bound must be finite and nonnegative.")
    return epsilon


def _source_calibration(source_error_bound: float, c_gap: float) -> Tuple[float, float]:
    """Validate the two scalars used by every source certification rule."""

    epsilon = _validate_error_bound(source_error_bound)
    gap_constant = float(c_gap)
    if not np.isfinite(gap_constant) or gap_constant <= 2:
        raise ValueError("c_gap must be finite and strictly larger than 2.")
    return epsilon, gap_constant


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
    tau0 = float(noise_standard_deviation)
    x0 = float(tail_parameter)
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
) -> bool:
    """Evaluate Eq. ``(bismart-rank-boundary-check)`` exactly.

    The inequality is strict.  Equality therefore fails, matching the paper.
    """

    # PSEUDOCODE 1: Validate the source-only calibration constants.
    epsilon, gap_constant = _source_calibration(source_error_bound, c_gap)

    # PSEUDOCODE 2: Compare the observed truncation gap with c_gap*epsilon_hat.
    return source_rank_boundary_gap(decomposition) > gap_constant * epsilon


def require_source_rank_boundary(
    decomposition: SourceDecomposition,
    source_error_bound: float,
    *,
    c_gap: float = 3.0,
) -> None:
    """Raise a recoverable branch failure unless the rank boundary passes."""

    if not source_rank_boundary_passes(
        decomposition, source_error_bound, c_gap=c_gap
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
) -> Tuple[int, ...]:
    """Return the certified internal cuts in Eq. ``(bismart-certified-cuts)``.

    Returned positions are sorted from left to right.  Position ``j`` lies
    after the first ``j`` singular directions, so it has the same numerical
    value as the paper's one-based boundary index.
    """

    # PSEUDOCODE 1: Calibrate the strict observed-gap threshold.
    epsilon, gap_constant = _source_calibration(source_error_bound, c_gap)
    threshold = gap_constant * epsilon

    # PSEUDOCODE 2: Compute all r0-1 adjacent gaps in source order.
    gaps = observed_internal_gaps(decomposition)

    # PSEUDOCODE 3: Keep precisely the boundaries whose gaps are strictly larger.
    return tuple(
        position
        for position, gap in enumerate(gaps, start=1)
        if float(gap) > threshold
    )


def build_nested_partitions(
    source_rank: int,
    cuts: Sequence[int],
    internal_gaps: Sequence[float],
) -> Tuple[BlockPartition, ...]:
    """Build Eq. ``(bismart-partition-library)`` from finest to coarsest.

    Certified cuts are deleted by increasing observed gap, with the smaller cut
    position breaking exact ties.  This ordering is part of the estimator and
    later contributes to deterministic validation tie-breaking.
    """

    # PSEUDOCODE 1: Validate source rank and the complete adjacent-gap vector.
    if isinstance(source_rank, (bool, np.bool_)) or not isinstance(
        source_rank, (int, np.integer)
    ):
        raise TypeError("source_rank must be an integer.")
    r0 = int(source_rank)
    if r0 < 1:
        raise ValueError("source_rank must be at least one.")
    gaps = np.asarray(internal_gaps, dtype=float)
    if gaps.ndim != 1 or gaps.size != max(r0 - 1, 0):
        raise ValueError(f"internal_gaps must have length {max(r0 - 1, 0)}.")
    if np.any(~np.isfinite(gaps)) or np.any(gaps < 0):
        raise ValueError("internal_gaps must be finite and nonnegative.")

    # PSEUDOCODE 2: Canonicalize the certified set and construct its finest partition.
    current_cuts = tuple(
        sorted(_nonnegative_integer(cut, name="cut position") for cut in cuts)
    )
    finest = BlockPartition.from_cut_positions(r0, current_cuts)
    library = [finest]

    # PSEUDOCODE 3: Rank cuts by increasing gap and then smaller position.
    removal_order = sorted(current_cuts, key=lambda cut: (float(gaps[cut - 1]), cut))

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
    epsilon, gap_constant = _source_calibration(source_error_bound, c_gap)
    boundary_gap = source_rank_boundary_gap(decomposition)
    boundary_passes = boundary_gap > gap_constant * epsilon

    # PSEUDOCODE 3: Certify internal boundaries using the same source-only scale.
    gaps = observed_internal_gaps(decomposition)
    cuts = certified_cut_positions(
        decomposition,
        epsilon,
        c_gap=gap_constant,
    )

    # PSEUDOCODE 4: Coarsen deterministically until the single-block safeguard.
    partitions = build_nested_partitions(decomposition.rank, cuts, gaps)

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

    # PSEUDOCODE 1: Validate a consecutive empirical block B=[a,b].
    epsilon = _validate_error_bound(source_error_bound)
    indices = _validate_block(block, source_rank=decomposition.rank)
    a, b = indices[0], indices[-1]
    values = decomposition.singular_values

    # PSEUDOCODE 2: Start with separation of d_hat_b from zero/source null space.
    terms = [float(values[b] - epsilon)]

    # PSEUDOCODE 3: Add the observed left boundary when B is not the first block.
    if a > 0:
        terms.append(float(values[a - 1] - values[a] - 2.0 * epsilon))

    # PSEUDOCODE 4: Add the observed right boundary when B is not the last block.
    if b < decomposition.rank - 1:
        terms.append(float(values[b] - values[b + 1] - 2.0 * epsilon))

    # PSEUDOCODE 5: The observable lower bound is the weakest present separation.
    return min(terms)


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
) -> WedinGateResult:
    """Evaluate the observable whole-partition Wedin gate.

    Passing means ``min_l gamma_lower(B_l) >= 4*epsilon_hat``.  Equality passes
    because the displayed appendix gate is non-strict.  Failure does not erase
    successful pilot/frozen candidates; it only skips RRR and refinement.
    """

    # PSEUDOCODE 1: Validate epsilon_hat and compute every block-specific lower gap.
    epsilon = _validate_error_bound(source_error_bound)
    gaps = observable_partition_gaps(decomposition, partition, epsilon)

    # PSEUDOCODE 2: Apply the common appendix threshold 4*epsilon_hat.
    threshold = 4.0 * epsilon
    failing = tuple(label for label, gap in enumerate(gaps) if gap < threshold)

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
) -> bool:
    """Boolean convenience wrapper around :func:`evaluate_wedin_gate`."""

    return evaluate_wedin_gate(
        decomposition,
        partition,
        source_error_bound,
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
