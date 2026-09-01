"""Block-invariant frozen screening for BI-SMART.

This module implements the *initialization-fold* part of Complete automatic
BI-SMART.  It follows Appendix equations ``(bismart-block-support-families)``
through ``(bismart-block-frozen-refit)``.  In particular, selection is over
whole source spectral blocks.  It never thresholds individual coordinates
inside a block, because doing so would make the result depend on an arbitrary
singular-vector basis when source singular values are tied.

The routines deliberately return an unsuccessful :class:`ScreenResult` for a
paper-defined algebraic failure.  A failed branch is not a Python exception:
the complete algorithm is supposed to omit that branch and continue with the
remaining partitions, budgets, and safeguards.  Invalid programmer inputs
still raise ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .linalg import spd_solve, strict_polar_factor, strict_truncated_svd
from .types import (
    BlockPartition,
    CandidateStatus,
    FailureReason,
    FloatArray,
    FoldData,
    NumericalFailure,
    ScreenResult,
    SourceDecomposition,
)


CellLabel = Tuple[int, int]


@dataclass(frozen=True)
class BlockThresholdResult:
    """Result of the exact whole-block energy knapsack.

    ``blocks`` contains zero-based block labels in increasing order.  Keeping
    these labels is important: the appendix passes the maximizing labels, not
    rows that merely happen to remain numerically nonzero after a polar step.
    """

    matrix: FloatArray
    blocks: Tuple[int, ...]
    dimension: int
    energy: float


@dataclass(frozen=True)
class CellThresholdResult:
    """Result of exact block-cell thresholding in the sparse pilot branch."""

    matrix: FloatArray
    cells: Tuple[CellLabel, ...]
    dimension: int
    energy: float


@dataclass(frozen=True)
class ScreeningComplexities:
    """Observable hybrid-pilot quantities in Eq. ``(bismart-block-complexities)``."""

    doubled_left_budget: int
    doubled_right_budget: int
    doubled_cell_budget: int
    cell_log_cardinality: float
    left_log_cardinality: float
    right_log_cardinality: float
    chi: float
    xi: float
    phi: float


@dataclass(frozen=True)
class _KnapsackState:
    """Best deterministic subset having one exact integer weight."""

    energy: float
    labels: tuple


def _validate_nonnegative_integer(value: int, *, name: str) -> int:
    """Return a Python integer while rejecting booleans and negative values."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative; got {result}.")
    return result


def _better_state(candidate: _KnapsackState, incumbent: Optional[_KnapsackState]) -> bool:
    """Apply the paper's deterministic maximum-energy and lexicographic rule.

    Energies are compared exactly here.  Approximate equality would add a new
    tuning tolerance that the manuscript does not define.  The item order and
    summation order are fixed, so the result is deterministic for one numeric
    backend.
    """

    if incumbent is None:
        return True
    if candidate.energy > incumbent.energy:
        return True
    if candidate.energy < incumbent.energy:
        return False
    return candidate.labels < incumbent.labels


def _exact_subset_knapsack(
    weights: Sequence[int],
    energies: Sequence[float],
    labels: Sequence[object],
    *,
    capacity: int,
    minimum_weight: int = 0,
) -> _KnapsackState:
    """Solve a nonnegative-value zero-one knapsack with deterministic ties.

    The dynamic program stores one best subset for every *exact* weight.  A
    final scan over ``minimum_weight <= weight <= capacity`` implements the
    inequality-constrained families in Eqs. ``(bismart-block-support-families)``
    and ``(bismart-cell-family)``.  Its arithmetic cost is ``O(items*capacity)``.
    """

    capacity = _validate_nonnegative_integer(capacity, name="capacity")
    minimum_weight = _validate_nonnegative_integer(minimum_weight, name="minimum_weight")
    if minimum_weight > capacity:
        raise NumericalFailure(
            FailureReason.SCREEN_FAILED,
            f"No support can have dimension in [{minimum_weight}, {capacity}].",
        )
    if not (len(weights) == len(energies) == len(labels)):
        raise ValueError("weights, energies, and labels must have the same length.")

    normalized_weights = tuple(_validate_nonnegative_integer(w, name="item weight") for w in weights)
    if any(weight == 0 for weight in normalized_weights):
        raise ValueError("Knapsack item weights must be positive.")
    normalized_energies = tuple(float(value) for value in energies)
    if any(not np.isfinite(value) or value < 0 for value in normalized_energies):
        raise ValueError("Knapsack energies must be finite and nonnegative.")

    # PSEUDOCODE 1: The empty set is the unique state of exact weight zero.
    states: list[Optional[_KnapsackState]] = [None] * (capacity + 1)
    states[0] = _KnapsackState(energy=0.0, labels=())

    # PSEUDOCODE 2: Visit items in the caller's fixed label order.  Descending
    # capacity prevents one item from being used more than once.
    for weight, energy, label in zip(normalized_weights, normalized_energies, labels):
        for total in range(capacity, weight - 1, -1):
            previous = states[total - weight]
            if previous is None:
                continue
            proposal = _KnapsackState(
                energy=previous.energy + energy,
                labels=previous.labels + (label,),
            )
            if _better_state(proposal, states[total]):
                states[total] = proposal

    # PSEUDOCODE 3: Maximize over all admissible realized dimensions, then use
    # the lexicographically first ordered label tuple at an exact energy tie.
    best: Optional[_KnapsackState] = None
    for total in range(minimum_weight, capacity + 1):
        state = states[total]
        if state is not None and _better_state(state, best):
            best = state
    if best is None:
        raise NumericalFailure(
            FailureReason.SCREEN_FAILED,
            f"No block subset has dimension in [{minimum_weight}, {capacity}].",
        )
    return best


def count_subsets_with_capacity(weights: Sequence[int], capacity: int) -> int:
    """Count subsets whose integer weight is at most ``capacity`` exactly.

    This is the capacity dynamic program described after Eq.
    ``(bismart-block-complexities)``.  Python integers retain exact counts even
    when the family is large.
    """

    capacity = _validate_nonnegative_integer(capacity, name="capacity")
    normalized = tuple(_validate_nonnegative_integer(w, name="item weight") for w in weights)
    if any(weight == 0 for weight in normalized):
        raise ValueError("Subset-counting weights must be positive.")

    counts = [0] * (capacity + 1)
    counts[0] = 1
    for weight in normalized:
        for total in range(capacity, weight - 1, -1):
            counts[total] += counts[total - weight]
    return int(sum(counts))


def screening_complexities(
    partition: BlockPartition,
    target_rank: int,
    left_budget: int,
    right_budget: int,
) -> ScreeningComplexities:
    """Compute the hybrid pilot decision and its observable complexities.

    This implements Eq. ``(bismart-block-complexities)`` literally, including
    cardinalities of all block/cell subsets under the doubled capacities.
    """

    target_rank = _validate_nonnegative_integer(target_rank, name="target_rank")
    left_budget = _validate_nonnegative_integer(left_budget, name="left_budget")
    right_budget = _validate_nonnegative_integer(right_budget, name="right_budget")
    r0 = partition.rank
    if target_rank < 1 or target_rank > r0:
        raise ValueError("target_rank must lie in [1, partition.rank].")
    if not (target_rank <= left_budget <= r0 and target_rank <= right_budget <= r0):
        raise ValueError("Both budgets must lie in [target_rank, partition.rank].")

    block_sizes = partition.block_sizes
    cell_weights = tuple(left * right for left in block_sizes for right in block_sizes)
    cell_budget = left_budget * right_budget
    doubled_cell_budget = min(2 * cell_budget, r0 * r0)
    doubled_left = min(2 * left_budget, r0)
    doubled_right = min(2 * right_budget, r0)

    h_cell = log(max(count_subsets_with_capacity(cell_weights, doubled_cell_budget), 2))
    h_left = log(max(count_subsets_with_capacity(block_sizes, doubled_left), 2))
    h_right = log(max(count_subsets_with_capacity(block_sizes, doubled_right), 2))
    chi = float(doubled_cell_budget + h_cell)
    xi = float(min(target_rank * r0, chi))
    phi = float(
        target_rank * (doubled_left + doubled_right) + h_left + h_right
    )
    return ScreeningComplexities(
        doubled_left_budget=doubled_left,
        doubled_right_budget=doubled_right,
        doubled_cell_budget=doubled_cell_budget,
        cell_log_cardinality=h_cell,
        left_log_cardinality=h_left,
        right_log_cardinality=h_right,
        chi=chi,
        xi=xi,
        phi=phi,
    )


def block_hard_threshold(
    matrix: NDArray[np.floating],
    partition: BlockPartition,
    *,
    target_rank: int,
    budget: int,
) -> BlockThresholdResult:
    """Retain the exact maximum-energy whole-block row support.

    Implements Eqs. ``(bismart-block-energies)`` and the operator
    ``H^blk_{B,K}``.  ``matrix`` must have ``r0`` rows; its column count is
    unrestricted, although the BI-SMART screen uses ``r`` columns.
    """

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != partition.rank:
        raise ValueError(
            f"matrix must have shape (r0, k) with r0={partition.rank}; got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must contain only finite values.")
    budget = _validate_nonnegative_integer(budget, name="budget")
    target_rank = _validate_nonnegative_integer(target_rank, name="target_rank")

    energies = tuple(float(np.sum(array[np.asarray(block, dtype=int), :] ** 2)) for block in partition.blocks)
    state = _exact_subset_knapsack(
        partition.block_sizes,
        energies,
        tuple(range(partition.n_blocks)),
        capacity=budget,
        minimum_weight=target_rank,
    )
    selected = tuple(int(label) for label in state.labels)
    indices = partition.union_indices(selected)
    thresholded = np.zeros_like(array)
    thresholded[np.asarray(indices, dtype=int), :] = array[np.asarray(indices, dtype=int), :]
    return BlockThresholdResult(
        matrix=thresholded,
        blocks=selected,
        dimension=len(indices),
        energy=float(state.energy),
    )


def cell_hard_threshold(
    matrix: NDArray[np.floating],
    partition: BlockPartition,
    *,
    capacity: int,
) -> CellThresholdResult:
    """Retain maximum-energy block cells under the exact cell-size budget.

    This is ``H^cell_{B,m}`` from Eq. ``(bismart-cell-family)``.  Cell labels
    are enumerated row-major, which fixes the appendix's lexicographic tie rule.
    """

    array = np.asarray(matrix, dtype=float)
    r0 = partition.rank
    if array.ndim != 2 or array.shape != (r0, r0):
        raise ValueError(f"matrix must have shape ({r0}, {r0}); got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must contain only finite values.")
    capacity = _validate_nonnegative_integer(capacity, name="capacity")

    labels: list[CellLabel] = []
    weights: list[int] = []
    energies: list[float] = []
    for left_label, left_block in enumerate(partition.blocks):
        left_indices = np.asarray(left_block, dtype=int)
        for right_label, right_block in enumerate(partition.blocks):
            right_indices = np.asarray(right_block, dtype=int)
            cell = array[np.ix_(left_indices, right_indices)]
            labels.append((left_label, right_label))
            weights.append(len(left_block) * len(right_block))
            energies.append(float(np.sum(cell**2)))

    state = _exact_subset_knapsack(
        weights,
        energies,
        labels,
        capacity=capacity,
        minimum_weight=0,
    )
    selected = tuple((int(left), int(right)) for left, right in state.labels)
    thresholded = np.zeros_like(array)
    for left_label, right_label in selected:
        left_indices = np.asarray(partition.blocks[left_label], dtype=int)
        right_indices = np.asarray(partition.blocks[right_label], dtype=int)
        thresholded[np.ix_(left_indices, right_indices)] = array[
            np.ix_(left_indices, right_indices)
        ]
    dimension = sum(
        len(partition.blocks[left]) * len(partition.blocks[right])
        for left, right in selected
    )
    return CellThresholdResult(
        matrix=thresholded,
        cells=selected,
        dimension=dimension,
        energy=float(state.energy),
    )


def compute_screening_statistic(
    initialization_fold: FoldData,
    source: SourceDecomposition,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> FloatArray:
    """Compute ``Z_in`` from Eq. ``(bismart-screen-statistic-inverse)``.

    The solve uses the reduced Gram matrix itself.  It intentionally does not
    left-whiten by an inverse square root, which would mix rows across blocks
    and destroy the block sparsity being screened.
    """

    if initialization_fold.n_features != source.source_shape[0]:
        raise ValueError("Initialization predictors and source rows are incompatible.")
    if initialization_fold.n_responses != source.source_shape[1]:
        raise ValueError("Initialization responses and source columns are incompatible.")

    n = initialization_fold.n_samples
    U0 = source.left_vectors
    V0 = source.right_vectors
    gram = (initialization_fold.X.T @ initialization_fold.X) / n
    reduced_gram = U0.T @ gram @ U0
    cross_covariance = (initialization_fold.X.T @ initialization_fold.Y) / n
    reduced_cross_covariance = U0.T @ cross_covariance @ V0
    return spd_solve(reduced_gram, reduced_cross_covariance, atol=atol, rtol=rtol)


def _failed_screen(
    partition: BlockPartition,
    failure: NumericalFailure,
    *,
    statistic: Optional[FloatArray] = None,
    pilot_matrix: Optional[FloatArray] = None,
    left_blocks: Iterable[int] = (),
    right_blocks: Iterable[int] = (),
    left_factor: Optional[FloatArray] = None,
    right_factor: Optional[FloatArray] = None,
) -> ScreenResult:
    """Convert a recoverable numerical failure into branch data."""

    return ScreenResult(
        status=CandidateStatus.UNSUCCESSFUL,
        partition=partition,
        left_blocks=tuple(left_blocks),
        right_blocks=tuple(right_blocks),
        statistic=statistic,
        left_factor=left_factor,
        right_factor=right_factor,
        pilot_matrix=pilot_matrix,
        frozen_matrix=None,
        failure_reason=failure.reason,
        message=str(failure),
    )


def run_block_screen(
    initialization_fold: FoldData,
    source: SourceDecomposition,
    partition: BlockPartition,
    *,
    target_rank: int,
    left_budget: int,
    right_budget: int,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> ScreenResult:
    """Run one appendix hybrid-pilot and one-sweep screening branch.

    PSEUDOCODE
    ----------
    1. Form ``Z_in`` using the unwhitened reduced-Gram inverse
       [Eq. ``(bismart-screen-statistic-inverse)``].
    2. Compare ``r*r0`` with ``chi_B``.  Use all of ``Z_in`` in the dense
       branch, or exact cell-knapsack thresholding in the sparse branch
       [Eqs. ``(bismart-block-complexities)``--``(bismart-block-pilot-input)``].
    3. Require a nonzero, unique rank-``r`` truncation and retain the ambient
       pilot fit [Eqs. ``(bismart-block-pilot-svd)``--``fit``].
    4. Threshold both pilot singular factors by whole-block knapsack and check
       both polar factors exactly as written in the appendix.  The left pilot
       factor is checked even though only the right pilot factor enters the
       subsequent alternating sweep; this preserves the manuscript's stated
       failure semantics.
    5. Perform one left and one right update, retaining the *maximizing block
       labels* from each threshold operation [Eqs. ``(bismart-block-left-sweep)``
       and ``(bismart-block-right-sweep)``].
    6. Refit the small core and return the frozen ambient matrix
       [Eq. ``(bismart-block-frozen-refit)``].

    A paper-defined numerical failure returns ``UNSUCCESSFUL``.  If the pilot
    was already formed before a later polar failure, it remains attached to
    the failed result so orchestration can retain that independently successful
    pilot candidate, as required by Appendix Algorithm 2.
    """

    target_rank = _validate_nonnegative_integer(target_rank, name="target_rank")
    left_budget = _validate_nonnegative_integer(left_budget, name="left_budget")
    right_budget = _validate_nonnegative_integer(right_budget, name="right_budget")
    if partition.rank != source.rank:
        raise ValueError("partition and source decomposition must use the same r0.")
    if target_rank < 1 or target_rank > source.rank:
        raise ValueError("target_rank must lie in [1, source.rank].")
    if not (target_rank <= left_budget <= source.rank):
        raise ValueError("left_budget must lie in [target_rank, source.rank].")
    if not (target_rank <= right_budget <= source.rank):
        raise ValueError("right_budget must lie in [target_rank, source.rank].")

    statistic: Optional[FloatArray] = None
    pilot_matrix: Optional[FloatArray] = None

    # PSEUDOCODE 1: A non-PD reduced initialization Gram makes this branch
    # unsuccessful without affecting other partitions or safeguards.
    try:
        statistic = compute_screening_statistic(
            initialization_fold,
            source,
            atol=atol,
            rtol=rtol,
        )
    except NumericalFailure as failure:
        return _failed_screen(partition, failure)

    # PSEUDOCODE 2: Choose the dense or cell-sparse hybrid pilot exactly from
    # the observable entropy comparison in Eq. (bismart-block-pilot-input).
    complexities = screening_complexities(
        partition,
        target_rank,
        left_budget,
        right_budget,
    )
    if target_rank * source.rank <= complexities.chi:
        pilot_input = statistic
    else:
        try:
            pilot_input = cell_hard_threshold(
                statistic,
                partition,
                capacity=left_budget * right_budget,
            ).matrix
        except NumericalFailure as failure:
            return _failed_screen(partition, failure, statistic=statistic)

    # PSEUDOCODE 3: The strict helper enforces both sigma_r > sigma_{r+1}
    # and sigma_r > 0, converting either condition to a recoverable failure.
    try:
        pilot_svd = strict_truncated_svd(
            pilot_input,
            target_rank,
            atol=atol,
            rtol=rtol,
        )
    except NumericalFailure as failure:
        return _failed_screen(partition, failure, statistic=statistic)

    pilot_coordinate_matrix = pilot_svd.approximation
    pilot_matrix = (
        source.left_vectors @ pilot_coordinate_matrix @ source.right_vectors.T
    )

    # PSEUDOCODE 4: Apply exact whole-block thresholding to both pilot factors.
    # The appendix computes/checks P_u^(0), although only P_v^(0) is consumed
    # by the sweep.  Follow that statement literally and document the quirk.
    try:
        initial_left = block_hard_threshold(
            pilot_svd.u,
            partition,
            target_rank=target_rank,
            budget=left_budget,
        )
        _unused_left_polar = strict_polar_factor(
            initial_left.matrix,
            atol=atol,
            rtol=rtol,
        )
        initial_right = block_hard_threshold(
            pilot_svd.vt.T,
            partition,
            target_rank=target_rank,
            budget=right_budget,
        )
        right_factor_0 = strict_polar_factor(
            initial_right.matrix,
            atol=atol,
            rtol=rtol,
        )
    except NumericalFailure as failure:
        return _failed_screen(
            partition,
            failure,
            statistic=statistic,
            pilot_matrix=pilot_matrix,
        )

    # PSEUDOCODE 5a: One left sweep, recording the maximizing block set before
    # polar normalization rather than inferring it from floating-point zeros.
    try:
        left_update = block_hard_threshold(
            statistic @ right_factor_0,
            partition,
            target_rank=target_rank,
            budget=left_budget,
        )
        left_factor_1 = strict_polar_factor(
            left_update.matrix,
            atol=atol,
            rtol=rtol,
        )
    except NumericalFailure as failure:
        return _failed_screen(
            partition,
            failure,
            statistic=statistic,
            pilot_matrix=pilot_matrix,
        )

    # PSEUDOCODE 5b: One right sweep conditional on the updated left factor.
    try:
        right_update = block_hard_threshold(
            statistic.T @ left_factor_1,
            partition,
            target_rank=target_rank,
            budget=right_budget,
        )
        right_factor_1 = strict_polar_factor(
            right_update.matrix,
            atol=atol,
            rtol=rtol,
        )
    except NumericalFailure as failure:
        return _failed_screen(
            partition,
            failure,
            statistic=statistic,
            pilot_matrix=pilot_matrix,
            left_blocks=left_update.blocks,
            left_factor=left_factor_1,
        )

    # PSEUDOCODE 6: Refit the rank-r core in source coordinates and map the
    # block-thresholded factors back to the ambient p-by-q coefficient space.
    frozen_core = left_factor_1.T @ statistic @ right_factor_1
    frozen_matrix = (
        source.left_vectors
        @ left_factor_1
        @ frozen_core
        @ right_factor_1.T
        @ source.right_vectors.T
    )
    return ScreenResult(
        status=CandidateStatus.SUCCESSFUL,
        partition=partition,
        left_blocks=left_update.blocks,
        right_blocks=right_update.blocks,
        statistic=statistic,
        left_factor=left_factor_1,
        right_factor=right_factor_1,
        pilot_matrix=pilot_matrix,
        frozen_matrix=frozen_matrix,
    )


__all__ = [
    "BlockThresholdResult",
    "CellThresholdResult",
    "ScreeningComplexities",
    "block_hard_threshold",
    "cell_hard_threshold",
    "compute_screening_statistic",
    "count_subsets_with_capacity",
    "run_block_screen",
    "screening_complexities",
]
