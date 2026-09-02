"""Focused tests for BI-SMART screening and fitting-fold initialization."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from bi_smart.initialization import initialize_restricted_rrr, target_only_rrr
from bi_smart.refinement import BISMARTState, fitted_source, fitted_target
from bi_smart.screening import (
    block_hard_threshold,
    cell_hard_threshold,
    run_block_screen,
)
from bi_smart.source_blocks import build_nested_partitions, build_source_block_library
from bi_smart.types import (
    BlockPartition,
    CandidateStatus,
    DEFAULT_PINV_RCOND,
    FailureReason,
    FoldData,
)


def _exact_source_target_problem() -> tuple:
    """Build a noiseless rank-two target inside three singleton source blocks."""

    observed_source = np.diag([9.0, 6.0, 3.0, 0.0])
    library = build_source_block_library(
        observed_source,
        source_rank=3,
        source_error_bound=0.1,
        c_gap=3.0,
    )
    partition = library.partitions[0]
    assert partition.blocks == ((0,), (1,), (2,))

    # Define the target from the decomposition returned by the package.  This
    # avoids relying on a platform-specific sign convention for SVD vectors.
    source = library.decomposition
    target = (
        source.left_vectors[:, (0, 2)]
        @ np.diag([4.0, 2.0])
        @ source.right_vectors[:, (0, 2)].T
    )
    design = np.tile(np.eye(4), (6, 1))
    initialization = FoldData(design, design @ target, name="initialization")
    fitting = FoldData(design.copy(), design @ target, name="fitting")
    return library, partition, target, initialization, fitting


def test_exact_block_knapsack_and_lexicographic_tie() -> None:
    """Block selection maximizes energy exactly and resolves ties by labels."""

    partition = BlockPartition.from_cut_positions(4, cuts=(1, 2, 3))

    # With a two-direction budget, the exact optimum is blocks 1 and 2.  This
    # checks dimension capacity rather than a threshold on individual entries.
    values = np.array([[1.0], [4.0], [3.0], [2.0]])
    selected = block_hard_threshold(
        values,
        partition,
        target_rank=2,
        budget=2,
    )
    assert selected.blocks == (1, 2)
    assert selected.dimension == 2
    np.testing.assert_array_equal(
        selected.matrix,
        np.array([[0.0], [4.0], [3.0], [0.0]]),
    )

    # Every feasible pair now has equal energy.  The deterministic first pair
    # of zero-based block labels is (0, 1).
    tied = block_hard_threshold(
        np.ones((4, 1)),
        partition,
        target_rank=2,
        budget=2,
    )
    assert tied.blocks == (0, 1)


def test_exact_cell_knapsack_uses_row_major_lexicographic_tie() -> None:
    """Equal-energy block cells use the first row-major label list."""

    partition = BlockPartition.from_cut_positions(3, cuts=(1, 2))
    selected = cell_hard_threshold(
        np.ones((3, 3)),
        partition,
        capacity=2,
    )

    assert selected.cells == ((0, 0), (0, 1))
    assert selected.dimension == 2
    np.testing.assert_array_equal(
        selected.matrix,
        np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )


def test_block_and_cell_knapsacks_match_brute_force() -> None:
    """Dynamic programs agree with exhaustive search for unequal item sizes."""

    partition = BlockPartition.from_cut_positions(6, cuts=(1, 3, 4))
    block_values = np.array([[1.0], [2.0], [0.0], [1.5], [1.0], [1.0]])
    block_result = block_hard_threshold(
        block_values,
        partition,
        target_rank=2,
        budget=4,
    )

    block_energies = tuple(
        float(np.sum(block_values[np.asarray(block), :] ** 2))
        for block in partition.blocks
    )
    feasible_blocks = []
    for size in range(partition.n_blocks + 1):
        for labels in combinations(range(partition.n_blocks), size):
            dimension = sum(len(partition.blocks[label]) for label in labels)
            if 2 <= dimension <= 4:
                energy = sum(block_energies[label] for label in labels)
                feasible_blocks.append((energy, labels))
    maximum_block_energy = max(energy for energy, _ in feasible_blocks)
    expected_blocks = min(
        labels for energy, labels in feasible_blocks if energy == maximum_block_energy
    )
    assert block_result.blocks == expected_blocks

    cell_values = np.arange(1.0, 37.0).reshape(6, 6)
    cell_result = cell_hard_threshold(cell_values, partition, capacity=8)
    cell_items = []
    for left, left_block in enumerate(partition.blocks):
        for right, right_block in enumerate(partition.blocks):
            cell = cell_values[np.ix_(left_block, right_block)]
            cell_items.append(
                (
                    (left, right),
                    len(left_block) * len(right_block),
                    float(np.sum(cell**2)),
                )
            )
    feasible_cells = []
    for size in range(len(cell_items) + 1):
        for indices in combinations(range(len(cell_items)), size):
            dimension = sum(cell_items[index][1] for index in indices)
            if dimension <= 8:
                labels = tuple(cell_items[index][0] for index in indices)
                energy = sum(cell_items[index][2] for index in indices)
                feasible_cells.append((energy, labels))
    maximum_cell_energy = max(energy for energy, _ in feasible_cells)
    expected_cells = min(
        labels for energy, labels in feasible_cells if energy == maximum_cell_energy
    )
    assert cell_result.cells == expected_cells


def test_knapsack_energy_ties_use_atol_and_rtol() -> None:
    """Numerical energy ties use labels, while resolved advantages still win."""

    partition = BlockPartition.from_cut_positions(3, cuts=(1, 2))
    near_tied = np.sqrt(np.array([[1.0], [1.0 + 5e-11], [0.25]]))

    tolerant = block_hard_threshold(
        near_tied,
        partition,
        target_rank=1,
        budget=1,
        atol=0.0,
        rtol=1e-10,
    )
    absolute_tie = block_hard_threshold(
        near_tied,
        partition,
        target_rank=1,
        budget=1,
        atol=1e-10,
        rtol=0.0,
    )
    literal = block_hard_threshold(
        near_tied,
        partition,
        target_rank=1,
        budget=1,
        atol=0.0,
        rtol=0.0,
    )
    assert tolerant.blocks == (0,)
    assert absolute_tie.blocks == (0,)
    assert literal.blocks == (1,)

    resolved = block_hard_threshold(
        np.sqrt(np.array([[1.0], [1.0 + 5e-7], [0.25]])),
        partition,
        target_rank=1,
        budget=1,
        atol=0.0,
        rtol=1e-10,
    )
    assert resolved.blocks == (1,)

    cell_values = np.zeros((3, 3))
    cell_values[0, 0] = 1.0
    cell_values[0, 1] = np.sqrt(1.0 + 5e-11)
    tolerant_cell = cell_hard_threshold(
        cell_values,
        partition,
        capacity=1,
        atol=0.0,
        rtol=1e-10,
    )
    literal_cell = cell_hard_threshold(
        cell_values,
        partition,
        capacity=1,
        atol=0.0,
        rtol=0.0,
    )
    assert tolerant_cell.cells == ((0, 0),)
    assert literal_cell.cells == ((0, 1),)


def test_threshold_and_partition_helpers_reject_complex_arrays() -> None:
    """No public array helper may silently discard imaginary components."""

    partition = BlockPartition.from_cut_positions(2, cuts=(1,))
    complex_matrix = np.eye(2, dtype=complex) * (1.0 + 1e-14j)
    with pytest.raises(ValueError, match="real-valued"):
        block_hard_threshold(
            complex_matrix,
            partition,
            target_rank=1,
            budget=1,
        )
    with pytest.raises(ValueError, match="real-valued"):
        cell_hard_threshold(complex_matrix, partition, capacity=1)
    with pytest.raises(ValueError, match="internal_gaps must be real-valued"):
        build_nested_partitions(2, (1,), np.array([1.0 + 1e-14j]))
    with pytest.raises(ValueError, match="gap_scales must be real-valued"):
        build_nested_partitions(
            2,
            (1,),
            np.array([1.0]),
            gap_scales=np.array([1.0 + 1e-14j]),
        )


def test_threshold_labels_are_stable_under_within_block_rotations() -> None:
    """Roundoff from equivalent block bases cannot change numerical ties."""

    partition = BlockPartition.from_cut_positions(4, cuts=(2,))
    block_values = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [np.sqrt(2.0), 0.0],
            [0.0, 0.0],
        ]
    )
    angle = 0.371
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    block_rotation = np.zeros((4, 4))
    block_rotation[:2, :2] = rotation
    block_rotation[2:, 2:] = rotation.T
    rotated_block_values = block_rotation.T @ block_values

    canonical_block = block_hard_threshold(
        block_values,
        partition,
        target_rank=2,
        budget=2,
    )
    rotated_block = block_hard_threshold(
        rotated_block_values,
        partition,
        target_rank=2,
        budget=2,
    )
    assert canonical_block.blocks == rotated_block.blocks == (0,)

    cell_values = np.zeros((4, 4))
    cell_values[:2, 2:] = np.eye(2)
    cell_values[2:, :2] = np.eye(2)
    rotated_cell_values = block_rotation.T @ cell_values @ block_rotation
    canonical_cell = cell_hard_threshold(cell_values, partition, capacity=4)
    rotated_cell = cell_hard_threshold(rotated_cell_values, partition, capacity=4)
    assert canonical_cell.cells == rotated_cell.cells == ((0, 1),)


def test_one_sweep_screen_succeeds_and_recovers_exact_block_target() -> None:
    """The noiseless screen retains the two active blocks and fitted matrix."""

    library, partition, target, initialization, _ = _exact_source_target_problem()
    result = run_block_screen(
        initialization,
        library.decomposition,
        partition,
        target_rank=2,
        left_budget=2,
        right_budget=2,
    )

    assert result.status is CandidateStatus.SUCCESSFUL
    assert result.left_blocks == (0, 2)
    assert result.right_blocks == (0, 2)
    assert result.statistic is not None
    assert result.left_factor is not None
    assert result.right_factor is not None
    assert result.pilot_matrix is not None
    assert result.frozen_matrix is not None
    np.testing.assert_allclose(result.pilot_matrix, target, atol=1e-11)
    np.testing.assert_allclose(result.frozen_matrix, target, atol=1e-11)


def test_screen_returns_recoverable_failure_for_singular_reduced_gram() -> None:
    """A non-PD initialization Gram fails one branch instead of raising."""

    library, partition, _, _, _ = _exact_source_target_problem()
    rank_one_design = np.ones((8, 4))
    invalid_fold = FoldData(
        rank_one_design,
        np.zeros((8, 4)),
        name="rank_deficient_initialization",
    )

    result = run_block_screen(
        invalid_fold,
        library.decomposition,
        partition,
        target_rank=2,
        left_budget=2,
        right_budget=2,
    )

    assert result.status is CandidateStatus.UNSUCCESSFUL
    assert result.failure_reason is FailureReason.NON_POSITIVE_DEFINITE
    assert result.pilot_matrix is None
    assert result.frozen_matrix is None


def test_restricted_rrr_has_requested_rank_shape_and_shared_joint_state() -> None:
    """Fitting-fold RRR exactly reconstructs the noiseless selected target."""

    library, partition, target, initialization, fitting = _exact_source_target_problem()
    screen = run_block_screen(
        initialization,
        library.decomposition,
        partition,
        target_rank=2,
        left_budget=2,
        right_budget=2,
    )
    result = initialize_restricted_rrr(
        fitting,
        library.decomposition,
        screen,
        target_rank=2,
    )

    assert result.successful
    assert result.candidate.status is CandidateStatus.SUCCESSFUL
    assert result.candidate.matrix is not None
    assert result.candidate.matrix.shape == (4, 4)
    assert np.linalg.matrix_rank(result.candidate.matrix, tol=1e-10) == 2
    np.testing.assert_allclose(result.candidate.matrix, target, atol=1e-11)

    assert isinstance(result.state, BISMARTState)
    assert result.state is not None
    assert int(result.state.active_u.sum()) == 2
    assert int(result.state.active_v.sum()) == 2
    np.testing.assert_allclose(fitted_target(result.state), target, atol=1e-11)
    np.testing.assert_allclose(
        fitted_source(result.state),
        library.decomposition.approximation,
        atol=1e-11,
    )


def test_restricted_rrr_zero_cutoff_fails_but_target_only_remains_available() -> None:
    """A zero strict RRR cutoff is local; target-only still returns a candidate."""

    library, partition, _, initialization, fitting = _exact_source_target_problem()
    screen = run_block_screen(
        initialization,
        library.decomposition,
        partition,
        target_rank=2,
        left_budget=2,
        right_budget=2,
    )
    zero_response_fold = FoldData(
        fitting.X,
        np.zeros_like(fitting.Y),
        name="zero_response_fitting",
    )

    failed = initialize_restricted_rrr(
        zero_response_fold,
        library.decomposition,
        screen,
        target_rank=2,
    )
    assert not failed.successful
    assert failed.candidate.failure_reason is FailureReason.ZERO_RANK_COMPONENT

    safeguard = target_only_rrr(zero_response_fold, target_rank=2)
    assert safeguard.status is CandidateStatus.SUCCESSFUL
    assert safeguard.matrix is not None
    assert safeguard.matrix.shape == (4, 4)
    np.testing.assert_array_equal(safeguard.matrix, np.zeros((4, 4)))


def test_target_only_rrr_recovers_noiseless_rank_two_target() -> None:
    """The Moore--Penrose safeguard is a global rank-at-most-r fit."""

    _, _, target, _, fitting = _exact_source_target_problem()
    candidate = target_only_rrr(fitting, target_rank=2)

    assert candidate.status is CandidateStatus.SUCCESSFUL
    assert candidate.matrix is not None
    assert candidate.matrix.shape == target.shape
    assert np.linalg.matrix_rank(candidate.matrix, tol=1e-10) == 2
    np.testing.assert_allclose(candidate.matrix, target, atol=1e-11)


def test_target_only_rrr_uses_an_explicit_reproducible_pinv_cutoff() -> None:
    """The configured cutoff controls both the solve and reported design rank."""

    design = np.diag([1.0, 1e-8])
    fold = FoldData(design, design.copy(), name="ill_conditioned_fitting")

    default = target_only_rrr(fold, target_rank=2)
    legacy_none = target_only_rrr(fold, target_rank=2, pinv_rcond=None)
    aggressive = target_only_rrr(fold, target_rank=2, pinv_rcond=1e-4)

    assert default.metadata["pinv_rcond"] == DEFAULT_PINV_RCOND
    assert legacy_none.metadata["pinv_rcond"] == DEFAULT_PINV_RCOND
    assert default.metadata["design_rank"] == 2
    assert aggressive.metadata["pinv_rcond"] == 1e-4
    assert aggressive.metadata["design_rank"] == 1
    np.testing.assert_allclose(default.matrix, np.eye(2), atol=1e-12)
    np.testing.assert_array_equal(legacy_none.matrix, default.matrix)
    np.testing.assert_array_equal(aggressive.matrix, np.diag([1.0, 0.0]))

    with pytest.raises(ValueError, match="real-valued"):
        target_only_rrr(fold, target_rank=2, pinv_rcond=1e-12 + 1j)
