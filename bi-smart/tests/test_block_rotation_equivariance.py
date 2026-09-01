"""Within-block basis-rotation equivariance for executable BI-SMART stages."""

from __future__ import annotations

import numpy as np

from bi_smart.initialization import initialize_restricted_rrr
from bi_smart.refinement import fitted_source, fitted_target
from bi_smart.screening import run_block_screen
from bi_smart.source_blocks import build_source_block_library
from bi_smart.types import (
    CandidateStatus,
    FailureReason,
    FoldData,
    SourceDecomposition,
)


def _rotation(angle: float) -> np.ndarray:
    """Return a deterministic two-dimensional orthogonal rotation."""

    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def test_exact_stages_are_equivariant_to_independent_tied_block_rotations() -> None:
    """Pilot, frozen, RRR, and initialized fitted pairs are basis invariant."""

    # The first two equal singular values form one mandatory multiplicity block;
    # the separated third value forms a second block.
    observed_source = np.diag([7.0, 7.0, 3.0, 0.0])
    library = build_source_block_library(
        observed_source,
        source_rank=3,
        source_error_bound=0.1,
        c_gap=3.0,
    )
    source = library.decomposition
    partition = library.partitions[0]
    assert partition.blocks == ((0, 1), (2,))

    # Apply different left and right orthogonal changes of basis in the tied
    # block.  Their spans/projectors are unchanged, but coordinate matrices
    # transform by R.T on the left and S on the right.
    left_rotation = _rotation(0.37)
    right_rotation = _rotation(-0.61)
    rotated_u = source.left_vectors.copy()
    rotated_v = source.right_vectors.copy()
    rotated_u[:, :2] = source.left_vectors[:, :2] @ left_rotation
    rotated_v[:, :2] = source.right_vectors[:, :2] @ right_rotation
    full_left_rotation = np.eye(3)
    full_right_rotation = np.eye(3)
    full_left_rotation[:2, :2] = left_rotation
    full_right_rotation[:2, :2] = right_rotation
    rotated_core = (
        full_left_rotation.T
        @ np.diag(source.singular_values)
        @ full_right_rotation
    )
    rotated_source = SourceDecomposition(
        u=rotated_u,
        singular_values=source.singular_values,
        vt=rotated_v.T,
        next_singular_value=source.next_singular_value,
        source_shape=source.source_shape,
        coordinate_core=rotated_core,
    )

    # Use a rank-one target wholly inside the tied block, but not aligned with
    # either canonical source direction.  Its ambient matrix stays fixed under
    # both coordinate systems.
    target_left = source.left_vectors[:, :2] @ np.array([0.6, 0.8])
    target_right = source.right_vectors[:, :2] @ np.array([-0.8, 0.6])
    target = 4.0 * np.outer(target_left, target_right)
    design = np.tile(np.eye(4), (8, 1))
    initialization = FoldData(design, design @ target, name="initialization")
    fitting = FoldData(design.copy(), design @ target, name="fitting")

    canonical_screen = run_block_screen(
        initialization,
        source,
        partition,
        target_rank=1,
        left_budget=2,
        right_budget=2,
    )
    rotated_screen = run_block_screen(
        initialization,
        rotated_source,
        partition,
        target_rank=1,
        left_budget=2,
        right_budget=2,
    )

    assert canonical_screen.status is CandidateStatus.SUCCESSFUL
    assert rotated_screen.status is CandidateStatus.SUCCESSFUL
    assert canonical_screen.left_blocks == rotated_screen.left_blocks == (0,)
    assert canonical_screen.right_blocks == rotated_screen.right_blocks == (0,)
    assert canonical_screen.pilot_matrix is not None
    assert rotated_screen.pilot_matrix is not None
    assert canonical_screen.frozen_matrix is not None
    assert rotated_screen.frozen_matrix is not None
    np.testing.assert_allclose(
        rotated_screen.pilot_matrix,
        canonical_screen.pilot_matrix,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        rotated_screen.frozen_matrix,
        canonical_screen.frozen_matrix,
        atol=1e-11,
    )
    np.testing.assert_allclose(canonical_screen.pilot_matrix, target, atol=1e-11)
    np.testing.assert_allclose(canonical_screen.frozen_matrix, target, atol=1e-11)

    canonical_rrr = initialize_restricted_rrr(
        fitting,
        source,
        canonical_screen,
        target_rank=1,
    )
    rotated_rrr = initialize_restricted_rrr(
        fitting,
        rotated_source,
        rotated_screen,
        target_rank=1,
    )
    assert canonical_rrr.successful
    assert rotated_rrr.successful
    assert canonical_rrr.candidate.matrix is not None
    assert rotated_rrr.candidate.matrix is not None
    np.testing.assert_allclose(
        rotated_rrr.candidate.matrix,
        canonical_rrr.candidate.matrix,
        atol=1e-11,
    )
    np.testing.assert_allclose(canonical_rrr.candidate.matrix, target, atol=1e-11)

    assert canonical_rrr.state is not None
    assert rotated_rrr.state is not None
    np.testing.assert_allclose(
        fitted_target(rotated_rrr.state),
        fitted_target(canonical_rrr.state),
        atol=1e-11,
    )
    np.testing.assert_allclose(fitted_target(canonical_rrr.state), target, atol=1e-11)
    np.testing.assert_allclose(
        fitted_source(rotated_rrr.state),
        fitted_source(canonical_rrr.state),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        fitted_source(canonical_rrr.state),
        source.approximation,
        atol=1e-11,
    )


def test_initializer_rejects_coordinate_core_that_crosses_partition_blocks() -> None:
    """A globally isospectral core cannot silently move mass across blocks."""

    observed_source = np.diag([7.0, 5.0, 3.0, 0.0])
    library = build_source_block_library(
        observed_source,
        source_rank=3,
        source_error_bound=0.1,
        c_gap=3.0,
    )
    source = library.decomposition
    partition = library.partitions[0]
    assert partition.blocks == ((0,), (1,), (2,))

    # This core has the correct global singular values (7, 5, 3), so the
    # SourceDecomposition is globally valid, but it swaps the first two
    # singleton blocks and is invalid for this certified partition.
    malformed_core = np.array(
        [
            [0.0, 7.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    malformed_source = SourceDecomposition(
        u=source.u,
        singular_values=source.singular_values,
        vt=source.vt,
        next_singular_value=source.next_singular_value,
        source_shape=source.source_shape,
        coordinate_core=malformed_core,
    )
    target = np.diag([4.0, 0.0, 0.0, 0.0])
    design = np.tile(np.eye(4), (6, 1))
    initialization = FoldData(design, design @ target, name="initialization")
    fitting = FoldData(design.copy(), design @ target, name="fitting")
    screen = run_block_screen(
        initialization,
        malformed_source,
        partition,
        target_rank=1,
        left_budget=1,
        right_budget=1,
    )
    assert screen.status is CandidateStatus.SUCCESSFUL

    result = initialize_restricted_rrr(
        fitting,
        malformed_source,
        screen,
        target_rank=1,
    )
    assert not result.successful
    assert result.candidate.failure_reason is FailureReason.INVALID_INPUT
    assert "across" in result.candidate.message
