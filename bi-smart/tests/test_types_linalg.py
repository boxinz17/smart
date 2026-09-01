"""Focused tests for BI-SMART value objects and strict linear algebra."""

import numpy as np
import pytest

from bi_smart.linalg import (
    spd_inverse_sqrt,
    spd_solve,
    strict_polar_factor,
    strict_spd_eigh,
    strict_truncated_svd,
)
from bi_smart.types import (
    BISMARTConfig,
    BlockPartition,
    FailureReason,
    FoldData,
    NumericalFailure,
    RefinementControls,
)


def test_config_normalizes_and_validates_structural_inputs():
    config = BISMARTConfig(
        target_rank=np.int64(2),
        source_rank=np.int64(5),
        source_error_bound=0.25,
        budget_path=((2, 3), (5, 5)),
        source_weights=(0.1, 1.0),
    )

    assert config.target_rank == 2
    assert config.source_rank == 5
    assert config.budget_path == ((2, 3), (5, 5))
    assert config.source_weights == (0.1, 1.0)
    assert config.c_gap == 3.0

    with pytest.raises(ValueError, match="1 <= target_rank <= source_rank"):
        BISMARTConfig(3, 2, 0.1, ((3, 3),), (1.0,))
    with pytest.raises(ValueError, match="budget pair"):
        BISMARTConfig(2, 5, 0.1, ((1, 5),), (1.0,))
    with pytest.raises(ValueError, match="strictly larger than 2"):
        BISMARTConfig(2, 5, 0.1, ((2, 2),), (1.0,), c_gap=2.0)
    with pytest.raises(ValueError, match="positive finite"):
        BISMARTConfig(2, 5, 0.1, ((2, 2),), (0.0,))
    with pytest.raises(ValueError, match="duplicate budget"):
        BISMARTConfig(2, 5, 0.1, ((2, 2), (2, 2)), (1.0,))
    with pytest.raises(ValueError, match="duplicate values"):
        BISMARTConfig(2, 5, 0.1, ((2, 2),), (1.0, 1.0))


def test_refinement_controls_represent_every_fixed_algorithm_constant():
    controls = RefinementControls(
        armijo_constant=1e-4,
        contraction=0.5,
        initial_step_size=1.0,
        radius_ratio=2.0,
        radius_half_width=2,
        max_backtracking_cap=4,
        iteration_cap=5,
    )
    assert controls.radius_half_width == 2
    assert controls.max_backtracking_cap == 4
    assert controls.iteration_cap == 5

    with pytest.raises(ValueError, match="strictly larger than one"):
        RefinementControls(1e-4, 0.5, 1.0, 1.0, 1, 2, 3)
    with pytest.raises(ValueError, match="both be positive"):
        RefinementControls(1e-4, 0.5, 1.0, 2.0, 1, 0, 3)


def test_fold_data_checks_rows_finiteness_and_reports_dimensions():
    fold = FoldData(
        X=np.arange(12.0).reshape(4, 3),
        Y=np.arange(8.0).reshape(4, 2),
        name="initialization",
    )
    assert (fold.n_samples, fold.n_features, fold.n_responses) == (4, 3, 2)

    with pytest.raises(ValueError, match="same number of rows"):
        FoldData(np.ones((3, 2)), np.ones((4, 1)), name="bad")
    with pytest.raises(ValueError, match="finite"):
        FoldData(np.array([[1.0], [np.nan]]), np.ones((2, 1)), name="bad")


def test_block_partition_construction_and_union_validation():
    partition = BlockPartition.from_cut_positions(rank=6, cuts=(1, 4))

    assert partition.blocks == ((0,), (1, 2, 3), (4, 5))
    assert partition.rank == 6
    assert partition.block_sizes == (1, 3, 2)
    assert partition.cut_positions == (1, 4)
    assert partition.union_indices((0, 2)) == (0, 4, 5)
    assert partition.union_dimension((0, 2)) == 3

    with pytest.raises(ValueError, match="unique"):
        BlockPartition.from_cut_positions(6, (1, 1))
    with pytest.raises(ValueError, match="ordered, consecutive"):
        BlockPartition(blocks=((0, 2), (1, 3)))
    with pytest.raises(TypeError, match="integer"):
        BlockPartition.from_cut_positions(6, (1.5,))
    with pytest.raises(ValueError, match="unique"):
        partition.union_indices((1, 1))


def test_strict_truncated_svd_accepts_only_nonzero_unique_cutoff():
    result = strict_truncated_svd(
        np.diag([5.0, 3.0, 1.0]),
        rank=2,
        atol=0.0,
        rtol=0.0,
    )
    assert result.rank == 2
    assert result.next_singular_value == pytest.approx(1.0)
    assert np.allclose(result.approximation, np.diag([5.0, 3.0, 0.0]))

    with pytest.raises(NumericalFailure) as tied:
        strict_truncated_svd(
            np.diag([5.0, 2.0, 2.0]), rank=2, atol=0.0, rtol=0.0
        )
    assert tied.value.reason is FailureReason.NON_UNIQUE_RANK_CUTOFF

    with pytest.raises(NumericalFailure) as zero:
        strict_truncated_svd(
            np.diag([5.0, 0.0, 0.0]), rank=2, atol=0.0, rtol=0.0
        )
    assert zero.value.reason is FailureReason.ZERO_RANK_COMPONENT


def test_spd_helpers_solve_and_reject_singular_or_asymmetric_matrices():
    gram = np.array([[4.0, 1.0], [1.0, 3.0]])
    right_hand_side = np.array([[1.0, 2.0], [3.0, 4.0]])

    inverse_sqrt = spd_inverse_sqrt(gram)
    assert np.allclose(inverse_sqrt @ gram @ inverse_sqrt, np.eye(2))
    assert np.allclose(gram @ spd_solve(gram, right_hand_side), right_hand_side)

    with pytest.raises(NumericalFailure) as singular:
        strict_spd_eigh(np.diag([1.0, 0.0]), atol=0.0, rtol=0.0)
    assert singular.value.reason is FailureReason.NON_POSITIVE_DEFINITE

    with pytest.raises(NumericalFailure) as asymmetric:
        strict_spd_eigh(
            np.array([[2.0, 1.0], [0.0, 2.0]]), atol=0.0, rtol=0.0
        )
    assert asymmetric.value.reason is FailureReason.NON_POSITIVE_DEFINITE


def test_strict_polar_factor_is_orthonormal_and_rejects_deficiency():
    argument = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    factor = strict_polar_factor(argument)
    assert factor.shape == (3, 2)
    assert np.allclose(factor.T @ factor, np.eye(2))

    with pytest.raises(NumericalFailure) as deficient:
        strict_polar_factor(
            np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            atol=0.0,
            rtol=0.0,
        )
    assert deficient.value.reason is FailureReason.RANK_DEFICIENT_POLAR
