"""Deterministic tests for source certification and automatic block geometry."""

import numpy as np
import pytest

from bi_smart.source_blocks import (
    build_nested_partitions,
    build_source_block_library,
    certified_cut_positions,
    compute_source_decomposition,
    evaluate_wedin_gate,
    gaussian_source_error_bound,
    observable_partition_gaps,
    passes_wedin_gate,
    source_rank_boundary_gap,
    source_rank_boundary_passes,
)
from bi_smart.types import BlockPartition, SourceDecomposition


def test_gaussian_source_certificate_matches_appendix_formula():
    certificate = gaussian_source_error_bound(
        noise_standard_deviation=0.2,
        n_rows=9,
        n_columns=16,
        tail_parameter=2.0,
    )
    assert certificate == pytest.approx(0.2 * (3.0 + 4.0 + 2.0))


def test_source_rank_boundary_uses_strict_inequality():
    # At r0=3 the retained/excluded gap is 2.0 - 0.5 = 1.5, exactly
    # c_gap * epsilon = 3 * 0.5.  The appendix's strict check must fail.
    tied_to_threshold = compute_source_decomposition(
        np.diag([8.0, 5.0, 2.0, 0.5]), source_rank=3
    )
    assert source_rank_boundary_gap(tied_to_threshold) == pytest.approx(1.5)
    assert not source_rank_boundary_passes(
        tied_to_threshold, source_error_bound=0.5, c_gap=3.0
    )

    above_threshold = compute_source_decomposition(
        np.diag([8.0, 5.0, 2.0, 0.49]), source_rank=3
    )
    assert source_rank_boundary_passes(
        above_threshold, source_error_bound=0.5, c_gap=3.0
    )


def test_zero_error_certificate_does_not_promote_roundoff_sized_gaps():
    """An exact population tie represented a few ulps apart stays unresolved."""

    rounded_delta = 8.0 * np.finfo(float).eps
    rounded_boundary_tie = SourceDecomposition(
        u=np.array([[1.0], [0.0]]),
        singular_values=np.array([1.0 + rounded_delta]),
        vt=np.array([[1.0, 0.0]]),
        next_singular_value=1.0,
        source_shape=(2, 2),
    )
    assert not source_rank_boundary_passes(
        rounded_boundary_tie,
        source_error_bound=0.0,
        atol=1e-12,
        rtol=1e-10,
    )
    # Zero tolerances recover the literal strict floating-point comparison.
    assert source_rank_boundary_passes(
        rounded_boundary_tie,
        source_error_bound=0.0,
        atol=0.0,
        rtol=0.0,
    )

    rounded_internal_tie = SourceDecomposition(
        u=np.eye(2),
        singular_values=np.array([1.0 + rounded_delta, 1.0]),
        vt=np.eye(2),
        next_singular_value=0.0,
        source_shape=(2, 2),
    )
    assert certified_cut_positions(
        rounded_internal_tie,
        source_error_bound=0.0,
        atol=1e-12,
        rtol=1e-10,
    ) == ()
    assert certified_cut_positions(
        rounded_internal_tie,
        source_error_bound=0.0,
        atol=0.0,
        rtol=0.0,
    ) == (1,)


def test_source_library_records_and_applies_comparison_tolerances():
    source = np.diag([1.0 + 1e-8, 1.0])
    library = build_source_block_library(
        source,
        source_rank=1,
        source_error_bound=0.0,
        atol=1e-7,
        rtol=0.0,
    )

    assert not library.boundary_passes
    assert library.atol == pytest.approx(1e-7)
    assert library.rtol == pytest.approx(0.0)


def test_certified_cuts_and_equal_gap_coarsening_are_deterministic():
    # All three internal observed gaps equal 3.  Their removal order must be
    # cut 1, then cut 2, then cut 3 by the smaller-index tie breaker.
    source = np.diag([11.0, 8.0, 5.0, 2.0, 0.0])
    library = build_source_block_library(
        source,
        source_rank=4,
        source_error_bound=0.3,
        c_gap=3.0,
    )

    assert library.boundary_passes
    assert library.certified_cuts == (1, 2, 3)
    assert certified_cut_positions(
        library.decomposition, source_error_bound=0.3, c_gap=3.0
    ) == (1, 2, 3)
    assert [partition.cut_positions for partition in library.partitions] == [
        (1, 2, 3),
        (2, 3),
        (3,),
        (),
    ]
    assert library.partitions[-1].blocks == ((0, 1, 2, 3),)


def test_nested_partitions_remove_smallest_gap_before_index_ties():
    # Cuts 1 and 3 both have gap 2; cut 4 has gap 1 and is removed first.
    partitions = build_nested_partitions(
        source_rank=5,
        cuts=(1, 3, 4),
        internal_gaps=(2.0, 0.1, 2.0, 1.0),
    )
    assert [partition.cut_positions for partition in partitions] == [
        (1, 3, 4),
        (1, 3),
        (3,),
        (),
    ]


def test_nested_partitions_break_numerical_gap_ties_by_cut_position():
    """Roundoff within the configured guard cannot reverse the canonical order."""

    partitions = build_nested_partitions(
        source_rank=4,
        cuts=(1, 3),
        internal_gaps=(2.0 + 5e-12, 0.1, 2.0),
        atol=1e-11,
        rtol=0.0,
    )
    assert [partition.cut_positions for partition in partitions] == [
        (1, 3),
        (3,),
        (),
    ]


@pytest.mark.parametrize(
    ("u", "vt", "invalid_name"),
    [
        (np.array([[1.0, 0.0], [0.0, 1.001]]), np.eye(2), "u"),
        (np.eye(2), np.array([[1.0, 0.0], [0.0, 1.001]]), "vt.T"),
    ],
)
def test_public_source_decomposition_requires_orthonormal_bases(
    u, vt, invalid_name
):
    """Direct construction enforces the Stiefel constraints used downstream."""

    with pytest.raises(ValueError, match=rf"{invalid_name} must have orthonormal"):
        SourceDecomposition(
            u=u,
            singular_values=np.array([3.0, 1.0]),
            vt=vt,
            next_singular_value=0.0,
            source_shape=(2, 2),
        )


def test_observable_block_gaps_include_only_existing_boundaries():
    decomposition = compute_source_decomposition(
        np.diag([10.0, 7.0, 4.0, 1.0]), source_rank=4
    )
    partition = BlockPartition(blocks=((0,), (1, 2), (3,)))

    # With epsilon=.5, the first and middle blocks have lower gap 2.0;
    # the last block is limited by separation from the null space: 1-.5=.5.
    assert observable_partition_gaps(decomposition, partition, 0.5) == pytest.approx(
        (2.0, 2.0, 0.5)
    )
    gate = evaluate_wedin_gate(decomposition, partition, 0.5)
    assert gate.threshold == pytest.approx(2.0)
    assert not gate.passes
    assert gate.failing_blocks == (2,)
    assert gate.minimum_gap == pytest.approx(0.5)


def test_wedin_gate_passes_at_its_nonstrict_threshold_and_fails_below():
    partition = BlockPartition.from_cut_positions(rank=1, cuts=())

    # For one block and epsilon=1, gamma_lower = d_hat_1 - 1.  A singular
    # value of 5 gives equality with the gate threshold 4, which must pass.
    at_threshold = compute_source_decomposition(
        np.array([[5.0, 0.0], [0.0, 0.0]]), source_rank=1
    )
    passing = evaluate_wedin_gate(
        at_threshold,
        partition,
        1.0,
        atol=0.0,
        rtol=0.0,
    )
    assert passing.block_gaps == pytest.approx((4.0,))
    assert passing.passes
    assert passing.failing_blocks == ()

    below_threshold = compute_source_decomposition(
        np.array([[4.9, 0.0], [0.0, 0.0]]), source_rank=1
    )
    failing = evaluate_wedin_gate(
        below_threshold,
        partition,
        1.0,
        atol=0.0,
        rtol=0.0,
    )
    assert failing.block_gaps == pytest.approx((3.9,))
    assert not failing.passes
    assert failing.failing_blocks == (0,)


def test_wedin_gate_treats_a_tolerance_sized_shortfall_as_equality():
    """Roundoff just below a non-strict boundary cannot flip the gate."""

    partition = BlockPartition.from_cut_positions(rank=1, cuts=())
    shortfall = 5e-11
    nearly_equal = compute_source_decomposition(
        np.diag([5.0 - shortfall, 0.0]),
        source_rank=1,
    )

    # The active singular-value scale is approximately five, so rtol=1e-10
    # gives a roughly 5e-10 guard and treats this shortfall as equality.
    tolerant = evaluate_wedin_gate(
        nearly_equal,
        partition,
        1.0,
        atol=0.0,
        rtol=1e-10,
    )
    assert tolerant.passes
    assert tolerant.failing_blocks == ()
    assert passes_wedin_gate(
        nearly_equal,
        partition,
        1.0,
        atol=0.0,
        rtol=1e-10,
    )

    # Zero tolerances recover the literal comparison, under which the same
    # representable value is genuinely below the threshold.
    literal = evaluate_wedin_gate(
        nearly_equal,
        partition,
        1.0,
        atol=0.0,
        rtol=0.0,
    )
    assert not literal.passes
    assert literal.failing_blocks == (0,)

    # The tolerance is a roundoff guard, not a relaxation of the statistical
    # gate: a shortfall beyond the local atol/rtol margin must still fail.
    outside_margin = compute_source_decomposition(
        np.diag([5.0 - 1e-9, 0.0]),
        source_rank=1,
    )
    genuinely_below = evaluate_wedin_gate(
        outside_margin,
        partition,
        1.0,
        atol=0.0,
        rtol=1e-10,
    )
    assert not genuinely_below.passes
    assert genuinely_below.failing_blocks == (0,)


@pytest.mark.parametrize(
    ("tolerance_name", "invalid_value", "message"),
    [
        ("atol", -1.0, "finite and nonnegative"),
        ("rtol", np.inf, "finite and nonnegative"),
        ("atol", 1e-12 + 1e-14j, "real-valued"),
    ],
)
def test_wedin_gate_rejects_invalid_tolerances(
    tolerance_name: str,
    invalid_value: object,
    message: str,
) -> None:
    """The standalone public gate enforces the package tolerance contract."""

    decomposition = compute_source_decomposition(np.diag([5.0, 0.0]), source_rank=1)
    partition = BlockPartition.from_cut_positions(rank=1, cuts=())
    kwargs = {tolerance_name: invalid_value}

    with pytest.raises(ValueError, match=message):
        evaluate_wedin_gate(decomposition, partition, 1.0, **kwargs)
