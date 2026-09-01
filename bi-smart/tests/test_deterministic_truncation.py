"""Tests for the target-only safeguard's fixed tied-cutoff convention."""

import numpy as np

import bi_smart.linalg as linalg
from bi_smart.initialization import target_only_rrr
from bi_smart.linalg import (
    deterministic_rank_at_most_approximation,
    truncated_svd,
)
from bi_smart.types import CandidateStatus, FoldData


def test_positive_tie_uses_coordinate_lexicographic_subspace():
    matrix = np.diag([3.0, 2.0, 2.0])

    approximation = deterministic_rank_at_most_approximation(
        matrix,
        rank=2,
        atol=0.0,
        rtol=0.0,
    )

    # The tied left projector is diag(0, 1, 1).  Scanning ambient coordinates
    # selects e_1 before e_2, so the fixed rank-two solution keeps coordinates
    # zero and one.  Its error still equals the optimal excluded singular value.
    assert np.allclose(approximation, np.diag([3.0, 2.0, 0.0]))
    assert np.linalg.matrix_rank(approximation) == 2
    assert np.linalg.norm(matrix - approximation, ord="fro") == 2.0


def test_tied_result_is_independent_of_raw_svd_basis(monkeypatch):
    left, _ = np.linalg.qr(
        np.array(
            [
                [1.0, 2.0, -1.0],
                [2.0, 0.5, 1.0],
                [-1.0, 1.0, 2.0],
            ]
        )
    )
    right, _ = np.linalg.qr(
        np.array(
            [
                [2.0, -1.0, 0.5],
                [1.0, 2.0, 1.0],
                [0.5, 1.0, -2.0],
            ]
        )
    )
    matrix = left @ np.diag([4.0, 2.0, 2.0]) @ right.T
    expected = deterministic_rank_at_most_approximation(matrix, rank=2)

    original_svd = linalg.np.linalg.svd
    angle = 0.713
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    def svd_with_rotated_tied_basis(value, *args, **kwargs):
        u, singular_values, vt = original_svd(value, *args, **kwargs)
        tied = np.flatnonzero(np.isclose(singular_values, 2.0, atol=1e-10, rtol=1e-10))
        assert tuple(tied) == (1, 2)
        u = u.copy()
        vt = vt.copy()
        u[:, tied] = u[:, tied] @ rotation
        vt[tied, :] = rotation.T @ vt[tied, :]
        return u, singular_values, vt

    monkeypatch.setattr(linalg.np.linalg, "svd", svd_with_rotated_tied_basis)
    rotated_basis_result = deterministic_rank_at_most_approximation(matrix, rank=2)

    # Rotating U and V together is a valid alternate SVD of the same tied
    # matrix.  A raw-basis truncation changes; the projector-based rule does not.
    assert np.allclose(rotated_basis_result, expected, atol=1e-11, rtol=1e-11)


def test_zero_cutoff_returns_unique_positive_rank_reconstruction():
    rank_one = np.array(
        [
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    approximation = deterministic_rank_at_most_approximation(
        rank_one,
        rank=2,
        atol=0.0,
        rtol=0.0,
    )
    assert np.allclose(approximation, rank_one)
    assert np.linalg.matrix_rank(approximation) == 1

    zero = np.zeros((3, 3))
    assert np.array_equal(
        deterministic_rank_at_most_approximation(zero, rank=2),
        zero,
    )


def test_nontied_cutoff_matches_ordinary_truncated_svd():
    matrix = np.array(
        [
            [4.0, 1.0, 0.0],
            [0.0, 2.0, 0.5],
            [0.0, 0.0, 0.25],
            [1.0, -1.0, 0.0],
        ]
    )
    expected = truncated_svd(matrix, rank=2).approximation
    actual = deterministic_rank_at_most_approximation(matrix, rank=2)
    assert np.allclose(actual, expected)


def test_target_only_rrr_uses_fixed_tie_and_zero_cutoff_rules():
    identity = np.eye(3)

    tied_fold = FoldData(identity, np.diag([3.0, 2.0, 2.0]), name="fitting")
    tied_candidate = target_only_rrr(tied_fold, target_rank=2)
    assert tied_candidate.status is CandidateStatus.SUCCESSFUL
    assert np.allclose(tied_candidate.matrix, np.diag([3.0, 2.0, 0.0]))

    rank_one = np.diag([3.0, 0.0, 0.0])
    zero_cutoff_fold = FoldData(identity, rank_one, name="fitting")
    zero_cutoff_candidate = target_only_rrr(zero_cutoff_fold, target_rank=2)
    assert zero_cutoff_candidate.status is CandidateStatus.SUCCESSFUL
    assert np.allclose(zero_cutoff_candidate.matrix, rank_one)


def test_helper_is_exported_from_linalg():
    assert "deterministic_rank_at_most_approximation" in linalg.__all__
