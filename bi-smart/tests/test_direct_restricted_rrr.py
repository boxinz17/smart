"""Focused tests for the full-sample, full-source-span RRR pilot API."""

from __future__ import annotations

import numpy as np
import pytest

import bi_smart
from bi_smart.initialization import initialize_restricted_rrr, restricted_rrr
from bi_smart.refinement import fitted_target
from bi_smart.source_blocks import compute_source_decomposition
from bi_smart.types import (
    BlockPartition,
    CandidateStatus,
    FailureReason,
    FoldData,
    ScreenResult,
)


def _orthonormal_columns(
    rng: np.random.Generator,
    rows: int,
    columns: int,
) -> np.ndarray:
    """Return a reproducible rectangular orthonormal basis."""

    q, _ = np.linalg.qr(rng.normal(size=(rows, columns)))
    return q[:, :columns]


def _noiseless_problem() -> tuple[np.ndarray, ...]:
    """Construct a rank-two target inside rank-four source subspaces."""

    rng = np.random.default_rng(20260902)
    n, p, q, source_rank = 40, 6, 5, 4
    left = _orthonormal_columns(rng, p, source_rank)
    right = _orthonormal_columns(rng, q, source_rank)
    observed_source = (left * np.array([9.0, 7.0, 5.0, 3.0])) @ right.T

    # The target coordinate is not diagonal in the source SVD coordinates;
    # this exercises a genuine reduced-rank regression rather than selecting
    # the first two source singular components verbatim.
    left_rotation = _orthonormal_columns(rng, source_rank, 2)
    right_rotation = _orthonormal_columns(rng, source_rank, 2)
    coordinate = (left_rotation * np.array([2.5, 1.25])) @ right_rotation.T
    coefficient = left @ coordinate @ right.T
    X = rng.normal(size=(n, p))
    Y = X @ coefficient
    return X, Y, observed_source, coefficient, left, right


def test_direct_rrr_is_exported_and_recovers_the_full_span_target() -> None:
    """The root API uses every row and exactly recovers a noiseless target."""

    X, Y, observed_source, coefficient, left, right = _noiseless_problem()
    result = bi_smart.restricted_rrr(
        X,
        Y,
        observed_source,
        target_rank=2,
        source_rank=4,
    )

    assert bi_smart.restricted_rrr is restricted_rrr
    assert result.successful
    assert result.coefficient is result.candidate.matrix
    assert result.coefficient is not None
    np.testing.assert_allclose(result.coefficient, coefficient, atol=2e-12)
    assert np.linalg.matrix_rank(result.coefficient, tol=1e-10) == 2

    # The hard restriction is a geometric property of the returned matrix,
    # independent of which bases the source SVD chose inside its spans.
    np.testing.assert_allclose(
        (np.eye(left.shape[0]) - left @ left.T) @ result.coefficient,
        0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.coefficient @ (np.eye(right.shape[0]) - right @ right.T),
        0.0,
        atol=2e-12,
    )

    assert result.state is not None
    assert result.state.block_sizes == (4,)
    assert np.all(result.state.active_u)
    assert np.all(result.state.active_v)
    np.testing.assert_allclose(fitted_target(result.state), coefficient, atol=2e-12)

    metadata = result.candidate.metadata
    assert metadata["method"] == "full_sample_restricted_rrr"
    assert metadata["n_samples"] == X.shape[0]
    assert metadata["source_rank"] == 4
    assert len(metadata["source_singular_values"]) == 4
    assert metadata["source_boundary_gap"] > 0.0
    assert metadata["reduced_gram_condition"] >= 1.0
    assert metadata["rrr_cutoff_gap"] > 0.0


def test_direct_rrr_matches_screened_rrr_on_the_complete_one_block_span() -> None:
    """Extracting the common core does not change the established estimator."""

    X, Y, observed_source, _, _, _ = _noiseless_problem()
    fitting = FoldData(X, Y, name="fitting")
    source = compute_source_decomposition(observed_source, source_rank=4)
    partition = BlockPartition.from_cut_positions(4, ())

    # Only the recorded full-block support is consumed by RRR.  The remaining
    # fields make this a valid successful ScreenResult but are intentionally
    # irrelevant to the fitting-fold numerical core.
    screen = ScreenResult(
        status=CandidateStatus.SUCCESSFUL,
        partition=partition,
        left_blocks=(0,),
        right_blocks=(0,),
        statistic=np.zeros((4, 4)),
        left_factor=np.zeros((4, 2)),
        right_factor=np.zeros((4, 2)),
        pilot_matrix=np.zeros(observed_source.shape),
        frozen_matrix=np.zeros(observed_source.shape),
    )
    screened = initialize_restricted_rrr(
        fitting,
        source,
        screen,
        target_rank=2,
    )
    direct = restricted_rrr(X, Y, observed_source, 2, 4)

    assert screened.successful and direct.successful
    np.testing.assert_allclose(direct.coefficient, screened.coefficient, atol=2e-12)
    assert (
        direct.candidate.metadata["rrr_cutoff_gap"]
        == pytest.approx(screened.candidate.metadata["rrr_cutoff_gap"])
    )


def test_direct_rrr_depends_on_source_spans_not_internal_source_bases() -> None:
    """Changing the source matrix inside fixed subspaces leaves the fit fixed."""

    rng = np.random.default_rng(81)
    n, p, q, source_rank = 35, 5, 4, 3
    left = _orthonormal_columns(rng, p, source_rank)
    right = _orthonormal_columns(rng, q, source_rank)
    left_rotation = _orthonormal_columns(rng, source_rank, source_rank)
    right_rotation = _orthonormal_columns(rng, source_rank, source_rank)
    source_a = (left * np.array([8.0, 5.0, 2.0])) @ right.T
    source_b = (
        left
        @ left_rotation
        @ np.diag([10.0, 6.0, 1.0])
        @ right_rotation.T
        @ right.T
    )
    coordinate = np.diag([3.0, 1.0, 0.0])
    coefficient = left @ coordinate @ right.T
    X = rng.normal(size=(n, p))
    Y = X @ coefficient

    fit_a = restricted_rrr(X, Y, source_a, 2, source_rank)
    fit_b = restricted_rrr(X, Y, source_b, 2, source_rank)

    assert fit_a.successful and fit_b.successful
    np.testing.assert_allclose(fit_a.coefficient, coefficient, atol=3e-12)
    np.testing.assert_allclose(fit_b.coefficient, coefficient, atol=3e-12)


def test_tied_source_boundary_is_recorded_but_does_not_veto_direct_rrr() -> None:
    """The minimal pilot intentionally omits source-boundary certification."""

    observed_source = np.diag([4.0, 2.0, 2.0])
    source_u, _, source_vt = np.linalg.svd(observed_source, full_matrices=False)
    coefficient = 3.0 * np.outer(source_u[:, 0], source_vt[0, :])
    X = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    )
    result = restricted_rrr(X, X @ coefficient, observed_source, 1, 2)

    assert result.successful
    assert result.candidate.metadata["source_boundary_gap"] == pytest.approx(0.0)
    np.testing.assert_allclose(result.coefficient, coefficient, atol=1e-12)


def test_direct_rrr_reports_singular_design_and_zero_cutoff_failures() -> None:
    """Undefined fits are explicit records and never receive fallback estimates."""

    observed_source = np.diag([5.0, 3.0, 1.0])
    singular = restricted_rrr(
        np.ones((6, 3)),
        np.ones((6, 3)),
        observed_source,
        2,
        3,
    )
    assert not singular.successful
    assert singular.coefficient is None
    assert singular.state is None
    assert singular.candidate.failure_reason is FailureReason.NON_POSITIVE_DEFINITE
    assert "reduced_gram_eigenvalues" in singular.candidate.metadata

    zero = restricted_rrr(
        np.eye(3),
        np.zeros((3, 3)),
        observed_source,
        2,
        3,
    )
    assert not zero.successful
    assert zero.coefficient is None
    assert zero.candidate.failure_reason is FailureReason.ZERO_RANK_COMPONENT
    assert zero.candidate.metadata["rrr_retained_singular_value"] == 0.0


def test_direct_rrr_reports_a_tied_target_rank_cutoff() -> None:
    """A nonunique exact-r truncation is not resolved by an arbitrary SVD basis."""

    result = restricted_rrr(
        np.eye(3),
        np.diag([3.0, 1.0, 1.0]),
        np.diag([5.0, 3.0, 1.0]),
        2,
        3,
        atol=0.0,
        rtol=1e-12,
    )

    assert not result.successful
    assert result.candidate.failure_reason is FailureReason.NON_UNIQUE_RANK_CUTOFF
    assert result.candidate.metadata["rrr_cutoff_gap"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("arguments", "exception", "message"),
    [
        ({"target_rank": 0, "source_rank": 2}, ValueError, "target_rank"),
        ({"target_rank": 3, "source_rank": 2}, ValueError, "target_rank"),
        ({"target_rank": 1, "source_rank": 0}, ValueError, "source_rank"),
        ({"target_rank": True, "source_rank": 2}, TypeError, "target_rank"),
        ({"target_rank": 1, "source_rank": False}, TypeError, "source_rank"),
        (
            {"target_rank": 1, "source_rank": 2, "atol": -1.0},
            ValueError,
            "atol",
        ),
    ],
)
def test_direct_rrr_rejects_invalid_controls(
    arguments: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    """Ranks and tolerances remain explicit validated inputs."""

    with pytest.raises(exception, match=message):
        restricted_rrr(
            np.eye(3),
            np.eye(3),
            np.diag([3.0, 2.0, 1.0]),
            **arguments,
        )


def test_direct_rrr_rejects_bad_matrices_without_silent_complex_casts() -> None:
    """Matrix compatibility and the real-valued contract are checked up front."""

    with pytest.raises(ValueError, match="same number of rows"):
        restricted_rrr(np.eye(3), np.ones((2, 3)), np.eye(3), 1, 2)
    with pytest.raises(ValueError, match="observed_source must have shape"):
        restricted_rrr(np.ones((4, 3)), np.ones((4, 2)), np.eye(3), 1, 2)
    with pytest.raises(ValueError, match="real-valued"):
        restricted_rrr(
            np.eye(3),
            np.eye(3),
            np.eye(3, dtype=complex) * (1.0 + 1.0j),
            1,
            2,
        )


def test_direct_rrr_never_calls_screen_gate_or_source_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pilot path is mechanically independent of the complete procedure."""

    import bi_smart.screening as screening
    import bi_smart.source_blocks as source_blocks

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("complete-procedure machinery was called")

    monkeypatch.setattr(screening, "run_block_screen", forbidden)
    monkeypatch.setattr(source_blocks, "build_source_block_library", forbidden)
    monkeypatch.setattr(source_blocks, "evaluate_wedin_gate", forbidden)

    X, Y, observed_source, coefficient, _, _ = _noiseless_problem()
    result = restricted_rrr(X, Y, observed_source, 2, 4)

    assert result.successful
    np.testing.assert_allclose(result.coefficient, coefficient, atol=2e-12)
