"""Exact-stage integration tests for the three-fold BI-SMART wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from bi_smart.estimator import BISMART
from bi_smart.types import (
    BISMARTConfig,
    CandidateStatus,
    FailureReason,
    FoldData,
    RefinementControls,
)


def _folds_for_target(target: np.ndarray) -> tuple[FoldData, FoldData, FoldData]:
    """Create deterministic full-rank folds with exact target responses."""

    design = np.tile(np.eye(target.shape[0]), (6, 1))
    return (
        FoldData(design.copy(), design @ target, name="initialization"),
        FoldData(design.copy(), design @ target, name="fitting"),
        FoldData(design.copy(), design @ target, name="validation"),
    )


def _config() -> BISMARTConfig:
    """Use one small explicit budget/weight path for deterministic tests."""

    return BISMARTConfig(
        target_rank=2,
        source_rank=3,
        source_error_bound=0.1,
        budget_path=((2, 2),),
        source_weights=(1.0,),
        c_gap=3.0,
    )


def test_exact_stage_end_to_end_retains_all_safeguard_kinds() -> None:
    """The executable pipeline selects an exact fit without refinement."""

    observed_source = np.diag([9.0, 6.0, 3.0, 0.0])
    target = np.diag([4.0, 0.0, 2.0, 0.0])
    initialization, fitting, validation = _folds_for_target(target)

    estimator = BISMART(observed_source, _config())
    result = estimator.fit_folds(initialization, fitting, validation)

    assert result.source_library is not None
    assert result.source_library.boundary_passes
    assert all(
        candidate.status is CandidateStatus.SUCCESSFUL
        for candidate in result.candidates
    )
    kinds = {candidate.kind for candidate in result.candidates}
    assert {"pilot", "frozen", "restricted_rrr", "target_only_rrr", "zero"} <= kinds
    assert all("validation_loss" in candidate.metadata for candidate in result.candidates)
    assert len({candidate.label for candidate in result.candidates}) == len(
        result.candidates
    )
    assert len({candidate.order_key for candidate in result.candidates}) == len(
        result.candidates
    )
    np.testing.assert_allclose(result.coefficient, target, atol=1e-11)

    estimates = estimator.get_estimates()
    np.testing.assert_allclose(estimates["C_hat"], target, atol=1e-11)
    np.testing.assert_allclose(
        estimates["U"] @ estimates["D"] @ estimates["V"].T,
        target,
        atol=1e-11,
    )
    assert estimates["selected_candidate"] == result.selected_candidate


def test_failed_source_rank_boundary_falls_back_to_target_only_and_zero() -> None:
    """An unresolved r0 boundary removes transfer but not mandatory safeguards."""

    # At r0=3, the observed outer gap is 1.00 - 0.95 = 0.05, below the
    # required strict threshold c_gap * epsilon_hat = 0.30.
    observed_source = np.diag([5.0, 4.0, 1.0, 0.95])
    target = np.diag([3.0, 2.0, 0.0, 0.0])
    initialization, fitting, validation = _folds_for_target(target)

    estimator = BISMART(observed_source, _config())
    result = estimator.fit_folds(initialization, fitting, validation)

    assert result.source_library is not None
    assert not result.source_library.boundary_passes
    assert tuple(candidate.kind for candidate in result.candidates) == (
        "target_only_rrr",
        "zero",
    )
    assert result.selected_candidate.kind == "target_only_rrr"
    np.testing.assert_allclose(result.coefficient, target, atol=1e-11)

    assert len(estimator.branch_diagnostics_) == 1
    diagnostic = estimator.branch_diagnostics_[0]
    assert diagnostic.status is CandidateStatus.UNSUCCESSFUL
    assert diagnostic.failure_reason is FailureReason.SOURCE_RANK_BOUNDARY


def test_refinement_grid_records_missing_backend_and_continues() -> None:
    """Every unsupported refinement call fails locally; safeguards still run."""

    observed_source = np.diag([9.0, 6.0, 3.0, 0.0])
    target = np.diag([4.0, 0.0, 2.0, 0.0])
    initialization, fitting, validation = _folds_for_target(target)
    base = _config()
    config = BISMARTConfig(
        target_rank=base.target_rank,
        source_rank=base.source_rank,
        source_error_bound=base.source_error_bound,
        budget_path=base.budget_path,
        source_weights=base.source_weights,
        refinement_controls=RefinementControls(
            armijo_constant=1e-4,
            contraction=0.5,
            initial_step_size=1.0,
            radius_ratio=2.0,
            radius_half_width=0,
            max_backtracking_cap=1,
            iteration_cap=1,
        ),
    )

    estimator = BISMART(observed_source, config)
    result = estimator.fit_folds(
        initialization,
        fitting,
        validation,
        enable_refinement=True,
    )

    # The missing quotient solve never erases exact t=0 fits or mandatory
    # safeguards, and it never leaks an unanalysed replacement optimizer.
    kinds = {candidate.kind for candidate in result.candidates}
    assert {"restricted_rrr", "target_only_rrr", "zero"} <= kinds
    assert "refinement" not in kinds
    assert any(
        diagnostic.failure_reason is FailureReason.NOT_IMPLEMENTED
        and "PSEUDOCODE" in diagnostic.message
        for diagnostic in estimator.branch_diagnostics_
    )
    np.testing.assert_allclose(result.coefficient, target, atol=1e-11)


def test_enabling_refinement_requires_explicit_controls() -> None:
    """The package does not invent the manuscript's unspecified constants."""

    observed_source = np.diag([9.0, 6.0, 3.0, 0.0])
    target = np.diag([4.0, 0.0, 2.0, 0.0])
    folds = _folds_for_target(target)

    with pytest.raises(ValueError, match="requires config.refinement_controls"):
        BISMART(observed_source, _config()).fit_folds(
            *folds,
            enable_refinement=True,
        )


def test_failed_wedin_gate_keeps_screen_candidates_and_skips_branch_rrr() -> None:
    """A local gate failure follows Algorithm 2's precise continuation rule."""

    observed_source = np.diag([9.0, 6.0, 3.0, 0.0])
    target = np.diag([4.0, 0.0, 2.0, 0.0])
    folds = _folds_for_target(target)
    config = BISMARTConfig(
        target_rank=2,
        source_rank=3,
        source_error_bound=0.6,
        budget_path=((2, 2),),
        source_weights=(1.0,),
        c_gap=3.0,
    )

    estimator = BISMART(observed_source, config)
    result = estimator.fit_folds(*folds)
    labels = {candidate.label for candidate in result.candidates}

    # The finest singleton partition has observable block gap 1.8 < 2.4.
    assert "transfer/budget=2,2/partition=0/pilot" in labels
    assert "transfer/budget=2,2/partition=0/frozen" in labels
    assert "transfer/budget=2,2/partition=0/restricted_rrr/t=0" not in labels
    assert any(
        diagnostic.label == "transfer/budget=2,2/partition=0/wedin_gate"
        and diagnostic.failure_reason is FailureReason.WEDIN_GATE
        for diagnostic in estimator.branch_diagnostics_
    )


def test_restricted_rrr_failure_continues_to_all_safeguards() -> None:
    """A fitting-fold inverse failure is branch-local, not estimator-global."""

    observed_source = np.diag([9.0, 6.0, 3.0, 0.0])
    target = np.diag([4.0, 0.0, 2.0, 0.0])
    initialization, _, validation = _folds_for_target(target)
    zero_design = np.zeros((12, 4))
    fitting = FoldData(
        zero_design,
        np.zeros((12, 4)),
        name="singular_fitting",
    )

    estimator = BISMART(observed_source, _config())
    result = estimator.fit_folds(initialization, fitting, validation)

    kinds = {candidate.kind for candidate in result.candidates}
    assert "restricted_rrr" not in kinds
    assert {"pilot", "frozen", "target_only_rrr", "zero"} <= kinds
    assert any(
        diagnostic.failure_reason is FailureReason.NON_POSITIVE_DEFINITE
        for diagnostic in estimator.branch_diagnostics_
    )


def test_extreme_radius_grid_values_fail_locally_without_aborting() -> None:
    """Floating overflow/underflow obeys the same per-call continuation rule."""

    observed_source = np.diag([9.0, 6.0, 3.0, 0.0])
    target = np.diag([4.0, 0.0, 2.0, 0.0])
    folds = _folds_for_target(target)
    base = _config()
    config = BISMARTConfig(
        target_rank=base.target_rank,
        source_rank=base.source_rank,
        source_error_bound=base.source_error_bound,
        budget_path=base.budget_path,
        source_weights=base.source_weights,
        refinement_controls=RefinementControls(
            armijo_constant=1e-4,
            contraction=0.5,
            initial_step_size=1.0,
            radius_ratio=1e308,
            radius_half_width=2,
            max_backtracking_cap=1,
            iteration_cap=1,
        ),
    )

    estimator = BISMART(observed_source, config)
    result = estimator.fit_folds(*folds, enable_refinement=True)

    assert {"target_only_rrr", "zero"} <= {
        candidate.kind for candidate in result.candidates
    }
    assert any(
        diagnostic.failure_reason is FailureReason.INVALID_INITIAL_THRESHOLDS
        and diagnostic.label.endswith("/radius")
        for diagnostic in estimator.branch_diagnostics_
    )
    assert any(
        diagnostic.failure_reason is FailureReason.NOT_IMPLEMENTED
        for diagnostic in estimator.branch_diagnostics_
    )


def test_relative_refinement_tolerance_accepts_regular_small_scale_state() -> None:
    """rtol is relative to the actual cores, with no unintended unit floor."""

    observed_source = np.diag([9e-11, 6e-11, 3e-11, 0.0])
    target = np.diag([4e-11, 0.0, 2e-11, 0.0])
    folds = _folds_for_target(target)
    config = BISMARTConfig(
        target_rank=2,
        source_rank=3,
        source_error_bound=0.0,
        budget_path=((2, 2),),
        source_weights=(1.0,),
        refinement_controls=RefinementControls(
            armijo_constant=1e-4,
            contraction=0.5,
            initial_step_size=1.0,
            radius_ratio=2.0,
            radius_half_width=0,
            max_backtracking_cap=1,
            iteration_cap=1,
        ),
        atol=0.0,
        rtol=1e-10,
    )

    estimator = BISMART(observed_source, config)
    estimator.fit_folds(*folds, enable_refinement=True)

    assert any(
        diagnostic.failure_reason is FailureReason.NOT_IMPLEMENTED
        for diagnostic in estimator.branch_diagnostics_
    )
    assert not any(
        diagnostic.failure_reason is FailureReason.INVALID_INITIAL_THRESHOLDS
        for diagnostic in estimator.branch_diagnostics_
    )
