"""Exact-stage integration tests for the three-fold BI-SMART wrapper."""

from __future__ import annotations

import numpy as np
import pytest

import bi_smart.estimator as estimator_module
from bi_smart.estimator import BISMART
from bi_smart.types import (
    BISMARTConfig,
    CandidateStatus,
    FailureReason,
    FoldData,
    NumericalFailure,
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


def test_config_threads_source_tolerances_and_target_pinv_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level numerical policy reaches both source and target-only stages."""

    observed_source = np.diag([9.0, 6.0, 3.0, 0.0])
    target = np.diag([4.0, 0.0, 2.0, 0.0])
    config = BISMARTConfig(
        target_rank=2,
        source_rank=3,
        source_error_bound=0.1,
        budget_path=((2, 2),),
        source_weights=(1.0,),
        pinv_rcond=5e-13,
        atol=2e-13,
        rtol=3e-11,
    )

    real_gate = estimator_module.evaluate_wedin_gate
    gate_tolerances: list[tuple[float, float]] = []

    def recording_gate(*args: object, **kwargs: object):
        gate_tolerances.append((float(kwargs["atol"]), float(kwargs["rtol"])))
        return real_gate(*args, **kwargs)

    monkeypatch.setattr(estimator_module, "evaluate_wedin_gate", recording_gate)

    estimator = BISMART(observed_source, config)
    result = estimator.fit_folds(*_folds_for_target(target))

    assert result.source_library is not None
    assert result.source_library.atol == config.atol
    assert result.source_library.rtol == config.rtol
    assert gate_tolerances
    assert set(gate_tolerances) == {(config.atol, config.rtol)}
    target_only = next(
        candidate
        for candidate in result.candidates
        if candidate.kind == "target_only_rrr"
    )
    assert target_only.metadata["pinv_rcond"] == config.pinv_rcond


def test_estimator_rejects_complex_observed_source() -> None:
    """The public wrapper does not silently discard source imaginary parts."""

    with pytest.raises(ValueError, match="real-valued"):
        BISMART(np.eye(4, dtype=complex) * (1.0 + 1e-14j), _config())


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


def test_refinement_grid_produces_certified_iterates_and_keeps_safeguards() -> None:
    """A regular exact fit produces a certified zero-step refinement iterate."""

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

    # At this exact fit the joint residual is zero.  The unique horizontal GN
    # direction is therefore zero, Armijo accepts equality, and t=1 represents
    # the same fitted matrix with a complete numerical certificate.
    kinds = {candidate.kind for candidate in result.candidates}
    assert {"restricted_rrr", "target_only_rrr", "zero"} <= kinds
    refinements = [
        candidate for candidate in result.candidates
        if candidate.kind == "refinement"
    ]
    assert refinements
    assert all(
        candidate.metadata["gn_jacobian_rank"] > 0
        for candidate in refinements
    )
    assert all(
        candidate.metadata["gn_jacobian_rank"]
        == candidate.metadata["gn_quotient_dimension"]
        for candidate in refinements
    )
    for candidate in refinements:
        np.testing.assert_allclose(candidate.matrix, target, atol=1e-11)
    np.testing.assert_allclose(result.coefficient, target, atol=1e-11)


def test_estimator_forwards_forced_matrix_free_backend_metadata() -> None:
    """Refinement controls select matrix-free GN and expose its certificate."""

    observed_source = np.diag([9.0, 6.0, 3.0, 0.0])
    target = np.diag([4.0, 0.0, 2.0, 0.0])
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
            gauss_newton_backend="matrix_free",
            matrix_free_max_iterations=20,
        ),
    )

    result = BISMART(observed_source, config).fit_folds(
        *_folds_for_target(target), enable_refinement=True
    )
    refinements = [
        candidate
        for candidate in result.candidates
        if candidate.kind == "refinement"
    ]

    assert refinements
    assert all(
        candidate.metadata["gn_solver_backend"] == "matrix_free"
        for candidate in refinements
    )
    assert all(
        candidate.metadata["gn_jacobian_rank"] is None
        for candidate in refinements
    )
    assert all(
        candidate.metadata["gn_structural_quotient_rank"]
        == candidate.metadata["gn_quotient_dimension"]
        for candidate in refinements
    )


def test_complete_refinement_grid_retains_t1_through_t_and_exact_order() -> None:
    """A regular run follows every deterministic grid axis before safeguards."""

    observed_source = np.diag([9.0, 6.0, 3.0, 0.0])
    target = np.diag([4.0, 0.0, 2.0, 0.0])
    folds = _folds_for_target(target)
    iteration_cap = 3
    config = BISMARTConfig(
        target_rank=2,
        source_rank=3,
        source_error_bound=0.1,
        budget_path=((2, 2), (3, 3)),
        source_weights=(0.5, 1.0),
        refinement_controls=RefinementControls(
            armijo_constant=1e-4,
            contraction=0.5,
            initial_step_size=1.0,
            radius_ratio=2.0,
            radius_half_width=1,
            max_backtracking_cap=2,
            iteration_cap=iteration_cap,
        ),
    )

    result = BISMART(observed_source, config).fit_folds(
        *folds, enable_refinement=True
    )

    # At the exact fit every calibrated call is regular and accepts its zero
    # direction immediately.  Consequently each successful call contributes
    # the complete t=1,...,T sequence, never only a partial prefix.
    iterations_by_call: dict[str, list[int]] = {}
    for candidate in result.candidates:
        if candidate.kind != "refinement":
            continue
        call_label, iteration_text = candidate.label.rsplit("/t=", 1)
        iterations_by_call.setdefault(call_label, []).append(int(iteration_text))
    assert iterations_by_call
    assert all(
        iterations == list(range(1, iteration_cap + 1))
        for iterations in iterations_by_call.values()
    )

    refinement_suffixes = [
        (3, weight_index, radius_index, cap_index, iteration)
        for weight_index in range(2)
        for radius_index in range(3)
        for cap_index in range(2)
        for iteration in range(1, iteration_cap + 1)
    ]

    def branch_keys(prefix: tuple[int, int, int], *, refines: bool) -> list[tuple]:
        keys: list[tuple] = [prefix + (0,)]
        if refines:
            keys.extend((prefix + (1,), prefix + (2,)))
            keys.extend(prefix + suffix for suffix in refinement_suffixes)
        return keys

    expected_order: list[tuple] = []
    # Budget (2,2) supports the finest singleton partition only.  The two
    # coarser branches retain their pilot before their local screen failure.
    expected_order.extend(branch_keys((0, 0, 0), refines=True))
    expected_order.extend(branch_keys((0, 0, 1), refines=False))
    expected_order.extend(branch_keys((0, 0, 2), refines=False))
    # Full budget supports all three nested partitions.
    for partition_index in range(3):
        expected_order.extend(
            branch_keys((0, 1, partition_index), refines=True)
        )
    # The independently rerun full-block safeguard precedes target-only and
    # zero, even though its matrix duplicates an earlier transfer branch.
    expected_order.extend(branch_keys((1, 0, 0), refines=True))
    expected_order.extend(((2, 0), (2, 1)))

    actual_order = [candidate.order_key for candidate in result.candidates]
    assert actual_order == expected_order
    assert actual_order == sorted(actual_order)
    assert result.candidates[-2].label == "safeguard/target_only_rrr"
    assert result.candidates[-1].label == "safeguard/zero"


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


def test_incomplete_refinement_call_discards_its_partial_iterates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A t=2 numerical failure removes the already accepted local t=1 step."""

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
            radius_ratio=2.0,
            radius_half_width=0,
            max_backtracking_cap=1,
            iteration_cap=2,
        ),
    )

    real_solver = estimator_module.solve_quotient_gauss_newton
    calls = 0

    def fail_only_second_direction(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise NumericalFailure(
                FailureReason.SINGULAR_NORMAL_EQUATIONS,
                "injected second-iteration failure",
            )
        return real_solver(*args, **kwargs)

    monkeypatch.setattr(
        estimator_module,
        "solve_quotient_gauss_newton",
        fail_only_second_direction,
    )
    estimator = BISMART(observed_source, config)
    result = estimator.fit_folds(*folds, enable_refinement=True)

    # The failed call loses its already accepted t=1 candidate, while a later
    # calibrated/full-block call continues and retains a complete t=1,t=2 pair.
    assert calls >= 4
    failed = next(
        diagnostic
        for diagnostic in estimator.branch_diagnostics_
        if diagnostic.failure_reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
        and diagnostic.label.endswith("/t=2")
    )
    failed_call = failed.label.rsplit("/t=", 1)[0]
    assert not any(
        candidate.kind == "refinement"
        and candidate.label.startswith(f"{failed_call}/t=")
        for candidate in result.candidates
    )
    assert any(candidate.kind == "refinement" for candidate in result.candidates)
    assert any(candidate.kind == "restricted_rrr" for candidate in result.candidates)


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
    # The finite exponent-zero radius still executes normally even though
    # other calibrated radii overflow or underflow.
    assert any(candidate.kind == "refinement" for candidate in result.candidates)


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

    # The regular-state checks are scale-relative and reach the quotient solve.
    # This particular product parameterization is numerically ill-conditioned:
    # core-coordinate columns have unit scale while rotation columns are about
    # 1e-11, so the configured 1e-10 Jacobian cutoff correctly reports excess
    # nullity rather than misclassifying the state or its initial thresholds.
    assert any(
        diagnostic.failure_reason is FailureReason.SINGULAR_NORMAL_EQUATIONS
        for diagnostic in estimator.branch_diagnostics_
    )
    assert not any(
        diagnostic.failure_reason in {
            FailureReason.INVALID_INITIAL_THRESHOLDS,
            FailureReason.IRREGULAR_ITERATE,
        }
        for diagnostic in estimator.branch_diagnostics_
    )
