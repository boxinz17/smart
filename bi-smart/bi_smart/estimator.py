"""High-level orchestration for the complete automatic BI-SMART scaffold.

The :class:`BISMART` wrapper is the analogue of ``smart.SMART``.  Its API is
deliberately three-fold because the theorem-ready appendix uses initialization,
fitting, and validation responses for disjoint purposes.  It does not invent a
two-fold cross-fitting aggregation that the manuscript has not yet specified.

Joint quotient Gauss--Newton refinement is opt-in because its finite trust and
line-search grids require explicit manuscript constants.  When enabled, each
fixed-support direction is computed by the configured dense or matrix-free
Moore--Penrose backend and every algebraic failure remains local to its
calibrated call.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .candidates import score_and_select_candidates, zero_candidate
from .initialization import initialize_restricted_rrr, target_only_rrr
from .refinement import (
    BISMARTState,
    RefinementError,
    default_core_thresholds,
    fitted_source,
    fitted_target,
    safeguarded_backtracking,
    solve_quotient_gauss_newton,
)
from .screening import run_block_screen
from .source_blocks import (
    build_source_block_library,
    evaluate_wedin_gate,
)
from .types import (
    BISMARTConfig,
    BISMARTResult,
    BlockPartition,
    Candidate,
    CandidateStatus,
    FailureReason,
    FoldData,
    NumericalFailure,
    ScreenResult,
    SourceBlockLibrary,
)


class BISMART:
    """Complete BI-SMART candidate generator and validation selector.

    Parameters
    ----------
    observed_source:
        Noisy fitted source coefficient matrix ``C0_tilde`` with shape
        ``(p, q)``.  Raw source observations are not required.
    config:
        Explicit ranks, source-error certificate, budget path, source weights,
        and numerical tolerances.

    Notes
    -----
    ``fit_folds`` is executable with ``enable_refinement=False`` (the default).
    It produces pilot, frozen, exact restricted-RRR, full-block, target-only,
    and zero candidates.  With refinement enabled, it additionally enumerates
    the appendix calibration grids and retains a call's iterates only when all
    requested quotient-rank, path, and Armijo checks pass.
    """

    def __init__(self, observed_source: Any, config: BISMARTConfig) -> None:
        raw_source = np.asarray(observed_source)
        if np.iscomplexobj(raw_source):
            raise ValueError(
                "observed_source must be real-valued; complex input is unsupported."
            )
        source = np.asarray(raw_source, dtype=float)
        if source.ndim != 2 or 0 in source.shape:
            raise ValueError(
                "observed_source must be a nonempty two-dimensional matrix."
            )
        if not np.all(np.isfinite(source)):
            raise ValueError("observed_source must contain only finite values.")
        if not isinstance(config, BISMARTConfig):
            raise TypeError("config must be a BISMARTConfig instance.")
        if config.source_rank > min(source.shape):
            raise ValueError(
                "config.source_rank cannot exceed min(observed_source.shape)."
            )

        self.observed_source = source
        self.config = config
        self.result_: Optional[BISMARTResult] = None
        self.branch_diagnostics_: tuple[Candidate, ...] = ()
        self.source_library_: Optional[SourceBlockLibrary] = None

    @staticmethod
    def _validate_folds(
        initialization: FoldData,
        fitting: FoldData,
        validation: FoldData,
        source_shape: tuple[int, int],
    ) -> None:
        """Validate dimensions that the paper assumes across all three folds."""

        # PSEUDOCODE 1: Require explicit FoldData objects so fold roles cannot
        # be accidentally swapped with raw positional matrices.
        folds = (initialization, fitting, validation)
        if any(not isinstance(fold, FoldData) for fold in folds):
            raise TypeError("initialization, fitting, and validation must be FoldData.")

        # PSEUDOCODE 2: All target folds use the same predictor/response axes as
        # the source coefficient matrix.  Statistical independence cannot be
        # mechanically tested and remains the caller's design responsibility.
        expected_p, expected_q = source_shape
        for fold in folds:
            if fold.n_features != expected_p or fold.n_responses != expected_q:
                raise ValueError(
                    f"{fold.name} has ({fold.n_features}, {fold.n_responses}) "
                    f"predictor/response dimensions; expected ({expected_p}, {expected_q})."
                )

    def _screen_candidates(
        self,
        screen: ScreenResult,
        *,
        prefix: str,
        order_prefix: tuple,
        metadata: dict,
    ) -> tuple[list[Candidate], list[Candidate]]:
        """Translate one screen result into candidates plus failure diagnostics."""

        candidates: list[Candidate] = []
        diagnostics: list[Candidate] = []

        # PSEUDOCODE 1: A strict pilot SVD is independently useful.  If a later
        # polar/sweep check fails, the appendix still permits retaining it.
        if screen.pilot_matrix is not None:
            candidates.append(
                Candidate.successful(
                    label=f"{prefix}/pilot",
                    matrix=screen.pilot_matrix,
                    order_key=order_prefix + (0,),
                    kind="pilot",
                    metadata=metadata,
                )
            )

        # PSEUDOCODE 2: The frozen candidate exists only after both whole-block
        # sweep updates and their rank checks succeed.
        if screen.frozen_matrix is not None:
            frozen_metadata = {
                **metadata,
                "left_blocks": screen.left_blocks,
                "right_blocks": screen.right_blocks,
            }
            candidates.append(
                Candidate.successful(
                    label=f"{prefix}/frozen",
                    matrix=screen.frozen_matrix,
                    order_key=order_prefix + (1,),
                    kind="frozen",
                    metadata=frozen_metadata,
                )
            )

        # PSEUDOCODE 3: Keep failed branch information outside the validation
        # library.  It is auditable through ``branch_diagnostics_``.
        if screen.status is CandidateStatus.UNSUCCESSFUL:
            diagnostics.append(
                Candidate.unsuccessful(
                    label=f"{prefix}/screen",
                    reason=screen.failure_reason or FailureReason.SCREEN_FAILED,
                    message=screen.message or "The block screen was unsuccessful.",
                    order_key=order_prefix + (9,),
                    kind="screen",
                    metadata=metadata,
                )
            )
        return candidates, diagnostics

    def _run_source_branch(
        self,
        *,
        initialization: FoldData,
        fitting: FoldData,
        library: SourceBlockLibrary,
        partition: BlockPartition,
        left_budget: int,
        right_budget: int,
        prefix: str,
        order_prefix: tuple,
        enable_refinement: bool,
    ) -> tuple[list[Candidate], list[Candidate]]:
        """Execute screen -> Wedin gate -> exact RRR -> refinement grid."""

        config = self.config
        metadata = {
            "partition": partition.blocks,
            "left_budget": left_budget,
            "right_budget": right_budget,
        }

        # PSEUDOCODE 1: Use initialization responses only for support discovery.
        screen = run_block_screen(
            initialization,
            library.decomposition,
            partition,
            target_rank=config.target_rank,
            left_budget=left_budget,
            right_budget=right_budget,
            atol=config.atol,
            rtol=config.rtol,
        )
        candidates, diagnostics = self._screen_candidates(
            screen,
            prefix=prefix,
            order_prefix=order_prefix,
            metadata=metadata,
        )
        if screen.status is not CandidateStatus.SUCCESSFUL:
            return candidates, diagnostics

        # PSEUDOCODE 2: A failed observable Wedin gate keeps pilot/frozen but
        # forbids restricted RRR and every joint-refinement call for this branch.
        gate = evaluate_wedin_gate(
            library.decomposition,
            partition,
            config.source_error_bound,
            atol=config.atol,
            rtol=config.rtol,
        )
        gate_metadata = {
            **metadata,
            "left_blocks": screen.left_blocks,
            "right_blocks": screen.right_blocks,
            "wedin_block_gaps": gate.block_gaps,
            "wedin_threshold": gate.threshold,
        }
        if not gate.passes:
            diagnostics.append(
                Candidate.unsuccessful(
                    label=f"{prefix}/wedin_gate",
                    reason=FailureReason.WEDIN_GATE,
                    message=(
                        "Observable Wedin gate failed for blocks "
                        f"{gate.failing_blocks}; pilot/frozen were retained."
                    ),
                    order_key=order_prefix + (10,),
                    kind="wedin_gate",
                    metadata=gate_metadata,
                )
            )
            return candidates, diagnostics

        # PSEUDOCODE 3: Conditional on the initialization-fold support, recompute
        # the exact restricted fit using only independent fitting responses.
        rrr = initialize_restricted_rrr(
            fitting,
            library.decomposition,
            screen,
            target_rank=config.target_rank,
            label=f"{prefix}/restricted_rrr/t=0",
            order_key=order_prefix + (2,),
            atol=config.atol,
            rtol=config.rtol,
        )
        if not rrr.successful:
            diagnostics.append(rrr.candidate)
            return candidates, diagnostics
        candidates.append(rrr.candidate)

        # PSEUDOCODE 4: Enumerate every omega/radius/cap call.  Each call tries
        # the quotient direction, finite path certificate, and Armijo steps.
        # Any rank or regularity failure is branch-local, so later calibrated
        # calls and mandatory safeguards still run.
        if enable_refinement:
            assert rrr.state is not None
            refinement_candidates, refinement_diagnostics = (
                self._run_refinement_grid(
                    fitting=fitting,
                    initial_state=rrr.state,
                    prefix=prefix,
                    order_prefix=order_prefix,
                    metadata=gate_metadata,
                )
            )
            candidates.extend(refinement_candidates)
            diagnostics.extend(refinement_diagnostics)

        return candidates, diagnostics

    def _run_refinement_grid(
        self,
        *,
        fitting: FoldData,
        initial_state: BISMARTState,
        prefix: str,
        order_prefix: tuple,
        metadata: dict,
    ) -> tuple[list[Candidate], list[Candidate]]:
        """Enumerate Algorithm 2's weight/radius/cap refinement calls.

        A call contributes its iterates only if all ``T`` iterations complete.
        If any direction, regularity, or line-search check fails, every partial
        iterate from that call is discarded and one diagnostic is returned.
        The exact RRR candidate is owned by the caller and is never removed.
        """

        config = self.config
        controls = config.refinement_controls
        if controls is None:  # Guarded once in fit_folds; retained defensively.
            raise ValueError(
                "enable_refinement=True requires config.refinement_controls."
            )

        candidates: list[Candidate] = []
        diagnostics: list[Candidate] = []
        initial_target = fitted_target(initial_state)
        source_approximation = fitted_source(initial_state)
        target_norm_squared = float(np.sum(initial_target * initial_target))
        source_norm_squared = float(
            np.sum(source_approximation * source_approximation)
        )
        # PSEUDOCODE 1: For each omega, compute the observable scale s_hat_omega
        # and the increasing radius grid s_hat_omega*q_rho^j, -J_rho <= j <=
        # J_rho.  Strict source/RRR checks imply a positive scale.
        for weight_index, omega in enumerate(config.source_weights):
            scale = float(
                np.sqrt(target_norm_squared + omega * source_norm_squared)
            )
            if not np.isfinite(scale) or scale <= 0.0:
                diagnostics.append(
                    Candidate.unsuccessful(
                        label=f"{prefix}/refinement/omega={weight_index}/scale",
                        reason=FailureReason.INVALID_INITIAL_THRESHOLDS,
                        message="The observable refinement calibration scale is not positive.",
                        order_key=order_prefix + (3, weight_index, -1, -1, 0),
                        kind="refinement_call",
                        metadata={**metadata, "omega": omega, "scale": scale},
                    )
                )
                continue

            radius_grid: list[float] = []
            for exponent in range(
                -controls.radius_half_width,
                controls.radius_half_width + 1,
            ):
                # A mathematically finite grid can overflow or underflow in
                # floating-point arithmetic.  Preserve its slot and mark each
                # affected calibrated call unsuccessful below; never let one
                # extreme radius erase later branches or safeguards.
                try:
                    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                        radius = float(
                            scale * controls.radius_ratio**exponent
                        )
                except OverflowError:
                    radius = float("inf")
                radius_grid.append(radius)

            # PSEUDOCODE 2: Traverse radii and B_max=1,...,B_cal in increasing
            # numerical order.  Indices remain in labels so every call is
            # distinct even when floating-point values format identically.
            for radius_index, radius in enumerate(radius_grid):
                for backtracking_cap in range(
                    1,
                    controls.max_backtracking_cap + 1,
                ):
                    call_label = (
                        f"{prefix}/refinement/omega={weight_index}"
                        f"/radius={radius_index}/cap={backtracking_cap}"
                    )
                    call_metadata = {
                        **metadata,
                        "omega": omega,
                        "omega_index": weight_index,
                        "calibration_scale": scale,
                        "trust_radius": radius,
                        "radius_index": radius_index,
                        "backtracking_cap": backtracking_cap,
                        "iteration_cap": controls.iteration_cap,
                    }

                    if not np.isfinite(radius) or radius <= 0.0:
                        diagnostics.append(
                            Candidate.unsuccessful(
                                label=f"{call_label}/radius",
                                reason=FailureReason.INVALID_INITIAL_THRESHOLDS,
                                message=(
                                    "The calibrated trust radius overflowed, "
                                    "underflowed, or was not strictly positive."
                                ),
                                order_key=order_prefix
                                + (3, weight_index, radius_index, backtracking_cap - 1, 0),
                                kind="refinement_call",
                                metadata=call_metadata,
                            )
                        )
                        continue

                    # PSEUDOCODE 3: Construct the data-dependent core and
                    # separation thresholds.  A zero required quantity makes
                    # this one calibrated call unsuccessful.
                    try:
                        thresholds = default_core_thresholds(
                            initial_state,
                            radius=radius,
                            atol=config.atol,
                            rtol=config.rtol,
                        )
                    except RefinementError as failure:
                        diagnostics.append(
                            Candidate.unsuccessful(
                                label=f"{call_label}/thresholds",
                                reason=FailureReason.INVALID_INITIAL_THRESHOLDS,
                                message=str(failure),
                                order_key=order_prefix
                                + (3, weight_index, radius_index, backtracking_cap - 1, 0),
                                kind="refinement_call",
                                metadata=call_metadata,
                            )
                        )
                        continue

                    current_state = initial_state
                    completed_iterates: list[Candidate] = []
                    failure_record: Optional[Candidate] = None

                    # PSEUDOCODE 4: Run exactly T fixed-support iterations.
                    # Partial iterates stay local until the entire call
                    # completes, matching Algorithm 2's all-or-nothing rule.
                    for iteration in range(1, controls.iteration_cap + 1):
                        iteration_order = order_prefix + (
                            3,
                            weight_index,
                            radius_index,
                            backtracking_cap - 1,
                            iteration,
                        )
                        try:
                            gauss_newton = solve_quotient_gauss_newton(
                                current_state,
                                fitting.X,
                                fitting.Y,
                                self.observed_source,
                                omega,
                                atol=config.atol,
                                rtol=config.rtol,
                                max_dense_work_bytes=(
                                    controls.max_dense_work_bytes
                                ),
                                solver_backend=controls.gauss_newton_backend,
                                matrix_free_max_iterations=(
                                    controls.matrix_free_max_iterations
                                ),
                            )
                            direction = gauss_newton.direction
                            backtracking = safeguarded_backtracking(
                                current_state,
                                direction,
                                initial_state,
                                fitting.X,
                                fitting.Y,
                                self.observed_source,
                                omega=omega,
                                thresholds=thresholds,
                                initial_step_size=controls.initial_step_size,
                                contraction=controls.contraction,
                                armijo_constant=controls.armijo_constant,
                                max_trials=backtracking_cap,
                                polar_atol=config.atol,
                                polar_rtol=config.rtol,
                            )
                        except NumericalFailure as failure:
                            failure_record = Candidate.unsuccessful(
                                label=f"{call_label}/t={iteration}",
                                reason=failure.reason,
                                message=str(failure),
                                order_key=iteration_order,
                                kind="refinement_call",
                                metadata=call_metadata,
                            )
                            break
                        except np.linalg.LinAlgError as failure:
                            failure_record = Candidate.unsuccessful(
                                label=f"{call_label}/t={iteration}",
                                reason=FailureReason.SINGULAR_NORMAL_EQUATIONS,
                                message=str(failure),
                                order_key=iteration_order,
                                kind="refinement_call",
                                metadata=call_metadata,
                            )
                            break
                        except RefinementError as failure:
                            failure_record = Candidate.unsuccessful(
                                label=f"{call_label}/t={iteration}",
                                reason=FailureReason.IRREGULAR_ITERATE,
                                message=str(failure),
                                order_key=iteration_order,
                                kind="refinement_call",
                                metadata=call_metadata,
                            )
                            break

                        if not backtracking.accepted or backtracking.state is None:
                            failure_record = Candidate.unsuccessful(
                                label=f"{call_label}/t={iteration}",
                                reason=FailureReason.LINE_SEARCH,
                                message=(
                                    backtracking.failure_reason
                                    or "No safeguarded Armijo trial was accepted."
                                ),
                                order_key=iteration_order,
                                kind="refinement_call",
                                metadata=call_metadata,
                            )
                            break

                        current_state = backtracking.state
                        completed_iterates.append(
                            Candidate.successful(
                                label=f"{call_label}/t={iteration}",
                                matrix=fitted_target(current_state),
                                order_key=iteration_order,
                                kind="refinement",
                                metadata={
                                    **call_metadata,
                                    **gauss_newton.diagnostics.as_metadata(),
                                    "iteration": iteration,
                                    "accepted_step_size": backtracking.step_size,
                                },
                            )
                        )

                    # PSEUDOCODE 5: Retain t=1,...,T only after a complete
                    # call.  On failure, discard its partial list and expose
                    # the reason while the caller retains exact RRR at t=0.
                    if failure_record is None:
                        candidates.extend(completed_iterates)
                    else:
                        diagnostics.append(failure_record)

        return candidates, diagnostics

    def fit_folds(
        self,
        initialization: FoldData,
        fitting: FoldData,
        validation: FoldData,
        *,
        enable_refinement: bool = False,
    ) -> BISMARTResult:
        """Run the theorem-ready three-fold BI-SMART scaffold.

        The returned candidate library contains successful, validation-scored
        candidates only, matching the appendix step that deletes unsuccessful
        branches before selection.  Failure records remain available in
        :attr:`branch_diagnostics_`.
        """

        self._validate_folds(
            initialization,
            fitting,
            validation,
            self.observed_source.shape,
        )
        config = self.config
        if enable_refinement and config.refinement_controls is None:
            raise ValueError(
                "enable_refinement=True requires config.refinement_controls; "
                "the appendix does not prescribe defaults for these controls."
            )

        # PSEUDOCODE 1: Source processing uses no target responses.
        library = build_source_block_library(
            self.observed_source,
            config.source_rank,
            config.source_error_bound,
            c_gap=config.c_gap,
            atol=config.atol,
            rtol=config.rtol,
        )
        self.source_library_ = library
        candidates: list[Candidate] = []
        diagnostics: list[Candidate] = []

        # PSEUDOCODE 2: If r0 cuts an unresolved source block, omit every
        # transfer branch and continue to mandatory target-only/zero safeguards.
        if library.boundary_passes:
            for budget_index, (left_budget, right_budget) in enumerate(
                config.budget_path
            ):
                for partition_index, partition in enumerate(library.partitions):
                    prefix = (
                        f"transfer/budget={left_budget},{right_budget}"
                        f"/partition={partition_index}"
                    )
                    branch_candidates, branch_diagnostics = self._run_source_branch(
                        initialization=initialization,
                        fitting=fitting,
                        library=library,
                        partition=partition,
                        left_budget=left_budget,
                        right_budget=right_budget,
                        prefix=prefix,
                        order_prefix=(0, budget_index, partition_index),
                        enable_refinement=enable_refinement,
                    )
                    candidates.extend(branch_candidates)
                    diagnostics.extend(branch_diagnostics)

            # PSEUDOCODE 3: Rerun the one-block/full-budget branch independently,
            # even if an identical configuration appeared above.  Duplicate
            # successful matrices retain distinct labels by design.
            full_partition = BlockPartition.from_cut_positions(config.source_rank, ())
            full_candidates, full_diagnostics = self._run_source_branch(
                initialization=initialization,
                fitting=fitting,
                library=library,
                partition=full_partition,
                left_budget=config.source_rank,
                right_budget=config.source_rank,
                prefix="safeguard/full_block",
                order_prefix=(1, 0, 0),
                enable_refinement=enable_refinement,
            )
            candidates.extend(full_candidates)
            diagnostics.extend(full_diagnostics)
        else:
            diagnostics.append(
                Candidate.unsuccessful(
                    label="source/rank_boundary",
                    reason=FailureReason.SOURCE_RANK_BOUNDARY,
                    message=(
                        "The r0 versus r0+1 source gap did not exceed "
                        "c_gap * source_error_bound; all source branches were omitted."
                    ),
                    order_key=(-1,),
                    kind="source_boundary",
                    metadata={
                        "boundary_gap": library.boundary_gap,
                        "required_gap": config.c_gap * config.source_error_bound,
                    },
                )
            )

        # PSEUDOCODE 4: These safeguards do not depend on source certification
        # or screening and therefore are always present.
        candidates.append(
            target_only_rrr(
                fitting,
                target_rank=config.target_rank,
                label="safeguard/target_only_rrr",
                order_key=(2, 0),
                pinv_rcond=config.pinv_rcond,
            )
        )
        candidates.append(
            zero_candidate(
                fitting.n_features,
                fitting.n_responses,
                order_key=(2, 1),
            )
        )

        # PSEUDOCODE 5: Validation responses are touched only here.  Input order
        # is already the paper's deterministic order, so the first minimum wins.
        selected, scored_candidates = score_and_select_candidates(
            candidates,
            validation,
        )
        result = BISMARTResult(
            selected_candidate=selected,
            candidates=scored_candidates,
            source_library=library,
        )
        self.result_ = result
        self.branch_diagnostics_ = tuple(diagnostics)
        return result

    # ``fit`` and ``run_full_selection`` make the wrapper feel familiar to
    # users of the existing ``smart.SMART`` package while retaining explicit
    # fold arguments mandated by the BI-SMART appendix.
    fit = fit_folds
    run_full_selection = fit_folds

    def get_result(self) -> BISMARTResult:
        """Return the fitted result or raise if no three-fold run has completed."""

        if self.result_ is None:
            raise RuntimeError("BI-SMART has not been fitted. Call fit_folds() first.")
        return self.result_

    def get_estimates(self) -> dict[str, Any]:
        """Return the SMART-compatible estimate keys plus BI-SMART audit data.

        ``U @ D @ V.T`` reconstructs ``C_hat``.  These factors are a reporting
        SVD only; at an internal singular-value tie their basis may follow the
        local linear-algebra backend, while the selected coefficient matrix and
        its validation result remain unchanged.
        """

        result = self.get_result()
        left, singular_values, right_t = np.linalg.svd(
            result.coefficient,
            full_matrices=False,
        )
        rank = self.config.target_rank
        return {
            "C_hat": result.coefficient,
            "U": left[:, :rank],
            "D": np.diag(singular_values[:rank]),
            "V": right_t[:rank, :].T,
            "selected_candidate": result.selected_candidate,
            "candidates": result.candidates,
            "branch_diagnostics": self.branch_diagnostics_,
        }


__all__ = ["BISMART"]
