"""Masked, constraint-aware refinement for full outside-support V2 settings.

This opt-in practical solver uses H = Z D^-1. The fixed anchor lower bound
is then a convex operator-norm ball for each *whole* H block, while the
original V2 L1 penalty applies only outside its declared free rows. Its
column weights are retained in both H and d updates. Neither the objective,
source frames, free rows nor the anchor/spectral bounds are relaxed.

Low-level certified proximal routines and failure-aware search are shared
with SparseSMART v1; all mask-dependent orchestration lives here. Spectral
projection uses V2's roundoff-safe implementation. Returned and callback
states always retain the original Z encoding. Step and residual norms use
H coordinates. This is an alternate numerical metric, not a new theorem.

Restrictive hard support caps are unsupported because operator-ball
projection can change entry support. They are rejected explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numbers
import time

import numpy as np

from sparse_smart.anchor_solver import (
    DEFAULT_LINE_SEARCH_STRATEGY, _BlockTrialResult, _ProxConvergenceError,
    _FailureAwareSearch, _new_work_counts, _work_groups, _cache_hit, _finite,
    _weighted_l1_ball_prox, _project_omega, _roundoff_feasible,
    _to_h, _to_z, _value_gradient_h,
)
from sparse_smart.calibration import Margins
from sparse_smart.chart import AnchorChart
from sparse_smart.objective import loss_context as _loss_context
from sparse_smart.solver import _integer, _record as _base_record, _result, RefinementResult
from sparse_smart.stopping import ValidationStopRequest

from .calibration import PracticalCalibration
from .solver import penalty_value as _penalty, objective_change as _objective_change
from .spectral import spectral_proximal_step
from .support import _check_masks, validate_support_limits


@dataclass(frozen=True)
class _MaskedBlockTrialResult(_BlockTrialResult):
    spectral_mapping: np.ndarray | None = None


def _mapping_movement(chart, state, trial, inverse_step):
    mapped = inverse_step * (state - trial.value)
    # Preserve the V2 diagonal cancellation safeguard at large coefficients.
    if trial.spectral_mapping is not None:
        mapped[chart.d_slice] = trial.spectral_mapping
    return float(np.linalg.norm(mapped))


def _record(chart, x, free_rows, *args, **kwargs):
    """Report only penalized entries; free nonanchor rows never count as support."""
    record = _base_record(chart, x, *args, **kwargs)
    counts = [int(np.count_nonzero(x[sl].reshape(mask.shape)[mask])) for sl, mask in
              ((chart.z_u_slice, free_rows.penalized_u),
               (chart.z_v_slice, free_rows.penalized_v))]
    return replace(record, support_u=counts[0], support_v=counts[1],
                   raw_support_u=counts[0], raw_support_v=counts[1],
                   support_tolerance_u=0., support_tolerance_v=0.)


def _evaluate_block_trial(chart, state, gradient, L, penalties, free_rows, margins, tolerance, *,
                          absolute_gap_tol=None, work=None, iteration=0, phase="mapping"):
    """Count actual work, excluding reference trials reused by the line search."""
    started = time.perf_counter()
    result, failure = None, None
    try:
        result = _block_trial(chart, state, gradient, L, penalties, free_rows, margins, tolerance,
                             absolute_gap_tol=absolute_gap_tol)
        return result
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, ArithmeticError) as error:
        failure = error
        raise
    finally:
        elapsed = time.perf_counter() - started
        owner = result if result is not None else failure
        for counts in _work_groups(work, iteration, phase):
            counts["block_calls"] += 1
            counts["block_seconds"] += elapsed
            if failure is not None:
                counts["block_failures"] += 1
                key = str(float(L))
                counts["failures_by_inverse"][key] = counts["failures_by_inverse"].get(key, 0) + 1
            for side in ("u", "v"):
                prox = getattr(owner, "prox_" + side, None)
                if prox is None:
                    continue
                item = counts[side]
                item["calls"] += 1
                item["iterations"] += prox.n_iter if prox.iterations_run is None else prox.iterations_run
                item["dykstra_iterations"] += prox.dykstra_iterations
                item["dual_iterations"] += prox.dual_iterations
                item["fallback_calls"] += int(prox.dual_iterations > 0)
                item["failures"] += int(not prox.converged)
                item["seconds"] += prox.elapsed_seconds
                reason = prox.termination_reason
                item["reasons"][reason] = item["reasons"].get(reason, 0) + 1
                # Compact last-call evidence; arrays and individual sweeps are never retained.
                item["last"] = dict(converged=prox.converged, n_iter=prox.n_iter,
                    iterations_run=prox.iterations_run, residual=_finite(prox.residual),
                    residual_threshold=_finite(prox.residual_threshold),
                    duality_gap=_finite(prox.duality_gap), gap_roundoff=_finite(prox.gap_roundoff),
                    gap_evaluated_iteration=prox.gap_evaluated_iteration,
                    relative_gap=_finite(prox.relative_gap),
                    dykstra_iterations=prox.dykstra_iterations, dual_iterations=prox.dual_iterations,
                    certificate_method=prox.certificate_method,
                    termination_reason=reason, inverse_step=float(L))
                if not prox.converged:
                    item["last_failure"] = item["last"].copy()


def _trial_key(chart, state, gradient, L, penalties, free_rows, margins, tolerance):
    """Fit-local, value-based identity for an unrefined block trial."""
    return (id(chart), state.tobytes(), gradient.tobytes(), float(L),
            tuple(penalties), free_rows.penalized_u.tobytes(), free_rows.penalized_v.tobytes(),
            margins, float(tolerance))


def _block_trial(chart, state, gradient, L, penalties, free_rows, margins, tolerance, *, absolute_gap_tol=None):
    """Constrained block-model minimizer and proximal uncertainty estimate."""
    omega_u, omega_v, d, h_u, h_v = chart.unpack(state)
    g_u, g_v, g_d, g_hu, g_hv = chart.unpack(gradient)
    column_penalty = (penalties[0] * (free_rows.penalized_u * np.abs(h_u)).sum(axis=0)
                      + penalties[1] * (free_rows.penalized_v * np.abs(h_v)).sum(axis=0))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        next_d, spectral_mapping = spectral_proximal_step(d, g_d + column_penalty, L,
            d_lower=margins.d_lower, d_upper=margins.d_upper, gap=margins.gap)
        next_u = _project_omega(omega_u - g_u / L, chart.rank)
        next_v = _project_omega(omega_v - g_v / L, chart.rank)
        radius = float(np.sqrt((1. - margins.anchor_min) * (1. + margins.anchor_min)))
        started = time.perf_counter()
        pu = _weighted_l1_ball_prox(h_u - g_hu / L, penalties[0] * free_rows.penalized_u * d / L,
                                    radius, tolerance=tolerance, absolute_gap_tol=absolute_gap_tol)
        pu = replace(pu, elapsed_seconds=time.perf_counter() - started)
        started = time.perf_counter()
        pv = _weighted_l1_ball_prox(h_v - g_hv / L, penalties[1] * free_rows.penalized_v * d / L,
                                    radius, tolerance=tolerance, absolute_gap_tol=absolute_gap_tol)
        pv = replace(pv, elapsed_seconds=time.perf_counter() - started)
    if not pu.converged or not pv.converged:
        raise _ProxConvergenceError(pu, pv)
    uncertainty = L * np.hypot(pu.error_bound, pv.error_bound)
    return _MaskedBlockTrialResult(chart.pack(next_u, next_v, next_d, pu.value, pv.value),
        float(uncertainty), pu.closed_form and pv.closed_form, pu, pv, spectral_mapping)


def _mapping(chart, state, gradient, reference_L, penalties, free_rows, margins, tolerance,
             stationarity_tol=None, *, trial_cache=None, cache_failed_trials=True,
             work=None, iteration=0):
    """All-constraint block-proximal residual, with inner-solve uncertainty.

    The gap term prevents an unresolved inner prox from manufacturing an outer
    convergence decision.  Unlike the legacy diagnostic, anchor and Cayley
    normal cones are included.  This is a first-order numerical diagnostic,
    not a global optimality or statistical guarantee.
    """
    raw = float(np.linalg.norm(gradient))
    if trial_cache is not None:
        trial_cache.clear()
    try:
        initial_trial = _evaluate_block_trial(chart, state, gradient, reference_L,
            penalties, free_rows, margins, tolerance, work=work, iteration=iteration)
        trial, uncertainty = initial_trial
        if trial_cache is not None:
            trial_cache.update(key=_trial_key(chart, state, gradient, reference_L,
                                              penalties, free_rows, margins, tolerance),
                               result=initial_trial)
        movement = _mapping_movement(chart, state, initial_trial, reference_L)
        refinements = 0
        # Refine only when inner uncertainty prevents a possible stationarity
        # decision. Ordinary, clearly nonstationary iterations keep the cheaper
        # relative proximal solve. The two attempts are bounded even if an
        # absolute certificate cannot be resolved in float64.
        while (stationarity_tol is not None and max(0., movement - uncertainty) <= stationarity_tol
               and movement + uncertainty > stationarity_tol and refinements < 2):
            target = min(stationarity_tol / 4, max((stationarity_tol - movement) / 2,
                                                  stationarity_tol / 100)) / (10 ** refinements)
            # Each of the two gaps <= target^2/(4 L_ref^2) ensures their
            # combined allowance L_ref sqrt(2 (gap_u+gap_v)) <= target.
            absolute_gap = max(np.finfo(float).tiny, .25 * (target / reference_L) ** 2)
            refinements += 1
            try:
                refined_result = _evaluate_block_trial(chart, state, gradient, reference_L,
                    penalties, free_rows, margins, tolerance, absolute_gap_tol=absolute_gap,
                    work=work, iteration=iteration)
                refined, allowance = refined_result
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, ArithmeticError):
                # Keep the original finite diagnostic if a stricter inner
                # solve cannot deliver a usable certificate.
                break
            refined_movement = _mapping_movement(chart, state, refined_result, reference_L)
            if refined_movement + allowance < movement + uncertainty:
                movement, uncertainty = refined_movement, allowance
            else:
                break
        residual = movement + uncertainty
        limited = bool(stationarity_tol is not None and max(0., movement - uncertainty) <= stationarity_tol
                       and residual > stationarity_tol and refinements)
        return residual, raw, reference_L, None, movement, uncertainty, refinements, limited
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, ArithmeticError) as error:
        if cache_failed_trials and trial_cache is not None and isinstance(error, _ProxConvergenceError):
            # Only the unrefined deterministic failure is reusable. A failed
            # tighter diagnostic above must not replace the original trial.
            trial_cache.update(key=_trial_key(chart, state, gradient, reference_L,
                penalties, free_rows, margins, tolerance),
                failure=_ProxConvergenceError(error.prox_u, error.prox_v))
        return float("inf"), raw, reference_L, str(error), None, None, 0, False


def refine(
    chart: AnchorChart, initial_state, design, response, *,
    calibration: PracticalCalibration, margins: Margins, free_rows,
    iterations: int = 500, max_backtracks: int = 60, loss_offset: float = 0.,
    stationarity_tol: float | None = None, iterate_callback=None,
    numerical_work: dict | None = None, cache_failed_trials: bool = True,
    line_search_strategy: str = DEFAULT_LINE_SEARCH_STRATEGY,
) -> RefinementResult:
    """Refine with full fixed-chart constraints, preserving original Z states.

    Only full outside-mask support caps are supported.  Each outer iteration is
    one simultaneous constrained block-model step with actual-objective
    backtracking.  The line-search radius is still checked in the original Z
    chart; decrease and recorded step norms use the H metric.  ``None`` requests
    the fixed iteration budget; otherwise the complete constrained residual
    (including a proximal-solve error allowance) enables stationarity stopping.
    No anchors or source frames are changed, and callback exceptions propagate.
    An explicit ValidationStopRequest callback return stops successfully
    without asserting stationarity; other callback return values are ignored.
    Certified stationarity takes precedence when both stop conditions coincide.
    An empty ``numerical_work`` dictionary requests per-update work summaries;
    otherwise only compact totals are attached to the result. Failure caching
    reuses only an identical, unrefined reference solve within this fit.
    """
    if numerical_work is not None and (not isinstance(numerical_work, dict) or numerical_work):
        raise ValueError("numerical_work must be an empty dictionary or None")
    if not isinstance(cache_failed_trials, (bool, np.bool_)):
        raise ValueError("cache_failed_trials must be boolean")
    if line_search_strategy not in {"reset_initial_inverse", "failure_aware"}:
        raise ValueError("unknown anchor line-search strategy")
    work = {} if numerical_work is None else numerical_work
    work.update(schema_version=1, solver="masked_anchor_projected", coordinate_metric="H=Z/D", line_search_strategy=line_search_strategy,
                cache_failed_trials=bool(cache_failed_trials), totals=_new_work_counts())
    work["search"] = dict(adapted_starts=0, recovery_probes=0, activations=0,
                          failure_threshold=4, consecutive_updates=2, recovery_interval=25)
    if numerical_work is not None:
        work["iterations"] = []

    def finish(*args, **kwargs):
        result = _result(*args, **kwargs)
        result.numerical_work = work
        return result

    _integer(iterations, "iterations")
    _integer(max_backtracks, "max_backtracks")
    if stationarity_tol is not None and (isinstance(stationarity_tol, (bool, np.bool_))
            or not isinstance(stationarity_tol, numbers.Real)
            or not np.isfinite(stationarity_tol) or stationarity_tol <= 0):
        raise ValueError("stationarity_tol must be positive and finite, or None")
    if iterate_callback is not None and not callable(iterate_callback):
        raise ValueError("iterate_callback must be callable or None")
    if not isinstance(margins, Margins) or not isinstance(calibration, PracticalCalibration):
        raise ValueError("margins and calibration must be Margins and PracticalCalibration")
    for name, array, columns in (("design", design, chart.n_u), ("response", response, chart.n_v)):
        if np.iscomplexobj(array):
            raise ValueError(f"{name} must be real")
        array = np.asarray(array, dtype=float)
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != columns or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with {columns} columns and at least one row")
    design, response = np.asarray(design, dtype=float), np.asarray(response, dtype=float)
    if design.shape[0] != response.shape[0]:
        raise ValueError("design and response must have the same number of rows")
    if not np.isfinite(loss_offset) or loss_offset < 0:
        raise ValueError("loss_offset must be finite and nonnegative")
    if np.iscomplexobj(initial_state):
        raise ValueError("initial_state must be real")
    x = np.asarray(initial_state, dtype=float).copy()
    domain_args = dict(d_lower=margins.d_lower, d_upper=margins.d_upper,
                       gap=margins.gap, anchor_min=margins.anchor_min)
    _check_masks(chart, free_rows)
    limits = validate_support_limits(free_rows, calibration.support_limits)
    if limits != free_rows.capacities:
        raise ValueError("masked anchor-projected refinement requires full outside-support limits; "
                         "restrictive hard sparsity caps are unsupported")
    reason = chart.domain_reason(x, **domain_args)
    if reason:
        return finish(x, "invalid_initial_state", reason, 0, [])
    penalties = np.asarray(calibration.penalties, dtype=float)
    initial_L = calibration.step_size_inverse
    if penalties.shape != (2,) or not np.all(np.isfinite(penalties)) or np.any(penalties < 0):
        raise ValueError("penalties must contain two finite nonnegative values")
    if not np.isfinite(initial_L) or initial_L <= 0:
        raise ValueError("invalid inverse step size")
    reference_L = float(np.clip(initial_L, 1., 1000.))
    tolerance = min(1e-11, stationarity_tol / (100 * reference_L)) if stationarity_tol else 1e-11
    tolerance = max(tolerance, 8 * np.finfo(float).eps)
    state = _to_h(chart, x)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            smooth, gradient = _value_gradient_h(chart, state, design, response)
            penalty_value = _penalty(chart, x, free_rows, penalties)
        if not np.isfinite(smooth + penalty_value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("nonfinite initial loss or gradient")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return finish(x, "numerical_failure", str(error), 0, [])
    trial_cache = {}
    diagnostic = _mapping(chart, state, gradient, reference_L, penalties, free_rows, margins, tolerance,
        stationarity_tol, trial_cache=trial_cache, cache_failed_trials=cache_failed_trials, work=work)
    history = [_record(chart, x, free_rows, 0, smooth, penalty_value, initial_L, 0., [], loss_offset,
                       diagnostic, effective_support=False)]
    if iterate_callback is not None:
        request = iterate_callback(0, x.copy(), history[-1])
        if isinstance(request, ValidationStopRequest):
            if (iterations and stationarity_tol is not None and diagnostic[3] is None
                    and diagnostic[0] <= stationarity_tol):
                return finish(x, "converged", "The complete fixed-chart constrained residual meets tolerance.",
                              0, history, termination_reason="stationarity")
            return finish(x, "completed", request.message, 0, history,
                          termination_reason="validation_stop")
    search = _FailureAwareSearch(float(initial_L), float(initial_L))
    for iteration in range(iterations):
        if stationarity_tol is not None and diagnostic[3] is None and diagnostic[0] <= stationarity_tol:
            return finish(x, "converged", "The complete fixed-chart constrained residual meets tolerance.",
                           iteration, history, termination_reason="stationarity")
        # Ordinary curvature still triggers a fresh search. Only repeated
        # exhausted proximal solves justify remembering the successful scale.
        start_L = (search.start(diagnostic[3] is not None)
                   if line_search_strategy == "failure_aware" else initial_L)
        if search.last_policy == "previous_half":
            work["search"]["adapted_starts"] += 1
        elif search.last_policy == "recovery_probe":
            work["search"]["recovery_probes"] += 1
        if "iterations" in work:
            _work_groups(work, iteration + 1, "line_search")
            work["iterations"][iteration + 1]["start_policy"] = search.last_policy
        L, rejects, accepted = start_L, [], False
        proximal_failures = 0
        loss_context = _loss_context(chart, x, design, response)
        relative_scale = max(1., float(np.linalg.norm(state)))
        backtrack = 0
        while backtrack <= max_backtracks:
            try:
                key = _trial_key(chart, state, gradient, L, penalties, free_rows, margins, tolerance)
                if trial_cache.get("key") == key:
                    failure = trial_cache.get("failure")
                    _cache_hit(work, iteration + 1, failure is not None)
                    if failure is not None:
                        # Keep traceback frames out of the fit-local cache.
                        raise _ProxConvergenceError(failure.prox_u, failure.prox_v)
                    block_trial = trial_cache["result"]
                else:
                    block_trial = _evaluate_block_trial(chart, state, gradient, L, penalties, free_rows, margins,
                        tolerance, work=work, iteration=iteration + 1, phase="line_search")
                trial, _ = block_trial
                trial = _roundoff_feasible(chart, trial, margins)
                original_trial = _to_z(chart, trial)
                step_norm = float(np.linalg.norm(trial - state))
                original_step = float(np.linalg.norm(original_trial - x))
                resolution = 64 * np.finfo(float).eps * relative_scale
                spectral_roundoff_blocked = np.any(
                    (trial[chart.d_slice] == state[chart.d_slice])
                    & (block_trial.spectral_mapping != 0.))
                mapping_unresolved = (diagnostic[3] is not None
                    or diagnostic[0] > reference_L * resolution
                    or spectral_roundoff_blocked
                    or (stationarity_tol is not None and diagnostic[0] > stationarity_tol))
                # A numerically exact fixed point can have a nonzero raw
                # gradient balanced by L1 or constraint normals. Recognize it
                # only when the independently scaled reference trial used
                # closed-form proximal maps; an unresolved Dykstra iterate
                # must not manufacture a no-op. Keep all uncertainty in the
                # diagnostic, and never relax an explicit stopping tolerance.
                reference_trial = trial_cache.get("result")
                exact_fixed_budget_noop = (stationarity_tol is None
                    and getattr(reference_trial, "closed_form", False)
                    and diagnostic[3] is None
                    and diagnostic[4] == 0.
                    and np.array_equal(reference_trial.value, state)
                    and getattr(block_trial, "closed_form", False)
                    and np.array_equal(trial, state))
                if step_norm <= resolution and mapping_unresolved and not exact_fixed_budget_noop:
                    return finish(x, "numerical_stagnation",
                        "The feasible trial is at numerical resolution while its constrained residual is unresolved.",
                        iteration, history, rejects[-1] if rejects else None, "numerical_stagnation")
                reason = chart.domain_reason(original_trial, **domain_args)
                if reason is None and original_step > margins.trial_radius:
                    reason = "trial radius exceeded"
                if reason is None:
                    trial_smooth = chart.loss(original_trial, design, response)
                    trial_penalty = _penalty(chart, original_trial, free_rows, penalties)
                    objective_change = _objective_change(chart, x, original_trial, design,
                                                         response, free_rows, penalties, context=loss_context)
                    required = .25 * L * step_norm**2
                    if not np.isfinite(trial_smooth + trial_penalty) or not np.isfinite(required):
                        reason = "nonfinite trial objective or decrease"
                    elif objective_change <= -required:
                        accepted = True
                    else:
                        reason = "insufficient objective decrease"
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, ArithmeticError) as error:
                reason = str(error)
                proximal_failures += int(isinstance(error, _ProxConvergenceError))
            if accepted:
                was_active = search.active
                if line_search_strategy == "failure_aware":
                    search.accepted(L, proximal_failures)
                    work["search"]["activations"] += int(search.active and not was_active)
                state, x = trial, original_trial
                smooth, penalty_value = trial_smooth, trial_penalty
                try:
                    _, gradient = _value_gradient_h(chart, state, design, response)
                    if not np.all(np.isfinite(gradient)):
                        raise FloatingPointError("nonfinite gradient")
                    diagnostic = _mapping(chart, state, gradient, reference_L, penalties, free_rows, margins, tolerance,
                        stationarity_tol, trial_cache=trial_cache, cache_failed_trials=cache_failed_trials,
                        work=work, iteration=iteration + 1)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    diagnostic = (float("inf"), float("inf"), reference_L, str(error))
                    history.append(_record(chart, x, free_rows, iteration + 1, smooth, penalty_value, L,
                                            step_norm, rejects, loss_offset, diagnostic,
                                            objective_change=objective_change,
                                            relative_step_norm=step_norm / relative_scale,
                                            line_search_start_inverse=start_L, effective_support=False))
                    if iterate_callback is not None:
                        iterate_callback(iteration + 1, x.copy(), history[-1])
                    return finish(x, "numerical_failure", str(error), iteration + 1, history)
                history.append(_record(chart, x, free_rows, iteration + 1, smooth, penalty_value, L,
                                        step_norm, rejects, loss_offset, diagnostic,
                                        objective_change=objective_change,
                                        relative_step_norm=step_norm / relative_scale,
                                        line_search_start_inverse=start_L, effective_support=False))
                if iterate_callback is not None:
                    request = iterate_callback(iteration + 1, x.copy(), history[-1])
                    if isinstance(request, ValidationStopRequest):
                        if (stationarity_tol is not None and diagnostic[3] is None
                                and diagnostic[0] <= stationarity_tol):
                            return finish(x, "converged", "The complete fixed-chart constrained residual meets tolerance.",
                                          iteration + 1, history, termination_reason="stationarity")
                        return finish(x, "completed", request.message, iteration + 1, history,
                                      termination_reason="validation_stop")
                break
            rejects.append(str(reason))
            if backtrack < max_backtracks:
                L *= 2.
                if not np.isfinite(L):
                    return finish(x, "numerical_failure", "Inverse step size overflowed.",
                                   iteration, history, rejects[-1])
            backtrack += 1
        if not accepted:
            return finish(x, "line_search_failed", "All feasible line-search trials were rejected.",
                           iteration, history, rejects[-1], "backtracking_exhausted")
    if iterations and stationarity_tol is not None and diagnostic[3] is None and diagnostic[0] <= stationarity_tol:
        return finish(x, "converged", "The complete fixed-chart constrained residual meets tolerance.",
                       iterations, history, termination_reason="stationarity")
    return finish(x, "completed", f"Completed {iterations} anchor-projected refinement updates.",
                   iterations, history, termination_reason="max_iterations")
