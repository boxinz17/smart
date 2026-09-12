"""Practical full-support refinement with all fixed-chart constraints projected.

The internal coordinates are ``(omega_u, omega_v, d, H_u, H_v)``, with
``Z_side = H_side * d``.  This is a change of optimization metric, not of the
chart, its anchors, or its objective.  In these coordinates the anchor floor
is the convex constraint ``||H_side||_op <= sqrt(1-anchor_min**2)`` and the
original L1 penalty is weighted by the positive singular values.

Each trial minimizes a separable first-order block model: a constrained
weighted-L1 proximal problem for each H, ordered projection for d including
its L1 column weights, and skew spectral-ball projection for each omega.
Backtracking checks the actual original objective, including the bilinear
penalty remainder.  Hard caps smaller than the entire complement are not
supported: projecting a spectral ball can change an entrywise support.

The reported step and gradient norms use the internal H metric.  Returned
states and callback states always use the original chart's Z encoding.
This numerical solver does not provide a statistical/theorem certificate.
"""
from __future__ import annotations
from .stopping import ValidationStopRequest

from dataclasses import dataclass, replace
import numbers
import time

import numpy as np

from .calibration import Margins, ResolvedCalibration
from .chart import AnchorChart, skew_coordinates, skew_matrix
from .solver import _integer, _penalty, _record, _result, RefinementResult
from .spectral import project_singular_values
from .objective import loss_context as _loss_context, objective_change as _objective_change


@dataclass(frozen=True)
class _ProxResult:
    value: np.ndarray
    n_iter: int
    converged: bool
    residual: float
    duality_gap: float
    gap_roundoff: float = 0.
    direct_error: float = 0.
    closed_form: bool = False
    iterations_run: int | None = None
    termination_reason: str = "unspecified"
    residual_threshold: float | None = None
    elapsed_seconds: float = 0.
    gap_evaluated_iteration: int | None = None
    relative_gap: float | None = None
    dykstra_iterations: int = 0
    dual_iterations: int = 0
    certificate_method: str = "dykstra"
    certified_error_bound: float | None = None

    @property
    def error_bound(self):
        if self.certified_error_bound is not None:
            return self.certified_error_bound
        return float(np.sqrt(2. * (self.duality_gap + self.gap_roundoff)) + self.direct_error)


@dataclass(frozen=True)
class _BlockTrialResult:
    value: np.ndarray
    uncertainty: float
    closed_form: bool
    prox_u: _ProxResult | None = None
    prox_v: _ProxResult | None = None

    def __iter__(self):
        # Preserve the internal two-value unpacking interface.
        return iter((self.value, self.uncertainty))


DEFAULT_LINE_SEARCH_STRATEGY = "failure_aware"


@dataclass
class _FailureAwareSearch:
    """Reuse step scale only during persistent expensive proximal failures."""

    initial_L: float
    previous_L: float
    failure_streak: int = 0
    active: bool = False
    adapted_steps: int = 0
    recovery_interval: int = 25
    last_policy: str = "reset_initial_inverse"

    def start(self, mapping_failed):
        # A recovered independently scaled mapping is an immediate reason to
        # try larger moves again. Large curvature alone never activates reuse.
        if not mapping_failed:
            self.active, self.failure_streak, self.adapted_steps = False, 0, 0
        if not self.active:
            self.last_policy = "reset_initial_inverse"
            return self.initial_L
        if self.adapted_steps >= self.recovery_interval:
            self.adapted_steps = 0
            self.last_policy = "recovery_probe"
            return self.initial_L
        self.adapted_steps += 1
        self.last_policy = "previous_half"
        return max(self.initial_L, self.previous_L / 2.)

    def accepted(self, inverse, proximal_failures):
        self.previous_L = inverse
        if self.last_policy == "recovery_probe" and proximal_failures < 4:
            self.active, self.failure_streak = False, 0
        if not self.active:
            self.failure_streak = self.failure_streak + 1 if proximal_failures >= 4 else 0
            if self.failure_streak >= 2:
                self.active, self.adapted_steps = True, 0


class _ProxConvergenceError(ArithmeticError):
    """A deterministic, exhausted block solve, safe to cache by exact inputs."""

    def __init__(self, prox_u, prox_v):
        super().__init__("weighted L1/anchor proximal subproblem did not converge")
        self.prox_u, self.prox_v = prox_u, prox_v


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _new_work_counts():
    return dict(block_calls=0, block_seconds=0., block_failures=0,
                cache_success_hits=0, cache_failure_hits=0, failures_by_inverse={},
                u=dict(calls=0, iterations=0, failures=0, seconds=0., reasons={},
                       dykstra_iterations=0, dual_iterations=0, fallback_calls=0),
                v=dict(calls=0, iterations=0, failures=0, seconds=0., reasons={},
                       dykstra_iterations=0, dual_iterations=0, fallback_calls=0))


def _work_groups(work, iteration, phase):
    if work is None:
        return ()
    groups = [work["totals"]]
    if "iterations" in work:
        while len(work["iterations"]) <= iteration:
            work["iterations"].append(dict(iteration=len(work["iterations"]),
                mapping=_new_work_counts(), line_search=_new_work_counts()))
        groups.append(work["iterations"][iteration][phase])
    return groups


def _cache_hit(work, iteration, failed):
    for counts in _work_groups(work, iteration, "line_search"):
        name = "cache_failure_hits" if failed else "cache_success_hits"
        counts[name] += 1


def _evaluate_block_trial(chart, state, gradient, L, penalties, margins, tolerance, *,
                          absolute_gap_tol=None, work=None, iteration=0, phase="mapping"):
    """Count actual work, excluding reference trials reused by the line search."""
    started = time.perf_counter()
    result, failure = None, None
    try:
        result = _block_trial(chart, state, gradient, L, penalties, margins, tolerance,
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


def _trial_key(chart, state, gradient, L, penalties, margins, tolerance):
    """Fit-local, value-based identity for an unrefined block trial."""
    return (id(chart), state.tobytes(), gradient.tobytes(), float(L),
            tuple(penalties), margins, float(tolerance))


def _project_operator_ball(value, radius):
    """Frobenius projection, including empty complement blocks."""
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("spectral projection needs a finite matrix")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("spectral projection radius must be finite and nonnegative")
    if not value.size or radius == 0:
        return np.zeros_like(value)
    left, singular, right = np.linalg.svd(value, full_matrices=False)
    if singular[0] <= radius:
        return value.copy()
    return (left * np.minimum(singular, radius)) @ right


def _weighted_l1_ball_prox(value, weights, radius, *, tolerance=1e-11,
                           max_iterations=2000, absolute_gap_tol=None):
    """Solve .5||H-value||^2 + sum(weights*|H|) + I_{op-ball}(H).

    The first 64 sweeps use Dykstra's two proximal steps. A primal/dual gap
    determines success; the split residual remains diagnostic. If its requested
    certificate is unresolved, the remaining sweep budget uses a certified
    dual solver, initialized with this same subproblem's clipped L1 dual.
    No warm-start dependence between estimator candidates is introduced.
    Gaps are floating-point
    estimates with a separate cancellation allowance. Closed-form proximal
    cases use a linear projection-roundoff allowance instead of a dual gap.
    """
    value = np.asarray(value, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("proximal input must be a finite matrix")
    try:
        weights = np.broadcast_to(weights, value.shape)
    except ValueError as error:
        raise ValueError("proximal weights must broadcast to the matrix") from error
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("proximal weights must be finite and nonnegative")
    if not np.isfinite(radius) or radius < 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("proximal radius/tolerance are invalid")
    _integer(max_iterations, "max_iterations", 1)
    if absolute_gap_tol is not None and (not np.isfinite(absolute_gap_tol) or absolute_gap_tol <= 0):
        raise ValueError("absolute_gap_tol must be positive and finite, or None")
    if not value.size or radius == 0:
        return _ProxResult(np.zeros_like(value), 0, True, 0., 0., closed_form=True,
                           iterations_run=0, termination_reason="zero_radius_or_empty",
                           certificate_method="closed_form")
    x = value.copy()
    p, q = np.zeros_like(x), np.zeros_like(x)
    scale = max(1., float(np.linalg.norm(value)))
    if not np.isfinite(scale):
        raise FloatingPointError("proximal input norm exceeds floating-point range")
    direct_error = 64 * np.finfo(float).eps * scale
    if not np.any(weights):
        # With no L1 term this is the closed-form ball projection. Avoid a
        # cancellation-prone Fenchel subtraction for an exact proximal map.
        return _ProxResult(_project_operator_ball(value, radius), 1, True, 0., 0.,
                           direct_error=direct_error, closed_form=True, iterations_run=1,
                           termination_reason="unweighted_projection", certificate_method="closed_form")
    soft = np.sign(value) * np.maximum(np.abs(value) - weights, 0.)
    if np.linalg.norm(soft, ord=2) < radius - direct_error:
        return _ProxResult(soft, 1, True, 0., 0., direct_error=direct_error, closed_form=True,
                           iterations_run=1, termination_reason="interior_soft_threshold",
                           certificate_method="closed_form")
    residual, gap, roundoff = float("inf"), float("inf"), 0.
    gap_evaluated_iteration = None
    best = None
    dykstra_budget = min(64, max_iterations)
    for iteration in range(1, dykstra_budget + 1):
        previous_p, previous_q = p, q
        incoming = x + p
        p = np.clip(incoming, -weights, weights)
        soft = np.sign(incoming) * np.maximum(np.abs(incoming) - weights, 0.)
        incoming = soft + q
        previous = x
        x = _project_operator_ball(incoming, radius)
        q = incoming - x
        residual = max(float(np.linalg.norm(x - soft)),
                       float(np.linalg.norm(x - previous)))
        if (residual <= tolerance * scale or iteration == 1
                or iteration % 10 == 0 or iteration == dykstra_budget):
            # p is dual-feasible for weighted L1; q is the spectral-ball normal
            # from the exact projection.  Evaluate the Fenchel gap stably.
            invariant = value - x - p - q
            l1_gap = float(np.sum(np.abs(x) * np.maximum(weights - np.sign(x) * p, 0.)))
            nuclear_q = float(np.linalg.svd(q, compute_uv=False).sum())
            pairing = float(np.sum(q * x))
            support_value = radius * nuclear_q
            ball_gap = support_value - pairing
            roundoff = 256 * np.finfo(float).eps * (abs(support_value) + float(np.sum(np.abs(q * x))))
            feasibility_slack = 64 * np.finfo(float).eps * max(1., radius)
            if np.linalg.norm(x, ord=2) > radius + feasibility_slack:
                raise ArithmeticError("anchor proximal certificate is not primal feasible")
            if not np.isfinite(ball_gap) or ball_gap < -roundoff:
                raise ArithmeticError("anchor proximal duality certificate is invalid")
            ball_gap = max(0., ball_gap)  # Only cancellation at floating precision.
            gap = .5 * float(np.sum(invariant * invariant)) + l1_gap + ball_gap
            gap_evaluated_iteration = iteration
            if not np.isfinite(gap):
                raise FloatingPointError("anchor proximal duality gap is nonfinite")
            if (gap + roundoff) / scale / scale <= tolerance:
                candidate = _ProxResult(x.copy(), iteration, True, residual, gap, roundoff, direct_error,
                    iterations_run=iteration, termination_reason="certified",
                    residual_threshold=tolerance * scale, gap_evaluated_iteration=iteration,
                    relative_gap=(gap + roundoff) / scale / scale, dykstra_iterations=iteration)
                if best is None or candidate.error_bound < best.error_bound:
                    best = candidate
                if absolute_gap_tol is None or gap + roundoff <= absolute_gap_tol:
                    return candidate
                if absolute_gap_tol < roundoff and gap <= roundoff:
                    # The certificate is now limited by cancellation, so
                    # further Dykstra sweeps cannot justify this target.
                    break
                # A stricter diagnostic target can be below floating precision.
                # Retain a valid relative certificate and its actual uncertainty;
                # never pretend that the requested absolute gap was attained.
                if (np.array_equal(x, previous) and np.array_equal(p, previous_p)
                        and np.array_equal(q, previous_q)):
                    break
    dykstra_used = iteration
    remaining = max_iterations - dykstra_used
    dual_used = 0
    if remaining:
        from .proximal_dual import solve_weighted_l1_ball_dual

        dual = solve_weighted_l1_ball_dual(value, weights, radius, tolerance=tolerance,
            absolute_gap_tol=absolute_gap_tol, max_iterations=remaining, initial_p=p)
        dual_used = dual.n_iter
        # gap_upper already includes the dual solver's numerical allowances and
        # certifies its returned feasible point; do not count the allowance twice.
        alternative = _ProxResult(dual.value, dykstra_used + dual.certificate_iteration,
            dual.relative_certified, dual.residual, dual.gap_upper,
            iterations_run=dykstra_used + dual_used, termination_reason=dual.termination_reason,
            residual_threshold=None, gap_evaluated_iteration=dykstra_used + dual.certificate_iteration,
            relative_gap=dual.gap_upper / scale / scale, dykstra_iterations=dykstra_used,
            dual_iterations=dual_used, certificate_method="dual_fista",
            certified_error_bound=dual.error_bound)
        if alternative.converged and (best is None or alternative.error_bound < best.error_bound):
            best = alternative
        if best is None:
            return alternative
    if best is not None:
        absolute_met = absolute_gap_tol is None or best.duality_gap + best.gap_roundoff <= absolute_gap_tol
        return replace(best, iterations_run=dykstra_used + dual_used,
            dykstra_iterations=dykstra_used, dual_iterations=dual_used,
            termination_reason="certified" if absolute_met else "absolute_target_unresolved")
    return _ProxResult(x, dykstra_used, False, residual, gap, gap_roundoff=roundoff,
        iterations_run=dykstra_used, termination_reason="duality_gap_not_certified",
        residual_threshold=tolerance * scale, gap_evaluated_iteration=gap_evaluated_iteration,
        relative_gap=_finite((gap + roundoff) / scale / scale), dykstra_iterations=dykstra_used)


def _to_h(chart, state):
    omega_u, omega_v, d, z_u, z_v = chart.unpack(state)
    return chart.pack(omega_u, omega_v, d, z_u / d, z_v / d)


def _to_z(chart, state):
    omega_u, omega_v, d, h_u, h_v = chart.unpack(state)
    return chart.pack(omega_u, omega_v, d, h_u * d, h_v * d)


def _value_gradient_h(chart, state, design, response):
    """Smooth loss and full chain-rule gradient at fixed H, not fixed Z."""
    original = _to_z(chart, state)
    value, gradient = chart.value_gradient(original, design, response)
    _, _, d, h_u, h_v = chart.unpack(state)
    g_u, g_v, g_d, g_zu, g_zv = chart.unpack(gradient)
    fixed_h_d = g_d + np.sum(g_zu * h_u, axis=0) + np.sum(g_zv * h_v, axis=0)
    return value, chart.pack(g_u, g_v, fixed_h_d, g_zu * d, g_zv * d)


def _project_omega(coordinates, rank):
    matrix = skew_matrix(coordinates, rank)
    projected = _project_operator_ball(matrix, .5)
    projected = .5 * (projected - projected.T)
    norm = float(np.linalg.norm(projected, ord=2))
    if norm > .5:
        projected *= np.nextafter(.5, 0.) / norm
    return skew_coordinates(projected)


def _block_trial(chart, state, gradient, L, penalties, margins, tolerance, *, absolute_gap_tol=None):
    """Constrained block-model minimizer and proximal uncertainty estimate."""
    omega_u, omega_v, d, h_u, h_v = chart.unpack(state)
    g_u, g_v, g_d, g_hu, g_hv = chart.unpack(gradient)
    column_penalty = penalties[0] * np.abs(h_u).sum(axis=0) + penalties[1] * np.abs(h_v).sum(axis=0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        next_d = project_singular_values(d - (g_d + column_penalty) / L,
            d_lower=margins.d_lower, d_upper=margins.d_upper, gap=margins.gap)
        next_u = _project_omega(omega_u - g_u / L, chart.rank)
        next_v = _project_omega(omega_v - g_v / L, chart.rank)
        radius = float(np.sqrt((1. - margins.anchor_min) * (1. + margins.anchor_min)))
        started = time.perf_counter()
        pu = _weighted_l1_ball_prox(h_u - g_hu / L, penalties[0] * d / L,
                                    radius, tolerance=tolerance, absolute_gap_tol=absolute_gap_tol)
        pu = replace(pu, elapsed_seconds=time.perf_counter() - started)
        started = time.perf_counter()
        pv = _weighted_l1_ball_prox(h_v - g_hv / L, penalties[1] * d / L,
                                    radius, tolerance=tolerance, absolute_gap_tol=absolute_gap_tol)
        pv = replace(pv, elapsed_seconds=time.perf_counter() - started)
    if not pu.converged or not pv.converged:
        raise _ProxConvergenceError(pu, pv)
    uncertainty = L * np.hypot(pu.error_bound, pv.error_bound)
    return _BlockTrialResult(chart.pack(next_u, next_v, next_d, pu.value, pv.value),
                             float(uncertainty), pu.closed_form and pv.closed_form, pu, pv)


def _roundoff_feasible(chart, state, margins):
    """Move only numerically active boundaries a few ulps inward.

    An H spectral projection can round just outside the original chart check,
    especially after the H -> Z -> H round trip.  This correction never lowers
    the declared anchor floor.  It is not an unrestricted radial trial rule.
    """
    corrected = state.copy()
    omega_u, omega_v, _, h_u, h_v = chart.unpack(corrected)
    # Skew coordinate extraction/reconstruction also has a rounding step.
    for omega in (omega_u, omega_v):
        norm = float(np.linalg.norm(skew_matrix(omega, chart.rank), ord=2))
        if norm >= .5:
            omega *= (.5 - 8 * np.finfo(float).eps) / norm
    radius_squared = (1. - margins.anchor_min) * (1. + margins.anchor_min)
    guard = 64 * np.finfo(float).eps * max(1, chart.rank)
    safe_radius = float(np.sqrt(max(0., radius_squared - guard)))
    for h in (h_u, h_v):
        if not h.size:
            continue
        norm = float(np.linalg.norm(h, ord=2))
        if norm > safe_radius:
            h *= safe_radius / norm
    return corrected


def _mapping(chart, state, gradient, reference_L, penalties, margins, tolerance,
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
            penalties, margins, tolerance, work=work, iteration=iteration)
        trial, uncertainty = initial_trial
        if trial_cache is not None:
            trial_cache.update(key=_trial_key(chart, state, gradient, reference_L,
                                              penalties, margins, tolerance),
                               result=initial_trial)
        movement = float(reference_L * np.linalg.norm(trial - state))
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
                refined, allowance = _evaluate_block_trial(chart, state, gradient, reference_L,
                    penalties, margins, tolerance, absolute_gap_tol=absolute_gap,
                    work=work, iteration=iteration)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, ArithmeticError):
                # Keep the original finite diagnostic if a stricter inner
                # solve cannot deliver a usable certificate.
                break
            refined_movement = float(reference_L * np.linalg.norm(refined - state))
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
                penalties, margins, tolerance),
                failure=_ProxConvergenceError(error.prox_u, error.prox_v))
        return float("inf"), raw, reference_L, str(error), None, None, 0, False


def refine_anchor_projected(
    chart: AnchorChart, initial_state, design, response, *,
    calibration: ResolvedCalibration, margins: Margins,
    iterations: int = 100, max_backtracks: int = 60, loss_offset: float = 0.,
    stationarity_tol: float | None = None, iterate_callback=None,
    numerical_work: dict | None = None, cache_failed_trials: bool = True,
    line_search_strategy: str = DEFAULT_LINE_SEARCH_STRATEGY,
) -> RefinementResult:
    """Refine with full fixed-chart constraints, preserving original Z states.

    Only full complement support caps are supported.  Each outer iteration is
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
    work.update(schema_version=1, line_search_strategy=line_search_strategy,
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
    if not isinstance(margins, Margins) or not isinstance(calibration, ResolvedCalibration):
        raise ValueError("margins and calibration must be Margins and ResolvedCalibration")
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
    reason = chart.domain_reason(x, **domain_args)
    if reason:
        return finish(x, "invalid_initial_state", reason, 0, [])
    limits = calibration.support_limits
    if not isinstance(limits, (tuple, list)) or len(limits) != 2:
        raise ValueError("support_limits must contain exactly two integer bounds")
    for sl, k in zip((chart.z_u_slice, chart.z_v_slice), limits):
        _integer(k, "support limit")
        if k != x[sl].size:
            raise ValueError("anchor-projected refinement requires full complement support limits")
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
            penalty_value = _penalty(chart, x, penalties)
        if not np.isfinite(smooth + penalty_value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("nonfinite initial loss or gradient")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return finish(x, "numerical_failure", str(error), 0, [])
    trial_cache = {}
    diagnostic = _mapping(chart, state, gradient, reference_L, penalties, margins, tolerance,
        stationarity_tol, trial_cache=trial_cache, cache_failed_trials=cache_failed_trials, work=work)
    history = [_record(chart, x, 0, smooth, penalty_value, initial_L, 0., [], loss_offset,
                       diagnostic, effective_support=True)]
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
                key = _trial_key(chart, state, gradient, L, penalties, margins, tolerance)
                if trial_cache.get("key") == key:
                    failure = trial_cache.get("failure")
                    _cache_hit(work, iteration + 1, failure is not None)
                    if failure is not None:
                        # Keep traceback frames out of the fit-local cache.
                        raise _ProxConvergenceError(failure.prox_u, failure.prox_v)
                    block_trial = trial_cache["result"]
                else:
                    block_trial = _evaluate_block_trial(chart, state, gradient, L, penalties, margins,
                        tolerance, work=work, iteration=iteration + 1, phase="line_search")
                trial, _ = block_trial
                trial = _roundoff_feasible(chart, trial, margins)
                original_trial = _to_z(chart, trial)
                step_norm = float(np.linalg.norm(trial - state))
                original_step = float(np.linalg.norm(original_trial - x))
                resolution = 64 * np.finfo(float).eps * relative_scale
                mapping_unresolved = (diagnostic[3] is not None
                    or diagnostic[0] > reference_L * resolution
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
                    trial_penalty = _penalty(chart, original_trial, penalties)
                    objective_change = _objective_change(chart, x, original_trial, design,
                                                         response, penalties, context=loss_context)
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
                    diagnostic = _mapping(chart, state, gradient, reference_L, penalties, margins, tolerance,
                        stationarity_tol, trial_cache=trial_cache, cache_failed_trials=cache_failed_trials,
                        work=work, iteration=iteration + 1)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    diagnostic = (float("inf"), float("inf"), reference_L, str(error))
                    history.append(_record(chart, x, iteration + 1, smooth, penalty_value, L,
                                            step_norm, rejects, loss_offset, diagnostic,
                                            objective_change=objective_change,
                                            relative_step_norm=step_norm / relative_scale,
                                            line_search_start_inverse=start_L, effective_support=True))
                    if iterate_callback is not None:
                        iterate_callback(iteration + 1, x.copy(), history[-1])
                    return finish(x, "numerical_failure", str(error), iteration + 1, history)
                history.append(_record(chart, x, iteration + 1, smooth, penalty_value, L,
                                        step_norm, rejects, loss_offset, diagnostic,
                                        objective_change=objective_change,
                                        relative_step_norm=step_norm / relative_scale,
                                        line_search_start_inverse=start_L, effective_support=True))
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
