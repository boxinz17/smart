"""Sparse refinement with explicit spectral and numerical termination rules."""
from __future__ import annotations

from dataclasses import dataclass
import numbers
import numpy as np

from .calibration import Margins, ResolvedCalibration
from .chart import AnchorChart
from .thresholding import threshold_step
from .spectral import project_singular_values
from .objective import loss_context, objective_change
from .support import coordinate_support


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    objective: float
    smooth_loss: float
    penalty_value: float
    step_size_inverse: float
    step_norm: float
    backtracks: int
    support_u: int
    support_v: int
    anchor_min_u: float
    anchor_min_v: float
    rejections: tuple[str, ...] = ()
    projected_gradient_norm: float | None = None
    raw_gradient_norm: float | None = None
    mapping_step_size_inverse: float | None = None
    mapping_domain_reason: str | None = None
    mapping_displacement: float | None = None
    proximal_uncertainty: float | None = None
    mapping_refinements: int = 0
    mapping_precision_limited: bool = False
    objective_change: float | None = None
    relative_step_norm: float | None = None
    line_search_start_inverse: float | None = None
    raw_support_u: int | None = None
    raw_support_v: int | None = None
    support_tolerance_u: float = 0.
    support_tolerance_v: float = 0.


@dataclass
class RefinementResult:
    state: np.ndarray
    status: str
    message: str
    n_iter: int
    history: list[IterationRecord]
    last_rejection: str | None = None
    termination_reason: str = "unspecified"
    projected_gradient_norm: float | None = None
    raw_gradient_norm: float | None = None
    mapping_displacement: float | None = None
    proximal_uncertainty: float | None = None
    mapping_refinements: int = 0
    mapping_precision_limited: bool = False
    numerical_work: dict | None = None

    @property
    def success(self) -> bool:
        return self.status in ("completed", "converged")


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _penalty(chart, x, penalty):
    return sum(lam * np.abs(x[sl]).sum() for sl, lam in
               zip((chart.z_u_slice, chart.z_v_slice), penalty))


def _record(chart, x, t, smooth, penalty_value, L, step_norm, rejects, offset, diagnostic,
            *, objective_change=None, relative_step_norm=None, line_search_start_inverse=None,
            effective_support=False):
    _, _, _, u_parts, v_parts = chart._parts(x)
    support_u, tolerance_u = coordinate_support(x[chart.z_u_slice], effective=effective_support)
    support_v, tolerance_v = coordinate_support(x[chart.z_v_slice], effective=effective_support)
    return IterationRecord(
        t, float(smooth + penalty_value + offset), float(smooth + offset),
        float(penalty_value), float(L), float(step_norm), len(rejects),
        len(support_u), len(support_v),
        float(np.min(u_parts[-1])), float(np.min(v_parts[-1])), tuple(rejects),
        *diagnostic,
        objective_change=objective_change, relative_step_norm=relative_step_norm,
        line_search_start_inverse=line_search_start_inverse,
        raw_support_u=int(np.count_nonzero(x[chart.z_u_slice])),
        raw_support_v=int(np.count_nonzero(x[chart.z_v_slice])),
        support_tolerance_u=tolerance_u, support_tolerance_v=tolerance_v,
    )


def _trial(chart, x, grad, L, penalties, limits, margins, spectral_step):
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        w = x - grad / L
    if not np.all(np.isfinite(w)):
        return w
    if spectral_step == "projected":
        w[chart.d_slice] = project_singular_values(w[chart.d_slice], d_lower=margins.d_lower,
                                                  d_upper=margins.d_upper, gap=margins.gap)
    for sl, k, lam in zip((chart.z_u_slice, chart.z_v_slice), limits, penalties):
        with np.errstate(over="ignore"):
            shrink = float(lam) / L
        w[sl] = np.zeros_like(w[sl]) if not np.isfinite(shrink) else threshold_step(w[sl], shrink, k)
    return w


def _mapping(chart, x, grad, reference_L, penalties, limits, margins, domain_args):
    """A spectral/support proximal mapping, separate from line-search steps.

    This mapping does not project nonlinear Cayley/anchor constraints. Its
    candidate domain check is reported, and an infeasible candidate cannot
    trigger the optional stationarity stop. The bounded reference inverse
    step does not grow with backtracking or huge prescribed constants.
    """
    raw = float(np.linalg.norm(grad))
    try:
        w = _trial(chart, x, grad, reference_L, penalties, limits, margins, "projected")
        if not np.all(np.isfinite(w)):
            return float("inf"), raw, reference_L, "nonfinite mapping trial"
        norm = float(reference_L * np.linalg.norm(w - x))
        return norm, raw, reference_L, chart.domain_reason(w, **domain_args)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return float("inf"), raw, reference_L, str(error)


def _stall_reason(rejects):
    recent = rejects[-16:]
    if any("singular-value gap" in reason for reason in recent):
        return "active_gap_stall"
    if any(any(token in reason for token in
               ("Cayley", "anchor", "square root", "positive definite", "singular values are outside"))
           for reason in recent):
        return "active_constraint_stall"
    return "numerical_stagnation"


def _result(x, status, message, n_iter, history, reason=None, termination_reason=None):
    last = history[-1] if history else None
    return RefinementResult(x, status, message, n_iter, history, reason,
                            termination_reason or status,
                            last.projected_gradient_norm if last else None,
                            last.raw_gradient_norm if last else None,
                            last.mapping_displacement if last else None,
                            last.proximal_uncertainty if last else None,
                            last.mapping_refinements if last else 0,
                            last.mapping_precision_limited if last else False)


def refine(
    chart: AnchorChart, initial_state, design, response, *,
    calibration: ResolvedCalibration, margins: Margins,
    iterations: int = 100, max_backtracks: int = 60, loss_offset: float = 0.,
    spectral_step: str = "projected", stationarity_tol: float | None = None,
    iterate_callback=None,
) -> RefinementResult:
    """Refine a supplied feasible state using fixed bases and fixed anchors.

    The caller owns the statistical quality of a supplied initial state.
    Passing domain checks alone does not certify membership of a truth basin.
    max_backtracks bounds doublings, so zero still permits the first trial.
    loss_offset is the constant response loss outside the right source span.
    It is included in reported losses but removed from acceptance comparisons.
    ``spectral_step='projected'`` uses exact bounded isotonic projection of the
    singular values; ``'reject'`` retains the manuscript trial rule. Setting
    stationarity_tol enables a stop based on the independently evaluated
    spectral/support proximal mapping, not the accepted step length. With None,
    exactly ``iterations`` updates are requested unless a numerical failure or
    stall occurs. The callback receives (iteration, state.copy(), record) at
    initialization and each accepted update; callback exceptions propagate.
    """
    _integer(iterations, "iterations")
    _integer(max_backtracks, "max_backtracks")
    if spectral_step not in ("projected", "reject"):
        raise ValueError("spectral_step must be 'projected' or 'reject'")
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
        return _result(x, "invalid_initial_state", reason, 0, [])
    limits = calibration.support_limits
    if not isinstance(limits, (tuple, list)) or len(limits) != 2:
        raise ValueError("support_limits must contain exactly two integer bounds")
    for sl, k in zip((chart.z_u_slice, chart.z_v_slice), limits):
        _integer(k, "support limit")
        if k > x[sl].size:
            raise ValueError("support limit exceeds chart coordinate count")
        if np.count_nonzero(x[sl]) > k:
            return _result(x, "invalid_initial_state", "initial support exceeds limit", 0, [])
    penalty = getattr(calibration, "penalties", calibration.penalty)
    if np.isscalar(penalty):
        penalty = (float(penalty), float(penalty))
    penalty = np.asarray(penalty, dtype=float)
    initial_L = calibration.step_size_inverse
    if penalty.shape != (2,) or not np.all(np.isfinite(penalty)) or np.any(penalty < 0) or not np.isfinite(initial_L) or initial_L <= 0:
        raise ValueError("invalid penalty or inverse step size")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            smooth, grad = chart.value_gradient(x, design, response)
            pen = _penalty(chart, x, penalty)
        if not np.isfinite(smooth + pen) or not np.all(np.isfinite(grad)):
            raise FloatingPointError("nonfinite initial loss or gradient")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return _result(x, "numerical_failure", str(error), 0, [])
    reference_L = float(np.clip(initial_L, 1., 1000.))
    diagnostic = _mapping(chart, x, grad, reference_L, penalty, limits, margins, domain_args)
    history = [_record(chart, x, 0, smooth, pen, initial_L, 0., [], loss_offset, diagnostic)]
    if iterate_callback is not None:
        iterate_callback(0, x.copy(), history[-1])
    for t in range(iterations):
        if stationarity_tol is not None and diagnostic[0] <= stationarity_tol and diagnostic[3] is None:
            return _result(x, "converged", "The spectral/support projected gradient mapping meets tolerance.",
                           t, history, termination_reason="stationarity")
        L = initial_L  # The manuscript resets L at every accepted iterate.
        context = loss_context(chart, x, design, response)
        rejects = []
        accepted = False
        for trial in range(max_backtracks + 1):
            try:
                w = _trial(chart, x, grad, L, penalty, limits, margins, spectral_step)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                return _result(x, "numerical_failure", str(error), t, history)
            if not np.all(np.isfinite(w)):
                reason = "nonfinite trial"
            else:
                step = w - x
                step_norm = float(np.linalg.norm(step))
                resolution = 64 * np.finfo(float).eps * max(1., float(np.linalg.norm(x)))
                mapping_resolved = (not np.isfinite(diagnostic[0]) or
                    diagnostic[0] > 64 * np.finfo(float).eps * max(1., diagnostic[0]))
                if step_norm <= resolution and mapping_resolved:
                    termination = _stall_reason(rejects)
                    return _result(x, "numerical_stagnation",
                        "The trial is at numerical resolution while the independently evaluated projected gradient mapping is nonzero.",
                        t, history, rejects[-1] if rejects else None, termination)
                reason = chart.domain_reason(w, **domain_args)
                if reason is None and step_norm > margins.trial_radius:
                    reason = "trial radius exceeded"
                if reason is None:
                    try:
                        trial_smooth = chart.loss(w, design, response)
                        trial_pen = _penalty(chart, w, penalty)
                        trial_obj = trial_smooth + trial_pen
                        change = objective_change(chart, x, w, design, response, penalty, context=context)
                        required = .25 * L * step_norm**2
                        if not np.isfinite(trial_obj) or not np.isfinite(required):
                            reason = "nonfinite trial objective or decrease"
                        elif change <= -required:
                            accepted = True
                        else:
                            reason = "insufficient objective decrease"
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                        reason = "trial loss evaluation failed"
            if accepted:
                x, smooth, pen = w, trial_smooth, trial_pen
                try:
                    _, grad = chart.value_gradient(x, design, response)
                    if not np.all(np.isfinite(grad)):
                        raise FloatingPointError("nonfinite gradient")
                    diagnostic = _mapping(chart, x, grad, reference_L, penalty, limits, margins, domain_args)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    diagnostic = (float("inf"), float("inf"), reference_L, str(error))
                    history.append(_record(chart, x, t + 1, smooth, pen, L, step_norm, rejects, loss_offset,
                                           diagnostic, objective_change=change))
                    if iterate_callback is not None:
                        iterate_callback(t + 1, x.copy(), history[-1])
                    return _result(x, "numerical_failure", str(error), t + 1, history)
                history.append(_record(chart, x, t + 1, smooth, pen, L, step_norm, rejects, loss_offset,
                                       diagnostic, objective_change=change))
                if iterate_callback is not None:
                    iterate_callback(t + 1, x.copy(), history[-1])
                break
            rejects.append(str(reason))
            if trial < max_backtracks:
                L *= 2.
                if not np.isfinite(L):
                    return _result(x, "numerical_failure", "Inverse step size overflowed.",
                                   t, history, rejects[-1])
        if not accepted:
            return _result(x, "line_search_failed", "All allowed line-search trials were rejected.",
                           t, history, rejects[-1], "backtracking_exhausted")
    if iterations and stationarity_tol is not None and diagnostic[0] <= stationarity_tol and diagnostic[3] is None:
        return _result(x, "converged", "The spectral/support projected gradient mapping meets tolerance.",
                       iterations, history, termination_reason="stationarity")
    return _result(x, "completed", f"Completed {iterations} refinement updates.", iterations, history,
                   termination_reason="max_iterations")
