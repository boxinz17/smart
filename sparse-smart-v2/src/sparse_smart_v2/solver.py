"""Chart proximal gradient with hard caps outside the declared free rows.

The quadratic model is solved exactly by soft thresholding followed by top-k
truncation. Domain tests follow that step; they never project or rotate it.
"""
from __future__ import annotations

import numbers
import numpy as np
from scipy.linalg import norm as stable_norm

from sparse_smart.objective import loss_context
from sparse_smart.solver import IterationRecord, RefinementResult
from sparse_smart.stopping import ValidationStopRequest

from .support import threshold_state


def _blocks(chart, state, free_rows):
    return (state[chart.z_u_slice].reshape(free_rows.penalized_u.shape)[free_rows.penalized_u],
            state[chart.z_v_slice].reshape(free_rows.penalized_v.shape)[free_rows.penalized_v])


def penalty_value(chart, state, free_rows, penalties):
    """The penalty on the actual outside weighted factor entries only."""
    return float(sum(lam * np.sum(np.abs(block), dtype=np.longdouble)
                     for lam, block in zip(penalties, _blocks(chart, state, free_rows))))


def objective_change(chart, state, trial, design, response, free_rows, penalties, *, context=None):
    """Stable full objective difference without subtracting two large losses."""
    P, d, Q, projected, residual = (loss_context(chart, state, design, response)
                                   if context is None else context)
    next_P, next_d, next_Q = chart.reconstruct(trial)
    delta_prediction = (((design @ (next_P - P)) * next_d) @ next_Q.T
                        + (projected * (next_d - d)) @ next_Q.T
                        + (projected * d) @ (next_Q - Q).T)
    delta = (np.sum(residual * delta_prediction, dtype=np.longdouble)
             + np.sum(delta_prediction * delta_prediction, dtype=np.longdouble) / 2) / len(design)
    for lam, old, new in zip(penalties, _blocks(chart, state, free_rows),
                            _blocks(chart, trial, free_rows)):
        delta += lam * np.sum(np.abs(new) - np.abs(old), dtype=np.longdouble)
    if not np.isfinite(delta):
        raise FloatingPointError("nonfinite objective change")
    return float(delta)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _mapping(chart, state, gradient, reference_L, calibration, free_rows, domain, radius):
    """Independent fixed-step hard-prox mapping, not the accepted step size.

    This is a feasible hard-prox fixed-point diagnostic. It is not a KKT
    certificate for the chart inequalities or a global optimality test.
    """
    raw = float(stable_norm(gradient, check_finite=False))
    try:
        trial = threshold_state(chart, state - gradient / reference_L, free_rows,
                                calibration.penalties, calibration.support_limits, reference_L)
        # Free-coordinate mapping equals the gradient exactly. Retaining that
        # expression avoids a false zero from rounded state subtraction.
        mapped = gradient.copy()
        for sl, mask, lam in ((chart.z_u_slice, free_rows.penalized_u, calibration.penalties[0]),
                              (chart.z_v_slice, free_rows.penalized_v, calibration.penalties[1])):
            flat = mask.ravel()
            old, proposal = state[sl][flat], trial[sl][flat]
            # For retained coordinates use the algebraic expression, avoiding
            # cancellation when a tiny soft-gradient step rounds back to x.
            values = reference_L * old
            retained = proposal != 0
            values[retained] = gradient[sl][flat][retained] + lam * np.sign(proposal[retained])
            mapped[sl][flat] = values
        norm = float(stable_norm(mapped, check_finite=False))
        displacement = float(stable_norm(trial - state, check_finite=False))
        reason = chart.domain_reason(trial, **domain)
        if reason is None and displacement > radius:
            reason = "reference mapping exceeds trial radius"
        if not np.isfinite(norm):
            reason = "nonfinite reference mapping"
        return norm, raw, reference_L, reason, displacement
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return float("inf"), raw, reference_L, str(error), float("inf")


def _record(chart, state, free_rows, t, smooth, penalty, L, step, rejections,
            diagnostic, *, change=None, initial_L=None):
    P, _, Q = chart.reconstruct(state)
    counts = tuple(int(np.count_nonzero(b)) for b in _blocks(chart, state, free_rows))
    return IterationRecord(
        iteration=t, objective=float(smooth + penalty), smooth_loss=float(smooth),
        penalty_value=float(penalty), step_size_inverse=float(L), step_norm=float(step),
        backtracks=len(rejections), support_u=counts[0], support_v=counts[1],
        anchor_min_u=float(np.linalg.svd(P[chart.anchors_u], compute_uv=False)[-1]),
        anchor_min_v=float(np.linalg.svd(Q[chart.anchors_v], compute_uv=False)[-1]),
        rejections=tuple(rejections), projected_gradient_norm=diagnostic[0],
        raw_gradient_norm=diagnostic[1], mapping_step_size_inverse=diagnostic[2],
        mapping_domain_reason=diagnostic[3], mapping_displacement=diagnostic[4],
        objective_change=change, relative_step_norm=step / max(1., float(np.linalg.norm(state))),
        line_search_start_inverse=initial_L, raw_support_u=counts[0], raw_support_v=counts[1],
    )


def _result(x, status, message, n_iter, history, reason=None, termination_reason=None):
    last = history[-1] if history else None
    return RefinementResult(
        state=x.copy(), status=status, message=str(message), n_iter=n_iter,
        history=history, last_rejection=reason, termination_reason=termination_reason or status,
        projected_gradient_norm=last.projected_gradient_norm if last else None,
        raw_gradient_norm=last.raw_gradient_norm if last else None,
        mapping_displacement=last.mapping_displacement if last else None,
        numerical_work={"accepted_updates": n_iter,
                        "accepted_update_backtracks": sum(r.backtracks for r in history),
                        "inner_proximal_solves": 0},
    )


def refine(chart, initial_state, design, response, *, calibration, margins, free_rows,
           iterations=500, max_backtracks=60, stationarity_tol=None, iterate_callback=None):
    """Run the note's finite-trial, isotropic omega/d/z update.

    Every iteration restarts at calibration.step_size_inverse. A maximum of
    max_backtracks doublings is allowed, including an initial trial when zero.
    Budget completion, validation stopping, fixed-point tolerance, and failed
    line search/numerical stagnation have distinct termination reasons.
    """
    _integer(iterations, "iterations")
    _integer(max_backtracks, "max_backtracks")
    if stationarity_tol is not None and (
        isinstance(stationarity_tol, (bool, np.bool_))
        or not isinstance(stationarity_tol, numbers.Real)
        or not np.isfinite(stationarity_tol) or stationarity_tol <= 0
    ):
        raise ValueError("stationarity_tol must be finite and positive or None")
    if np.iscomplexobj(initial_state) or np.iscomplexobj(design) or np.iscomplexobj(response):
        raise ValueError("state, design and response must be real")
    state = np.asarray(initial_state, dtype=float).copy()
    design, response = chart._data(design, response)
    domain = dict(d_lower=margins.d_lower, d_upper=margins.d_upper,
                  gap=margins.gap, anchor_min=margins.anchor_min)
    # A zero-penalty identity proposal validates masks/caps without altering
    # the caller's initial state. Initial infeasibility is never repaired here.
    threshold_state(chart, state, free_rows, (0., 0.), calibration.support_limits, 1.)
    reason = chart.domain_reason(state, **domain)
    if reason:
        return _result(state, "invalid_initial_state", reason, 0, [])
    if any(np.count_nonzero(b) > k for b, k in
           zip(_blocks(chart, state, free_rows), calibration.support_limits)):
        return _result(state, "invalid_initial_state", "initial outside support exceeds cap", 0, [])
    initial_L = float(calibration.step_size_inverse)
    # Keep the diagnostic independent of very large accepted/backtracked L.
    reference_L = float(np.clip(initial_L, 1., 1000.))
    penalties = calibration.penalties
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            smooth, gradient = chart.value_gradient(state, design, response)
            penalty = penalty_value(chart, state, free_rows, penalties)
        if not np.all(np.isfinite(gradient)) or not np.isfinite(smooth + penalty):
            raise FloatingPointError("nonfinite initial objective or gradient")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return _result(state, "numerical_failure", error, 0, [])
    diagnostic = _mapping(chart, state, gradient, reference_L, calibration, free_rows,
                          domain, margins.trial_radius)
    history = [_record(chart, state, free_rows, 0, smooth, penalty, initial_L, 0., [],
                       diagnostic, initial_L=initial_L)]

    def finish_if_requested(t):
        request = iterate_callback(t, state.copy(), history[-1]) if iterate_callback else None
        stationary = diagnostic[0] == 0. or (stationarity_tol is not None and diagnostic[0] <= stationarity_tol)
        if iterations and diagnostic[3] is None and stationary:
            message = ("Feasible fixed-step hard-prox mapping is exactly zero."
                       if diagnostic[0] == 0. else "Feasible fixed-step hard-prox mapping meets tolerance.")
            return _result(state, "converged", message,
                           t, history, termination_reason="stationarity")
        if isinstance(request, ValidationStopRequest):
            return _result(state, "completed", request.message, t, history,
                           termination_reason="validation_stop")
        return None

    result = finish_if_requested(0)
    if result is not None:
        return result
    for t in range(iterations):
        L, rejections = initial_L, []
        context = loss_context(chart, state, design, response)
        for backtrack in range(max_backtracks + 1):
            accepted = False
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    trial = threshold_state(chart, state - gradient / L, free_rows,
                                            penalties, calibration.support_limits, L)
                step = float(np.linalg.norm(trial - state))
                resolution = 64 * np.finfo(float).eps * max(1., float(np.linalg.norm(state)))
                if step <= resolution:
                    return _result(state, "numerical_stagnation",
                        "Trial displacement reached numerical resolution; no optimization convergence is inferred.",
                        t, history, rejections[-1] if rejections else None,
                        "numerical_stagnation")
                reason = chart.domain_reason(trial, **domain)
                if reason is None and step > margins.trial_radius:
                    reason = "trial radius exceeded"
                if reason is None:
                    trial_smooth = chart.loss(trial, design, response)
                    trial_penalty = penalty_value(chart, trial, free_rows, penalties)
                    change = objective_change(chart, state, trial, design, response,
                                              free_rows, penalties, context=context)
                    required = .25 * L * step**2
                    if not np.isfinite(trial_smooth + trial_penalty) or not np.isfinite(required):
                        reason = "nonfinite objective or sufficient decrease"
                    elif change <= -required:
                        accepted = True
                    else:
                        reason = "insufficient objective decrease"
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                reason = str(error)
            if accepted:
                state, smooth, penalty = trial, trial_smooth, trial_penalty
                try:
                    _, gradient = chart.value_gradient(state, design, response)
                    if not np.all(np.isfinite(gradient)):
                        raise FloatingPointError("nonfinite gradient")
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    diagnostic = (float("inf"), float("inf"), reference_L, str(error), float("inf"))
                    history.append(_record(chart, state, free_rows, t + 1, smooth, penalty,
                        L, step, rejections, diagnostic, change=change, initial_L=initial_L))
                    return _result(state, "numerical_failure", error, t + 1, history)
                diagnostic = _mapping(chart, state, gradient, reference_L, calibration,
                                      free_rows, domain, margins.trial_radius)
                history.append(_record(chart, state, free_rows, t + 1, smooth, penalty,
                    L, step, rejections, diagnostic, change=change, initial_L=initial_L))
                result = finish_if_requested(t + 1)
                if result is not None:
                    return result
                break
            rejections.append(str(reason))
            if backtrack < max_backtracks:
                L *= 2.
                if not np.isfinite(L):
                    return _result(state, "numerical_failure", "Inverse step overflowed.",
                                   t, history, rejections[-1])
        else:
            return _result(state, "line_search_failed", "All line-search trials were rejected.",
                           t, history, rejections[-1], "backtracking_exhausted")
    return _result(state, "completed", f"Completed {iterations} refinement updates.",
                   iterations, history, termination_reason="max_iterations")
