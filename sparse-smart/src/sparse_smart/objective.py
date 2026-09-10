"""Stable differences of the original fixed-chart objective."""

import numpy as np


def loss_context(chart, state, design, response):
    """Cache the current prediction factors and residual for backtracking."""
    P, d, Q = chart.reconstruct(state)
    projected = design @ P
    residual = (projected * d) @ Q.T - response
    return P, d, Q, projected, residual


def objective_change(chart, state, trial, design, response, penalties, *, context=None):
    """Compute the original objective difference without subtracting losses.

    Factor differences telescope the prediction change. The quadratic identity
    and per-entry L1 changes retain small decreases even when separately
    recorded losses round to the same float. No acceptance slack is added.
    """
    P, d, Q, projected, residual = (loss_context(chart, state, design, response)
                                   if context is None else context)
    next_P, next_d, next_Q = chart.reconstruct(trial)
    delta_prediction = (((design @ (next_P - P)) * next_d) @ next_Q.T
                        + (projected * (next_d - d)) @ next_Q.T
                        + (projected * d) @ (next_Q - Q).T)
    # Extended accumulation reduces cancellation without requiring a different
    # linear-algebra backend or changing the optimization coordinates.
    delta = (np.sum(residual * delta_prediction, dtype=np.longdouble)
             + np.sum(delta_prediction * delta_prediction, dtype=np.longdouble) / 2) / len(design)
    for sl, penalty in zip((chart.z_u_slice, chart.z_v_slice), penalties):
        delta += penalty * np.sum(np.abs(trial[sl]) - np.abs(state[sl]), dtype=np.longdouble)
    value = float(delta)
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite objective change")
    return value
