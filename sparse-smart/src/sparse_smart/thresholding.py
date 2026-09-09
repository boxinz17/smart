"""Explicit minimizer of the manuscript's isotropic quadratic subproblem."""
from __future__ import annotations

import numbers
import numpy as np


def _array(values):
    if np.iscomplexobj(values):
        raise ValueError("thresholding requires real values")
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("thresholding requires finite values")
    return array


def soft_threshold(values, threshold: float) -> np.ndarray:
    """Entrywise soft thresholding, preserving the input shape."""
    x = _array(values)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def hard_threshold(values, limit: int) -> np.ndarray:
    """Keep at most limit entries, breaking exact ties by C-order index."""
    x = _array(values)
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, numbers.Integral):
        raise ValueError("support limit must be an integer")
    if not 0 <= limit <= x.size:
        raise ValueError("support limit must lie between zero and coordinate count")
    flat = x.ravel(order="C")
    result = np.zeros_like(flat)
    if limit:
        order = np.argsort(-np.abs(flat), kind="stable")[:limit]
        result[order] = flat[order]
    return result.reshape(x.shape)


def threshold_step(values, threshold: float, limit: int) -> np.ndarray:
    """Soft threshold, then select the largest remaining magnitudes."""
    return hard_threshold(soft_threshold(values, threshold), limit)

