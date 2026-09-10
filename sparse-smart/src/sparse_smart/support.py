"""Literal and effective coordinate support for numerical diagnostics.

Effective support is a reporting convention for the approximate anchor
proximal solver, not a support guarantee or a coefficient-error certificate.
Coefficients and hard support feasibility are never changed by this helper.
"""

import numpy as np


def coordinate_support(values, *, effective=False):
    """Return C-order support indices and the threshold used to report them.

    Literal support uses a zero threshold. Effective support uses
    ``max(1e-10 * scale, 64 * eps * scale)``, where
    ``scale = max(1, max(abs(values)))`` separately for each Z block.
    """
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("support coordinates must be finite")
    tolerance = 0.
    if effective:
        scale = max(1., float(np.max(np.abs(values), initial=0.)))
        tolerance = max(1e-10 * scale, 64 * np.finfo(float).eps * scale)
    return np.flatnonzero(np.abs(values.ravel(order="C")) > tolerance), float(tolerance)
