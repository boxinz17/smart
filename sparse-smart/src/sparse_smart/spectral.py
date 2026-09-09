"""Euclidean projection onto bounded singular values with a minimum gap."""

import numpy as np


def project_singular_values(values, *, d_lower, d_upper, gap):
    """Project onto d_lower <= d[r-1], d[0] <= d_upper, d[j]-d[j+1] >= gap.

    Adding j*gap transforms the problem into decreasing isotonic regression
    with common bounds [d_lower+(r-1)*gap, d_upper]. Pool adjacent violators,
    then clip its block means. Final ulp corrections enforce comparisons in
    floating arithmetic without changing the projection at numerical precision.
    """
    if np.iscomplexobj(values):
        raise ValueError("singular values must be real")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("singular values must be a nonempty finite vector")
    bounds = np.asarray([d_lower, d_upper, gap], dtype=float)
    if not np.all(np.isfinite(bounds)) or d_lower < 0 or d_upper < d_lower or gap < 0:
        raise ValueError("spectral bounds require 0 <= d_lower <= d_upper and gap >= 0")
    rank = values.size
    lower = d_lower + (rank - 1) * gap
    if not np.isfinite(lower) or lower > d_upper:
        raise ValueError("singular-value bounds and minimum gaps have no feasible point")
    transformed = values + np.arange(rank) * gap
    if not np.all(np.isfinite(transformed)):
        raise ValueError("transformed spectral coordinates overflow")
    means, counts = [], []
    for value in transformed:
        means.append(float(value))
        counts.append(1)
        while len(means) > 1 and means[-2] < means[-1]:
            count = counts[-2] + counts[-1]
            mean = means[-2] * (counts[-2] / count) + means[-1] * (counts[-1] / count)
            means[-2:] = [mean]
            counts[-2:] = [count]
    isotonic = np.repeat(np.clip(means, lower, d_upper), counts)
    projected = isotonic - np.arange(rank) * gap
    # Bias at most a few ulps into the feasible set when subtraction rounds a
    # mathematically active gap below its declared value.
    projected = np.clip(projected, d_lower, d_upper)
    for j in range(1, rank):
        if projected[j - 1] - projected[j] < gap:
            projected[j] = projected[j - 1] - gap
            if projected[j - 1] - projected[j] < gap:
                projected[j] = np.nextafter(projected[j], -np.inf)
    if projected[-1] < d_lower:
        projected[-1] = d_lower
        for j in range(rank - 2, -1, -1):
            if projected[j] - projected[j + 1] < gap:
                projected[j] = projected[j + 1] + gap
                if projected[j] - projected[j + 1] < gap:
                    projected[j] = np.nextafter(projected[j], np.inf)
    if projected[0] > d_upper or projected[-1] < d_lower or np.any(projected[:-1] - projected[1:] < gap):
        raise ValueError("spectral constraints cannot be represented at the supplied floating precision")
    return projected
