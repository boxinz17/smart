"""Deterministic free-row expansion and entrywise masked support constraints."""

from dataclasses import dataclass

import numpy as np

from sparse_smart.calibration import penalty_pair
from sparse_smart.support import coordinate_support

from .calibration import _integer_pair, _real


@dataclass(frozen=True)
class FreeRows:
    rows_u: np.ndarray
    rows_v: np.ndarray
    penalized_u: np.ndarray
    penalized_v: np.ndarray

    def __post_init__(self):
        for name in ("rows_u", "rows_v"):
            value = np.asarray(getattr(self, name))
            if (value.ndim != 1 or value.dtype.kind not in "iu" or np.any(value < 0)
                    or np.any(np.diff(value.astype(np.int64)) <= 0)):
                raise ValueError(f"{name} must contain sorted distinct nonnegative integer rows")
            value = np.array(value, dtype=np.intp, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in ("penalized_u", "penalized_v"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 2 or value.dtype.kind != "b":
                raise ValueError(f"{name} must be a two-dimensional boolean mask")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def capacities(self):
        return int(np.count_nonzero(self.penalized_u)), int(np.count_nonzero(self.penalized_v))


def choose_free_rows(chart, free_directions=None):
    """Protect anchors, then add earliest remaining source rows (zero-based).

    Masks have the chart's z-block shapes, with True precisely at entries
    outside the expanded free sets. Support capacities count entries, not rows.
    """
    if free_directions is None:
        free_directions = (chart.rank, chart.rank)
    counts = _integer_pair(free_directions, "free_directions")
    rows, masks = [], []
    for side, count in zip(("u", "v"), counts):
        dimension = getattr(chart, f"n_{side}")
        if not chart.rank <= count <= dimension:
            raise ValueError(f"free_directions for side {side} must lie between rank and {dimension}")
        anchors = getattr(chart, f"anchors_{side}")
        complement = getattr(chart, f"complement_{side}")
        free = np.sort(np.concatenate((anchors, complement[:count - chart.rank])))
        rows.append(free)
        masks.append(np.repeat((~np.isin(complement, free))[:, None], chart.rank, axis=1))
    return FreeRows(rows[0], rows[1], masks[0], masks[1])


def validate_support_limits(free_rows, support_limits):
    limits = _integer_pair(support_limits, "support_limits")
    for side, limit, capacity in zip(("u", "v"), limits, free_rows.capacities):
        if limit > capacity:
            raise ValueError(f"support limit for side {side} exceeds masked capacity {capacity}")
    return limits


def _check_masks(chart, free_rows):
    if not isinstance(free_rows, FreeRows):
        raise TypeError("free_rows must be FreeRows")
    for side in ("u", "v"):
        rows = getattr(free_rows, f"rows_{side}")
        mask = getattr(free_rows, f"penalized_{side}")
        dimension = getattr(chart, f"n_{side}")
        anchors, complement = getattr(chart, f"anchors_{side}"), getattr(chart, f"complement_{side}")
        if np.any(rows >= dimension) or not np.isin(anchors, rows).all():
            raise ValueError(f"free rows on side {side} must contain its anchors and be in range")
        expected = np.repeat((~np.isin(complement, rows))[:, None], chart.rank, axis=1)
        if not np.array_equal(mask, expected):
            raise ValueError(f"penalized mask on side {side} does not match chart rows")


def threshold_state(chart, state, free_rows, penalties, support_limits, inverse_step):
    """Soft/hard threshold only masked entries of an already stepped state.

    Free coordinates are copied verbatim. Equal magnitudes retain the earliest
    C-order masked index. This is the algebraic sparse-model update; it neither
    reconstructs factors nor assumes chart feasibility after entry deletion.
    """
    _check_masks(chart, free_rows)
    limits = validate_support_limits(free_rows, support_limits)
    penalties = penalty_pair(penalties)
    inverse_step = _real(inverse_step, "inverse_step", positive=True)
    blocks = chart.unpack(state)
    result = chart.pack(*(np.array(block, copy=True) for block in blocks))
    for side, penalty, limit in zip(("u", "v"), penalties, limits):
        mask = getattr(free_rows, f"penalized_{side}")
        block = result[getattr(chart, f"z_{side}_slice")].reshape(mask.shape)
        selected = block[mask]
        values = np.sign(selected) * np.maximum(np.abs(selected) - penalty / inverse_step, 0)
        if limit < values.size:
            order = np.argsort(-np.abs(values), kind="stable")
            values[order[limit:]] = 0
        block[mask] = values
    return result


__all__ = ["FreeRows", "choose_free_rows", "threshold_state", "validate_support_limits", "coordinate_support"]
