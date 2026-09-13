"""Adaptive chart anchors within the exact, unchanged free source rows.

This is a change of coordinates for the current factors, not a new
initializer, source basis, penalty mask, or constrained optimum. A bounded
QR-and-row-swap search need not find every possible usable anchor. Historical
states must retain their own chart when a switch is accepted.
"""

from __future__ import annotations

import hashlib
from typing import NamedTuple

import numpy as np

from sparse_smart.anchors import AnchorFailure, _pivot_columns
from sparse_smart.chart import AnchorChart, skew_matrix

from .calibration import _real
from .support import FreeRows, _check_masks


_TRIGGER_MULTIPLE = 1.25
_NUMERICAL_GAIN_EPSILON_MULTIPLE = 256
_ROTATION_TRIGGER = .45
_MAX_SWAP_ROUNDS = 16
_MAX_PRIMARY_SEARCH_STARTS = 2
CHART_CONTINUATION_RULE = "restricted_neighbor_fallback_numerical_gain_and_rotation_recenter_v4"


class AnchorSwitch(NamedTuple):
    chart: AnchorChart
    state: np.ndarray
    free_rows: FreeRows
    event: dict


def _margin(factor, rows):
    return float(np.linalg.svd(factor[rows], compute_uv=False)[-1])


def _center_sha256(center):
    return hashlib.sha256(np.asarray(center, dtype="<f8").tobytes(order="C")).hexdigest()


def _improve_candidate(factor, free, start):
    """Run one bounded improving search; never discard a different basin."""
    rows = start.copy()
    score = _margin(factor, rows)
    swap_rounds = 0
    for _ in range(_MAX_SWAP_ROUNDS):
        best_rows, best_score = rows, score
        remaining = np.setdiff1d(free, rows)
        tolerance = 64 * np.finfo(float).eps * max(1., score)
        for removed in rows:
            for added in remaining:
                trial = np.sort(np.append(rows[rows != removed], added))
                value = _margin(factor, trial)
                if value > best_score + tolerance:
                    best_rows, best_score = trial, value
        if np.array_equal(rows, best_rows):
            break
        rows, score = best_rows, best_score
        swap_rounds += 1
    return rows, score, swap_rounds


def _candidate_search(factor, free, old):
    """Preserve successful two-start searches; fall back only without gain.

    The fallback uses the one-row-exchange neighbors of the current anchors
    as deterministic starts. Each start receives at most 16 improving swaps;
    the fallback stops at its first numerically meaningful gain. There are
    at most rank*(free_count-rank) additional starts, a polynomial bound.
    These are local searches, not enumeration of all rank-sized subsets.
    """
    old_margin = _margin(factor, old)
    threshold = old_margin + _NUMERICAL_GAIN_EPSILON_MULTIPLE * np.finfo(float).eps * max(1., old_margin)
    max_fallback = len(old) * (len(free) - len(old))
    work = dict(primary_starts=0, primary_best_margin=old_margin, fallback_triggered=False,
                fallback_starts=0, maximum_fallback_starts=max_fallback,
                maximum_search_starts=_MAX_PRIMARY_SEARCH_STARTS + max_fallback,
                winning_start=None, winning_seed=None, winning_swap_rounds=0,
                fallback_accepted=False)
    if len(free) == len(old):
        return None, work
    starts = [("current_anchors", old.copy())]
    try:
        # The restricted factor is not orthonormal. Only the deterministic
        # pivoting primitive is applicable, not select_anchor's entry proof.
        pivots = _pivot_columns(factor[free].T, free)
        qr = np.sort(free[pivots])
        if not np.array_equal(qr, old):
            starts.append(("restricted_cpqr", qr))
    except AnchorFailure:
        # A pivoting failure need not rule out improving the valid old chart.
        pass
    best = None
    seen = set()
    for name, start in starts:
        result = _improve_candidate(factor, free, start)
        seen.add(tuple(start.tolist()))
        work["primary_starts"] += 1
        if best is None or result[1] > best[1]:
            best = result
            work.update(winning_start=name, winning_seed=start.tolist(), winning_swap_rounds=result[2])
    work["primary_best_margin"] = best[1]
    # This exact return preserves the existing successful two-start path.
    if best[1] > threshold:
        return best, work
    work["fallback_triggered"] = True
    remaining = np.setdiff1d(free, old)
    for removed in old:
        for added in remaining:
            start = np.sort(np.append(old[old != removed], added))
            key = tuple(start.tolist())
            if key in seen:
                continue
            seen.add(key)
            result = _improve_candidate(factor, free, start)
            work["fallback_starts"] += 1
            if result[1] > best[1]:
                best = result
                work.update(winning_start="current_anchor_one_exchange", winning_seed=start.tolist(),
                            winning_swap_rounds=result[2])
            if result[1] > threshold:
                work["fallback_accepted"] = True
                return best, work
    return best, work


def _candidate(factor, free, old):
    """Compatibility result: rows, margin, winning swap rounds, or None."""
    return _candidate_search(factor, free, old)[0]


def _mask(chart, side, rows):
    complement = getattr(chart, f"complement_{side}")
    return np.repeat((~np.isin(complement, rows))[:, None], chart.rank, axis=1)


def _repack_side(old_chart, new_chart, state, factor, d, side):
    old_complement = getattr(old_chart, f"complement_{side}")
    new_complement = getattr(new_chart, f"complement_{side}")
    old_z = state[getattr(old_chart, f"z_{side}_slice")].reshape(-1, old_chart.rank)
    new_z = factor[new_complement] * d
    # Shared nonanchor rows retain the actual weighted coordinates verbatim.
    # In particular every row outside J is shared, so its L1 value and exact
    # support are unchanged, including zeros and very small nonzero entries.
    _, old_indices, new_indices = np.intersect1d(
        old_complement, new_complement, assume_unique=True, return_indices=True)
    new_z[new_indices] = old_z[old_indices]
    return new_z


def reanchor(chart, state, free_rows, *, anchor_min, d_lower, d_upper, gap):
    """Return a new chart/state/free-mask/event tuple, or no useful switch.

    An anchor replacement is considered at margin <= 1.25*anchor_min. Its
    rows must lie within the exact existing free set, and its margin must
    exceed the current margin by 256 machine epsilons times max(1, margin),
    while still satisfying anchor_min. There is no extra absolute floor or
    relative gain requirement. Independent current-anchor and QR starts,
    each with at most 16 single-row swap rounds, use only current factors.
    Only if neither start gives meaningful gain, at most r*(|J|-r) current-
    anchor one-exchange starts are tried, stopping at the first useful result.

    Independently, a side with Cayley operator norm >= .45 is recentered at
    its current rotation. If no useful anchor replacement is available its
    anchor rows and weighted complement coordinates remain exactly unchanged.

    Only a switched side gets zero Cayley coordinates and a new polar center.
    The returned state is read-only; input states, chart geometry and free
    row sets are not changed. Invalid inputs or a failed reconstruction check
    raise explicitly. None means no candidate passed the switching rule.

    This practical rule is not the initializer's stronger 8*anchor_min
    certificate, nor a guarantee of future feasibility or convergence.
    """
    if not isinstance(chart, AnchorChart):
        raise TypeError("chart must be an AnchorChart")
    _check_masks(chart, free_rows)
    anchor_min = _real(anchor_min, "anchor_min", positive=True)
    d_lower = _real(d_lower, "d_lower", positive=True)
    d_upper = _real(d_upper, "d_upper", positive=True)
    gap = _real(gap, "gap")
    domain = dict(anchor_min=anchor_min, d_lower=d_lower, d_upper=d_upper, gap=gap)
    reason = chart.domain_reason(state, **domain)
    if reason is not None:
        raise ValueError(f"cannot reanchor an infeasible current state: {reason}")
    state = np.array(state, dtype=float, copy=True)
    omega_u, omega_v, d, z_u, z_v = chart.unpack(state)
    P, _, Q = chart.reconstruct(state)
    anchors = {side: getattr(chart, f"anchors_{side}").copy() for side in ("u", "v")}
    centers = {side: getattr(chart, f"center_{side}").copy() for side in ("u", "v")}
    factors, side_events, switched, reanchored = {"u": P, "v": Q}, {}, [], []
    omegas = {"u": omega_u, "v": omega_v}
    for side in ("u", "v"):
        old = anchors[side]
        margin = _margin(factors[side], old)
        rotation_norm = float(np.linalg.norm(skew_matrix(omegas[side], chart.rank), 2))
        rotation_triggered = rotation_norm >= _ROTATION_TRIGGER
        detail = dict(old_anchors=old.tolist(), new_anchors=old.tolist(),
                      old_min_singular_value=margin, new_min_singular_value=margin,
                      old_cayley_operator_norm=rotation_norm, new_cayley_operator_norm=rotation_norm,
                      old_center_sha256=_center_sha256(centers[side]),
                      new_center_sha256=_center_sha256(centers[side]),
                      switched=False, swap_rounds=0, action="none", anchor_search=None)
        side_events[side] = detail
        trigger_slack = 64 * np.finfo(float).eps * max(1., margin)
        anchor_triggered = margin <= _TRIGGER_MULTIPLE * anchor_min + trigger_slack
        rows, new_margin, rounds = old, margin, 0
        if anchor_triggered:
            candidate, search = _candidate_search(factors[side], getattr(free_rows, f"rows_{side}"), old)
            detail["anchor_search"] = search
            if candidate is not None:
                candidate_rows, candidate_margin, candidate_rounds = candidate
                gain_slack = _NUMERICAL_GAIN_EPSILON_MULTIPLE * np.finfo(float).eps * max(1., margin)
                if (not np.array_equal(candidate_rows, old) and candidate_margin >= anchor_min
                        and candidate_margin > margin + gain_slack):
                    rows, new_margin, rounds = candidate_rows, candidate_margin, candidate_rounds
                    reanchored.append(side)
        if side not in reanchored and not rotation_triggered:
            continue
        left, _, right = np.linalg.svd(factors[side][rows], full_matrices=False)
        anchors[side], centers[side] = rows, left @ right
        switched.append(side)
        action = ("anchor_switch_and_recenter" if side in reanchored and rotation_triggered
                  else "anchor_switch" if side in reanchored else "rotation_recenter")
        detail.update(new_anchors=rows.tolist(), new_min_singular_value=new_margin,
                      new_cayley_operator_norm=0., new_center_sha256=_center_sha256(centers[side]),
                      switched=True, swap_rounds=rounds, action=action)
    if not switched:
        return None

    new_chart = AnchorChart(chart.n_u, chart.n_v, anchors["u"], anchors["v"],
                            centers["u"], centers["v"])
    new_free = FreeRows(free_rows.rows_u, free_rows.rows_v,
                       _mask(new_chart, "u", free_rows.rows_u),
                       _mask(new_chart, "v", free_rows.rows_v))
    new_state = new_chart.pack(
        np.zeros_like(omega_u) if "u" in switched else omega_u,
        np.zeros_like(omega_v) if "v" in switched else omega_v, d,
        _repack_side(chart, new_chart, state, P, d, "u") if "u" in reanchored else z_u,
        _repack_side(chart, new_chart, state, Q, d, "v") if "v" in reanchored else z_v)
    reason = new_chart.domain_reason(new_state, **domain)
    if reason is not None:
        raise FloatingPointError(f"replacement anchor chart is infeasible: {reason}")
    new_P, new_d, new_Q = new_chart.reconstruct(new_state)
    if (not np.array_equal(d, new_d)
            or not np.allclose(P, new_P, rtol=1e-10, atol=1e-12)
            or not np.allclose(Q, new_Q, rtol=1e-10, atol=1e-12)):
        raise FloatingPointError("anchor switch failed to preserve the original factors")
    for side in ("u", "v"):
        old_values = state[getattr(chart, f"z_{side}_slice")].reshape(
            getattr(free_rows, f"penalized_{side}").shape)[getattr(free_rows, f"penalized_{side}")]
        new_values = new_state[getattr(new_chart, f"z_{side}_slice")].reshape(
            getattr(new_free, f"penalized_{side}").shape)[getattr(new_free, f"penalized_{side}")]
        if not np.array_equal(old_values, new_values):
            raise FloatingPointError("anchor switch changed weighted coordinates outside the free rows")
    event = dict(kind="adaptive_anchor_switch", rule=CHART_CONTINUATION_RULE,
                 switched_sides=switched, sides=side_events,
                 free_rows={"u": free_rows.rows_u.tolist(), "v": free_rows.rows_v.tolist()},
                 anchor_min=anchor_min, trigger_multiple=_TRIGGER_MULTIPLE,
                 acceptance_multiple=None, acceptance_reference="current_anchor_margin",
                 relative_margin_multiple=1., minimum_relative_improvement=0.,
                 numerical_gain_epsilon_multiple=_NUMERICAL_GAIN_EPSILON_MULTIPLE,
                 strict_gain_comparison=True, rotation_trigger=_ROTATION_TRIGGER,
                 maximum_swap_rounds=_MAX_SWAP_ROUNDS,
                 maximum_primary_search_starts=_MAX_PRIMARY_SEARCH_STARTS,
                 maximum_fallback_search_starts={side: chart.rank * (len(getattr(free_rows, f"rows_{side}")) - chart.rank)
                                                for side in ("u", "v")},
                 maximum_search_starts={side: _MAX_PRIMARY_SEARCH_STARTS + chart.rank *
                                        (len(getattr(free_rows, f"rows_{side}")) - chart.rank) for side in ("u", "v")},
                 swap_round_limit_scope="per_start",
                 search_starts=["current_anchors", "restricted_cpqr", "current_anchor_one_exchange"],
                 fallback_trigger_rule="primary_no_positive_numerical_gain",
                 fallback_stop_at_first_acceptable=True,
                 free_rows_unchanged=True, outside_weighted_coordinates_unchanged=True,
                 factor_frobenius_differences={"u": float(np.linalg.norm(new_P - P)),
                                                "v": float(np.linalg.norm(new_Q - Q))},
                 initializer_certificate=False)
    new_state.setflags(write=False)
    return AnchorSwitch(new_chart, new_state, new_free, event)


__all__ = ["AnchorSwitch", "CHART_CONTINUATION_RULE", "reanchor"]
