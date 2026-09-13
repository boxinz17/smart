import hashlib
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sparse_smart.chart import AnchorChart
from sparse_smart_v2.anchors import CHART_CONTINUATION_RULE, reanchor
from sparse_smart_v2.solver import penalty_value
from sparse_smart_v2.support import FreeRows, threshold_state


DOMAIN = dict(anchor_min=.04, d_lower=.05, d_upper=12., gap=.01)


def _free(chart, left, right):
    return FreeRows(np.asarray(left), np.asarray(right),
        np.repeat((~np.isin(chart.complement_u, left))[:, None], chart.rank, axis=1),
        np.repeat((~np.isin(chart.complement_v, right))[:, None], chart.rank, axis=1))


def _problem(*, both_weak=False):
    P, Q = np.zeros((9, 2)), np.zeros((8, 2))
    P[[0, 4, 5], 0] = [.045, .8, np.sqrt(1 - .045**2 - .8**2)]
    P[[2, 7, 8], 1] = [.6, .7, np.sqrt(1 - .6**2 - .7**2)]
    if both_weak:
        Q[[1, 4, 0], 0] = [.045, .8, np.sqrt(1 - .045**2 - .8**2)]
        Q[[3, 6, 7], 1] = [.6, .7, np.sqrt(1 - .6**2 - .7**2)]
    else:
        Q[[1, 6], 0] = [.85, np.sqrt(1 - .85**2)]
        Q[[3, 7], 1] = [.9, np.sqrt(1 - .9**2)]
    chart = AnchorChart(9, 8, [0, 2], [1, 3], np.eye(2), np.eye(2))
    state = chart.initial_state(P, np.array([3., 1.]), Q)
    state[chart.omega_u_slice], state[chart.omega_v_slice] = .07, .09
    return chart, state, _free(chart, [0, 2, 4, 7], [1, 3, 4, 6])


def _outside(chart, state, free, side):
    mask = getattr(free, f"penalized_{side}")
    return state[getattr(chart, f"z_{side}_slice")].reshape(mask.shape)[mask]


def test_switch_preserves_factors_exact_free_sets_penalty_and_hard_support():
    chart, state, free = _problem()
    old_state = state.copy()
    P, d, Q = chart.reconstruct(state)
    old_centers = chart.center_u.copy(), chart.center_v.copy()
    old_anchors = chart.anchors_u.copy(), chart.anchors_v.copy()
    result = reanchor(chart, state, free, **DOMAIN)
    assert result is not None and result.chart is not chart
    assert result.event["switched_sides"] == ["u"]
    detail = result.event["sides"]["u"]
    assert detail["new_min_singular_value"] > detail["old_min_singular_value"] + 256 * np.finfo(float).eps
    assert detail["action"] == "anchor_switch"
    assert result.event["rule"] == CHART_CONTINUATION_RULE
    assert not result.state.flags.writeable
    assert not result.free_rows.rows_u.flags.writeable
    assert_array_equal(result.free_rows.rows_u, free.rows_u)
    assert_array_equal(result.free_rows.rows_v, free.rows_v)
    assert np.isin(result.chart.anchors_u, free.rows_u).all()
    # Re-expanding from a count would incorrectly add low-index rows here.
    assert_array_equal(result.free_rows.rows_u, [0, 2, 4, 7])
    assert_array_equal(result.state[result.chart.omega_u_slice], 0.)
    assert_array_equal(result.state[result.chart.omega_v_slice], state[chart.omega_v_slice])
    assert_array_equal(result.chart.center_v, chart.center_v)
    assert_array_equal(result.state[result.chart.z_v_slice], state[chart.z_v_slice])
    for side in ("u", "v"):
        assert_array_equal(_outside(chart, state, free, side),
                           _outside(result.chart, result.state, result.free_rows, side))
    caps = tuple(np.count_nonzero(_outside(chart, state, free, side)) for side in ("u", "v"))
    assert caps == (2, 1)
    assert_array_equal(threshold_state(result.chart, result.state, result.free_rows,
                                     (0., 0.), caps, 20.), result.state)
    assert penalty_value(chart, state, free, (.03, .07)) == penalty_value(
        result.chart, result.state, result.free_rows, (.03, .07))
    new_P, new_d, new_Q = result.chart.reconstruct(result.state)
    assert_array_equal(new_d, d)
    assert_allclose(new_P, P, atol=2e-13, rtol=0)
    assert_allclose(new_Q, Q, atol=2e-13, rtol=0)
    assert_allclose((new_P * new_d) @ new_Q.T, (P * d) @ Q.T, atol=5e-13, rtol=0)
    assert result.chart.domain_reason(result.state, **DOMAIN) is None
    assert_array_equal(state, old_state)
    for actual, saved in zip((chart.center_u, chart.center_v), old_centers):
        assert_array_equal(actual, saved)
    for actual, saved in zip((chart.anchors_u, chart.anchors_v), old_anchors):
        assert_array_equal(actual, saved)
    json.dumps(result.event, allow_nan=False)


def test_both_sides_switch_deterministically_and_stop_switching_after_margin_improves():
    chart, state, free = _problem(both_weak=True)
    first, second = (reanchor(chart, state, free, **DOMAIN) for _ in range(2))
    assert first.event["switched_sides"] == ["u", "v"]
    assert_array_equal(first.state, second.state)
    assert first.event == second.event
    assert_array_equal(first.chart.anchors_u, second.chart.anchors_u)
    assert_array_equal(first.chart.anchors_v, second.chart.anchors_v)
    assert reanchor(first.chart, first.state, first.free_rows, **DOMAIN) is None


@pytest.mark.parametrize("left", [[0, 2], [0, 1, 2, 3]])
def test_no_usable_replacement_never_uses_penalized_rows(left):
    chart, state, _ = _problem()
    # Good rows 4 and 7 exist physically but are outside the allowed J.
    free = _free(chart, left, [1, 3])
    assert reanchor(chart, state, free, **DOMAIN) is None


def test_no_switch_away_from_trigger_and_invalid_inputs_are_explicit():
    chart, state, free = _problem()
    assert reanchor(chart, state, free, **dict(DOMAIN, anchor_min=.02)) is None
    with pytest.raises(ValueError, match="infeasible current state"):
        reanchor(chart, state, free, **dict(DOMAIN, anchor_min=.06))
    with pytest.raises(ValueError):
        reanchor(chart, state, free, **dict(DOMAIN, anchor_min=True))
    invalid = FreeRows(free.rows_u, free.rows_v, ~free.penalized_u, free.penalized_v)
    with pytest.raises(ValueError, match="does not match"):
        reanchor(chart, state, invalid, **DOMAIN)


def _small_anchor_improvement(new_margin):
    P = np.zeros((3, 1))
    P[:, 0] = [.044, new_margin, np.sqrt(1 - .044**2 - new_margin**2)]
    chart = AnchorChart(3, 2, [0], [0], np.eye(1), np.eye(1))
    state = chart.initial_state(P, np.array([2.]), np.array([[1.], [0.]]))
    return chart, state, _free(chart, [0, 1], [0])


@pytest.mark.parametrize("new_margin", [.047, .045, .044 + 1e-8])
def test_positive_numerical_gain_accepts_anchor_without_arbitrary_relative_buffer(new_margin):
    chart, state, free = _small_anchor_improvement(new_margin)
    result = reanchor(chart, state, free, **DOMAIN)
    assert result is not None
    detail = result.event["sides"]["u"]
    assert_array_equal(result.chart.anchors_u, [1])
    assert detail["old_min_singular_value"] == pytest.approx(.044, abs=1e-13)
    assert detail["new_min_singular_value"] == pytest.approx(new_margin, abs=1e-13)
    assert DOMAIN["anchor_min"] < detail["new_min_singular_value"] < 2 * DOMAIN["anchor_min"]
    assert result.event["acceptance_multiple"] is None
    assert result.event["relative_margin_multiple"] == 1.
    assert result.event["minimum_relative_improvement"] == 0.
    assert result.event["numerical_gain_epsilon_multiple"] == 256
    assert result.event["strict_gain_comparison"] is True
    assert_array_equal(_outside(chart, state, free, "u"),
                       _outside(result.chart, result.state, result.free_rows, "u"))


@pytest.mark.parametrize("new_margin", [.044, .044 + 1e-14])
def test_roundoff_only_anchor_gains_are_rejected(new_margin):
    chart, state, free = _small_anchor_improvement(new_margin)
    assert reanchor(chart, state, free, **DOMAIN) is None


def test_independent_current_anchor_start_recovers_basin_discarded_by_better_qr_seed():
    from sparse_smart.anchors import _pivot_columns
    from sparse_smart_v2.anchors import _candidate, _improve_candidate
    # Fixed geometry fixture: CPQR starts better, but its one-swap local optimum
    # is worse than the result reachable from the original anchor subset.
    factor = np.array([
        [-.08776769197285073, -.06188088108845473, -.8155635396769753],
        [.23380687070379838, -.42600991592083787, -.1397104297646383],
        [.35340365803961016, .04247348463285573, -.08834366413029097],
        [.05701967607245774, -.08521516972923583, .22997150256760582],
        [-.08472934068421832, -.24300669454787502, .3763854901388206],
        [.7001284644813401, -.45973606444385895, .008187482306943734]])
    free, old = np.arange(6), np.array([1, 2, 5])
    qr = np.sort(free[_pivot_columns(factor[free].T, free)])
    margin = lambda rows: np.linalg.svd(factor[rows], compute_uv=False)[-1]
    assert margin(qr) > margin(old)
    qr_only = _improve_candidate(factor, free, qr)
    proposed = _candidate(factor, free, old)
    assert qr_only[1] == pytest.approx(.26815444951111883)
    assert proposed[1] == pytest.approx(.3107721766821967)
    assert proposed[1] > qr_only[1]
    assert 0 <= proposed[2] <= 16
    repeated = _candidate(factor, free, old)
    assert_array_equal(proposed[0], repeated[0])
    assert proposed[1:] == repeated[1:]


def test_qr_failure_still_allows_bounded_improvement_from_current_anchors(monkeypatch):
    import sparse_smart_v2.anchors as module
    from sparse_smart.anchors import AnchorFailure
    def fail(*args):
        raise AnchorFailure("synthetic pivot failure")
    monkeypatch.setattr(module, "_pivot_columns", fail)
    chart, state, free = _small_anchor_improvement(.045)
    result = reanchor(chart, state, free, **DOMAIN)
    assert result is not None
    assert_array_equal(result.chart.anchors_u, [1])
    assert result.event["maximum_primary_search_starts"] == 2
    assert result.event["maximum_fallback_search_starts"] == {"u": 1, "v": 0}
    assert result.event["maximum_search_starts"] == {"u": 3, "v": 2}
    assert result.event["swap_round_limit_scope"] == "per_start"


def test_rotation_only_recenters_with_no_alternative_rows_and_preserves_penalty_support():
    chart, state, _ = _problem()
    free = _free(chart, [0, 2], [1, 3])  # |J|=r on both sides.
    state[chart.omega_u_slice] = np.sqrt(2) * .48
    old_state = state.copy()
    P, d, Q = chart.reconstruct(state)
    result = reanchor(chart, state, free, **DOMAIN)
    assert result is not None
    assert result.event["switched_sides"] == ["u"]
    detail = result.event["sides"]["u"]
    assert detail["action"] == "rotation_recenter"
    assert detail["old_cayley_operator_norm"] == pytest.approx(.48)
    assert detail["new_cayley_operator_norm"] == 0
    digest = lambda center: hashlib.sha256(np.asarray(center, dtype="<f8").tobytes(order="C")).hexdigest()
    assert detail["old_center_sha256"] == digest(chart.center_u)
    assert detail["new_center_sha256"] == digest(result.chart.center_u)
    assert detail["old_center_sha256"] != detail["new_center_sha256"]
    assert result.event["sides"]["v"]["action"] == "none"
    assert result.event["sides"]["v"]["old_center_sha256"] == result.event["sides"]["v"]["new_center_sha256"]
    for side in ("u", "v"):
        assert_array_equal(getattr(result.chart, f"anchors_{side}"), getattr(chart, f"anchors_{side}"))
        assert_array_equal(result.state[getattr(result.chart, f"z_{side}_slice")],
                           state[getattr(chart, f"z_{side}_slice")])
    assert_array_equal(result.state[result.chart.omega_u_slice], 0.)
    assert_array_equal(result.state[result.chart.omega_v_slice], state[chart.omega_v_slice])
    assert_array_equal(result.chart.center_v, chart.center_v)
    caps = tuple(np.count_nonzero(_outside(chart, state, free, side)) for side in ("u", "v"))
    assert_array_equal(threshold_state(result.chart, result.state, result.free_rows,
                                     (0., 0.), caps, 20.), result.state)
    assert penalty_value(chart, state, free, (.2, .3)) == penalty_value(
        result.chart, result.state, result.free_rows, (.2, .3))
    new_P, new_d, new_Q = result.chart.reconstruct(result.state)
    assert_allclose(new_P, P, atol=2e-13, rtol=0)
    assert_array_equal(new_d, d)
    assert_array_equal(new_Q, Q)
    assert_allclose((new_P * d) @ new_Q.T, (P * d) @ Q.T, atol=5e-13, rtol=0)
    assert_array_equal(state, old_state)
    assert reanchor(result.chart, result.state, result.free_rows, **DOMAIN) is None


def test_combined_anchor_and_rotation_trigger_prefers_better_rows_and_recenters():
    chart, state, free = _problem()
    state[chart.omega_u_slice] = np.sqrt(2) * .48
    result = reanchor(chart, state, free, **DOMAIN)
    assert result.event["sides"]["u"]["action"] == "anchor_switch_and_recenter"
    assert not np.array_equal(result.chart.anchors_u, chart.anchors_u)
    assert result.event["sides"]["u"]["new_cayley_operator_norm"] == 0
    assert_array_equal(result.free_rows.rows_u, free.rows_u)
    for side in ("u", "v"):
        assert_array_equal(_outside(chart, state, free, side),
                           _outside(result.chart, result.state, result.free_rows, side))


def test_rotation_only_does_not_search_alternative_anchor_rows_when_anchor_is_safe(monkeypatch):
    import sparse_smart_v2.anchors as module
    chart, state, free = _problem()
    first = reanchor(chart, state, free, **DOMAIN)
    chart, state, free = first.chart, first.state.copy(), first.free_rows
    state[chart.omega_v_slice] = np.sqrt(2) * .46
    def unexpected_search(*args):
        raise AssertionError("safe anchors must not be reselected for a rotation-only transition")
    monkeypatch.setattr(module, "_candidate_search", unexpected_search)
    result = reanchor(chart, state, free, **DOMAIN)
    assert result.event["switched_sides"] == ["v"]
    assert result.event["sides"]["v"]["action"] == "rotation_recenter"
    assert_array_equal(result.chart.anchors_v, chart.anchors_v)


def _fallback_geometry():
    # A fixed restricted factor block with two distinct anchor-search basins.
    # It contains no observations, responses, source matrices, or validation
    # values. Complete it to orthonormal columns and add an exactly zero row.
    block = np.array([
        [.12204004490296466, -.0971675272022093, .07400010234922684, -.048455068828278096, .1831258206832474],
        [.2148713112742655, .07566164298395166, -.016196610042517885, .1270270948901088, .0551269786092837],
        [-.08862286861616132, -.0051299601297364675, -.09056589424063388, .01634190514063991, -.12149425371373011],
        [-.7468614466999112, .1711570170169082, -.06102088493189675, -.09064929402998609, -.002365548247998543],
        [-.2059436886257928, -.5423990785505505, .045797924354352565, -.06132872221763106, .06120456408202265],
        [-.06773541682357571, -.2853100621613879, -.002278347344801449, .047958675682178466, .0531466487477508],
        [.09053824389570696, .2374876585752632, .05266480317682289, -.045499149008501515, -.09825378303815885],
        [-.03510250368772771, .004868149296874686, -.026158027120699666, .0426553342886883, .14374887657903984],
        [.03818476111874486, -.1395204706316253, .18874702156430645, -.20876952914757557, .024150260598597366],
        [.1424352155570814, -.19053493274982453, -.039876999759127806, .03531644690934479, -.19606505779435324]])
    block *= 1.0001  # Keep the accepted anchor strictly inside the domain.
    values, vectors = np.linalg.eigh(np.eye(5) - block.T @ block)
    assert np.min(values) > 0
    P = np.vstack((block, (vectors * np.sqrt(values)) @ vectors.T, np.zeros((1, 5))))
    Q = np.vstack((np.eye(5), np.zeros((1, 5))))
    old = np.array([0, 1, 2, 5, 8])
    left, _, right = np.linalg.svd(P[old], full_matrices=False)
    chart = AnchorChart(len(P), len(Q), old, np.arange(5), left @ right, np.eye(5))
    state = chart.initial_state(P, np.array([5., 4., 3., 2., 1.]), Q)
    return chart, state, _free(chart, np.arange(10), np.arange(5))


def test_neighbor_fallback_escapes_two_start_local_optimum_preserving_exact_penalty_support():
    from sparse_smart_v2.anchors import _candidate_search
    chart, state, free = _fallback_geometry()
    P, d, Q = chart.reconstruct(state)
    candidate, work = _candidate_search(P, free.rows_u, chart.anchors_u)
    assert work["primary_best_margin"] == pytest.approx(.040004, abs=1e-12)
    assert work["fallback_triggered"] and work["fallback_accepted"]
    assert work["fallback_starts"] == 1
    assert work["maximum_fallback_starts"] == 25
    assert work["maximum_search_starts"] == 27
    assert work["winning_start"] == "current_anchor_one_exchange"
    assert work["winning_seed"] == [1, 2, 3, 5, 8]
    assert work["winning_swap_rounds"] == 1
    assert_array_equal(candidate[0], [1, 2, 3, 5, 7])
    assert candidate[1] == pytest.approx(.04054423919072826 * 1.0001)
    result = reanchor(chart, state, free, **DOMAIN)
    assert result is not None
    assert result.event["sides"]["u"]["anchor_search"] == work
    assert result.event["sides"]["v"]["anchor_search"] is None
    assert result.event["maximum_fallback_search_starts"] == {"u": 25, "v": 0}
    assert result.event["fallback_stop_at_first_acceptable"] is True
    for side in ("u", "v"):
        old_values = _outside(chart, state, free, side)
        new_values = _outside(result.chart, result.state, result.free_rows, side)
        assert_array_equal(old_values, new_values)
        assert_array_equal(old_values == 0, new_values == 0)
        assert np.any(old_values == 0)
    caps = tuple(np.count_nonzero(_outside(chart, state, free, side)) for side in ("u", "v"))
    assert_array_equal(threshold_state(result.chart, result.state, result.free_rows,
                                      (0., 0.), caps, 20.), result.state)
    assert penalty_value(chart, state, free, (.0025, .01)) == penalty_value(
        result.chart, result.state, result.free_rows, (.0025, .01))
    for new_factor, old_factor in zip(result.chart.reconstruct(result.state), (P, d, Q)):
        assert_allclose(new_factor, old_factor, atol=1e-12, rtol=0)
    assert result.chart.domain_reason(result.state, **DOMAIN) is None
    repeat = reanchor(chart, state, free, **DOMAIN)
    assert_array_equal(repeat.state, result.state)
    assert repeat.event == result.event


def test_successful_primary_search_is_unchanged_and_skips_fallback():
    from sparse_smart_v2.anchors import _candidate_search, _improve_candidate
    from sparse_smart.anchors import _pivot_columns
    chart, state, free = _problem()
    factor = chart.reconstruct(state)[0]
    old = chart.anchors_u
    qr = np.sort(free.rows_u[_pivot_columns(factor[free.rows_u].T, free.rows_u)])
    primary = [_improve_candidate(factor, free.rows_u, start) for start in (old, qr)]
    expected = primary[0] if primary[0][1] >= primary[1][1] else primary[1]
    candidate, work = _candidate_search(factor, free.rows_u, old)
    assert not work["fallback_triggered"] and work["fallback_starts"] == 0
    assert_array_equal(candidate[0], expected[0])
    assert candidate[1:] == expected[1:]


def test_unsuccessful_fallback_has_polynomial_start_bound_and_does_not_reclassify_failure():
    from sparse_smart_v2.anchors import _candidate_search
    chart, state, free = _small_anchor_improvement(.043)
    P = chart.reconstruct(state)[0]
    candidate, work = _candidate_search(P, free.rows_u, chart.anchors_u)
    assert work["fallback_triggered"] and not work["fallback_accepted"]
    assert work["fallback_starts"] <= chart.rank * (len(free.rows_u) - chart.rank)
    assert work["primary_starts"] + work["fallback_starts"] <= work["maximum_search_starts"]
    assert work["winning_swap_rounds"] <= 16
    assert reanchor(chart, state, free, **DOMAIN) is None
