from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sparse_smart.chart import AnchorChart
from sparse_smart_v2.calibration import Margins, PracticalCalibration
from sparse_smart_v2.estimator import SparseSMARTv2
from sparse_smart_v2.support import FreeRows


def problem():
    X = 2 * np.eye(4)
    source = np.diag([5., 4., 3., 2.])
    P, Q = np.array([0., .8, .6, 0.]), np.array([0., .6, .8, 0.])
    return X, X @ (2 * np.outer(P, Q)), source


def model(**options):
    defaults = dict(rank=1, source_rank=3, free_directions=(3, 3),
        rrr_shortcut=False,
        margins=Margins(.1, 8., .1), calibration=PracticalCalibration(.08, 0., 20., (1, 1)),
        iterations=5, adaptive_anchors=True, validation_interval=1, checkpoint_iterations=(2, 3))
    defaults.update(options)
    return SparseSMARTv2(**defaults)


def controlled_switch(chart, state, free_rows, **kwargs):
    """Exact rank-one coordinate change, independent of trigger heuristics."""
    P, d, Q = chart.reconstruct(state)
    u = 2 if chart.anchors_u[0] == 1 else 1
    v = 1 if chart.anchors_v[0] == 2 else 2
    fresh = AnchorChart(chart.n_u, chart.n_v, [u], [v],
                        np.sign(P[[u]]), np.sign(Q[[v]]))
    replacement = fresh.initial_state(P, d, Q)
    masks = [np.repeat((~np.isin(getattr(fresh, f"complement_{side}"),
                               getattr(free_rows, f"rows_{side}")))[:, None], fresh.rank, axis=1)
             for side in ("u", "v")]
    new_free = FreeRows(free_rows.rows_u, free_rows.rows_v, masks[0], masks[1])
    event = dict(old_anchors={"u": chart.anchors_u.tolist(), "v": chart.anchors_v.tolist()},
                 new_anchors={"u": fresh.anchors_u.tolist(), "v": fresh.anchors_v.tolist()})
    return SimpleNamespace(chart=fresh, state=replacement, free_rows=new_free, event=event)


def interrupt_after(monkeypatch, *, failure="numerical_stagnation", reason="left anchor singular value is below anchor_min",
                    repeat=False, segment_length=2):
    import sparse_smart_v2.estimator as module
    original = module.refine
    calls = []

    def segmented(*args, **kwargs):
        calls.append(kwargs["iterations"])
        fail = len(calls) == 1 or repeat
        if fail:
            kwargs["iterations"] = min(kwargs["iterations"], segment_length)
        result = original(*args, **kwargs)
        if fail and result.termination_reason != "validation_stop":
            return replace(result, status=failure, message="Controlled domain rejection",
                           termination_reason=failure, last_rejection=reason)
        return result

    monkeypatch.setattr(module, "refine", segmented)
    monkeypatch.setattr(module, "reanchor", controlled_switch)
    return calls


def test_switch_preserves_global_budget_validation_incumbent_and_checkpoint_charts(monkeypatch):
    X, Y, source = problem()
    warm = model(iterations=0, checkpoint_iterations=()).fit(X, Y, source=source)
    calls = interrupt_after(monkeypatch)
    fitted = model().fit(X, Y, source=source, validation_data=(X, warm.predict(X)))
    assert fitted.success_, fitted.message_
    assert calls == [5, 3]
    assert fitted.n_iter_ == 5
    assert [r.iteration for r in fitted.history_] == list(range(6))
    assert fitted.history_chart_epochs_ == [0, 0, 0, 1, 1, 1]
    assert fitted.message_ == "Completed 5 refinement updates."
    assert [r["iteration"] for r in fitted.validation_history_] == list(range(6))
    assert [r["chart_epoch"] for r in fitted.validation_history_] == [0, 0, 0, 1, 1, 1]
    assert fitted.selected_iteration_ == 0
    assert fitted.selected_chart_ is fitted.initial_chart_
    assert fitted.chart_ is not fitted.initial_chart_
    np.testing.assert_allclose(fitted.coefficient_, warm.coefficient_, atol=1e-13)
    assert fitted.numerical_work_["accepted_updates"] == 5
    assert fitted.numerical_work_["anchor_switches"] == 1
    assert fitted.anchor_switches_[0]["iteration"] == 2
    assert fitted.metadata_["adaptive_anchors"] is True
    np.testing.assert_array_equal(fitted.free_rows_.rows_u, [0, 1, 2])
    np.testing.assert_array_equal(fitted.selected_free_rows_.rows_u, [0, 1, 2])
    before, after = fitted.checkpoint_model(2), fitted.checkpoint_model(3)
    assert len(before.anchor_switches_) == 0 and len(after.anchor_switches_) == 1
    assert before.chart_.anchors_u.tolist() != after.chart_.anchors_u.tolist()
    for prefix in (before, after):
        assert prefix.success_ and prefix.selected_iteration_ == 0
        np.testing.assert_allclose(prefix.predict(X), warm.predict(X), atol=1e-13)
        cp = fitted.checkpoints_[prefix.n_iter_]
        P, d, Q = cp.chart.reconstruct(cp.state)
        expected = ((fitted.source_.left @ P) * d) @ (fitted.source_.right @ Q).T
        np.testing.assert_allclose(prefix.last_coefficient_, expected, atol=1e-13)
        assert prefix.numerical_work_["anchor_switches"] == len(prefix.anchor_switches_)
    after.chart_.center_u[:] = 0
    np.testing.assert_allclose(fitted.chart_.center_u, [[1.]], atol=1e-13)


def test_validation_patience_is_not_restarted_after_switch(monkeypatch):
    X, Y, source = problem()
    warm = model(iterations=0, checkpoint_iterations=()).fit(X, Y, source=source)
    interrupt_after(monkeypatch)
    fitted = model(validation_patience=3, validation_min_iterations=0).fit(
        X, Y, source=source, validation_data=(X, warm.predict(X)))
    assert fitted.success_ and fitted.termination_reason_ == "validation_stop"
    assert fitted.n_iter_ == 3
    assert fitted.validation_stopping_["stop_iteration"] == 3
    assert [row["iteration"] for row in fitted.validation_history_] == [0, 1, 2, 3]
    assert fitted.checkpoint_model(3).termination_reason_ == "validation_stop"


def test_stationarity_at_new_segment_initial_state_uses_new_chart_diagnostics(monkeypatch):
    import sparse_smart_v2.estimator as module
    X, Y, source = problem()
    original = module.refine
    calls = []

    def segmented(*args, **kwargs):
        calls.append(1)
        kwargs["iterations"] = 2 if len(calls) == 1 else 0
        result = original(*args, **kwargs)
        if len(calls) == 1:
            return replace(result, status="numerical_stagnation", termination_reason="numerical_stagnation",
                           last_rejection="left anchor singular value is below anchor_min")
        # Controlled fixed point in the new chart: no additional accepted step.
        endpoint = replace(result.history[-1], projected_gradient_norm=0., mapping_domain_reason=None,
                           mapping_displacement=0.)
        return replace(result, status="converged", termination_reason="stationarity", history=[endpoint],
                       projected_gradient_norm=0., mapping_displacement=0.)

    monkeypatch.setattr(module, "refine", segmented)
    monkeypatch.setattr(module, "reanchor", controlled_switch)
    fitted = model().fit(X, Y, source=source, validation_data=(X, Y))
    assert fitted.success_ and fitted.optimization_converged_
    assert fitted.n_iter_ == 2 and len(fitted.history_) == 3
    assert fitted.history_chart_epochs_ == [0, 0, 0]
    assert fitted.history_[-1].projected_gradient_norm > 0
    assert fitted.result_.projected_gradient_norm == fitted.terminal_record_.projected_gradient_norm == 0
    assert fitted.checkpoints_[2].endpoint_record.projected_gradient_norm == 0
    prefix = fitted.checkpoint_model(2)
    assert prefix.optimization_converged_ and prefix.result_.projected_gradient_norm == 0
    assert prefix.terminal_record_.iteration == 2 and prefix.numerical_work_["anchor_switches"] == 1
    assert [row["iteration"] for row in prefix.validation_history_] == [0, 1, 2]


@pytest.mark.parametrize("options,reason", [
    ({"adaptive_anchors": False}, "left anchor singular value is below anchor_min"),
    ({"max_anchor_switches": 0}, "left anchor singular value is below anchor_min"),
    ({}, "adjacent singular-value gap is below gap"),
    ({}, "insufficient objective decrease"),
])
def test_disabled_exhausted_and_nonanchor_failures_are_not_recovered(monkeypatch, options, reason):
    X, Y, source = problem()
    calls = interrupt_after(monkeypatch, reason=reason)
    fitted = model(**options).fit(X, Y, source=source)
    assert not fitted.success_ and fitted.status_ == "numerical_stagnation"
    assert fitted.n_iter_ == 2 and calls == [5]
    assert fitted.anchor_switches_ == []


@pytest.mark.parametrize("failure", ["numerical_stagnation", "line_search_failed"])
@pytest.mark.parametrize("reason", [
    "left anchor singular value is below anchor_min",
    "right anchor singular value is below anchor_min",
    "left Cayley coordinate has operator norm above 1/2",
    "right Cayley coordinate has operator norm above 1/2",
])
def test_only_declared_chart_boundary_rejections_enable_continuation(monkeypatch, failure, reason):
    X, Y, source = problem()
    calls = interrupt_after(monkeypatch, failure=failure, reason=reason)
    fitted = model().fit(X, Y, source=source)
    assert fitted.success_ and calls == [5, 3]
    assert fitted.n_iter_ == 5 and len(fitted.anchor_switches_) == 1


@pytest.mark.parametrize("failure,reason", [
    ("numerical_failure", "left Cayley coordinate has operator norm above 1/2"),
    ("invalid_initial_state", "left anchor singular value is below anchor_min"),
    ("numerical_stagnation", "left complement Gram is not positive definite"),
    ("line_search_failed", "left Cayley coordinate has operator norm above 1/2 (unrecognized)"),
])
def test_other_failure_classes_and_nonexact_messages_are_not_recovered(monkeypatch, failure, reason):
    X, Y, source = problem()
    calls = interrupt_after(monkeypatch, failure=failure, reason=reason)
    fitted = model().fit(X, Y, source=source)
    assert not fitted.success_ and fitted.status_ == failure and calls == [5]
    assert not fitted.anchor_switches_


def test_actual_rotation_only_recenter_retains_chart_aware_checkpoint_predictions(monkeypatch):
    import sparse_smart_v2.estimator as module
    from sparse_smart_v2.anchors import CHART_CONTINUATION_RULE
    original = module.refine
    calls = []

    def boundary_segment(chart, state, design, response, **kwargs):
        calls.append(kwargs["iterations"])
        if len(calls) > 1:
            return original(chart, state, design, response, **kwargs)
        callback = kwargs["iterate_callback"]
        # Put a controlled accepted endpoint near the rotation boundary. The
        # continuation uses the actual geometry helper, not a mocked reanchor.
        first = original(chart, state, design, response, **dict(kwargs, iterations=0))
        rotated = state.copy()
        rotated[chart.omega_u_slice] = .48 * np.sqrt(2)
        endpoint = original(chart, rotated, design, response,
                            **dict(kwargs, iterations=0, iterate_callback=None))
        record = replace(endpoint.history[0], iteration=1)
        callback(1, rotated, record)
        return replace(endpoint, status="numerical_stagnation", termination_reason="numerical_stagnation",
                       n_iter=1, history=[first.history[0], record],
                       last_rejection="left Cayley coordinate has operator norm above 1/2")

    X, source = 2 * np.eye(4), np.diag([5., 4., 3., 2.])
    Y = X @ np.diag([3., 1., 0., 0.])
    opts = dict(rank=2, source_rank=3, free_directions=(3, 3), iterations=2,
                checkpoint_iterations=(1,), calibration=PracticalCalibration(.08, 0., 20., (2, 2)))
    warm = model(**dict(opts, iterations=0, checkpoint_iterations=())).fit(X, Y, source=source)
    monkeypatch.setattr(module, "refine", boundary_segment)
    fitted = model(**opts).fit(X, Y, source=source, validation_data=(X, warm.predict(X)))
    assert fitted.success_, fitted.message_
    assert calls == [2, 1] and fitted.n_iter_ == 2
    assert fitted.history_chart_epochs_ == [0, 0, 1]
    assert [row["iteration"] for row in fitted.validation_history_] == [0, 1, 2]
    assert fitted.metadata_["chart_continuation_rule"] == CHART_CONTINUATION_RULE
    assert len(fitted.anchor_switches_) == 1
    event = fitted.anchor_switches_[0]
    assert event["rule"] == CHART_CONTINUATION_RULE
    assert event["sides"]["u"]["action"] == "rotation_recenter"
    assert event["sides"]["v"]["action"] == "none"
    assert fitted.numerical_work_["chart_transitions"] == fitted.numerical_work_["anchor_switches"] == 1
    assert fitted.numerical_work_["anchor_row_switches"] == 0
    assert fitted.numerical_work_["rotation_recenters"] == 1
    before, after = fitted.checkpoint_model(1), fitted.checkpoint_model(2)
    for side in ("u", "v"):
        np.testing.assert_array_equal(getattr(before.chart_, f"anchors_{side}"),
                                      getattr(after.chart_, f"anchors_{side}"))
        np.testing.assert_array_equal(getattr(before.free_rows_, f"rows_{side}"),
                                      getattr(after.free_rows_, f"rows_{side}"))
    assert np.linalg.norm(before.chart_.center_u - after.chart_.center_u) > .1
    assert len(before.anchor_switches_) == 0 and len(after.anchor_switches_) == 1
    for prefix in (before, after):
        assert prefix.selected_iteration_ == 0
        np.testing.assert_allclose(prefix.predict(X), warm.predict(X), atol=2e-12)
        cp = fitted.checkpoints_[prefix.n_iter_]
        P, d, Q = cp.chart.reconstruct(cp.state)
        physical = ((fitted.source_.left @ P) * d) @ (fitted.source_.right @ Q).T
        np.testing.assert_allclose(prefix.last_coefficient_, physical, atol=2e-12)
    assert after.numerical_work_["rotation_recenters"] == 1


def test_switch_limit_is_bounded_even_when_segments_accept_no_updates(monkeypatch):
    X, Y, source = problem()
    calls = interrupt_after(monkeypatch, repeat=True, segment_length=0)
    fitted = model(max_anchor_switches=2).fit(X, Y, source=source)
    assert not fitted.success_ and fitted.status_ == "numerical_stagnation"
    assert calls == [5, 5, 5]
    assert fitted.n_iter_ == 0 and len(fitted.history_) == 1
    assert len(fitted.anchor_switches_) == 2
    assert fitted.numerical_work_["accepted_updates"] == 0
    assert fitted.checkpoint_model(0).anchor_switches_ == []


def test_unavailable_reanchor_retains_original_failure(monkeypatch):
    import sparse_smart_v2.estimator as module
    X, Y, source = problem()
    calls = interrupt_after(monkeypatch, failure="line_search_failed")
    monkeypatch.setattr(module, "reanchor", lambda *args, **kwargs: None)
    fitted = model().fit(X, Y, source=source)
    assert not fitted.success_ and fitted.status_ == "line_search_failed"
    assert fitted.anchor_switches_ == [] and calls == [5]


@pytest.mark.parametrize("options", [{"adaptive_anchors": 1}, {"adaptive_anchors": "yes"},
    {"max_anchor_switches": True}, {"max_anchor_switches": -1}, {"max_anchor_switches": 1.5}])
def test_invalid_adaptive_configuration_raises(options):
    X, Y, source = problem()
    with pytest.raises(ValueError):
        model(**options).fit(X, Y, source=source)


def test_repeat_fit_clears_anchor_switch_history(monkeypatch):
    X, Y, source = problem()
    interrupt_after(monkeypatch)
    fitted = model().fit(X, Y, source=source)
    assert len(fitted.anchor_switches_) == 1
    fitted.iterations, fitted.checkpoint_iterations = 0, ()
    fitted.fit(X, Y, source=source)
    assert fitted.success_ and not fitted.anchor_switches_
    assert fitted.chart_ is fitted.selected_chart_ is fitted.initial_chart_
