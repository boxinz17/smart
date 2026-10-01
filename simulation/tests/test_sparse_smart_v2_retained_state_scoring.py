"""Own-chart, completed-archive inventory fixtures; no estimator fitting."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
import sparse_smart_v2_retained_state_scoring as scoring
from sparse_smart.chart import AnchorChart


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


# Exercise the actual original writer, not a guessed approximation of its NPZ.
FROZEN = SIM / "result/sparse-smart-v2-paper100-all-20260913T073736Z/source/simulation/run_sparse_smart_v2_pilot.py"
WRITER_PATH = FROZEN if FROZEN.is_file() else SIM / "run_sparse_smart_v2_pilot.py"
writer = module(WRITER_PATH, "_retained_state_original_writer")


def seal(folder, result, arrays, history):
    writer._npz(folder / "states.npz", arrays)
    writer._history(folder / "history.json.gz", history)
    result["files"] = {name: writer._sha(folder / name) for name in ("states.npz", "history.json.gz")}
    writer._json(folder / "result.json", result)


@pytest.fixture
def retained(tmp_path):
    case = dict(p=3, q=3, rank=1, free_directions=[2, 2], support_limits=[1, 1])
    configuration = dict(iterations=10, checkpoint_interval=5,
        margins=dict(anchor_min=.04, d_lower=.001, d_upper=12., gap=.0001))
    source = dict(left=np.eye(3), right=np.eye(3))
    charts = [AnchorChart(3, 3, [a], [a], np.eye(1), np.eye(1)) for a in (0, 1)]
    states, coefficients = {}, {}
    for t in range(8):
        P = np.array([[np.cos(.6 + .015*t)], [np.sin(.6 + .015*t)], [0.]])
        Q = np.array([[.6], [.8], [0.]])
        d = np.array([1. + .08*t])
        states[t] = charts[int(t >= 5)].initial_state(P, d, Q)
        coefficients[t] = (P*d) @ Q.T
    X = np.vstack([np.eye(3), np.zeros((1, 3))])
    Y = X @ coefficients[3]
    Y[-1] = [.1, .2, .1]
    data = dict(X=X, Y=Y, C_star=coefficients[3], X_validation=X,
                Y_validation=X @ coefficients[3])
    validation, incumbent = [], None
    for t in range(8):
        loss = float(np.mean((data["Y_validation"] - X @ coefficients[t]) ** 2))
        delta = None if incumbent is None else loss - validation[incumbent]["loss"]
        validation.append(dict(iteration=t, loss=loss, chart_epoch=int(t >= 5),
                               incumbent_iteration=incumbent, selection_loss_difference=delta))
        if incumbent is None or delta < 0:
            incumbent = t
    center_sha = hashlib.sha256(np.asarray(np.eye(1), dtype="<f8").tobytes()).hexdigest()
    events = [dict(iteration=4, sides={side: dict(old_anchors=[0], new_anchors=[1],
              old_center_sha256=center_sha, new_center_sha256=center_sha) for side in ("u", "v")})]
    checkpoints = {}
    for t in (0, 5, 7):
        selected = min(t, 3)
        checkpoints[t] = SimpleNamespace(iteration=t, selected_iteration=selected, state=states[t],
            selected_state=states[selected], chart=charts[int(t >= 5)], selected_chart=charts[0],
            chart_epoch=int(t >= 5), selected_chart_epoch=0, history_length=t+1,
            validation_history_length=t+1, anchor_switches=events if t >= 5 else [],
            best_validation_loss=validation[selected]["loss"])
    model = SimpleNamespace(method_="sparse_smart_v2", last_state_=states[7], state_=states[3],
        chart_=charts[1], selected_chart_=charts[0], checkpoints_=checkpoints,
        free_rows_=SimpleNamespace(rows_u=np.array([0, 1]), rows_v=np.array([0, 1])))
    arrays = writer._state_arrays(model)
    history = dict(history=[dict(iteration=t) for t in range(8)],
        history_chart_epochs=[int(t >= 5) for t in range(8)], validation_history=validation,
        checkpoints={str(t): writer._checkpoint_metadata(cp) for t, cp in checkpoints.items()})
    result = dict(task=dict(task_id=17, penalty_u=.01, penalty_v=.01, init_penalty=.03),
        status="complete", execution_success=True, success=True, n_iter=7, selected_iteration=3,
        fit_method="sparse_smart_v2", termination_reason="validation_stop", optimization_converged=False,
        anchor_switches=events, selected_metrics=scoring.audit._metrics(coefficients[3], data),
        terminal_metrics=scoring.audit._metrics(coefficients[7], data))
    folder = tmp_path / "00017"
    folder.mkdir()
    seal(folder, result, arrays, history)
    return SimpleNamespace(folder=folder, result=result, arrays=arrays, history=history,
                           data=data, source=source, case=case, config=configuration)


def score(fixture):
    return scoring.score_retained_states(fixture.folder, fixture.result, fixture.data,
        fixture.source, fixture.case, fixture.config)


def test_original_writer_inventory_recovers_incumbent_actual_iteration_and_exact_endpoints(retained):
    result = score(retained)
    assert result["endpoints"] == scoring.old.score_endpoints(retained.folder, retained.result,
        retained.data, retained.source, retained.case, retained.config)
    inv = result["inventory"]
    assert inv["current_coverage_complete"] and inv["all_retained_coverage_complete"]
    assert inv["expected_checkpoint_iterations"] == inv["available_checkpoint_iterations"] == [0, 5, 7]
    assert inv["scheduled_checkpoint_iterations"] == [0, 5, 10]
    assert inv["unreached_checkpoint_iterations"] == [10]
    assert inv["unsaved_validation_iterations"] == [1, 2, 4, 6]
    assert inv["raw_saved_state_count"] == 8
    assert {r["iteration"] for r in result["current_states"]} == {0, 5, 7}
    assert {r["iteration"] for r in result["all_states"]} == {0, 3, 5, 7}
    incumbent = min(result["all_states"], key=scoring.state_key)
    assert incumbent["iteration"] == 3
    assert incumbent["state_pointer"]["state_array"] == "checkpoint_5_selected_state"
    assert incumbent["state_pointer"]["chart_prefix"] == "checkpoint_5_selected"
    assert incumbent["origins"] == [
        dict(kind="checkpoint_incumbent", saved_at_iteration=5, actual_iteration=3),
        dict(kind="checkpoint_incumbent", saved_at_iteration=7, actual_iteration=3),
        dict(kind="selected", saved_at_iteration=7, actual_iteration=3)]
    assert all("coefficient" not in row for row in result["all_states"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("damage, message", [
    ("current", "saved-state arrays"), ("incumbent", "saved-state arrays"),
    ("metadata", "checkpoint metadata"), ("geometry", "saved-state arrays"),
    ("actual_iteration", "incumbent iteration"), ("extra_state", "saved-state arrays"),
])
def test_missing_expected_state_metadata_or_chart_is_not_silently_skipped(retained, damage, message):
    if damage == "current":
        del retained.arrays["checkpoint_5_state"]
    elif damage == "incumbent":
        del retained.arrays["checkpoint_5_selected_state"]
    elif damage == "metadata":
        del retained.history["checkpoints"]["5"]
    elif damage == "geometry":
        del retained.arrays["checkpoint_5_selected_anchors_u"]
    elif damage == "actual_iteration":
        retained.history["checkpoints"]["5"]["selected_iteration"] = 5
    else:
        retained.arrays["checkpoint_2_state"] = retained.arrays["checkpoint_0_state"]
    seal(retained.folder, retained.result, retained.arrays, retained.history)
    with pytest.raises(scoring.RetainedStateError, match=message) as failure:
        score(retained)
    assert not failure.value.inventory["all_retained_coverage_complete"]
    if damage in ("current", "metadata"):
        assert not failure.value.inventory["current_coverage_complete"]


@pytest.mark.parametrize("name", ["states.npz", "history.json.gz"])
def test_corrupt_hash_and_missing_artifacts_block_scoring(retained, name):
    path = retained.folder / name
    path.write_bytes(path.read_bytes() + b"corrupted")
    with pytest.raises(scoring.RetainedStateError, match="SHA256"):
        score(retained)
    path.unlink()
    with pytest.raises(scoring.RetainedStateError):
        score(retained)


def test_completed_states_ignore_stale_live_checkpoint(retained):
    (retained.folder / "live-checkpoint.json").write_text("invalid stale live pointer")
    before = score(retained)
    (retained.folder / "live-checkpoint-0.npz").write_bytes(b"not an archive")
    assert score(retained) == before


def test_dedup_preserves_actual_iteration_and_chart_not_only_coefficient(retained):
    result = score(retained)
    rows = result["all_states"]
    assert len(rows) == 4
    # Same coefficient/state at another accepted iteration must remain distinct.
    template = next(r for r in rows if r["iteration"] == 3)
    changed = deepcopy(template)
    changed["iteration"] = 4
    changed["state_id"] += "different-iteration"
    assert len(scoring._deduplicate([template, changed])) == 2
    # Within a task, role-normalized array hashes retain geometry identities.
    metadata = dict(iteration=3, bic=template["bic"])
    names = [("state", "checkpoint_5_selected_state"), ("anchors_u", "checkpoint_5_selected_anchors_u")]
    first = scoring._fingerprint(retained.arrays, names, metadata)
    retained.arrays["checkpoint_5_selected_anchors_u"] = np.array([1])
    assert first != scoring._fingerprint(retained.arrays, names, metadata)


def test_bic_ties_use_iteration_then_task_then_state_without_evaluation():
    rows = [dict(bic=dict(score=1.), iteration=t, task_id=task, state_id=s, metrics=dict(validation_mse=v))
            for t, task, s, v in [(3, 0, "a", 0.), (2, 5, "b", 0.), (2, 4, "z", 100.), (2, 4, "a", 999.)]]
    assert min(rows, key=scoring.state_key) is rows[-1]
    for row in rows:
        row["metrics"] = dict(validation_mse=float("nan"), coefficient_rmse=-1.)
    assert min(rows, key=scoring.state_key) is rows[-1]


def test_scorer_receives_training_only_and_has_no_fitting_dependency(retained, monkeypatch):
    original = scoring.old.bic_score
    calls = []
    def bic(X, Y, coefficient, **kwargs):
        assert X is retained.data["X"] and Y is retained.data["Y"]
        assert set(kwargs) <= {"rank", "design_rank", "direct_rrr", "free_directions", "weighted_u", "weighted_v", "penalized_u", "penalized_v"}
        calls.append(True)
        return original(X, Y, coefficient, **kwargs)
    monkeypatch.setattr(scoring.old, "bic_score", bic)
    result = score(retained)
    assert result["inventory"]["all_retained_coverage_complete"] and len(calls) == 4


def test_failed_trajectory_does_not_become_eligible_from_saved_prefix(retained):
    retained.result.update(success=False, termination_reason="numerical_stagnation")
    seal(retained.folder, retained.result, retained.arrays, retained.history)
    with pytest.raises(scoring.RetainedStateError, match="originally eligible"):
        score(retained)


def test_rrr_three_physical_copies_deduplicate_with_certificate(tmp_path):
    case = dict(p=3, q=3, rank=1, free_directions=[2, 2], support_limits=[1, 1])
    coefficient = np.diag([1., 0, 0])
    certificate = dict(certified=True, scope="numerical_design_range_at_recorded_svd_tolerance",
                       coefficient_convention="minimum_norm_lift_of_selected_fitted_response")
    checkpoint = SimpleNamespace(coefficient=coefficient, iteration=0, selected_iteration=0,
                                 termination_reason="target_rrr_closed_form", certificate=certificate)
    model = SimpleNamespace(method_="target_rrr", coefficient_=coefficient,
        factors_=dict(P=np.eye(3)[:, :1], d=np.array([1.]), Q=np.eye(3)[:, :1]), checkpoints_={0: checkpoint})
    arrays = writer._state_arrays(model)
    history = dict(checkpoints={"0": writer._checkpoint_metadata(checkpoint)},
                   validation_history=[dict(iteration=0)])
    X = np.vstack([np.eye(3), np.zeros((1, 3))])
    Y = X @ coefficient
    Y[-1, 0] = .1
    data = dict(X=X, Y=Y, C_star=coefficient, X_validation=X, Y_validation=Y)
    result = dict(task=dict(task_id=1), status="complete", execution_success=True, success=True,
        fit_method="target_rrr", n_iter=0, selected_iteration=0, rrr_certificate=certificate,
        metadata=dict(rrr_certificate=certificate), termination_reason="target_rrr_closed_form", optimization_converged=True,
        selected_metrics=scoring.audit._metrics(coefficient, data), terminal_metrics=scoring.audit._metrics(coefficient, data))
    seal(tmp_path, result, arrays, history)
    output = scoring.score_retained_states(tmp_path, result, data, {}, case, dict(iterations=2000, checkpoint_interval=250))
    assert len(output["current_states"]) == len(output["all_states"]) == 1
    assert output["inventory"]["raw_saved_state_count"] == 3
    assert output["inventory"]["unreached_checkpoint_iterations"] == []
    assert output["all_states"][0]["bic"]["model_dimension"] == 5
    result["rrr_certificate"]["certified"] = False
    seal(tmp_path, result, arrays, history)
    with pytest.raises(scoring.RetainedStateError, match="certificate"):
        scoring.score_retained_states(tmp_path, result, data, {}, case, dict(iterations=2000, checkpoint_interval=250))


def test_feasible_state_with_wrong_declared_chart_epoch_is_rejected(retained):
    old_chart = scoring.audit.saved_chart(retained.arrays, "checkpoint_5_selected", retained.case)
    factors = old_chart.reconstruct(retained.arrays["checkpoint_5_selected_state"])
    wrong_epoch_chart = AnchorChart(3, 3, [1], [1], np.eye(1), np.eye(1))
    retained.arrays["checkpoint_5_selected_state"] = wrong_epoch_chart.initial_state(*factors)
    for field in scoring.GEOMETRY:
        retained.arrays[f"checkpoint_5_selected_{field}"] = getattr(wrong_epoch_chart, field)
    # The reconstructed coefficient is valid, but it belongs to epoch 1 while
    # the checkpoint explicitly identifies its validation incumbent as epoch 0.
    seal(retained.folder, retained.result, retained.arrays, retained.history)
    with pytest.raises(scoring.RetainedStateError, match="switch epoch"):
        score(retained)


def test_duplicate_state_ids_from_different_configurations_are_never_merged(retained):
    row = score(retained)["all_states"][0]
    other = deepcopy(row)
    other["task_id"] += 1
    assert len(scoring._deduplicate([row, other])) == 2


@pytest.mark.parametrize("reason, converged, cap", [("stationarity", True, 10), ("max_iterations", False, 7)])
def test_successful_early_stationarity_or_budget_terminal_is_expected_even_off_checkpoint_grid(retained, reason, converged, cap):
    retained.result.update(termination_reason=reason, optimization_converged=converged)
    retained.config["iterations"] = cap
    seal(retained.folder, retained.result, retained.arrays, retained.history)
    inventory = score(retained)["inventory"]
    assert inventory["expected_checkpoint_iterations"] == [0, 5, 7]
    assert inventory["unreached_checkpoint_iterations"] == ([10] if reason == "stationarity" else [])


def test_rank_two_rotation_recentering_preserves_own_state_and_rejects_terminal_center_alias(tmp_path):
    """A genuine SO(2) center change cannot be applied to an earlier coordinate."""
    def rotation(angle):
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    case = dict(p=4, q=4, rank=2, free_directions=[2, 2], support_limits=[4, 4])
    configuration = dict(iterations=2, checkpoint_interval=2,
        margins=dict(anchor_min=.04, d_lower=.001, d_upper=8., gap=.1))
    source = dict(left=np.eye(4), right=np.eye(4))
    ru, rv = rotation(.3), rotation(-.24)
    P, Q = np.vstack([ru, np.zeros((2, 2))]), np.vstack([rv, np.zeros((2, 2))])
    d = np.array([2., 1.])
    before = AnchorChart(4, 4, [0, 1], [0, 1], np.eye(2), np.eye(2))
    after = AnchorChart(4, 4, [0, 1], [0, 1], ru, rv)
    early = before.pack(np.array([-np.sqrt(2.) * np.tan(.3 / 2)]),
                        np.array([np.sqrt(2.) * np.tan(.24 / 2)]), d,
                        np.zeros((2, 2)), np.zeros((2, 2)))
    terminal = after.initial_state(P, d, Q)
    assert not np.allclose(early, terminal)
    def center_hash(array):
        return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes(order="C")).hexdigest()
    event = dict(iteration=1, sides={side: dict(old_anchors=[0, 1], new_anchors=[0, 1],
        old_center_sha256=center_hash(np.eye(2)), new_center_sha256=center_hash(center))
        for side, center in (("u", ru), ("v", rv))})
    truth = (P*d) @ Q.T
    X = np.vstack([np.eye(4), np.zeros((1, 4))])
    Y = X @ truth
    Y[-1] = [.1, -.1, .1, -.1]
    data = dict(X=X, Y=Y, C_star=truth, X_validation=X, Y_validation=X @ truth)
    validation = [dict(iteration=t, loss=0., chart_epoch=int(t == 2),
        incumbent_iteration=None if t == 0 else 0, selection_loss_difference=None if t == 0 else 0.) for t in range(3)]
    checkpoints = {t: SimpleNamespace(iteration=t, selected_iteration=0,
        state=early if t == 0 else terminal, selected_state=early,
        chart=before if t == 0 else after, selected_chart=before,
        chart_epoch=int(t == 2), selected_chart_epoch=0, history_length=t+1,
        validation_history_length=t+1, anchor_switches=[] if t == 0 else [event], best_validation_loss=0.)
        for t in (0, 2)}
    model = SimpleNamespace(method_="sparse_smart_v2", last_state_=terminal, state_=early,
        chart_=after, selected_chart_=before, checkpoints_=checkpoints,
        free_rows_=SimpleNamespace(rows_u=np.array([0, 1]), rows_v=np.array([0, 1])))
    arrays = writer._state_arrays(model)
    history = dict(history=[dict(iteration=t) for t in range(3)], history_chart_epochs=[0, 0, 1],
        validation_history=validation, checkpoints={str(t): writer._checkpoint_metadata(cp) for t, cp in checkpoints.items()})
    result = dict(task=dict(task_id=23, penalty_u=.01, penalty_v=.01, init_penalty=.03),
        status="complete", execution_success=True, success=True, n_iter=2, selected_iteration=0,
        fit_method="sparse_smart_v2", termination_reason="max_iterations", optimization_converged=False,
        anchor_switches=[event], selected_metrics=scoring.audit._metrics(truth, data),
        terminal_metrics=scoring.audit._metrics(truth, data))
    seal(tmp_path, result, arrays, history)
    report = scoring.score_retained_states(tmp_path, result, data, source, case, configuration)
    assert report["inventory"]["all_retained_coverage_complete"]
    assert {s["iteration"] for s in report["all_states"]} == {0, 2}
    for key, prefix in (("checkpoint_0_state", "checkpoint_0_terminal"), ("terminal_state", "terminal")):
        coefficient, _ = scoring._state(arrays, key, prefix, case, source, configuration, direct=False)
        np.testing.assert_allclose(coefficient, truth, atol=2e-14)
    # The earlier state remains feasible under this incorrect center, making
    # geometry provenance essential: a domain check alone would not catch it.
    wrong, _ = scoring._state(arrays, "checkpoint_0_state", "terminal", case, source, configuration, direct=False)
    assert np.linalg.norm(wrong-truth) > .1
    # Replacing only the stored early center by the terminal center is rejected
    # against the authenticated epoch chain before the state can be selected.
    arrays["checkpoint_0_terminal_center_u"] = arrays["terminal_center_u"]
    seal(tmp_path, result, arrays, history)
    with pytest.raises(scoring.RetainedStateError, match="geometry chain|declared chart epoch"):
        scoring.score_retained_states(tmp_path, result, data, source, case, configuration)
