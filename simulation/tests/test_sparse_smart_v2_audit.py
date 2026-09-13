"""Artifact auditing with real chart events but no optimization or simulations."""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("smart_v2_audit_under_test", CODE / "simulation/audit_sparse_smart_v2_campaign.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

from sparse_smart.chart import AnchorChart
from sparse_smart_v2.anchors import CHART_CONTINUATION_RULE, reanchor
from sparse_smart_v2.solver import penalty_value
from sparse_smart_v2.support import FreeRows


DOMAIN = dict(anchor_min=.04, d_lower=.05, d_upper=12., gap=.01)


@pytest.mark.parametrize("solver", ["masked_chart_spectral_soft_hard", "masked_anchor_projected"])
def test_endpoint_gradient_metric_matches_independent_finite_difference(solver):
    chart = AnchorChart(3, 3, [0], [0], np.eye(1), np.eye(1))
    state = chart.initial_state(np.array([[.8], [.6], [0.]]), np.array([2.]),
                                np.array([[.6], [0.], [.8]]))
    design = np.array([[1., 2., 0.], [-1., .5, 1.], [.5, 0., 1.]])
    response = np.arange(9.).reshape(3, 3)/7
    coordinates = state.copy()
    if solver == "masked_anchor_projected":
        for sl in (chart.z_u_slice, chart.z_v_slice): coordinates[sl] /= state[chart.d_slice][0]
    def loss(x):
        z = x.copy()
        if solver == "masked_anchor_projected":
            for sl in (chart.z_u_slice, chart.z_v_slice): z[sl] *= x[chart.d_slice][0]
        return chart.loss(z, design, response)
    delta = 1e-6
    derivatives=[]
    for i in range(len(coordinates)):
        direction=np.zeros_like(coordinates);direction[i]=delta
        derivatives.append((loss(coordinates+direction)-loss(coordinates-direction))/(2*delta))
    smooth,norm=audit.smooth_gradient_norm(chart,state,design,response,solver)
    assert smooth == pytest.approx(loss(coordinates))
    assert norm == pytest.approx(np.linalg.norm(derivatives), rel=2e-8, abs=2e-8)
    with pytest.raises(ValueError, match="unsupported refinement"):
        audit.smooth_gradient_norm(chart,state,design,response,"unknown")


def _free(chart, rows_u, rows_v):
    return FreeRows(np.asarray(rows_u), np.asarray(rows_v),
        np.repeat((~np.isin(chart.complement_u, rows_u))[:, None], chart.rank, axis=1),
        np.repeat((~np.isin(chart.complement_v, rows_v))[:, None], chart.rank, axis=1))


def _rotation_case(*, weak=False):
    P = np.zeros((4, 2))
    if weak:
        P[[0, 2], 0] = [.045, np.sqrt(1-.045**2)]
    else:
        P[[0, 2], 0] = [.8, .6]
    P[[1, 3], 1] = [.8, .6]
    Q = np.vstack((np.eye(2), np.zeros((1, 2))))
    chart = AnchorChart(4, 3, [0, 1], [0, 1], np.eye(2), np.eye(2))
    state = chart.initial_state(P, np.array([3., 1.]), Q)
    state[chart.omega_u_slice] = np.sqrt(2) * .48
    free = _free(chart, [0, 1], [0, 1])
    changed = reanchor(chart, state, free, **DOMAIN)
    assert changed is not None
    return chart, state, free, changed


def _fallback_case():
    # Fixed restricted geometry, completed to an orthonormal factor. These
    # numbers contain no training data or validation-selected quantities.
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
        [.1424352155570814, -.19053493274982453, -.039876999759127806, .03531644690934479, -.19606505779435324]]) * 1.0001
    values, vectors = np.linalg.eigh(np.eye(5)-block.T@block)
    P = np.vstack((block, (vectors*np.sqrt(values))@vectors.T, np.zeros((1, 5))))
    Q = np.vstack((np.eye(5), np.zeros((1, 5))))
    old = np.array([0, 1, 2, 5, 8])
    left, _, right = np.linalg.svd(P[old], full_matrices=False)
    chart = AnchorChart(len(P), len(Q), old, np.arange(5), left@right, np.eye(5))
    state = chart.initial_state(P, np.arange(5., 0., -1.), Q)
    free = _free(chart, np.arange(10), np.arange(5))
    changed = reanchor(chart, state, free, **DOMAIN)
    assert changed is not None and changed.event["sides"]["u"]["anchor_search"]["fallback_accepted"]
    return chart, state, free, changed


def _transcript(case, *, reason="left Cayley coordinate has operator norm above 1/2"):
    chart, state, free, changed = case
    event = deepcopy(changed.event)
    event.update(iteration=1, switch_index=1, previous_status="numerical_stagnation", previous_rejection=reason)
    result = dict(n_iter=1, anchor_switches=[event],
                  metadata=dict(anchor_switches=[event], chart_continuation_rule=CHART_CONTINUATION_RULE))
    config = dict(margins=DOMAIN, adaptive_anchors=True, max_anchor_switches=16,
                  chart_continuation_rule=CHART_CONTINUATION_RULE)
    return result, config


@pytest.mark.parametrize("weak", [False, True])
def test_real_rotation_event_with_fixed_free_rank_passes_and_binds_full_center(weak):
    case = _rotation_case(weak=weak)
    result, config = _transcript(case)
    chart, _, free, changed = case
    events, epochs = audit.switch_epochs(result, config, chart, (free.rows_u, free.rows_v))
    assert len(events) == 1 and len(epochs) == 2
    assert epochs[0]["u"]["anchors"] == epochs[1]["u"]["anchors"]
    assert epochs[0]["u"]["center_sha256"] != epochs[1]["u"]["center_sha256"]
    search = events[0]["sides"]["u"]["anchor_search"]
    if weak:
        assert search["primary_starts"] == 0 and not search["fallback_triggered"]
    else:
        assert search is None
    audit.check_chart_epoch(chart, epochs[0], name="initial")
    audit.check_chart_epoch(changed.chart, epochs[1], name="terminal")
    with pytest.raises(ValueError, match="center"):
        audit.check_chart_epoch(changed.chart, epochs[0], name="wrong epoch")


def test_real_fallback_event_passes_and_records_bounded_work():
    case = _fallback_case()
    result, config = _transcript(case, reason="left anchor singular value is below anchor_min")
    chart, _, free, changed = case
    _, epochs = audit.switch_epochs(result, config, chart, (free.rows_u, free.rows_v))
    search = result["anchor_switches"][0]["sides"]["u"]["anchor_search"]
    assert search["primary_starts"] == 2 and search["fallback_starts"] == 1
    assert search["winning_seed"] == [1, 2, 3, 5, 8]
    audit.check_chart_epoch(changed.chart, epochs[1], name="fallback terminal")


@pytest.mark.parametrize("key,value", [
    ("primary_starts", 3), ("fallback_starts", 26), ("maximum_fallback_starts", 24),
    ("maximum_search_starts", 26), ("fallback_triggered", False), ("fallback_accepted", False),
    ("fallback_triggered", 1), ("fallback_accepted", 1),
    ("winning_start", "restricted_cpqr"), ("winning_seed", [0, 1, 2, 3, 15]),
    ("winning_seed", [0, 1, 2, 3, 4]),
    ("winning_swap_rounds", 17), ("primary_best_margin", .041),
])
def test_fallback_search_transcript_corruption_is_rejected(key, value):
    case = _fallback_case()
    result, config = _transcript(case, reason="left anchor singular value is below anchor_min")
    result["anchor_switches"][0]["sides"]["u"]["anchor_search"][key] = value
    with pytest.raises(ValueError):
        audit.switch_epochs(result, config, case[0], (case[2].rows_u, case[2].rows_v))


@pytest.mark.parametrize("change", ["gain", "old_hash", "new_rows", "failure_trigger", "free_rows"])
def test_event_gain_identity_and_trigger_corruption_is_rejected(change):
    case = _fallback_case()
    result, config = _transcript(case, reason="left anchor singular value is below anchor_min")
    event = result["anchor_switches"][0]
    detail = event["sides"]["u"]
    if change == "gain":
        detail["new_min_singular_value"] = detail["old_min_singular_value"] + 256*np.finfo(float).eps
    elif change == "old_hash":
        detail["old_center_sha256"] = "0"*64
    elif change == "new_rows":
        detail["new_anchors"][-1] = 15
    elif change == "failure_trigger":
        event["previous_status"] = "stationarity"
    else:
        event["free_rows"]["u"][-1] = 15
    with pytest.raises(ValueError):
        audit.switch_epochs(result, config, case[0], (case[2].rows_u, case[2].rows_v))


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def _seal(task_root, result):
    result["files"] = {name: audit.sha(task_root/name) for name in ("states.npz", "history.json.gz")}
    _write_json(task_root/"result.json", result)
    status = {key: result[key] for key in ("schema_version", "method", "task", "case", "plan_fingerprint",
              "source_manifest_sha256", "success", "execution_success", "fit_status")}
    status.update(status="finished", result_sha256=audit.sha(task_root/"result.json"))
    _write_json(task_root/"status.json", status)


def _task_fixture(tmp_path):
    geometry = _rotation_case()
    chart, state, free, changed = geometry
    result, config = _transcript(geometry)
    P, d, Q = chart.reconstruct(state)
    coefficient = (P*d)@Q.T
    data = dict(X=np.eye(4), Y=coefficient+.01, C_star=coefficient-.02,
                X_validation=np.eye(4), Y_validation=coefficient+.03)
    source = dict(left=np.eye(4), right=np.eye(3))
    case = dict(case_id="synthetic", p=4, q=3, rank=2, free_directions=[2, 2], support_limits=[4, 2])
    task = dict(task_id=0, case_id="synthetic", init_penalty=.03, penalty_u=.0025, penalty_v=.01)
    initial = dict(init_penalty=.03, eligible=True,
                   metadata=dict(anchors={s:getattr(chart, f"anchors_{s}").tolist() for s in ("u", "v")}))
    metrics = audit._metrics(coefficient, data)
    def record(which_chart, which_state, iteration):
        loss, gradient = which_chart.value_gradient(which_state, data["X"], data["Y"])
        pen = penalty_value(which_chart, which_state, free, (.0025, .01))
        factors = which_chart.reconstruct(which_state)
        row = dict(iteration=iteration, smooth_loss=loss, penalty_value=pen,
                   objective=loss+pen, raw_gradient_norm=float(np.linalg.norm(gradient)))
        for side, factor in (("u", factors[0]), ("v", factors[2])):
            row[f"anchor_min_{side}"] = float(np.linalg.svd(factor[getattr(which_chart,f"anchors_{side}")],compute_uv=False)[-1])
            mask = getattr(free,f"penalized_{side}")
            row[f"support_{side}"] = int(np.count_nonzero(which_state[getattr(which_chart,f"z_{side}_slice")].reshape(mask.shape)[mask]))
        return row
    row0, row1 = record(chart, state, 0), record(chart, state, 1)
    terminal_record = record(changed.chart, changed.state, 1)
    validation = [dict(iteration=0, loss=metrics["validation_mse"], incumbent_iteration=None,
                       selection_loss_difference=None, chart_epoch=0),
                  dict(iteration=1, loss=metrics["validation_mse"], incumbent_iteration=0,
                       selection_loss_difference=0., chart_epoch=0)]
    checkpoints = {}
    for t in (0, 1):
        checkpoints[str(t)] = dict(iteration=t, history_length=t+1, validation_history_length=t+1,
            selected_iteration=0, chart_epoch=t, selected_chart_epoch=0,
            anchor_switches=result["anchor_switches"][:t], endpoint_record=row0 if t==0 else terminal_record,
            best_validation_loss=metrics["validation_mse"])
    history = dict(history=[row0,row1], history_chart_epochs=[0,0], validation_history=validation,
                   terminal_record=terminal_record, checkpoints=checkpoints)
    arrays = dict(states_schema_version=np.array(2), free_rows_u=free.rows_u, free_rows_v=free.rows_v)
    def save_state(name, prefix, which_chart, which_state, factors=False):
        arrays[name] = which_state
        for field in audit.GEOMETRY_FIELDS:
            arrays[f"{prefix}_{field}"] = getattr(which_chart,field)
        if factors:
            for label, factor in zip(("P","d","Q"), which_chart.reconstruct(which_state)):
                arrays[f"{prefix}_{label}"] = factor
    save_state("selected_state","selected",chart,state,True)
    save_state("terminal_state","terminal",changed.chart,changed.state,True)
    for t in (0,1):
        save_state(f"checkpoint_{t}_state",f"checkpoint_{t}_terminal",chart if t==0 else changed.chart,
                   state if t==0 else changed.state)
        save_state(f"checkpoint_{t}_selected_state",f"checkpoint_{t}_selected",chart,state)
    for field in audit.GEOMETRY_FIELDS:
        arrays[field] = getattr(changed.chart,field)
    result.update(schema_version=2,method="synthetic_audit_fixture",task=task,case=case,
        plan_fingerprint="synthetic-plan",source_manifest_sha256="synthetic-source",status="complete",
        success=True,execution_success=True,fit_status="completed",selected_iteration=0,
        fingerprints={},source_fingerprint="synthetic-frame",initialization_record=initial,
        selected_metrics=metrics,terminal_metrics=audit._metrics((changed.chart.reconstruct(changed.state)[0]*d)@Q.T,data),
        avg_err=metrics["coefficient_rmse"])
    result["metadata"].update(validation_mse=metrics["validation_mse"],
        selected_anchors={s:getattr(chart,f"anchors_{s}").tolist() for s in ("u","v")},
        terminal_anchors={s:getattr(changed.chart,f"anchors_{s}").tolist() for s in ("u","v")})
    plan = dict(tasks=[task],schema_version=2,method=result["method"],plan_fingerprint=result["plan_fingerprint"],
                source_manifest_sha256=result["source_manifest_sha256"],configuration=config)
    meta = dict(case=case,fingerprints={},source_fingerprint="synthetic-frame",initializers=[initial])
    task_root = tmp_path/"00000"
    task_root.mkdir()
    np.savez(task_root/"states.npz",**arrays)
    (task_root/"history.json.gz").write_bytes(gzip.compress(json.dumps(history,allow_nan=False).encode()))
    _seal(task_root,result)
    return task_root,plan,{"synthetic":(data,source,meta)},result,arrays


def test_full_task_audit_checks_all_saved_chart_epochs_and_metadata(tmp_path):
    task_root, plan, cache, _, _ = _task_fixture(tmp_path)
    report = audit.audit_task(task_root, plan, cache)
    assert report["success"] and report["accepted_states_checked"] == 6
    assert report["anchor_switches_checked"] == 1 and report["checkpoints_checked"] == 2


def test_full_task_rejects_resealed_task_case_identity_mismatch(tmp_path):
    task_root, plan, cache, result, _ = _task_fixture(tmp_path)
    # Make result, status, and the planned task agree on the wrong case ID.
    # The result's actual case and its cached metadata still identify the
    # original data, so checking each pair separately would miss this mismatch.
    result["task"] = {**result["task"], "case_id": "different_planned_case"}
    plan["tasks"][0] = deepcopy(result["task"])
    _seal(task_root, result)
    with pytest.raises(ValueError, match="case"):
        audit.audit_task(task_root, plan, cache)


@pytest.mark.parametrize("tampering", ["selected_anchor_metadata", "selected_center", "terminal_center"])
def test_full_task_rejects_rehashed_chart_metadata_corruption(tmp_path, tampering):
    task_root, plan, cache, result, arrays = _task_fixture(tmp_path)
    if tampering == "selected_anchor_metadata":
        result["metadata"]["selected_anchors"]["u"] = [0,2]
    elif tampering == "selected_center":
        arrays["selected_center_u"] = arrays["terminal_center_u"].copy()
    else:
        arrays["terminal_center_u"] = arrays["selected_center_u"].copy()
        arrays["center_u"] = arrays["terminal_center_u"].copy()
    np.savez(task_root/"states.npz",**arrays)
    _seal(task_root,result)
    with pytest.raises(ValueError, match="metadata|center"):
        audit.audit_task(task_root,plan,cache)
