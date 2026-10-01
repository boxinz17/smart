"""Checkpoint/terminal selection and complete provenance/state audits."""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import summarize_sparse_smart_v2_checkpoint_bic as summary
from sparse_smart.chart import AnchorChart
from sparse_smart_v2.support import choose_free_rows

spec = importlib.util.spec_from_file_location("checkpoint_legacy_fixture", HERE / "test_summarize_sparse_smart_v2_bic.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
base = summary.base


@pytest.fixture
def campaign(tmp_path):
    root, reference, plan, result = legacy.campaign.__wrapped__(tmp_path)
    group, task = plan["groups"][0], plan["tasks"][0]
    group.update(selected_target_rank=1, automatic_free_counts=[1, 2, 3])
    task["init_penalty"] = .003
    config = plan["configuration"]
    config.update(selection="bic_checkpoint", selection_rule="bic_checkpoint", iterations=200,
        init_penalties=[.003, .1], penalties_u=[0., .001, .0025, .01], penalties_v=[0., .001, .0025, .01],
        penalty_pairs=[[.001, .001], [.0025, .0025], [.01, .01]],
        checkpoint_iterations=[0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200])
    plan["n_candidates"] = 7
    plan["plan_fingerprint"] = base.digest({k: v for k, v in plan.items() if k != "plan_fingerprint"})
    legacy.write(root / "plan.json", plan)
    group_path = root / "groups" / group["group_id"] / "group.json"
    meta = base.read(group_path)
    meta.update(group=group, plan_fingerprint=plan["plan_fingerprint"])
    legacy.write(group_path, meta)
    prep = base.read(root / "preparation.json")
    prep.update(plan_fingerprint=plan["plan_fingerprint"], plan_sha256=base.sha(root / "plan.json"))
    prep["groups"][0]["group_json_sha256"] = base.sha(group_path)
    legacy.write(root / "preparation.json", prep)
    data = base.load_npz(Path(meta["files"]["data.npz"]["path"]))
    p, q = group["p"], group["q"]
    direction = np.array([np.cos(.08), np.sin(.08), 0., 0.])
    base_coefficient = np.outer(direction, direction)
    fitted_response = data["X"] @ base_coefficient
    optimum = float(np.sum(fitted_response * data["Y"]) / np.sum(fitted_response ** 2))
    states, outcomes = {}, []

    def state_and_score(free, scale, anchor):
        chart = AnchorChart(p, q, [anchor], [anchor], np.eye(1), np.eye(1))
        state = chart.initial_state(direction[:, None], np.array([scale * optimum]), direction[:, None])
        P, d, Q = chart.reconstruct(state)
        mask = choose_free_rows(chart, (free, free))
        coefficient = (P * d) @ Q.T
        arrays = dict(coefficient=coefficient, P=P, d=d, Q=Q, state=state,
            weighted_u=chart.unpack(state)[-2], weighted_v=chart.unpack(state)[-1],
            penalized_u=mask.penalized_u, penalized_v=mask.penalized_v,
            free_rows_u=mask.rows_u, free_rows_v=mask.rows_v, anchors_u=chart.anchors_u,
            anchors_v=chart.anchors_v, center_u=chart.center_u, center_v=chart.center_v)
        score = base.bic_score(data["X"], data["Y"], coefficient, rank=1, free_directions=(free, free),
            **{key: arrays[key] for key in ("weighted_u", "weighted_v", "penalized_u", "penalized_v")}).as_dict()
        error = coefficient - data["C_star"]
        metrics = dict(coefficient_rmse=float(np.linalg.norm(error)/4), coefficient_frobenius_squared=float(np.sum(error**2)),
            training_prediction_mse=float(np.mean((data["X"] @ error)**2)),
            validation_mse=float(np.mean((data["Y_validation"]-data["X_validation"]@coefficient)**2)))
        return arrays, score, metrics

    for free in task["free_counts"]:
        for penalty, early_scale, terminal_scale in ((.001, 1., .55), (.0025, .95, .98), (.01, .7, .7)):
            cid = f"c{len(outcomes)}"
            early, early_score, early_metrics = state_and_score(free, early_scale, 0)
            terminal, terminal_score, terminal_metrics = state_and_score(free, terminal_scale, 1)
            checkpoints = [dict(iteration=t, selection=early_score if t == 0 else terminal_score)
                           for t in config["checkpoint_iterations"]]
            best = min(checkpoints, key=lambda point: (point["selection"]["score"], point["iteration"]))
            chosen_arrays, chosen_metrics = (early, early_metrics) if best["iteration"] == 0 else (terminal, terminal_metrics)
            prefix, terminal_prefix = cid+"_", "t"+cid+"_"
            states.update({prefix+key: value for key, value in chosen_arrays.items()})
            states.update({terminal_prefix+key: value for key, value in terminal.items()})
            outcomes.append(dict(candidate_id=cid, rank=1, free_directions=[free, free], init_penalty=.003,
                penalty_u=penalty, penalty_v=penalty, success=True, execution_success=True,
                fit_method="sparse_smart_v2", fit_status="completed", termination_reason="max_iterations",
                n_iter=200, selected_iteration=best["iteration"], optimization_converged=False,
                selection=best["selection"], metrics=chosen_metrics, state_key=prefix,
                terminal_selection=terminal_score, terminal_metrics=terminal_metrics, terminal_state_key=terminal_prefix,
                checkpoint_scores=checkpoints))
    rrr = copy.deepcopy(result["outcomes"][-1])
    old_states = base.load_npz(root / "tasks/000000/states.npz")
    rrr.update(init_penalty=.003, terminal_selection=rrr["selection"], terminal_metrics=rrr["metrics"],
               terminal_state_key=rrr["state_key"], checkpoint_scores=[dict(iteration=0, selection=rrr["selection"])])
    states[rrr["state_key"]+"coefficient"] = old_states[rrr["state_key"]+"coefficient"]
    outcomes.append(rrr)
    np.savez(root / "tasks/000000/states.npz", **states)
    result.update(plan_fingerprint=plan["plan_fingerprint"], task=task, outcomes=outcomes,
                  elapsed_seconds=1., process_cpu_seconds=.8, selection_rule="bic_checkpoint")
    legacy.seal(root, result)
    return root, reference, plan, result


def test_audit_and_summary_use_distinct_checkpoint_terminal_winners(campaign):
    root, _, _, _ = campaign
    audit = summary.audit(root)
    assert audit["success"], audit["errors"]
    assert audit["completed_tasks"] == 1 and audit["audited_states"] == 13
    report = summary.summarize(root, make_plots=False)
    assert report["success"], report["errors"]
    assert report["complete_execution_coverage"] and report["complete_tuning_coverage"]
    assert report["outcome_counts"] == {"eligible": 7}
    methods = report["case_results"][0]["methods"]
    assert methods["automatic_checkpoint_bic"]["candidate_id"] == "c0"
    assert methods["automatic_checkpoint_bic"]["selected_iteration"] == 0
    assert methods["automatic_terminal_bic"]["candidate_id"] == "c1"
    assert methods["automatic_terminal_bic"]["selected_iteration"] == 200
    assert methods["fixed_checkpoint_bic"]["candidate_id"] == "c0"
    assert all(row["passed"] for row in report["selected_state_audits"])
    assert report["task_cpu_seconds"] == .8
    assert not report["evaluation_used_for_selection"]
    assert not list((root / "summary").glob("*.npz"))


@pytest.mark.parametrize("damage", ["source", "missing", "checkpoint_coverage", "wrong_minimum", "early_chart",
    "early_metrics", "terminal_metrics", "missing_required_pointer", "launcher", "validation_stop"])
def test_corruption_blocks_gate_and_summary(campaign, damage):
    root, _, _, result = campaign
    folder = root / "tasks/000000"
    if damage == "source":
        (root / "source/example.py").write_text("changed source")
    elif damage == "missing":
        (folder / "result.json").unlink()
    elif damage == "launcher":
        (folder / "launcher-exit-code.txt").write_text("1\n")
    else:
        if damage == "checkpoint_coverage": result["outcomes"][0]["checkpoint_scores"].pop(1)
        elif damage == "wrong_minimum": result["outcomes"][0]["selected_iteration"] = 200
        elif damage == "early_metrics": result["outcomes"][0]["metrics"]["coefficient_rmse"] += 1
        elif damage == "terminal_metrics": result["outcomes"][1]["terminal_metrics"]["coefficient_rmse"] += 1
        elif damage == "missing_required_pointer": result["outcomes"][0]["state_key"] = None
        elif damage == "validation_stop": result["outcomes"][0]["termination_reason"] = "validation_stop"
        elif damage == "early_chart":
            arrays = base.load_npz(folder / "states.npz")
            arrays["c0_anchors_u"] = arrays["tc0_anchors_u"]
            np.savez(folder / "states.npz", **arrays)
        legacy.seal(root, result)
    gate = summary.audit(root)
    assert not gate["success"] and gate["errors"]
    report = summary.summarize(root, make_plots=False)
    assert not report["success"] and report["errors"]


@pytest.mark.parametrize("kind, expected", [("initializer_exclusion", True), ("execution_failure", False)])
def test_numerical_exclusion_distinct_from_execution_failure(campaign, kind, expected):
    root, _, _, result = campaign
    row = result["outcomes"][2]
    row.update(success=False, execution_success=kind != "execution_failure", selection=None,
               metrics=None, terminal_selection=None, terminal_metrics=None, checkpoint_scores=[],
               state_key=None, terminal_state_key=None, fit_status="initialization_outside_domain")
    legacy.seal(root, result)
    audit = summary.audit(root)
    assert audit["success"] is expected
    report = summary.summarize(root, make_plots=False)
    assert report["success"] is expected
    assert not report["complete_tuning_coverage"]
    assert report["outcome_counts"][kind] == 1
    if not expected:
        assert report["case_results"][0]["methods"]["automatic_checkpoint_bic"] is None


def test_automatic_scope_excludes_fixed_only_dimensions_and_ignores_evaluation():
    group = dict(selected_target_rank=6, automatic_free_counts=[6, 7, 10])
    rows = []
    for index, (rank, free, score) in enumerate(((5, [10,10], -100), (6, [5,5], -90),
                                               (6, [6,6], -80), (6, [7,7], -70))):
        rows.append(dict(task_id=index, candidate_id=str(index), rank=rank, free_directions=free,
            eligible=True, fit_method="sparse_smart_v2", selection=dict(score=score),
            metrics=dict(coefficient_rmse=-score, validation_mse=-score),
            terminal_selection=dict(score=-score), terminal_metrics=dict(coefficient_rmse=0),
            n_iter=200, terminal_state_key="terminal_"))
    assert summary.select(rows, group)["candidate_id"] == "2"
    assert summary.select(rows, group, terminal=True)["candidate_id"] == "3"
    assert summary.select(rows, group, dict(rank=5,free_directions=[10,10]))["candidate_id"] == "0"
    rows.append(dict(rows[0], task_id=5, candidate_id="rrr", rank=6, fit_method="target_rrr", free_directions=[100,50]))
    assert summary.select(rows, group)["candidate_id"] == "rrr"


def test_seed_blocks_average_settings_before_uncertainty():
    records = []
    for seed, delta in ((10, 1.), (11, 2.), (12, 3.)):
        for setting, base_error in ((0, 10.), (1, 20.)):
            def winner(error):
                return dict(metrics=dict(coefficient_rmse=error, training_prediction_mse=error), selection=dict(score=0.))
            records.append(dict(case=dict(case_id=f"s{seed}k{setting}", group_id="g", model_id=0,
                experiment_id=0, setting_index=setting, seed_id=seed, n_train=200*(setting+1)),
                complete_execution_coverage=True, methods=dict(automatic_checkpoint_bic=winner(base_error+delta),
                previous_validation=winner(base_error))))
    _, _, _, blocks = summary.build_tables(records)
    cohort = [row for row in blocks if row["cohort"] == "seeds_10_99" and row["metric"] == "coefficient_rmse"][0]
    assert cohort["n"] == 3 and cohort["mean"] == 2.
    assert cohort["se"] == pytest.approx(1/np.sqrt(3))


def test_terminal_policy_auditor_does_not_accept_checkpoint_campaign(campaign):
    root, _, _, _ = campaign
    with pytest.raises(ValueError, match="policy differs"):
        base.load_plan(root)
