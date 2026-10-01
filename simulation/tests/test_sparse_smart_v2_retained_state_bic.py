"""Five-arm retrospective orchestration on the existing tiny runner fixture."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


previous_tests = module(HERE / "test_sparse_smart_v2_retrospective_bic.py", "_retained_previous_tests")
campaign = previous_tests.campaign
historical = previous_tests.retro
runner = module(HERE.parent / "run_sparse_smart_v2_retained_state_bic.py", "_retained_runner_test")


def setup(campaign, monkeypatch, tmp_path, *, rrr=True, damage_before_historical=None):
    old_path, cid, old_root, originals = previous_tests.complete(campaign, monkeypatch, tmp_path, rrr=rrr)
    if damage_before_historical:
        damage_before_historical(old_root, originals)
    old_result = historical.run_case(old_path, cid)
    assert old_result["success"], old_result["errors"]
    root = tmp_path / "all-retained"
    (root / "source").mkdir(parents=True)
    (root / "source/fixture.py").write_text("# retained-state source fixture\n")
    runner.write_json(root / "source-manifest.json", dict(schema_version=1,
        files={"fixture.py": runner.legacy.sha(root / "source/fixture.py")}))
    plan = deepcopy(historical.load_plan(old_path))
    plan.pop("plan_fingerprint")
    plan.update(method=runner.METHOD, root=str(root), selection_arms=list(runner.ARMS),
        previous_retrospective_root=str(old_path.parent), previous_plan_sha256=runner.legacy.sha(old_path),
        previous_source_manifest_sha256=runner.legacy.sha(old_path.parent / "source-manifest.json"),
        source_manifest_sha256=runner.legacy.sha(root / "source-manifest.json"))
    plan["plan_fingerprint"] = runner.legacy.digest(plan)
    runner.write_json(root / "plan.json", plan)
    ready = runner.prepare(root / "plan.json")
    assert ready["success"] and ready["n_cases"] == 1
    assert runner.prepare(root / "plan.json") == ready
    return root / "plan.json", cid, old_root, originals, old_result


@pytest.mark.parametrize("rrr", [False, True])
def test_prepare_five_arms_exact_controls_monotonicity_and_retained_coverage(campaign, monkeypatch, tmp_path, rrr):
    path, cid, old_root, originals, baseline = setup(campaign, monkeypatch, tmp_path, rrr=rrr)
    monkeypatch.setattr(previous_tests.fixtures.pilot, "fit", lambda *a, **k: pytest.fail("retrospective must not refit"))
    report = runner.run_case(path, cid)
    assert report["success"], report["errors"]
    assert set(report["arms"]) == set(runner.ARMS) and all(report["arms"].values())
    assert report["historical_comparison"]["passed"] and report["monotonicity"]["passed"]
    assert all(check["passed"] for check in report["monotonicity"]["checks"].values())
    for arm in runner.HISTORICAL_ARMS:
        assert report["arms"][arm] == baseline["arms"][arm]
    assert report["complete_execution_coverage"] and not report["complete_tuning_coverage"]
    assert report["inventory"]["current_coverage_complete"] and report["inventory"]["all_retained_coverage_complete"]
    assert report["inventory"]["eligible_task_ids"] == report["original_eligible_task_ids"]
    assert len(report["candidates"]) == 2
    for candidate in report["candidates"]:
        assert set(candidate["current_state_ids"]) <= {s["state_id"] for s in candidate["states"]}
        inv = candidate["inventory"]
        assert inv["expected_checkpoint_iterations"] == inv["available_checkpoint_iterations"]
        assert inv["raw_saved_state_count"] >= len(candidate["states"])
    assert report["fitting_performed"] is False and report["bic_uses_validation_or_truth"] is False
    assert report["validation_used_for_original_stopping"] is True
    assert not any(p.suffix in (".npz", ".gz") for p in path.parent.rglob("*"))
    direct = [c for c in report["candidates"] if c["selected"]["fit_method"] == "target_rrr"]
    assert len(direct) == int(rrr)
    if rrr:
        assert len(direct[0]["states"]) == len(direct[0]["current_state_ids"]) == 1
        assert direct[0]["inventory"]["raw_saved_state_count"] == 3
        assert direct[0]["states"][0]["iteration"] == 0


@pytest.mark.parametrize("missing", ["checkpoint_0_state", "checkpoint_0_selected_state", "checkpoint_0_selected_center_u"])
def test_missing_checkpoint_while_historical_endpoints_exist_publishes_no_partial_arms(campaign, monkeypatch, tmp_path, missing):
    def damage(root, originals):
        result = next(row for row in originals if row["success"] and row["fit_method"] != "target_rrr")
        folder = root / "tasks" / f"{result['task']['task_id']:05d}"
        with np.load(folder / "states.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files if key != missing}
        previous_tests.fixtures.pilot._npz(folder / "states.npz", arrays)
        result["files"]["states.npz"] = runner.legacy.sha(folder / "states.npz")
        previous_tests.fixtures.seal(folder, result)
    path, cid, _, _, _ = setup(campaign, monkeypatch, tmp_path, damage_before_historical=damage)
    report = runner.run_case(path, cid)
    assert not report["success"]
    assert report["arms"] == dict.fromkeys(runner.ARMS)
    assert report["counts"]["corrupt"] == 1 and len(report["candidates"]) == 1
    assert not report["inventory"]["all_retained_coverage_complete"]
    error = next(e for e in report["errors"] if "inventory" in e)
    assert missing in error["inventory"]["missing_expected_arrays"]


def test_restart_reuses_exact_output_and_rejects_reference_mutation(campaign, monkeypatch, tmp_path):
    path, cid, old_root, _, _ = setup(campaign, monkeypatch, tmp_path)
    first = runner.run_case(path, cid)
    assert first["success"], first["errors"]
    monkeypatch.setattr(runner, "score_retained_states", lambda *a, **k: pytest.fail("verified restart must not score again"))
    assert runner.run_case(path, cid) == first
    output = path.parent / "cases" / f"{cid}.json"
    before = output.read_bytes()
    (old_root / "tasks/00001/process-status.tsv").write_text("changed")
    with pytest.raises(ValueError):
        runner.run_case(path, cid)
    assert output.read_bytes() == before


def test_changed_historical_baseline_is_rejected(campaign, monkeypatch, tmp_path):
    path, cid, _, _, _ = setup(campaign, monkeypatch, tmp_path)
    plan = runner.legacy.read(path)
    previous = Path(plan["previous_retrospective_root"]) / "cases" / f"{cid}.json"
    previous.write_bytes(previous.read_bytes() + b" ")
    report = runner.run_case(path, cid)
    assert not report["success"] and report["arms"] == dict.fromkeys(runner.ARMS)
    assert "SHA256" in report["errors"][0]["error"]


def test_ties_keep_historical_rules_and_use_actual_iteration_for_new_arms():
    def row(task, iteration, score, sid, validation):
        return dict(task_id=task, iteration=iteration, bic=dict(score=score), state_id=sid,
                    metrics=dict(validation_mse=validation, coefficient_rmse=validation))
    selected1, terminal1 = row(1, 9, 2., "s1", .2), row(1, 10, 3., "t1", .2)
    selected2, terminal2 = row(2, 7, 2., "s2", .3), row(2, 10, 3., "t2", .3)
    first, early = row(1, 2, 1., "a", 999.), row(2, 1, 1., "b", 1000.)
    candidates = [dict(selected=selected1, terminal=terminal1, states=[first, selected1, terminal1],
                       current_state_ids=["a", "t1"]),
                  dict(selected=selected2, terminal=terminal2, states=[early, selected2, terminal2],
                       current_state_ids=["b", "t2"])]
    arms = runner.select_arms(candidates)
    assert arms["bic_validation_selected"] is selected1  # historical task-ID tie
    assert arms["bic_validation_terminal"] is terminal1
    assert arms["bic_saved_current"] is arms["bic_all_retained"] is early  # actual iteration wins
    assert runner.monotonicity(arms)["passed"]
    for candidate in candidates:
        for state in candidate["states"]:
            state["metrics"]["coefficient_rmse"] = -1000.
            state["metrics"]["validation_mse"] = np.inf
    new = runner.select_arms(candidates)
    assert all(new[arm] is arms[arm] for arm in runner.ARMS[1:])
    assert runner.select_arms([]) == dict.fromkeys(runner.ARMS)
