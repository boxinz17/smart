"""Coverage summaries against the real runner's small operational fixtures."""
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


fixtures = module(HERE / "test_sparse_smart_v2_pilot.py", "summary_pilot_fixtures")
pilot = fixtures.pilot
campaign = fixtures.campaign
summary = module(HERE.parent / "summarize_sparse_smart_v2_campaign.py", "summary_under_test")


def execution_records(folder, result, code=0):
    slurm = result["slurm"]
    for prefix in ("process", "launcher"):
        (folder / f"{prefix}-exit-code.txt").write_text(f"{code}\n")
        if prefix == "process":
            columns = "job_id\tstep_id\tpid\texit_code\tfinished_utc\n"
            row = f"{slurm['job_id']}\t{slurm['step_id']}\t42\t{code}\t2026-09-13T08:00:00Z\n"
        else:
            columns = "job_id\tparallel_sequence\texit_code\tfinished_utc\n"
            row = f"{slurm['job_id']}\t1\t{code}\t2026-09-13T08:00:00Z\n"
        (folder / f"{prefix}-status.tsv").write_text(columns + row)


def seal(folder, result):
    pilot._json(folder / "result.json", result)
    status = pilot._read(folder / "status.json")
    for key in ("task", "success", "execution_success", "fit_status", "case"):
        status[key] = result[key]
    status["result_sha256"] = pilot._sha(folder / "result.json")
    pilot._json(folder / "status.json", status)


def completed_campaign(campaign, monkeypatch, *, rrr=True):
    root, _, config, _, _ = campaign
    config["rrr_shortcut"] = rrr
    config["refinement_solver"] = "masked_anchor_projected"
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "7")
    plan = pilot.plan(root)
    pilot.prepare(root)
    results = []
    for task in plan["tasks"]:
        result = pilot.fit(root, task["task_id"])
        assert result["execution_success"]
        execution_records(root / "tasks" / f"{task['task_id']:05d}", result)
        results.append(result)
    return root, plan, results


@pytest.mark.parametrize("rrr", [False, True])
def test_actual_runner_legacy_and_deduplicated_rrr_are_summarized(campaign, monkeypatch, rrr):
    root, plan, results = completed_campaign(campaign, monkeypatch, rrr=rrr)
    report = summary.summarize(root, root / "summary")
    assert report["success"], report["errors"]
    assert report["complete_execution_coverage"]
    assert not report["complete_tuning_coverage"]  # Expected strict initializer exclusions.
    assert report["planned_tasks"] == (3 if rrr else 4)
    expected = min((r for r in results if r["success"]),
                   key=lambda r: (r["selected_metrics"]["validation_mse"], r["task"]["task_id"]))
    winner = report["case_results"][0]["winner"]
    assert winner["task_id"] == expected["task"]["task_id"]
    assert winner["coefficient_rmse"] == expected["selected_metrics"]["coefficient_rmse"]
    assert report["setting_results"][0]["available_seeds"] == 1
    assert report["outcome_counts"]["initializer_exclusion"] == (1 if rrr else 2)
    assert len((root / "summary/task-outcomes.jsonl").read_text().splitlines()) == len(plan["tasks"])
    assert report["truth_used_for_selection"] is False
    if rrr:
        assert winner["fit_method"] == "target_rrr" and winner["selected_iteration"] == 0
        assert winner["optimization_converged"]


@pytest.mark.parametrize("damage", ["missing_result", "bad_hash", "wrong_task", "bad_exit", "bad_step", "missing_states"])
def test_incomplete_or_corrupt_outputs_do_not_get_complete_coverage(campaign, monkeypatch, damage):
    root, _, results = completed_campaign(campaign, monkeypatch)
    folder = root / "tasks/00000"
    if damage == "missing_result":
        (folder / "result.json").unlink()
    elif damage == "bad_hash":
        (folder / "result.json").write_text((folder / "result.json").read_text() + " ")
    elif damage == "wrong_task":
        results[0]["task"]["grid_index"] = 119
        seal(folder, results[0])
    elif damage == "bad_exit":
        execution_records(folder, results[0], 7)
    elif damage == "bad_step":
        result = dict(results[0], slurm=dict(job_id="123", step_id="8"))
        seal(folder, result)
    else:
        (folder / "states.npz").unlink()
    report = summary.summarize(root, root / "summary")
    assert not report["success"] and not report["complete_execution_coverage"]
    assert report["case_results"][0]["winner"]["task_id"] != 0
    assert report["errors"]


def test_validation_selection_ignores_truth_and_uses_frozen_task_id_for_ties(campaign, monkeypatch):
    root, _, results = completed_campaign(campaign, monkeypatch, rrr=False)
    eligible = [r for r in results if r["success"]]
    assert len(eligible) == 2
    for i, result in enumerate(eligible):
        result["selected_metrics"]["validation_mse"] = .25
        result["selected_metrics"]["coefficient_rmse"] = 100. if i == 0 else .0001
        result["avg_err"] = result["selected_metrics"]["coefficient_rmse"]
        seal(root / "tasks" / f"{result['task']['task_id']:05d}", result)
    report = summary.summarize(root, root / "summary")
    assert report["success"], report["errors"]
    assert report["case_results"][0]["winner"]["task_id"] == eligible[0]["task"]["task_id"]
    assert report["case_results"][0]["winner"]["coefficient_rmse"] == 100.


def test_numerical_stall_is_complete_execution_but_ineligible(campaign, monkeypatch):
    root, _, results = completed_campaign(campaign, monkeypatch)
    result = results[1]
    assert result["success"]
    result.update(success=False, fit_status="numerical_stagnation", avg_err=None,
                  termination_reason="numerical_stagnation")
    seal(root / "tasks/00001", result)
    report = summary.summarize(root, root / "summary")
    assert report["success"] and report["complete_execution_coverage"]
    assert report["outcome_counts"]["numerical_stagnation"] == 1
    assert not report["complete_tuning_coverage"]
    assert report["case_results"][0]["winner"]["fit_method"] == "target_rrr"


def test_all_failed_refinements_report_missing_winner_without_reclassifying_execution(campaign, monkeypatch):
    root, _, results = completed_campaign(campaign, monkeypatch, rrr=False)
    for result in results:
        if result["success"]:
            result.update(success=False, fit_status="numerical_failure", avg_err=None)
            seal(root / "tasks" / f"{result['task']['task_id']:05d}", result)
    report = summary.summarize(root, root / "summary")
    assert not report["success"] and report["complete_execution_coverage"]
    assert report["cases_with_winner"] == 0 and report["case_results"][0]["winner"] is None
    assert report["setting_results"][0]["available_seeds"] == 0
    assert report["outcome_counts"]["other_scientific_failure"] == 2


def test_provenance_failure_has_explicit_false_coverage_and_no_raw_copy(campaign, monkeypatch):
    root, _, _ = completed_campaign(campaign, monkeypatch)
    (root / "work-items.tsv").write_text("0\n")
    report = summary.summarize(root, root / "summary")
    assert not report["success"] and not report["complete_execution_coverage"]
    assert report["case_results"] == []
    assert all(p.suffix not in (".npz", ".gz") for p in (root / "summary").iterdir())
    assert json.loads((root / "summary/summary.json").read_text())["success"] is False
