"""No-fit checks for tuning-task planning and canonical cell collection."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


_fixture_spec = importlib.util.spec_from_file_location(
    "_discovery_tuning_fixtures", Path(__file__).with_name("test_discovery_budget_study.py"))
fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(fixtures)
study = fixtures.study


def _refresh_plan_fingerprint(plan):
    identity = {key: value for key, value in plan.items() if key not in (
        "plan_scheme", "plan_fingerprint", "created", "source_root", "no_fits_performed")}
    plan["plan_fingerprint"] = study.digest(identity)


def _small_sharded_plan(root, *, seeds=(0,), size=1, experiment=0, setting=0):
    config = study.configuration(iteration_budgets=(2, 4), checkpoint_interval=2,
        init_penalties=(.03,), penalties_u=(.01, .04), penalties_v=(.0025, .01, .04))
    plan = study.make_plan(models=(0,), experiments=(experiment,), seed_ids=seeds,
        setting_index=setting, config=config, tuning_task_size=size)
    study.write_plan(plan, root)
    return plan


def _save_task(root, plan, task, **kwargs):
    return fixtures.save_cell(root, study.task_plan(task, plan), task, **kwargs)


def _stub_merger(monkeypatch, root, plan, *, statuses=None):
    """Verify merger inputs and return only canonical identity metadata, never fits."""
    calls = []
    module = ModuleType("merge_sparse_smart_budget_shards")

    def merge_records(records, *, config, source_artifacts):
        assert json.loads(json.dumps(asdict(config))) == plan["configuration"]
        first = records[0]
        cell = next(cell for cell in plan["cells"] if first["rd_seed_id"] == cell["seed"]
                    and first["setting"] == cell["simulation_setting"]
                    and first["model"] == f"model{cell['model']+1}"
                    and first["experiment"] == f"exp{cell['experiment']+1}")
        tasks = [task for task in study.planned_tasks(plan) if task["cell_task_id"] == cell["task_id"]]
        assert len(records) == len(source_artifacts) == len(tasks)
        for record, artifact, task in zip(records, source_artifacts, tasks):
            assert artifact["task_id"] == task["task_id"]
            assert artifact["grid_candidate_ids"] == task["grid_candidate_ids"]
            assert artifact["plan_fingerprint"] == plan["plan_fingerprint"]
            assert artifact["source_relative"] == (
                Path("tasks")/task["task_id"]/"results"/study.relative_result(task)).as_posix()
            source = root/artifact["source_relative"]
            assert artifact["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
            assert study.read_json(source) == record
            assert record["configuration"]["runner"] == task["configuration"]
            manifest = artifact["cell_manifest"]
            assert manifest["sha256"] == hashlib.sha256(Path(manifest["path"]).read_bytes()).hexdigest()
        calls.append(dict(cell=cell, records=deepcopy(records), source_artifacts=deepcopy(source_artifacts)))
        status = (statuses or {}).get(cell["task_id"], "complete")
        result = fixtures.fake_record(plan, cell, status=status)
        result["tuning_shard_merge"] = dict(source_artifacts=deepcopy(source_artifacts))
        return result

    module.merge_records = merge_records
    monkeypatch.setitem(sys.modules, "merge_sparse_smart_budget_shards", module)
    return calls


@pytest.mark.parametrize("seeds, size, expected_tasks", [((0,), 1, 6309), ((0, 1, 2), 1, 18927), ((0,), 2, 3789)])
def test_tuning_tasks_expand_execution_without_changing_scientific_cells(seeds, size, expected_tasks):
    plan = study.make_plan(seed_ids=seeds, tuning_task_size=size)
    assert plan["expected_cells"] == 72*len(seeds)
    assert plan["expected_applicable"] == 63*len(seeds)
    assert plan["expected_inapplicable"] == 9*len(seeds)
    assert plan["expected_tasks"] == len(study.planned_tasks(plan)) == expected_tasks
    assert len({task["task_id"] for task in study.planned_tasks(plan)}) == expected_tasks
    assert all("_g" not in cell["task_id"] for cell in plan["cells"])
    excluded = [task for task in study.planned_tasks(plan) if task["inapplicability_reason"] is not None]
    assert len(excluded) == 9*len(seeds)
    assert all(task["task_id"] == task["cell_task_id"] and task["grid_candidate_ids"] == []
               and "configuration" not in task for task in excluded)
    study.validate_plan(plan)


@pytest.mark.parametrize("penalties, size, expected_tasks", [
    ((.0025, .01, .04, .16, .32), 1, 100),
    ((.0025, .01, .04, .16, .32), 2, 60),
    ((.0025, .01, .04, .16, .32), 4, 40),
    ((.0025, .01, .04, .16, .32), 20, 20),
    ((.001, .0025, .01, .04, .16, .32), 1, 144),
    ((.001, .0025, .01, .04, .16, .32), 2, 72),
    ((.001, .0025, .01, .04, .16, .32), 4, 48),
    ((.001, .0025, .01, .04, .16, .32), 20, 24),
])
def test_expanded_grid_chunks_do_not_cross_initialization_or_u_boundaries(penalties, size, expected_tasks):
    config = study.configuration(penalties_u=penalties, penalties_v=penalties)
    plan = study.make_plan(models=(0,), experiments=(0,), seed_ids=(0,), setting_index=0,
                           config=config, tuning_task_size=size)
    tasks = study.planned_tasks(plan)
    assert plan["expected_cells"] == 1 and plan["expected_tasks"] == expected_tasks
    grid = study.resolved_configuration(plan["cells"][0]["simulation_setting"], config)["candidate_grid"]
    assert [identifier for task in tasks for identifier in task["grid_candidate_ids"]] == list(range(4*len(penalties)**2))
    for task in tasks:
        ids = task["grid_candidate_ids"]
        assert 1 <= len(ids) <= size and ids == list(range(ids[0], ids[-1]+1))
        assert task["task_id"] == f"{task['cell_task_id']}_g{ids[0]}"
        assert len(task["configuration"]["init_penalties"]) == len(task["configuration"]["penalties_u"]) == 1
        subset = study.resolved_configuration(task["simulation_setting"], task["configuration"])["candidate_grid"]
        assert subset == [grid[index] for index in ids]
        assert study.task_plan(task, plan)["configuration"] == task["configuration"]
    study.validate_plan(plan)


def test_work_table_stays_five_columns_and_shell_overrides_match_exact_subsets(tmp_path):
    plan = _small_sharded_plan(tmp_path, size=2, seeds=(0, 1))
    tasks = study.planned_tasks(plan)
    rows = [line.split("\t") for line in (tmp_path/"work-items.tsv").read_text().splitlines()]
    assert rows == [[str(task[key]) for key in study.TABLE_FIELDS] for task in tasks]
    assert all(len(row) == 5 for row in rows)
    assert len(tasks) == 8 and len(list((tmp_path/"task-configs").glob("*.sh"))) == 4
    for task in tasks:
        script = tmp_path/"task-configs"/f"g{task['grid_candidate_ids'][0]}.sh"
        result = subprocess.run(["bash", "-c", 'source "$1"\nprintf "%s\\0" "${tuning_args[@]}"',
                                 "tuning-task-check", str(script)], capture_output=True, check=True)
        arguments = result.stdout.decode().rstrip("\0").split("\0")
        expected = []
        for key in ("init_penalties", "penalties_u", "penalties_v"):
            expected.extend(("--"+key.replace("_", "-"), ",".join(map(str, task["configuration"][key]))))
        assert arguments == expected
    first = tmp_path/"task-configs"/f"g{tasks[0]['grid_candidate_ids'][0]}.sh"
    first.write_text("tuning_args=(--penalties-v 999)\n")
    with pytest.raises(ValueError, match="configuration differs"):
        study.write_plan(plan, tmp_path)


def test_cli_defaults_to_single_combination_tasks_and_all_preserves_legacy_layout(tmp_path):
    default_root, all_root = tmp_path/"default", tmp_path/"all"
    assert study.main(["plan", "--output-root", str(default_root), "--seed-ids", "0"]) == 0
    default = study.read_json(default_root/"study-plan.json")
    assert default["tuning_task_size"] == 1 and default["expected_tasks"] == 6309
    assert default["configuration"]["init_penalties"] == [.01, .03, .1, .3]
    assert study.main(["plan", "--output-root", str(all_root), "--seed-ids", "0",
                       "--tuning-task-size", "all"]) == 0
    legacy = study.read_json(all_root/"study-plan.json")
    assert "work_items" not in legacy and "tuning_task_size" not in legacy and "expected_tasks" not in legacy
    assert study.planned_tasks(legacy) == legacy["cells"]
    assert len((all_root/"work-items.tsv").read_text().splitlines()) == 72
    assert not (all_root/"task-configs").exists()


def test_sharded_planner_still_runs_without_site_packages(tmp_path):
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import discovery_budget_study as s; "
        "s.main(['plan','--output-root',sys.argv[2],'--models','0','--experiments','0',"
        "'--seed-ids','0','--setting-index','0']); "
        "assert not any(x in sys.modules for x in "
        "('numpy','smart','sparse_smart','run_sparse_smart_budget_study','merge_sparse_smart_budget_shards'))"
    )
    result = subprocess.run([sys.executable, "-S", "-c", code, str(study.HERE), str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len((tmp_path/"work-items.tsv").read_text().splitlines()) == 100


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "all"])
def test_invalid_api_tuning_task_size_is_rejected(size):
    with pytest.raises(ValueError, match="[Tt]uning task size"):
        study.make_plan(seed_ids=(0,), tuning_task_size=size)


@pytest.mark.parametrize("mutation", [
    lambda plan: plan["work_items"].append(deepcopy(plan["work_items"][0])),
    lambda plan: plan["work_items"][0]["grid_candidate_ids"].append(99),
    lambda plan: plan["work_items"][0]["configuration"].update(penalties_v=[99.]),
])
def test_tuning_mapping_tampering_is_rejected_even_with_refreshed_plan_digest(tmp_path, mutation):
    plan = _small_sharded_plan(tmp_path)
    mutation(plan)
    plan["expected_tasks"] = len(plan["work_items"])
    _refresh_plan_fingerprint(plan)
    with pytest.raises(ValueError, match="tuning task mapping"):
        study.validate_plan(plan)


def test_complete_tuning_tasks_merge_to_one_canonical_cell_with_source_hashes(tmp_path, monkeypatch):
    plan = _small_sharded_plan(tmp_path, size=2)
    tasks = study.planned_tasks(plan)
    # Completion order is immaterial; merger inputs must follow canonical IDs.
    for task in reversed(tasks):
        _save_task(tmp_path, plan, task)
    calls = _stub_merger(monkeypatch, tmp_path, plan)
    report = study.aggregate(tmp_path, merge_shards=True)
    assert len(calls) == 1 and report["expected_cells"] == report["recorded_cells"] == 1
    assert report["expected_tasks"] == report["recorded_tasks"] == len(tasks)
    assert report["missing_tasks"] == report["missing_cells"] == []
    assert report["execution_complete"] and report["budget_coverage_complete"]
    cell = plan["cells"][0]
    canonical = study.read_json(tmp_path/"results"/study.relative_result(cell))
    assert study.validate_record_identity(canonical, cell, plan) == "complete"
    assert canonical["tuning_shard_merge"]["plan_fingerprint"] == plan["plan_fingerprint"]
    manifest = study.read_json(tmp_path/"results/budget_study_manifest.json")
    assert manifest["configuration"] == plan["configuration"] and manifest["expected_cells"] == 1
    assert len(list((tmp_path/"results").rglob("BudgetStudy_result_*.json"))) == 1
    assert len(report["task_artifacts"]) == len(tasks)


def test_shards_require_explicit_merge_before_canonical_cell_is_published(tmp_path, monkeypatch):
    plan = _small_sharded_plan(tmp_path)
    for task in study.planned_tasks(plan):
        _save_task(tmp_path, plan, task)
    calls = _stub_merger(monkeypatch, tmp_path, plan)
    report = study.aggregate(tmp_path)
    assert not calls and report["recorded_tasks"] == plan["expected_tasks"]
    assert report["recorded_cells"] == 0 and not report["execution_complete"]
    assert report["missing_cells"] == [plan["cells"][0]["task_id"]]
    assert any(error["error"] == "tuning_merge_required" for error in report["errors"])
    assert not list((tmp_path/"results").rglob("BudgetStudy_result_*.json"))


def test_missing_one_tuning_task_marks_its_whole_cell_missing_and_other_cell_still_merges(tmp_path, monkeypatch):
    plan = _small_sharded_plan(tmp_path, seeds=(0, 1))
    tasks = study.planned_tasks(plan)
    missing = tasks[1]
    for task in tasks:
        if task != missing:
            _save_task(tmp_path, plan, task)
    calls = _stub_merger(monkeypatch, tmp_path, plan)
    report = study.aggregate(tmp_path, merge_shards=True)
    assert len(calls) == 1 and calls[0]["cell"]["seed"] == 1
    assert report["recorded_cells"] == 1 and not report["execution_complete"]
    assert report["missing_tasks"] == [missing["task_id"]]
    assert report["missing_cells"] == [missing["cell_task_id"]]
    assert not (tmp_path/"results"/study.relative_result(plan["cells"][0])).exists()


@pytest.mark.parametrize("corruption", ["subset_configuration", "duplicate_result", "manifest_configuration"])
def test_bad_tuning_task_is_excluded_without_merging_an_incomplete_grid(tmp_path, monkeypatch, corruption):
    plan = _small_sharded_plan(tmp_path)
    tasks = study.planned_tasks(plan)
    saved = [_save_task(tmp_path, plan, task) for task in tasks]
    path, record, manifest_path = saved[0]
    if corruption == "subset_configuration":
        record["configuration"]["runner"]["penalties_v"] = [.16]
        study.atomic_json(record, path)
    elif corruption == "duplicate_result":
        study.atomic_json(record, path.with_name("BudgetStudy_result_duplicate.json"))
    else:
        manifest = study.read_json(manifest_path)
        manifest["configuration"] = plan["configuration"]
        study.atomic_json(manifest, manifest_path)
    calls = _stub_merger(monkeypatch, tmp_path, plan)
    report = study.aggregate(tmp_path, merge_shards=True)
    assert not calls and report["recorded_cells"] == 0
    assert report["missing_tasks"] == [tasks[0]["task_id"]]
    assert report["missing_cells"] == [tasks[0]["cell_task_id"]]
    assert not report["execution_complete"] and report["errors"]


def test_inapplicable_cell_is_collected_once_without_invoking_tuning_merger(tmp_path, monkeypatch):
    plan = _small_sharded_plan(tmp_path, experiment=2, setting=0)
    assert plan["expected_tasks"] == 1
    _save_task(tmp_path, plan, study.planned_tasks(plan)[0], status="inapplicable")
    calls = _stub_merger(monkeypatch, tmp_path, plan)
    report = study.aggregate(tmp_path, merge_shards=True)
    assert not calls and report["recorded_cells"] == report["recorded_tasks"] == 1
    assert report["execution_complete"] and report["status_counts"] == {"inapplicable": 1}


def test_merged_partial_cell_and_execution_error_remain_distinct(tmp_path, monkeypatch):
    plan = _small_sharded_plan(tmp_path)
    tasks = study.planned_tasks(plan)
    for task in tasks:
        _save_task(tmp_path, plan, task)
    calls = _stub_merger(monkeypatch, tmp_path, plan,
                         statuses={plan["cells"][0]["task_id"]: "partial"})
    report = study.aggregate(tmp_path, merge_shards=True)
    assert len(calls) == 1 and report["execution_complete"] and not report["budget_coverage_complete"]
    (tmp_path/"tasks"/tasks[0]["task_id"]/"launcher-exit-code.txt").write_text("7\n")
    report = study.aggregate(tmp_path, merge_shards=True)
    assert report["recorded_cells"] == 1 and not report["execution_complete"]
    assert any(error.get("exit_code") == 7 for error in report["errors"])


def test_all_mode_keeps_existing_cell_collection_without_merger(tmp_path, monkeypatch):
    plan = fixtures.small_plan(tmp_path)
    assert "work_items" not in plan
    fixtures.save_cell(tmp_path, plan, plan["cells"][0])
    module = ModuleType("merge_sparse_smart_budget_shards")
    module.merge_records = lambda *args, **kwargs: pytest.fail("All mode must not merge tuning shards")
    monkeypatch.setitem(sys.modules, "merge_sparse_smart_budget_shards", module)
    report = study.aggregate(tmp_path, merge_shards=True)
    assert report["execution_complete"] and report["recorded_cells"] == 1
