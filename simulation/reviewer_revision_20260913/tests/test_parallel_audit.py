"""Parallel audit coverage fixtures; saved arrays only, with no estimator fits."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "simulation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reviewer_revision_20260913 import audit
import parallel_audit
from test_audit import fixture, metrics, save_json


def _multicase_fixture(root, seeds=(1, 2)):
    """Clone saved seed artifacts into a second matched stress case."""
    fixture(root, seeds=seeds)
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    original_tasks = deepcopy(plan["tasks"])
    second_case = dict(plan["cases"][0], case_id="tiny_second", level=1)
    plan["cases"].append(second_case)
    plan["n_cases"] = 2
    plan["tasks"].extend(dict(task, case_index=1) for task in original_tasks)
    plan["n_tasks"] = len(plan["tasks"])
    save_json(plan_path, plan)
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    for index in range(len(original_tasks)):
        shutil.copytree(root / "tasks" / f"task-{index:06d}",
                        root / "tasks" / f"task-{index + len(original_tasks):06d}")
    for index, task in enumerate(plan["tasks"]):
        path = root / "tasks" / f"task-{index:06d}" / "result.json"
        result = json.loads(path.read_text())
        case = plan["cases"][task["case_index"]]
        result.update(task=task, case=case, plan_sha256=digest)
        result["metadata"].update(seed=task["seed"], case_id=case["case_id"])
        save_json(path, result)
    return plan


def _worker_results(root, workers=2):
    plan = json.loads((root / "plan.json").read_text())
    return [audit.audit(root, task_indices=indices, write=False, allow_tiny_local=True)
            for indices in parallel_audit.seed_partitions(plan, workers)]


def _assert_same_existing_fields(serial, parallel):
    """Ignore added coverage metadata and report ordering, preserve all values."""
    coverage = {"audited_task_indices", "subset_valid", "is_subset", "subset",
                "coverage_complete", "n_audited_tasks", "n_workers", "parallel_workers"}

    def check(expected, actual, path=""):
        if isinstance(expected, dict):
            assert isinstance(actual, dict), path
            for key, value in expected.items():
                if key not in coverage:
                    assert key in actual, path + "/" + key
                    check(value, actual[key], path + "/" + key)
        elif isinstance(expected, list):
            assert isinstance(actual, list), path
            if path.endswith(("audit_scope", "limitations")):
                assert all(value in actual for value in expected), path
            else:
                # Lists of report rows may be produced in worker completion
                # order; comparison normalizes ordering but not contents.
                assert sorted(json.dumps(value, sort_keys=True) for value in expected) == \
                       sorted(json.dumps(value, sort_keys=True) for value in actual), path
        else:
            assert expected == actual, path

    check(serial, parallel)


def test_seed_partitions_keep_all_cases_of_each_seed_together(tmp_path):
    plan = _multicase_fixture(tmp_path, seeds=(1, 2, 3, 4))
    partitions = parallel_audit.seed_partitions(plan, 3)
    assert len(partitions) <= 3 and all(partitions)
    flattened = [index for group in partitions for index in group]
    assert sorted(flattened) == list(range(8))
    assert len(flattened) == len(set(flattened))
    assert all(indices == sorted(indices) for indices in partitions)
    worker_for_index = {index: worker for worker, indices in enumerate(partitions) for index in indices}
    for seed in plan["seed_ids"]:
        assert len({worker_for_index[index] for index, task in enumerate(plan["tasks"]) if task["seed"] == seed}) == 1
    assert parallel_audit.seed_partitions(plan, 3) == partitions


def test_subset_is_never_a_complete_passing_audit(tmp_path):
    plan = _multicase_fixture(tmp_path)
    subset = parallel_audit.seed_partitions(plan, 2)[0]
    result = audit.audit(tmp_path, task_indices=subset, write=False, allow_tiny_local=True)
    assert result["audit"]["audited_task_indices"] == subset
    assert result["audit"]["subset_valid"] is True
    assert result["audit"]["audit_passed"] is False
    assert result["audit"]["all_planned_tasks_complete"] is False
    assert {row["task_index"] for row in result["per_replication"]} == set(subset)
    with pytest.raises(ValueError, match="write=False"):
        audit.audit(tmp_path, task_indices=subset, write=True, allow_tiny_local=True)


def test_parallel_reproduces_serial_results_and_written_artifacts(tmp_path):
    _multicase_fixture(tmp_path)
    serial = audit.audit(tmp_path, write=False, allow_tiny_local=True)
    result = parallel_audit.parallel_audit(tmp_path, workers=2, write=True, allow_tiny_local=True)
    assert serial["audit"]["audit_passed"] and result["audit"]["audit_passed"]
    _assert_same_existing_fields(serial, result)
    assert serial == result
    for name in ("audit.json", "summary.json", "per_replication.csv", "aggregate.csv", "paired.csv"):
        assert (tmp_path / name).stat().st_size > 0
    written = json.loads((tmp_path / "audit.json").read_text())
    assert written["audit_passed"] is True and written["all_planned_tasks_complete"] is True


def test_missing_artifact_is_covered_and_matches_serial_denominators(tmp_path):
    fixture(tmp_path, seeds=(1, 2, 3), missing=(2,))
    serial = audit.audit(tmp_path, write=False, allow_tiny_local=True)
    merged = parallel_audit.merge_worker_results(tmp_path, _worker_results(tmp_path), write=False)
    assert not merged["audit"]["audit_passed"]
    _assert_same_existing_fields(serial, merged)
    assert merged["summary"]["status_counts"]["missing_task"] == 5
    assert merged["summary"]["n_planned_tasks"] == 3


@pytest.mark.parametrize("corruption", ("missing", "duplicate", "plan_hash"))
def test_malformed_coverage_or_plan_hash_fails_closed_without_overwrite(tmp_path, corruption):
    _multicase_fixture(tmp_path)
    results = _worker_results(tmp_path)
    if corruption == "missing":
        results.pop()
    elif corruption == "duplicate":
        results.append(deepcopy(results[0]))
    else:
        results[0]["audit"]["plan_sha256"] = "0" * 64
    sentinel = b"existing independently reviewed audit\n"
    for name in ("audit.json", "summary.json", "per_replication.csv", "aggregate.csv", "paired.csv"):
        (tmp_path / name).write_bytes(sentinel)
    with pytest.raises(ValueError):
        parallel_audit.merge_worker_results(tmp_path, results, write=True)
    assert all((tmp_path / name).read_bytes() == sentinel for name in
               ("audit.json", "summary.json", "per_replication.csv", "aggregate.csv", "paired.csv"))


def test_worker_partitions_cannot_split_one_seed_across_cases(tmp_path):
    _multicase_fixture(tmp_path)
    # Full disjoint coverage alone is insufficient: split-by-case workers
    # cannot independently verify shared random draws across matched cases.
    workers = [audit.audit(tmp_path, task_indices=indices, write=False, allow_tiny_local=True)
               for indices in ([0, 1], [2, 3])]
    with pytest.raises(ValueError):
        parallel_audit.merge_worker_results(tmp_path, workers, write=False)


def test_duplicate_method_row_is_rejected(tmp_path):
    _multicase_fixture(tmp_path)
    results = _worker_results(tmp_path)
    results[0]["per_replication"].append(deepcopy(results[0]["per_replication"][0]))
    with pytest.raises(ValueError, match="row coverage"):
        parallel_audit.merge_worker_results(tmp_path, results, write=False)


def test_global_issues_are_deduplicated_and_order_matches_serial(tmp_path):
    _multicase_fixture(tmp_path)
    (tmp_path / "tasks/task-999999").mkdir()
    serial = audit.audit(tmp_path, write=False, allow_tiny_local=True)
    merged = parallel_audit.merge_worker_results(tmp_path, _worker_results(tmp_path), write=False)
    assert serial == merged
    assert sum(item["code"] == "unplanned_task_artifact" for item in merged["audit"]["issues"]) == 1


def test_cross_case_pairing_corruption_is_detected_after_parallel_audit(tmp_path):
    _multicase_fixture(tmp_path)
    directory = tmp_path / "tasks/task-000002"
    data, truth = audit._arrays(directory / "data.npz"), audit._arrays(directory / "truth.npz")
    data["X"][0, 0] += .25
    data["Y"] = data["X"] @ truth["C_star"]
    np.savez(directory / "data.npz", **data)
    result_path = directory / "result.json"
    record = json.loads(result_path.read_text())
    record["fit_data_fingerprint"] = audit._fingerprint(data)
    coefficients = audit._arrays(directory / "coefficients.npz")
    for method, coefficient in coefficients.items():
        record["methods"][method]["metrics"] = metrics(coefficient, data, truth)
    save_json(result_path, record)
    serial = audit.audit(tmp_path, write=False, allow_tiny_local=True)
    assert {issue["code"] for issue in serial["audit"]["issues"]} == {"paired_data_mismatch"}
    result = parallel_audit.parallel_audit(tmp_path, workers=2, write=False, allow_tiny_local=True)
    assert not result["audit"]["audit_passed"]
    assert "paired_data_mismatch" in {issue["code"] for issue in result["audit"]["issues"]}
    _assert_same_existing_fields(serial, result)
