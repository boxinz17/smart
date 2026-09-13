"""Parallel equivalence and fail-closed coverage using saved tiny artifacts."""
from copy import deepcopy
import hashlib
import json

import pytest

from test_rank_audit import fixture, save
from reviewer_revision_20260913 import parallel_rank_audit as parallel, rank_audit


OUTPUTS = ("rank-audit.json", "rank-summary.json", "rank-per-replication.csv",
           "rank-aggregate.csv", "rank-paired.csv", "rank-selection.csv", "rank-runtime.csv")


def shards_for(root, workers=2):
    plan = json.loads((root / "rank-plan.json").read_text())
    digest = hashlib.sha256((root / "rank-plan.json").read_bytes()).hexdigest()
    shards = [parallel._worker(str(root), indices, digest, True)
              for indices in parallel.partition_by_seed(plan, workers)]
    return digest, shards


def test_partition_keeps_all_conditions_for_each_seed_together():
    plan = dict(tasks=[dict(seed=seed) for _case in range(6) for seed in range(30)])
    groups = parallel.partition_by_seed(plan, 8)
    assert len(groups) == 8
    assert sorted(index for group in groups for index in group) == list(range(180))
    assert max(map(len, groups)) - min(map(len, groups)) <= 6
    for seed in range(30):
        assert sum(any(plan["tasks"][index]["seed"] == seed for index in group)
                   for group in groups) == 1


def test_serial_process_parallel_and_output_bytes_match(tmp_path):
    fixture(tmp_path, seeds=(1, 2, 3))
    serial = rank_audit.audit(tmp_path, allow_tiny_local=True)
    before = {name: (tmp_path / name).read_bytes() for name in OUTPUTS}
    plan_before = (tmp_path / "rank-plan.json").read_bytes()
    actual = parallel.audit(tmp_path, workers=2, allow_tiny_local=True)
    assert actual == serial
    assert {name: (tmp_path / name).read_bytes() for name in OUTPUTS} == before
    assert (tmp_path / "rank-plan.json").read_bytes() == plan_before


def test_subset_is_read_only_and_cannot_claim_study_completion(tmp_path):
    fixture(tmp_path)
    result = rank_audit.audit(tmp_path, task_indices=[1], write=False, allow_tiny_local=True)
    assert result["audit"]["subset_valid"]
    assert result["audit"]["audited_task_indices"] == [1]
    assert not result["audit"]["audit_passed"]
    assert not result["audit"]["all_planned_tasks_complete"]
    assert not any((tmp_path / name).exists() for name in OUTPUTS)
    with pytest.raises(ValueError, match="write=False"):
        rank_audit.audit(tmp_path, task_indices=[0], allow_tiny_local=True)
    for indices in ([0, 0], [-1], [2], [True]):
        with pytest.raises(ValueError):
            rank_audit.audit(tmp_path, task_indices=indices, write=False, allow_tiny_local=True)


@pytest.mark.parametrize("mutation,match", [
    ("missing_worker", "exactly once"), ("duplicate_worker", "exactly once"),
    ("missing_report", "task-report coverage"), ("duplicate_report", "task-report coverage"),
    ("missing_row", "method-outcome coverage"), ("duplicate_row", "method-outcome coverage"),
    ("wrong_hash", "plan hash"), ("wrong_audit_hash", "plan hash"),
    ("wrong_seed", "identity"), ("foreign_issue", "unaudited task")])
def test_bad_worker_coverage_never_publishes(tmp_path, mutation, match):
    fixture(tmp_path)
    digest, shards = shards_for(tmp_path)
    if mutation == "missing_worker":
        shards.pop()
    elif mutation == "duplicate_worker":
        shards.append(deepcopy(shards[0]))
    elif mutation == "missing_report":
        shards[0]["result"]["audit"]["tasks"].pop()
    elif mutation == "duplicate_report":
        shards[0]["result"]["audit"]["tasks"].append(deepcopy(shards[0]["result"]["audit"]["tasks"][0]))
    elif mutation == "missing_row":
        shards[0]["result"]["per_replication"].pop()
    elif mutation == "duplicate_row":
        shards[0]["result"]["per_replication"].append(deepcopy(shards[0]["result"]["per_replication"][0]))
    elif mutation == "wrong_hash":
        shards[0]["plan_sha256"] = "wrong"
    elif mutation == "wrong_audit_hash":
        shards[0]["result"]["audit"]["plan_sha256"] = "wrong"
    elif mutation == "wrong_seed":
        shards[0]["result"]["per_replication"][0]["seed"] = 999
    elif mutation == "foreign_issue":
        shards[0]["result"]["audit"]["issues"].append(dict(task_index=1, code="forged"))
    with pytest.raises(ValueError, match=match):
        parallel.merge_results(tmp_path, shards, plan_sha256=digest, allow_tiny_local=True)
    assert not any((tmp_path / name).exists() for name in OUTPUTS)


def test_changed_original_plan_rejected_before_publication(tmp_path):
    fixture(tmp_path)
    digest, shards = shards_for(tmp_path)
    (tmp_path / "rank-plan.json").write_text((tmp_path / "rank-plan.json").read_text() + "\n")
    with pytest.raises(RuntimeError, match="plan changed"):
        parallel.merge_results(tmp_path, shards, plan_sha256=digest, allow_tiny_local=True)
    assert not (tmp_path / "rank-audit.json").exists()


def test_merge_independently_rejects_splitting_one_seed(tmp_path):
    fixture(tmp_path, seeds=(1, 1))
    digest = hashlib.sha256((tmp_path / "rank-plan.json").read_bytes()).hexdigest()
    shards = [parallel._worker(str(tmp_path), [index], digest, True) for index in (0, 1)]
    with pytest.raises(ValueError, match="seed must remain"):
        parallel.merge_results(tmp_path, shards, plan_sha256=digest, allow_tiny_local=True)
    assert not (tmp_path / "rank-audit.json").exists()


def test_explicit_fit_failure_stays_in_parallel_denominators(tmp_path):
    from reviewer_revision_20260913 import audit as common
    import numpy as np
    fixture(tmp_path)
    directory = tmp_path / "rank_tasks/task-000000"
    path = directory / "benchmarks/target_rrr/candidates.json"
    archive = json.loads(path.read_text())
    for row in archive["candidates"]:
        row.update(status="failed", coefficient_saved=False, error="fixture_failure")
        (path.parent / f"candidate-{row['candidate_index']:04d}.npz").unlink()
    archive.update(n_eligible=0, selected_index=None)
    save(path, archive)
    result = json.loads((directory / "result.json").read_text())
    result["methods"]["target_rrr"].update(success=False, error="fixture_failure", n_eligible=0,
                                            audit=archive["candidates"])
    save(directory / "result.json", result)
    coefficients = common._arrays(directory / "coefficients.npz")
    del coefficients["target_rrr"]
    np.savez(directory / "coefficients.npz", **coefficients)
    serial = rank_audit.audit(tmp_path, write=False, allow_tiny_local=True)
    digest, shards = shards_for(tmp_path)
    actual = parallel.merge_results(tmp_path, shards, plan_sha256=digest,
                                    write=False, allow_tiny_local=True)
    assert actual == serial
    assert actual["audit"]["audit_passed"]
    assert actual["summary"]["status_counts"] == {"failed": 1, "success": 11}
    pair = next(row for row in actual["paired"] if row["method"] == "v2" and row["reference"] == "target_rrr")
    assert pair["n_planned"] == 2 and pair["n_paired"] == 1 and pair["n_unavailable_pairs"] == 1


def test_audit_failures_and_missing_tasks_match_serial_denominators(tmp_path):
    fixture(tmp_path, seeds=(1, 2, 3), missing=(2,))
    path = tmp_path / "rank_tasks/task-000000/v2/cell-000/candidates.json"
    saved = json.loads(path.read_text())
    saved["candidates"][0]["validation_mse"] = 1234.
    save(path, saved)
    serial = rank_audit.audit(tmp_path, write=False, allow_tiny_local=True)
    digest, shards = shards_for(tmp_path)
    actual = parallel.merge_results(tmp_path, list(reversed(shards)), plan_sha256=digest,
                                    write=False, allow_tiny_local=True)
    assert actual == serial
    assert not actual["audit"]["audit_passed"]
    assert actual["summary"]["n_planned_tasks"] == 3
    assert actual["summary"]["n_planned_method_outcomes"] == 18


def test_global_plan_issues_are_deduplicated_in_serial_order(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "rank-plan.json"
    plan = json.loads(path.read_text())
    plan["configuration"]["tuning_uses_truth"] = True
    save(path, plan)
    serial = rank_audit.audit(tmp_path, write=False, allow_tiny_local=True)
    digest, shards = shards_for(tmp_path)
    actual = parallel.merge_results(tmp_path, list(reversed(shards)), plan_sha256=digest,
                                    write=False, allow_tiny_local=True)
    assert actual == serial


def test_worker_exception_does_not_publish(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "rank-plan.json"
    plan = json.loads(path.read_text())
    del plan["configuration"]["rank_grid"]
    save(path, plan)
    with pytest.raises(KeyError, match="rank_grid"):
        parallel.audit(tmp_path, workers=2, allow_tiny_local=True)
    assert not any((tmp_path / name).exists() for name in OUTPUTS)


def test_production_guard_is_checked_before_creating_workers(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Production rank auditing"):
        parallel.audit(tmp_path, workers=2)
