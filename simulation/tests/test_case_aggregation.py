"""Case-level aggregation of synthetic saved fits; no optimization is performed."""
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "_case_aggregation_shards", Path(__file__).with_name("test_budget_tuning_shards.py"))
shards = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shards)
fixtures, runner = shards.fixtures, shards.runner

import aggregate_sparse_smart_cases as cases
import discovery_budget_study as study


def _cancelled_fixture(tmp_path, *, publication_mode="compact", authorize=True):
    raw, output = tmp_path / "raw", tmp_path / "analysis"
    plan, paths = _run(raw)
    task = study.planned_tasks(plan)[1]
    base = raw / "tasks" / task["task_id"]
    saved = _files(base)
    shutil.rmtree(base / "results")
    _write_markers(raw, task["task_id"], process=None, exit_code=None, launcher="137")
    policy = [dict(run_root=str(raw), task_id=task["task_id"], reason="Intentionally cancelled slow fit",
                   launcher_exit_code=137)]
    index = cases.prepare_campaign([raw], output, publication_mode=publication_mode,
                                   allowed_missing_tasks=policy if authorize else None)
    return raw, output, plan, task, saved, index, policy


def _slurm_cancelled_fixture(tmp_path, *, evidence_change=None, reference_change=None):
    raw, _, plan, task, saved, _, _ = _cancelled_fixture(tmp_path, authorize=False)
    base = raw / "tasks" / task["task_id"]
    for name in cases.MARKERS[:3]:
        (base / name).unlink(missing_ok=True)
    environment = f"Job: 11859590\nStep: 1620\nHost: test-node\nTask: {task['task_id']}\n"
    (base / "environment.txt").write_text(environment)
    manifest = json.loads(saved["results/budget_study_manifest.json"])
    manifest.update(attempt_status="running", cells=[], errors=[])
    study.atomic_json(manifest, base / "results/budget_study_manifest.json")
    evidence = dict(schema_version=1, kind="slurm_cancelled_task", run_root=str(raw),
        task_id=task["task_id"], job_id="11859590", step_id="11859590.1620",
        environment_sha256=hashlib.sha256(environment.encode()).hexdigest(),
        sacct=dict(job=dict(JobIDRaw="11859590", State="CANCELLED by 1234", ExitCode="0:15"),
                   step=dict(JobIDRaw="11859590.1620", State="CANCELLED", ExitCode="0:15")))
    if evidence_change:
        evidence_change(evidence)
    path = tmp_path / "cancellation-evidence.json"
    study.atomic_json(evidence, path)
    reference = dict(path=str(path), sha256=cases._sha(path), job_id="11859590",
                     step_id="11859590.1620", task_id=task["task_id"])
    if reference_change:
        reference_change(reference)
    policy = [dict(run_root=str(raw), task_id=task["task_id"], reason="User cancelled the whole allocation",
                   cancellation_evidence=reference)]
    output = tmp_path / "slurm-analysis"
    index = cases.prepare_campaign([raw], output, publication_mode="compact", allowed_missing_tasks=policy)
    return raw, output, plan, task, saved, index, path


def test_whole_job_cancellation_uses_hashed_accounting_without_creating_exit_markers(tmp_path, monkeypatch):
    raw, output, plan, task, saved, index, path = _slurm_cancelled_fixture(tmp_path)
    before = _files(raw)
    first = _aggregate(output, plan)
    assert first["scientific"]["audit_passed"]
    assert first["scientific"]["status"] == "partial"
    assert first["execution"]["available_tasks_success"]
    assert first["execution"]["allowed_missing_task_ids"] == [task["task_id"]]
    assert first["missing_tuning"]["available_candidate_count"] == 6
    assert first["missing_tuning"]["planned_candidate_count"] == 8
    receipt = _read(_generation(output, first) / "receipt.json")
    inputs = next(row for row in receipt["inputs"]["tasks"] if row["task_id"] == task["task_id"])
    assert inputs["cancellation_evidence"]["sha256"] == cases._sha(path)
    assert any(row["path"].endswith("/environment.txt") and row["present"] for row in inputs["files"])
    import merge_sparse_smart_budget_shards as merger
    with monkeypatch.context() as patch:
        patch.setattr(merger, "merge_records_with_audit", lambda *args, **kwargs: pytest.fail("Resume repeated audit"))
        resumed = _aggregate(output, plan)
        assert resumed["resumed"] and resumed["generation"] == first["generation"]
    assert _files(raw) == before
    for relative, content in saved.items():
        target = raw / "tasks" / task["task_id"] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    restored = _aggregate(output, plan)
    assert restored["scientific"]["audit_passed"] and restored["scientific"]["budget_coverage_complete"]
    assert restored["execution"]["allowed_missing_task_ids"] == []


@pytest.mark.parametrize("kind", ["job_state", "step_state", "job_id", "step_id", "task_id", "run_root",
                                  "environment_hash", "exit_code"])
def test_whole_job_cancellation_rejects_inconsistent_frozen_accounting(tmp_path, kind):
    def change(evidence):
        if kind == "job_state": evidence["sacct"]["job"]["State"] = "COMPLETED"
        if kind == "step_state": evidence["sacct"]["step"]["State"] = "FAILED"
        if kind == "job_id": evidence["sacct"]["job"]["JobIDRaw"] = "123"
        if kind == "step_id": evidence["sacct"]["step"]["JobIDRaw"] = "11859590.1621"
        if kind == "task_id": evidence["task_id"] = "m0_e0_s0_k0_g999"
        if kind == "run_root": evidence["run_root"] += "-other"
        if kind == "environment_hash": evidence["environment_sha256"] = "0" * 64
        if kind == "exit_code": evidence["sacct"]["step"]["ExitCode"] = "unknown"
    raw, output, plan, _, _, _, _ = _slurm_cancelled_fixture(tmp_path, evidence_change=change)
    before = _files(raw)
    audit = _aggregate(output, plan)
    assert not audit["scientific"]["audit_passed"]
    assert not audit["execution"]["available_tasks_success"]
    assert audit["paper"] is None
    assert _files(raw) == before


@pytest.mark.parametrize("key", ["Job", "Step", "Task"])
def test_whole_job_cancellation_checks_environment_identity_even_when_hash_matches(tmp_path, key):
    raw, output, plan, task, _, _, path = _slurm_cancelled_fixture(tmp_path)
    environment = raw / "tasks" / task["task_id"] / "environment.txt"
    lines = environment.read_text().splitlines()
    environment.write_text("\n".join(line if not line.startswith(key + ":") else f"{key}: 999"
                                      for line in lines) + "\n")
    evidence = _read(path)
    evidence["environment_sha256"] = cases._sha(environment)
    study.atomic_json(evidence, path)
    index = _read(output / "campaign-index.json")
    index["allowed_missing_tasks"][0]["cancellation_evidence"]["sha256"] = cases._sha(path)
    index["index_fingerprint"] = study.digest({k: v for k, v in index.items() if k != "index_fingerprint"})
    study.atomic_json(index, output / "campaign-index.json")
    audit = _aggregate(output, plan)
    assert not audit["scientific"]["audit_passed"] and audit["paper"] is None


@pytest.mark.parametrize("marker", cases.MARKERS[:3])
@pytest.mark.parametrize("value", ["0", "1", "143"])
def test_whole_job_cancellation_does_not_excuse_any_present_exit_marker(tmp_path, marker, value):
    raw, output, plan, task, _, _, _ = _slurm_cancelled_fixture(tmp_path)
    (raw / "tasks" / task["task_id"] / marker).write_text(value + "\n")
    audit = _aggregate(output, plan)
    assert not audit["scientific"]["audit_passed"]
    assert not audit["execution"]["available_tasks_success"]


@pytest.mark.parametrize("kind", ["accounting_changed", "environment_changed", "success_manifest", "corrupt_result"])
def test_whole_job_cancellation_never_reuses_success_after_evidence_changes(tmp_path, kind):
    raw, output, plan, task, _, _, path = _slurm_cancelled_fixture(tmp_path)
    first = _aggregate(output, plan)
    assert first["scientific"]["audit_passed"]
    base = raw / "tasks" / task["task_id"]
    if kind == "accounting_changed": path.write_text(path.read_text() + "\n")
    if kind == "environment_changed": (base / "environment.txt").write_text("changed\n")
    if kind == "success_manifest":
        study.atomic_json(dict(attempt_status="completed", cells=[dict(status="complete", success=True)]),
                          base / "results/budget_study_manifest.json")
    if kind == "corrupt_result":
        target = base / "results" / study.relative_result(task)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("not JSON")
    audit = _aggregate(output, plan)
    assert not audit["scientific"]["audit_passed"] and audit["paper"] is None
    assert audit["generation"] != first["generation"]


@pytest.mark.parametrize("kind", ["accounting", "environment"])
def test_whole_job_cancellation_rechecks_evidence_before_publication(tmp_path, monkeypatch, kind):
    raw, output, plan, task, _, _, path = _slurm_cancelled_fixture(tmp_path)
    import merge_sparse_smart_budget_shards as merger
    original = merger.merge_records_with_audit
    def mutate_after_audit(*args, **kwargs):
        result = original(*args, **kwargs)
        target = path if kind == "accounting" else raw / "tasks" / task["task_id"] / "environment.txt"
        target.write_text(target.read_text() + "\n")
        return result
    monkeypatch.setattr(merger, "merge_records_with_audit", mutate_after_audit)
    with pytest.raises(ValueError, match="Inputs changed during case audit"):
        _aggregate(output, plan)
    assert not (output / "cases" / task["cell_task_id"] / "current.json").exists()


@pytest.mark.parametrize("kind", ["relative_path", "bad_sha", "step_job_mismatch", "task_mismatch"])
def test_invalid_whole_job_evidence_reference_rejected_during_preparation(tmp_path, kind):
    def change(reference):
        if kind == "relative_path": reference["path"] = "evidence.json"
        if kind == "bad_sha": reference["sha256"] = "invalid"
        if kind == "step_job_mismatch": reference["step_id"] = "123.1620"
        if kind == "task_mismatch": reference["task_id"] = "m0_e0_s0_k0_g999"
    with pytest.raises(ValueError):
        _slurm_cancelled_fixture(tmp_path, reference_change=change)


@pytest.mark.parametrize("publication_mode", ["full", "compact"])
def test_allowed_missing_shard_retains_full_grid_and_audited_available_winner(tmp_path, publication_mode):
    raw, output, plan, task, _, index, _ = _cancelled_fixture(tmp_path, publication_mode=publication_mode)
    before = _files(raw)
    audit = _aggregate(output, plan)
    assert index["schema_version"] == 3
    assert audit["scientific"]["audit_passed"] is True
    assert audit["scientific"]["status"] == "partial"
    assert audit["scientific"]["budget_coverage_complete"] is False
    assert audit["missing_tuning"]["unavailable_grid_candidate_ids"] == task["grid_candidate_ids"]
    assert audit["missing_tuning"]["planned_candidate_count"] == 8
    assert audit["missing_tuning"]["available_candidate_count"] == 6
    assert audit["execution"]["fit_success"] is False
    assert audit["execution"]["launch_success"] is False
    assert audit["execution"]["available_tasks_success"] is True
    assert audit["execution"]["allowed_missing_task_ids"] == [task["task_id"]]
    assert cases._permitted_execution(audit["execution"])
    assert len(audit["paper"]["candidate_grid"]) == 8
    for cap, selected in zip(audit["summary"]["caps"], audit["paper"]["selected"], strict=True):
        assert cap["success"] and not cap["coverage_complete"]
        assert set(task["grid_candidate_ids"]) <= set(cap["unresolved_candidate_ids"])
        assert selected["grid_candidate_id"] not in task["grid_candidate_ids"]
    if publication_mode == "compact":
        selected = _read(_generation(output, audit) / "selected-factors.json")
        assert selected["factors"]
        assert all(row["grid_candidate_id"] not in task["grid_candidate_ids"] for row in selected["factors"].values())
    assert _files(raw) == before


def test_missing_shard_without_permission_still_fails(tmp_path):
    raw, output, plan, _, _, index, _ = _cancelled_fixture(tmp_path, authorize=False)
    audit = _aggregate(output, plan)
    assert index["schema_version"] == 2
    assert not audit["scientific"]["audit_passed"]
    assert audit["paper"] is None


def test_permitted_missing_publication_resumes_and_restored_shard_is_used(tmp_path, monkeypatch):
    raw, output, plan, task, saved, _, _ = _cancelled_fixture(tmp_path)
    first = _aggregate(output, plan)
    assert first["scientific"]["audit_passed"]
    import merge_sparse_smart_budget_shards as merger
    with monkeypatch.context() as patch:
        patch.setattr(merger, "merge_records_with_audit", lambda *args, **kwargs: pytest.fail("Resume repeated audit"))
        resumed = _aggregate(output, plan)
        assert resumed["resumed"] and resumed["generation"] == first["generation"]
    base = raw / "tasks" / task["task_id"]
    for relative, content in saved.items():
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    restored = _aggregate(output, plan)
    assert restored["scientific"]["audit_passed"] and restored["scientific"]["budget_coverage_complete"]
    assert restored["missing_tuning"]["unavailable_tasks"] == []
    assert restored["execution"]["allowed_missing_task_ids"] == []
    assert restored["execution"]["fit_success"] and restored["execution"]["launch_success"]
    assert restored["generation"] != first["generation"]


@pytest.mark.parametrize("marker,value", [("launcher-exit-code.txt", "0"), ("launcher-exit-code.txt", None),
    ("launcher-exit-code.txt", "bad"), ("process-exit-code.txt", "0"), ("exit-code.txt", "1"),
    ("archive-status.txt", "complete")])
def test_permission_does_not_excuse_wrong_cancellation_evidence(tmp_path, marker, value):
    raw, output, plan, task, _, _, _ = _cancelled_fixture(tmp_path)
    path = raw / "tasks" / task["task_id"] / marker
    if value is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(value + "\n")
    audit = _aggregate(output, plan)
    assert not audit["scientific"]["audit_passed"]
    assert not cases._permitted_execution(audit["execution"])
    assert audit["paper"] is None


@pytest.mark.parametrize("kind", ["corrupt_result", "success_manifest"])
def test_permission_never_hides_present_corrupt_or_previously_complete_results(tmp_path, kind):
    raw, output, plan, task, _, _, _ = _cancelled_fixture(tmp_path)
    results = raw / "tasks" / task["task_id"] / "results"
    if kind == "corrupt_result":
        path = results / study.relative_result(task)
        path.parent.mkdir(parents=True)
        path.write_text("not JSON")
    else:
        study.atomic_json(dict(attempt_status="completed", cells=[dict(status="complete", success=True)]),
                          results / "budget_study_manifest.json")
    audit = _aggregate(output, plan)
    assert not audit["scientific"]["audit_passed"]
    assert audit["paper"] is None
    assert audit["missing_tuning"]["unavailable_tasks"] == []


def test_unrelated_execution_failure_is_not_excused_by_missing_task_policy(tmp_path):
    raw, output, plan, task, _, _, _ = _cancelled_fixture(tmp_path)
    other = next(row for row in study.planned_tasks(plan) if row["task_id"] != task["task_id"])
    _write_markers(raw, other["task_id"], process="17", exit_code="17", launcher="17")
    audit = _aggregate(output, plan)
    assert audit["scientific"]["audit_passed"]
    assert not audit["execution"]["available_tasks_success"]
    assert not cases._permitted_execution(audit["execution"])


@pytest.mark.parametrize("kind", ["valid", "empty_object", "wrong_seed", "wrong_configuration", "driver_error"])
def test_present_incomplete_manifest_must_match_cancelled_task(tmp_path, kind):
    raw, output, plan, task, saved, _, _ = _cancelled_fixture(tmp_path)
    manifest = json.loads(saved["results/budget_study_manifest.json"])
    manifest.update(attempt_status="running", cells=[], errors=[])
    if kind == "empty_object": manifest = {}
    if kind == "wrong_seed": manifest["seed_ids"] = [99]
    if kind == "wrong_configuration": manifest["configuration"]["inverse_step"] += 1
    if kind == "driver_error": manifest["errors"] = [dict(error="unrelated failure")]
    path = raw / "tasks" / task["task_id"] / "results/budget_study_manifest_attempts/attempt.json"
    study.atomic_json(manifest, path)
    before = _files(raw)
    audit = _aggregate(output, plan)
    assert audit["scientific"]["audit_passed"] is (kind == "valid")
    assert _files(raw) == before


def test_missing_permission_is_bound_to_frozen_task_grid_ids(tmp_path):
    _, output, plan, task, _, index, _ = _cancelled_fixture(tmp_path)
    index["allowed_missing_tasks"][0]["grid_candidate_ids"] = [0, 1]
    assert task["grid_candidate_ids"] != [0, 1]
    index["index_fingerprint"] = study.digest({key: value for key, value in index.items() if key != "index_fingerprint"})
    study.atomic_json(index, output / "campaign-index.json")
    audit = _aggregate(output, plan)
    assert not audit["scientific"]["audit_passed"]
    assert audit["paper"] is None


@pytest.mark.parametrize("kind", ["duplicate", "unknown_task", "relative_root", "wrong_ids", "all_tasks"])
def test_invalid_missing_task_policy_rejected_before_publication(tmp_path, kind):
    raw, output = tmp_path / "raw", tmp_path / "analysis"
    plan, _ = _run(raw)
    tasks = study.planned_tasks(plan)
    item = dict(run_root=str(raw), task_id=tasks[0]["task_id"], reason="Cancelled", launcher_exit_code=137)
    policy = [item]
    if kind == "duplicate": policy.append(deepcopy(item))
    if kind == "unknown_task": item["task_id"] = "m0_e2_s0_k0_g999"
    if kind == "relative_root": item["run_root"] = "raw"
    if kind == "wrong_ids": item.update(case_key=tasks[0]["cell_task_id"], grid_candidate_ids=[7])
    if kind == "all_tasks": policy = [dict(item, task_id=task["task_id"]) for task in tasks]
    with pytest.raises(ValueError):
        cases.prepare_campaign([raw], output, allowed_missing_tasks=policy)
    assert not (output / "campaign-index.json").exists()


def _read(path):
    return json.loads(Path(path).read_text())


def _run(root, *, split=True, record=None, failed_ids=(), legacy_validation_policy=False):
    """Persist honest numerical fixtures with the plan's source provenance."""
    records = (shards._shards(failed_ids=failed_ids) if split else
               [deepcopy(record) if record is not None else shards._synthetic(failed_ids=failed_ids)])
    config = asdict(shards.CONFIG) if split else records[0]["configuration"]["runner"]
    if legacy_validation_policy:
        for key in ("validation_interval", "validation_patience", "validation_min_iterations",
                    "validation_min_relative_improvement"):
            config.pop(key)
            for record in records:
                record["configuration"]["runner"].pop(key)
        for record in records:
            record["configuration_fingerprint"] = study.digest({key: record[key]
                for key in shards.merger._IDENTITY_KEYS})
    setting_index = next(i for i, setting in enumerate(study.experiment_settings(0, 2))
                         if setting == records[0]["setting"])
    plan = study.make_plan(models=(0,), experiments=(2,),
        seed_ids=(records[0]["rd_seed_id"],), setting_index=setting_index,
        config=json.loads(json.dumps(config)), tuning_task_size=2 if split else None)
    study.write_plan(plan, root)
    paths = []
    for task, record in zip(study.planned_tasks(plan), records, strict=True):
        applicable = task["inapplicability_reason"] is None
        source = plan["source"]["implementation"][str(applicable).lower()]
        record.update(implementation_fingerprint_scheme=study.SCHEME,
            implementation_manifest=deepcopy(source["manifest"]),
            implementation_fingerprint=source["fingerprint"])
        task_root = root/"tasks"/task["task_id"]
        output = task_root/"results"
        result_path = output/study.relative_result(task)
        study.atomic_json(record, result_path)
        row = dict(model=record["model"], experiment=record["experiment"],
            setting=record["setting"]["suffix"], seed_id=task["seed"],
            status=record["status"], success=record["success"], path=str(result_path))
        manifest = dict(schema_version=1, method=study.METHOD, models=[task["model"]],
            experiments=[task["experiment"]], seed_ids=[task["seed"]], profile=plan["profile"],
            setting_index=task["setting"], expected_cells=1,
            configuration=task.get("configuration", plan["configuration"]),
            seed_file_sha256=plan["seed_file_sha256"], attempt_status="completed", cells=[row], errors=[])
        study.atomic_json(manifest, output/"budget_study_manifest.json")
        for name in ("process-exit-code.txt", "exit-code.txt", "launcher-exit-code.txt"):
            (task_root/name).write_text("0\n")
        (task_root/"archive-status.txt").write_text("deferred\n")
        paths.append(result_path)
    return plan, paths


def _write_markers(root, task_id, *, process="0", exit_code="0", launcher="0", archive="deferred"):
    for name, value in (("process-exit-code.txt", process), ("exit-code.txt", exit_code),
                        ("launcher-exit-code.txt", launcher), ("archive-status.txt", archive)):
        path = root/"tasks"/task_id/name
        if value is None:
            path.unlink()
        else:
            path.write_text(value+"\n")


def _files(root):
    """Byte inventory to ensure aggregation does not mutate raw inputs."""
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def _prepare(tmp_path, *, split=True, record=None, failed_ids=(), publication_mode="full"):
    raw, output = tmp_path/"raw", tmp_path/"analysis"
    plan, paths = _run(raw, split=split, record=record, failed_ids=failed_ids)
    index = cases.prepare_campaign([raw], output, publication_mode=publication_mode)
    return raw, output, plan, paths, index


def _aggregate(output, plan):
    return cases.aggregate_case(output/"campaign-index.json", plan["cells"][0]["task_id"],
                                generate_data_fn=fixtures.generator)


def _generation(output, envelope):
    return output/"cases"/envelope["case_key"]/"generations"/envelope["generation"]


def test_split_case_is_numerically_audited_and_matches_unsplit_selection(tmp_path):
    raw, output, plan, _, index = _prepare(tmp_path)
    original = _files(raw)
    envelope = _aggregate(output, plan)
    expected = shards._audit(shards._synthetic())
    assert envelope["case_key"] == plan["cells"][0]["task_id"]
    assert envelope["resumed"] is False
    assert envelope["scientific"]["audit_passed"] is True
    assert envelope["scientific"]["budget_coverage_complete"] is True
    assert envelope["scientific"]["status"] == "complete"
    assert envelope["summary"]["caps"] == [
        {**cap, "selection_score": envelope["summary"]["caps"][i]["selection_score"]}
        for i, cap in enumerate(expected["caps"])]
    assert envelope["summary"]["validation_audit"] == expected["validation_audit"]
    assert envelope["summary"]["verified_factor_states"] == 40
    assert envelope["execution"]["fit_success"] is True
    assert envelope["execution"]["launch_success"] is True
    assert _files(raw) == original
    assert _read(output/"campaign-index.json") == index
    assert (output/"cases"/envelope["case_key"]/"current.json").is_file()
    assert _generation(output, envelope).is_dir()


def test_legacy_split_campaign_keeps_saved_disabled_validation_policy(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "analysis"
    plan, _ = _run(raw, legacy_validation_policy=True)
    before = _files(raw)
    cases.prepare_campaign([raw], output)
    audit = _aggregate(output, plan)
    assert audit['scientific']['audit_passed'], audit['scientific']['issues']
    assert audit['scientific']['status'] == 'complete'
    assert audit['scientific']['budget_coverage_complete']
    assert 'validation_patience' not in audit['configuration']
    assert audit['configuration']['n_validation'] == 100
    assert _files(raw) == before


def test_unsplit_case_keeps_the_same_full_numerical_audit(tmp_path):
    _, output, plan, _, _ = _prepare(tmp_path, split=False)
    envelope = _aggregate(output, plan)
    expected = shards._audit(shards._synthetic())
    assert envelope["scientific"]["audit_passed"] is True
    assert envelope["summary"]["caps"] == expected["caps"]
    assert envelope["summary"]["verified_factor_states"] == 40


def test_partial_trajectory_is_audited_without_claiming_budget_completion(tmp_path):
    _, output, plan, _, _ = _prepare(tmp_path, failed_ids=(6, 7))
    envelope = _aggregate(output, plan)
    assert envelope["scientific"]["audit_passed"] is True
    assert envelope["scientific"]["status"] == "partial"
    assert envelope["scientific"]["budget_coverage_complete"] is False
    assert envelope["summary"]["failed_trajectories"] == 2


@pytest.mark.parametrize("publication_mode", ["full", "compact"])
def test_unchanged_inputs_reuse_verified_generation_without_reauditing(tmp_path, monkeypatch, publication_mode):
    _, output, plan, _, _ = _prepare(tmp_path, publication_mode=publication_mode)
    first = _aggregate(output, plan)
    published = _files(_generation(output, first))
    def forbidden(*args, **kwargs):
        raise AssertionError("Unchanged published case must not repeat numerical auditing")
    monkeypatch.setattr(shards.summary, "validate_record", forbidden)
    monkeypatch.setattr(shards.merger, "merge_records_with_audit", forbidden)
    second = _aggregate(output, plan)
    assert second["resumed"] is True
    assert {key: value for key, value in second.items() if key != "resumed"} == {
        key: value for key, value in first.items() if key != "resumed"}
    assert _files(_generation(output, second)) == published


@pytest.mark.parametrize("publication_mode", ["full", "compact"])
def test_changed_raw_bytes_create_new_generation_and_preserve_previous(tmp_path, publication_mode):
    _, output, plan, paths, _ = _prepare(tmp_path, publication_mode=publication_mode)
    first = _aggregate(output, plan)
    previous = _generation(output, first)
    saved = _files(previous)
    # Timing metadata is audit-neutral, but its changed bytes must invalidate reuse.
    record = _read(paths[0])
    record["elapsed_time_sec"] += 1.
    study.atomic_json(record, paths[0])
    second = _aggregate(output, plan)
    assert second["scientific"]["audit_passed"] is True
    assert second["resumed"] is False
    assert second["generation"] != first["generation"]
    assert _files(previous) == saved


@pytest.mark.parametrize("publication_mode", ["full", "compact"])
def test_changed_analysis_identity_creates_new_generation(tmp_path, monkeypatch, publication_mode):
    _, output, plan, _, _ = _prepare(tmp_path, publication_mode=publication_mode)
    first = _aggregate(output, plan)
    saved = _files(_generation(output, first))
    original = cases.analysis_provenance
    def changed(*args, **kwargs):
        return {**original(*args, **kwargs), "fixture_revision": "reviewed-audit-update"}
    monkeypatch.setattr(cases, "analysis_provenance", changed)
    second = _aggregate(output, plan)
    assert second["resumed"] is False
    assert second["scientific"]["audit_passed"] is True
    assert second["generation"] != first["generation"]
    assert _files(_generation(output, first)) == saved


@pytest.mark.parametrize("publication_mode", ["full", "compact"])
def test_missing_shard_does_not_publish_an_incomplete_grid_winner(tmp_path, publication_mode):
    _, output, plan, paths, _ = _prepare(tmp_path, publication_mode=publication_mode)
    paths[-1].unlink()
    envelope = _aggregate(output, plan)
    assert envelope["scientific"]["audit_passed"] is False
    assert envelope["scientific"]["issues"]
    assert envelope["summary"] is None
    assert envelope["paper"] is None
    assert not (_generation(output, envelope)/"merged.json").exists()
    outputs = _read(_generation(output, envelope)/"receipt.json")["outputs"]
    assert outputs["merged"] is None
    assert outputs["selected_factors"] is None
    assert not (_generation(output, envelope)/"selected-factors.json").exists()


@pytest.mark.parametrize("mutation", ["factor", "seed", "manifest", "source",
                                      "manifest_path", "manifest_status"])
def test_real_numerical_and_identity_checks_reject_tampered_shards(tmp_path, mutation):
    _, output, plan, paths, _ = _prepare(tmp_path)
    path = paths[0]
    record = _read(path)
    if mutation == "factor":
        record["trajectories"][0]["factor_states"]["2"]["singular_values"][0] += 1.
    elif mutation == "seed":
        record["random_seed"] += 1
    elif mutation == "source":
        record["implementation_manifest"]["files"][0]["sha256"] = "b"*64
    else:
        manifest_path = path.parents[2]/"budget_study_manifest.json"
        manifest = _read(manifest_path)
        if mutation == "manifest_path":
            manifest["cells"][0]["path"] = "/unrelated/result.json"
        elif mutation == "manifest_status":
            manifest["cells"][0].update(status="all_candidates_failed", success=False)
        else:
            manifest["configuration"]["penalties_u"] = [.99]
        study.atomic_json(manifest, manifest_path)
    if not mutation.startswith("manifest"):
        study.atomic_json(record, path)
    envelope = _aggregate(output, plan)
    assert envelope["scientific"]["audit_passed"] is False
    assert envelope["scientific"]["issues"]
    assert envelope["summary"] is None


def test_case_lock_rejects_duplicate_case_but_not_an_unrelated_case(tmp_path):
    first_root, second_root, output = tmp_path/"first", tmp_path/"second", tmp_path/"analysis"
    first_plan, _ = _run(first_root)
    second_plan, _ = _run(second_root, split=False, record=fixtures.fixture(seed=1))
    cases.prepare_campaign([first_root, second_root], output)
    first_key, second_key = (plan["cells"][0]["task_id"] for plan in (first_plan, second_plan))
    case_root = output/"cases"/first_key
    case_root.mkdir(parents=True, exist_ok=True)
    with (case_root/".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="[Ll]ock|[Aa]lready"):
            cases.aggregate_case(output/"campaign-index.json", first_key, generate_data_fn=fixtures.generator)
        other = cases.aggregate_case(output/"campaign-index.json", second_key, generate_data_fn=fixtures.generator)
        assert other["scientific"]["audit_passed"] is True
    assert _aggregate(output, first_plan)["scientific"]["audit_passed"] is True


@pytest.mark.parametrize("markers, archive_only, fit_success, launch_success", [
    (dict(exit_code="74", launcher="74", archive="failed"), True, True, True),
    (dict(exit_code="0", launcher="74", archive="failed"), True, True, True),
    (dict(exit_code="0", launcher="74", archive="deferred"), False, True, False),
    (dict(exit_code="1", launcher="74", archive="failed"), False, True, False),
    (dict(process="1", exit_code="0", launcher="74", archive="failed"), False, False, False),
    (dict(exit_code="74", launcher="74", archive="deferred"), False, True, False),
    (dict(process="1", exit_code="74", launcher="74", archive="failed"), False, False, False),
    (dict(process=None), False, False, True),
    (dict(exit_code=None), False, True, False),
    (dict(launcher=None), False, True, False),
    (dict(process="malformed"), False, False, True),
    (dict(exit_code="74", launcher="74", archive=None), False, True, False),
])
def test_execution_classification_keeps_archive_only_separate_from_fit_failures(
        tmp_path, markers, archive_only, fit_success, launch_success):
    raw, output, plan, _, _ = _prepare(tmp_path)
    task = study.planned_tasks(plan)[0]
    _write_markers(raw, task["task_id"], **markers)
    envelope = _aggregate(output, plan)
    assert envelope["execution"]["archive_only_failures"] == ([task["task_id"]] if archive_only else [])
    assert envelope["execution"]["fit_success"] is fit_success
    assert envelope["execution"]["launch_success"] is launch_success
    # A saved numerical result remains independently audited even if execution
    # provenance disqualifies it from the primary results.
    assert envelope["scientific"]["audit_passed"] is True
    assert envelope["summary"]["verified_factor_states"] == 40


@pytest.mark.parametrize("artifact", ["audit", "merged", "receipt", "receipt_outputs"])
def test_corrupted_published_generation_is_not_silently_reused(tmp_path, artifact):
    _, output, plan, _, _ = _prepare(tmp_path)
    first = _aggregate(output, plan)
    directory = _generation(output, first)
    path = directory/("receipt.json" if artifact.startswith("receipt") else artifact+".json")
    if artifact == "receipt_outputs":
        receipt = _read(path)
        receipt["outputs"] = {}
        study.atomic_json(receipt, path)
        (directory/"merged.json").write_text('{"corrupted":true}\n')
    elif artifact == "receipt":
        receipt = _read(path)
        receipt["generation"] = "f"*64
        study.atomic_json(receipt, path)
    else:
        path.write_bytes(path.read_bytes()+b"\n")
    corrupted = _files(directory)
    with pytest.raises(ValueError, match="[Hh]ash|[Gg]eneration|[Rr]eceipt|[Oo]utput|[Aa]rtifact"):
        _aggregate(output, plan)
    assert _files(directory) == corrupted
    assert not (directory.parent.parent/"current.json").exists()


@pytest.mark.parametrize("publication_mode", ["full", "compact"])
def test_inapplicable_case_remains_explicit_without_any_fitted_trajectory(tmp_path, publication_mode):
    record = fixtures.fixture()
    setting = runner.experiment_settings(0, 2)[0]
    config = shards.CONFIG
    data = shards.summary._data(setting, record["random_seed"], config, {}, fixtures.generator)
    record.update(setting=asdict(setting), n_train=setting.n,
        configuration=fixtures._json_value(runner.resolved_configuration(setting, config)),
        generator_arguments=dict(n=setting.n, p=setting.p, q=setting.q, sigma0=setting.sigma0,
            sigma=.5, r_star=5, r0_star=10, random_seed=record["random_seed"]),
        status="inapplicable", success=False, applicable=False,
        failure_reason=setting.inapplicability_reason(), trajectories=[], selection_history=[],
        cap_outcomes=[], tuning_diagnostics={}, selected_budget=None, selected_candidate_id=None,
        selected_iteration=None, validation_loss=None, avg_err=None, fit_time_sec=0.)
    for key in ("training_observed_input_fingerprint", "validation_observed_input_fingerprint",
                "evaluation_truth_fingerprint", "validation_seed_metadata"):
        record[key] = data[key]
    record["input_fingerprint"] = fixtures._digest_json(dict(
        training_observed=record["training_observed_input_fingerprint"],
        validation_observed=record["validation_observed_input_fingerprint"],
        evaluation_truth=record["evaluation_truth_fingerprint"]))
    shards.schedule_fixture._refresh_identity(record)
    _, output, plan, _, _ = _prepare(tmp_path, split=False, record=record,
                                    publication_mode=publication_mode)
    envelope = _aggregate(output, plan)
    assert envelope["scientific"]["audit_passed"] is True
    assert envelope["scientific"]["status"] == "inapplicable"
    assert envelope["summary"]["applicable"] is False
    assert envelope["summary"]["caps"] == []
    assert envelope["paper"]["selected"] == []
    directory = _generation(output, envelope)
    if publication_mode == "full":
        assert _read(directory/"merged.json")["trajectories"] == []
        assert not (directory/"selected-factors.json").exists()
    else:
        assert not (directory/"merged.json").exists()
        selected = _read(directory/"selected-factors.json")
        assert selected["factors"] == {} and selected["selections"] == []


def test_preparation_rejects_overlapping_case_rosters_and_raw_output_paths(tmp_path):
    raw, duplicate = tmp_path/"raw", tmp_path/"duplicate"
    _run(raw)
    _run(duplicate)
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        cases.prepare_campaign([raw, duplicate], tmp_path/"analysis")
    with pytest.raises(ValueError, match="separate|raw"):
        cases.prepare_campaign([raw], raw/"analysis")


def test_frozen_work_table_change_is_rejected_before_publication(tmp_path):
    raw, output, plan, _, _ = _prepare(tmp_path)
    first = _aggregate(output, plan)
    previous = _generation(output, first)
    saved = _files(previous)
    table = raw/"work-items.tsv"
    table.write_bytes(table.read_bytes()+b"m0_e2_s99_k2_g0\t0\t2\t99\t2\n")
    with pytest.raises(ValueError, match="[Tt]able|[Pp]lan"):
        _aggregate(output, plan)
    assert _files(previous) == saved
    assert not (previous.parent.parent/"current.json").exists()


def test_changed_input_during_audit_withdraws_stale_pointer_and_keeps_old_generation(tmp_path):
    raw, output, plan, _, _ = _prepare(tmp_path)
    first = _aggregate(output, plan)
    previous = _generation(output, first)
    saved = _files(previous)
    task = study.planned_tasks(plan)[0]
    def mutate_input(**kwargs):
        result = fixtures.generator(**kwargs)
        (raw/"tasks"/task["task_id"]/"archive-status.txt").write_text("complete\n")
        return result
    with pytest.raises(ValueError, match="[Ii]nputs changed"):
        cases.aggregate_case(output/"campaign-index.json", first["case_key"], generate_data_fn=mutate_input)
    assert _files(previous) == saved
    assert not (previous.parent.parent/"current.json").exists()


def test_preparation_cli_needs_no_numerical_packages_and_does_not_read_fit_results(tmp_path):
    raw, output = tmp_path/"raw", tmp_path/"analysis"
    _, paths = _run(raw)
    # Preparation freezes plans, so even unreadable scientific JSON is irrelevant.
    paths[0].write_text("Not a scientific JSON record")
    program = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import aggregate_sparse_smart_cases as c; "
        "assert c.main(['prepare','--run-roots',sys.argv[2],'--output-root',sys.argv[3]]) == 0; "
        "assert not any(name in sys.modules for name in ('numpy','smart','sparse_smart','run_sparse_smart_budget_study'))"
    )
    result = subprocess.run([sys.executable, "-S", "-c", program, str(study.HERE), str(raw), str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    index = _read(output/"campaign-index.json")
    assert len(index["cases"]) == 1
    assert index["schema_version"] == 2
    assert index["publication_mode"] == "compact"
    assert not (output/"cases").exists()
    dry_program = (
        "import sys, subprocess; sys.path.insert(0, sys.argv[1]); "
        "subprocess.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError('No scheduler or child process')); "
        "import aggregate_sparse_smart_cases as c; "
        "assert c.main(['run','--index',sys.argv[2],'--workers','2','--dry-run']) == 0; "
        "assert not any(name in sys.modules for name in ('numpy','smart','sparse_smart','run_sparse_smart_budget_study'))"
    )
    before = _files(output)
    dry = subprocess.run([sys.executable, "-S", "-c", dry_program, str(study.HERE),
                          str(output/"campaign-index.json")], capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["no_audits_performed"] is True
    assert _files(output) == before


def test_process_pool_publishes_independent_missing_case_diagnostics(tmp_path, monkeypatch):
    raw, output = tmp_path/"raw", tmp_path/"analysis"
    plan = study.make_plan(models=(0,), experiments=(2,), seed_ids=(0, 1), setting_index=2,
        config=json.loads(json.dumps(asdict(shards.CONFIG))), tuning_task_size=2)
    study.write_plan(plan, raw)
    cases.prepare_campaign([raw], output)
    keys = [cell["task_id"] for cell in reversed(plan["cells"])]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    reports = cases.aggregate_cases(output/"campaign-index.json", workers=2, case_keys=keys)
    assert [report["case_key"] for report in reports] == keys
    for report in reports:
        assert report["scientific"]["audit_passed"] is False
        assert report["execution"]["fit_success"] is False
        assert report["execution"]["launch_success"] is False
        assert "summary" not in report and "paper" not in report
        assert (_generation(output, report)/"audit.json").is_file()
        assert not (_generation(output, report)/"merged.json").exists()


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_case_pool_rejects_invalid_worker_limits(tmp_path, workers):
    _, output, _, _, _ = _prepare(tmp_path)
    with pytest.raises(ValueError, match="[Ww]orkers"):
        cases.aggregate_cases(output/"campaign-index.json", workers=workers)
    assert not (output/"cases").exists()


@pytest.mark.parametrize("selection", ["duplicate", "unknown", "empty"])
def test_case_pool_rejects_invalid_or_duplicate_selection(tmp_path, selection):
    _, output, plan, _, _ = _prepare(tmp_path)
    key = plan["cells"][0]["task_id"]
    keys = [key, key] if selection == "duplicate" else ["m0_e2_s99_k2"] if selection == "unknown" else []
    with pytest.raises(ValueError, match="[Ss]election|[Dd]uplicate"):
        cases.aggregate_cases(output/"campaign-index.json", case_keys=keys)
    assert not (output/"cases").exists()


@pytest.mark.parametrize("failed_ids", [(), tuple(range(8))], ids=["complete", "retained-prefix"])
def test_compact_publication_keeps_full_audit_and_only_verified_selected_factors(
        tmp_path, monkeypatch, failed_ids):
    raw, full_output, plan, paths, full_index = _prepare(tmp_path, failed_ids=failed_ids)
    compact_output = tmp_path/"compact-analysis"
    compact_index = cases.prepare_campaign([raw], compact_output, publication_mode="compact")
    before = _files(raw)

    def forbidden(*args, **kwargs):
        raise AssertionError("Aggregation must not load or run an estimator")

    monkeypatch.setattr(runner, "run_setting", forbidden)
    monkeypatch.setattr(runner.external_runner.old_runner, "_load_sparse_api", forbidden)
    full, compact = _aggregate(full_output, plan), _aggregate(compact_output, plan)
    for key in ("identity", "configuration", "scientific", "summary", "execution"):
        assert compact[key] == full[key]
    assert compact["scientific"]["audit_passed"] is True
    assert compact["summary"]["verified_factor_states"] == (32 if failed_ids else 40)
    assert full_index["publication_mode"] == "full"
    assert compact_index["publication_mode"] == "compact"
    assert full_index["index_fingerprint"] != compact_index["index_fingerprint"]
    assert full["generation"] != compact["generation"]
    assert _files(raw) == before

    directory = _generation(compact_output, compact)
    assert not (directory/"merged.json").exists()
    selected = _read(directory/"selected-factors.json")
    receipt = _read(directory/"receipt.json")
    assert compact["schema_version"] == receipt["schema_version"] == 2
    assert compact["publication_mode"] == receipt["publication_mode"] == "compact"
    assert set(receipt["outputs"]) == {"audit", "merged", "selected_factors"}
    assert receipt["outputs"]["merged"] is None
    assert receipt["outputs"]["selected_factors"] == dict(path="selected-factors.json",
        sha256=hashlib.sha256((directory/"selected-factors.json").read_bytes()).hexdigest())
    assert selected["schema_version"] == 1
    assert selected["method"] == "SparseSMARTSelectedFactors"
    for key in ("case_key", "generation", "index_fingerprint", "plan_fingerprint", "identity"):
        assert selected[key] == compact[key]
    assert selected["no_fits_performed"] is True
    assert isinstance(selected["factor_encoding"], str) and selected["factor_encoding"]

    full_record = _read(_generation(full_output, full)/"merged.json")
    expected_keys = {f"g{cap['winner_grid_candidate_id']}_i{cap['selected_factor_key']}"
                     for cap in full_record["cap_outcomes"] if cap["success"]}
    assert set(selected["factors"]) == expected_keys
    assert [row["iteration_budget"] for row in selected["selections"]] == list(shards.CONFIG.iteration_budgets)
    data = fixtures.generated(0)
    for choice, cap, paper in zip(selected["selections"], full_record["cap_outcomes"],
                                  compact["paper"]["selected"], strict=True):
        assert choice["success"] == cap["success"]
        assert choice["coverage_complete"] == cap["coverage_complete"]
        factor_id = choice["factor_id"]
        assert paper["selected_factor_id"] == factor_id
        exported = selected["factors"][factor_id]
        assert exported["grid_candidate_id"] == cap["winner_grid_candidate_id"]
        assert exported["factor_key"] == cap["selected_factor_key"]
        assert exported["selected_iteration"] == cap["selected_iteration"]
        assert exported["params"] == paper["params"]
        source = exported["source"]
        assert source["run_root"] == str(raw)
        source_path = raw/source["source_relative"]
        assert source_path in paths
        assert source["sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert source_path.is_relative_to(raw/"tasks"/source["task_id"])
        original = _read(source_path)["trajectories"][source["local_grid_candidate_id"]]
        assert original["params"] == exported["params"]
        assert exported["state"] == original["factor_states"][exported["factor_key"]]
        state = exported["state"]
        assert state["iteration"] == exported["selected_iteration"]
        prediction = runner._factor_prediction(state, data["X"])
        assert float(np.mean((prediction-data["Y"])**2)) == pytest.approx(cap["validation_mse"])
        coefficient = (np.asarray(state["left"])*state["singular_values"]) @ np.asarray(state["right"]).T
        error = np.linalg.norm(coefficient-data["truth"])/np.sqrt(coefficient.size)
        assert error == pytest.approx(cap["coefficient_error"])
    if failed_ids:
        assert len(selected["factors"]) == 1
        assert len(selected["selections"]) == 2
        assert compact["scientific"]["budget_coverage_complete"] is False


def test_compact_export_retains_initializer_winner_once_across_all_budgets(tmp_path):
    raw, output, plan, _, _ = _prepare(tmp_path, split=False, record=fixtures.fixture(tied=True),
                                      publication_mode="compact")
    before = _files(raw)
    audit = _aggregate(output, plan)
    assert audit["scientific"]["audit_passed"] is True
    selected = _read(_generation(output, audit)/"selected-factors.json")
    assert list(selected["factors"]) == ["g0_i0"]
    assert all(choice["factor_id"] == "g0_i0" for choice in selected["selections"])
    assert selected["factors"]["g0_i0"]["selected_iteration"] == 0
    assert selected["factors"]["g0_i0"]["state"]["iteration"] == 0
    assert _files(raw) == before


def test_compact_all_candidates_failed_is_audited_without_exporting_a_winner(tmp_path):
    record = shards._initial_failures(shards._synthetic(), tuple(range(8)))
    _, output, plan, _, _ = _prepare(tmp_path, split=False, record=record, publication_mode="compact")
    audit = _aggregate(output, plan)
    assert audit["scientific"]["audit_passed"] is True
    assert audit["scientific"]["status"] == "all_candidates_failed"
    selected = _read(_generation(output, audit)/"selected-factors.json")
    assert selected["factors"] == {}
    assert len(selected["selections"]) == len(shards.CONFIG.iteration_budgets)
    assert all(not choice["success"] and choice["factor_id"] is None
               for choice in selected["selections"])


@pytest.mark.parametrize("corruption", ["changed_factor", "missing_file", "wrong_output_roster"])
def test_compact_generation_corruption_withdraws_current_without_rewriting_history(tmp_path, corruption):
    _, output, plan, _, _ = _prepare(tmp_path, publication_mode="compact")
    first = _aggregate(output, plan)
    directory = _generation(output, first)
    path = directory/"selected-factors.json"
    if corruption == "changed_factor":
        selected = _read(path)
        next(iter(selected["factors"].values()))["state"]["singular_values"][0] += 1.
        study.atomic_json(selected, path)
    elif corruption == "missing_file":
        path.unlink()
    else:
        receipt = _read(directory/"receipt.json")
        receipt["outputs"]["selected_factors"] = None
        study.atomic_json(receipt, directory/"receipt.json")
    before = _files(directory)
    with pytest.raises((ValueError, FileNotFoundError)):
        _aggregate(output, plan)
    assert not (directory.parent.parent/"current.json").exists()
    assert _files(directory) == before


@pytest.mark.parametrize("publication_mode", ["full", "compact"])
def test_fresh_case_reads_each_scientific_input_twice_including_final_rehash(
        tmp_path, monkeypatch, publication_mode):
    _, output, plan, paths, _ = _prepare(tmp_path, publication_mode=publication_mode)
    reads = {path: 0 for path in paths}
    original_open = Path.open

    def track_open(path, mode="r", *args, **kwargs):
        if path in reads and "r" in mode:
            reads[path] += 1
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)
    audit = _aggregate(output, plan)
    assert audit["scientific"]["audit_passed"] is True
    assert set(reads.values()) == {2}


def test_fused_reads_still_reject_changed_scientific_bytes_during_audit(tmp_path):
    _, output, plan, paths, _ = _prepare(tmp_path, publication_mode="compact")
    first = _aggregate(output, plan)
    saved = _files(_generation(output, first))

    def mutate_input(**kwargs):
        generated = fixtures.generator(**kwargs)
        record = _read(paths[0])
        record["elapsed_time_sec"] += 1.
        study.atomic_json(record, paths[0])
        return generated

    with pytest.raises(ValueError, match="[Ii]nputs changed"):
        cases.aggregate_case(output/"campaign-index.json", first["case_key"], generate_data_fn=mutate_input)
    assert _files(_generation(output, first)) == saved
    assert not (_generation(output, first).parent.parent/"current.json").exists()


@pytest.mark.parametrize("models", [(1,), (2, 0)])
def test_scoped_preparation_selects_models_from_mixed_producer_plans_without_reading_fits(tmp_path, models):
    raw, output = tmp_path/"mixed-raw", tmp_path/"scoped-analysis"
    plan = study.make_plan(models=(0, 1, 2), experiments=(0,), seed_ids=(0, 1),
        config=json.loads(json.dumps(asdict(shards.CONFIG))), tuning_task_size=2)
    study.write_plan(plan, raw)
    before = _files(raw)
    index = cases.prepare_campaign([raw], output, models=models, publication_mode="compact")
    assert index["schema_version"] == 2
    assert index["scope"] == {"models": sorted(models)}
    assert index["publication_mode"] == "compact"
    expected = {cell["task_id"] for cell in plan["cells"] if cell["model"] in models}
    assert {entry["case_key"] for entry in index["cases"]} == expected
    assert all(entry["plan_fingerprint"] == plan["plan_fingerprint"] for entry in index["cases"])
    assert cases.load_index(output/"campaign-index.json") == index
    assert _files(raw) == before
    assert not (output/"cases").exists()
    assert cases.prepare_campaign([raw], output, models=models, publication_mode="compact") == index


@pytest.mark.parametrize("models", [[], [1, 1], [-1], [3], [True], [1.0]])
def test_preparation_rejects_invalid_model_scope(tmp_path, models):
    raw, output = tmp_path/"raw", tmp_path/"analysis"
    _run(raw)
    with pytest.raises(ValueError):
        cases.prepare_campaign([raw], output, models=models)
    assert not (output/"campaign-index.json").exists()


@pytest.mark.parametrize("models", [(1,), (0, 1)])
def test_preparation_rejects_scope_with_no_cases_for_any_requested_model(tmp_path, models):
    raw, output = tmp_path/"raw", tmp_path/"analysis"
    _run(raw)
    with pytest.raises(ValueError):
        cases.prepare_campaign([raw], output, models=models)
    assert not (output/"campaign-index.json").exists()


@pytest.mark.parametrize("mutation", ["scope_excludes_case", "duplicate_scope", "invalid_mode"])
def test_index_loader_rejects_rehashed_invalid_scope_or_publication_mode(tmp_path, mutation):
    _, output, _, _, index = _prepare(tmp_path)
    if mutation == "scope_excludes_case":
        index["scope"] = {"models": [1]}
    elif mutation == "duplicate_scope":
        index["scope"] = {"models": [0, 0]}
    else:
        index["publication_mode"] = "unknown"
    index["index_fingerprint"] = study.digest({key: value for key, value in index.items()
                                               if key != "index_fingerprint"})
    study.atomic_json(index, output/"campaign-index.json")
    with pytest.raises(ValueError):
        cases.load_index(output/"campaign-index.json")


def test_publication_mode_is_frozen_and_legacy_index_remains_readable(tmp_path):
    raw, output, _, _, index = _prepare(tmp_path)
    before = (output/"campaign-index.json").read_bytes()
    with pytest.raises(ValueError):
        cases.prepare_campaign([raw], output, publication_mode="compact")
    assert (output/"campaign-index.json").read_bytes() == before
    legacy = {key: value for key, value in index.items()
              if key not in ("scope", "publication_mode", "index_fingerprint")}
    legacy["schema_version"] = 1
    legacy["index_fingerprint"] = study.digest(legacy)
    legacy_path = tmp_path/"legacy-index.json"
    study.atomic_json(legacy, legacy_path)
    assert cases.load_index(legacy_path) == legacy


def test_model_two_cli_scope_defaults_to_compact_without_numerical_imports(tmp_path):
    raw, output = tmp_path/"mixed-raw", tmp_path/"model-two-analysis"
    plan = study.make_plan(models=(0, 1, 2), experiments=(0,), seed_ids=(0,),
        config=json.loads(json.dumps(asdict(shards.CONFIG))), tuning_task_size=2)
    study.write_plan(plan, raw)
    program = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import aggregate_sparse_smart_cases as c; "
        "assert c.main(['prepare','--run-roots',sys.argv[2],'--output-root',sys.argv[3],"
        "'--models','1']) == 0; "
        "assert not any(name in sys.modules for name in ('numpy','smart','sparse_smart','run_sparse_smart_budget_study'))"
    )
    result = subprocess.run([sys.executable, "-S", "-c", program, str(study.HERE), str(raw), str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    index = _read(output/"campaign-index.json")
    assert index["scope"] == {"models": [1]}
    assert index["publication_mode"] == "compact"
    assert {entry["case_key"] for entry in index["cases"]} == {
        cell["task_id"] for cell in plan["cells"] if cell["model"] == 1}
    assert not (output/"cases").exists()


def test_legacy_index_publishes_and_reuses_original_full_output_contract(tmp_path):
    _, output, plan, _, index = _prepare(tmp_path)
    legacy = {key: value for key, value in index.items()
              if key not in ("scope", "publication_mode", "index_fingerprint")}
    legacy["schema_version"] = 1
    legacy["index_fingerprint"] = study.digest(legacy)
    study.atomic_json(legacy, output/"campaign-index.json")
    first = _aggregate(output, plan)
    directory = _generation(output, first)
    receipt = _read(directory/"receipt.json")
    assert first["schema_version"] == receipt["schema_version"] == 1
    assert set(receipt["outputs"]) == {"audit", "merged"}
    assert (directory/"merged.json").is_file()
    assert not (directory/"selected-factors.json").exists()
    assert first["scientific"]["audit_passed"] is True
    second = _aggregate(output, plan)
    assert second["resumed"] is True
    assert second["generation"] == first["generation"]


def test_index_source_must_match_frozen_plan_before_collecting_results(tmp_path, monkeypatch):
    _, output, plan, _, index = _prepare(tmp_path)
    index["cases"][0]["source"]["implementation"]["true"]["fingerprint"] = "0"*64
    index["index_fingerprint"] = study.digest({k: v for k, v in index.items() if k != "index_fingerprint"})
    study.atomic_json(index, output/"campaign-index.json")
    def forbidden(*args, **kwargs):
        raise AssertionError("Incorrect planned source must fail before collecting raw scientific files")
    monkeypatch.setattr(cases, "_collect", forbidden)
    with pytest.raises(ValueError, match="source differs"):
        _aggregate(output, plan)


@pytest.mark.parametrize("field,value", [("scope", {"models": [0]}), ("publication_mode", "compact")])
def test_legacy_index_cannot_masquerade_as_new_publication_contract(tmp_path, field, value):
    _, output, _, _, index = _prepare(tmp_path)
    legacy = {k: v for k, v in index.items() if k not in ("scope", "publication_mode", "index_fingerprint")}
    legacy.update(schema_version=1, **{field: value})
    legacy["index_fingerprint"] = study.digest(legacy)
    study.atomic_json(legacy, output/"campaign-index.json")
    with pytest.raises(ValueError, match="Legacy"):
        cases.load_index(output/"campaign-index.json")


@pytest.mark.parametrize("artifact", ["audit", "receipt"])
def test_legacy_publication_rejects_extra_mode_even_after_rehash(tmp_path, artifact):
    _, output, plan, _, index = _prepare(tmp_path)
    legacy = {k: v for k, v in index.items() if k not in ("scope", "publication_mode", "index_fingerprint")}
    legacy["schema_version"] = 1
    legacy["index_fingerprint"] = study.digest(legacy)
    study.atomic_json(legacy, output/"campaign-index.json")
    first = _aggregate(output, plan)
    directory = _generation(output, first)
    receipt = _read(directory/"receipt.json")
    if artifact == "audit":
        audit = _read(directory/"audit.json")
        audit["publication_mode"] = "compact"
        study.atomic_json(audit, directory/"audit.json")
        receipt["outputs"]["audit"]["sha256"] = cases._sha(directory/"audit.json")
    else:
        receipt["publication_mode"] = "compact"
    study.atomic_json(receipt, directory/"receipt.json")
    study.atomic_json(dict(generation=first["generation"], receipt_sha256=cases._sha(directory/"receipt.json")),
                      directory.parent.parent/"current.json")
    with pytest.raises(ValueError, match="Legacy"):
        _aggregate(output, plan)
