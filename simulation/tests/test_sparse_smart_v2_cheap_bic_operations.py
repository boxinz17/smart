"""Small numerical and fake-scheduler tests; no campaign or SSH operations."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM.parent / "sparse-smart-v2/src"))
import sparse_smart_v2_cheap_bic_reuse as reuse
from sparse_smart_v2_bic_plan import digest, sha
from run_sparse_smart_v2_pilot import _fingerprints, _metrics
from sparse_smart_v2.rrr import target_rrr
from sparse_smart_v2.selection import bic_score


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.fixture
def rrr_case(tmp_path):
    rng = np.random.default_rng(17)
    x, y = rng.normal(size=(8, 3)), rng.normal(size=(8, 2))
    data = dict(X=x, Y=y, C0=rng.normal(size=(3, 2)), C_star=np.zeros((3, 2)),
                X_validation=x.copy(), Y_validation=y.copy())
    endpoint = target_rrr(x, y, 1)
    old_root = tmp_path / "old"
    folder = old_root / "tasks/000000"
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "states.npz", c0000_coefficient=endpoint.coefficient)
    task = dict(task_id=0, group_id="g", rank=1, include_rrr=True)
    row = dict(candidate_id="t0_rrr_u0_v0", state_key="c0000_", fit_method="target_rrr",
               success=True, execution_success=True, n_iter=0, selected_iteration=0,
               optimization_converged=True, fit_status="converged", termination_reason="target_rrr_closed_form",
               rrr_certificate=endpoint.certificate, init_penalty=.003, penalty_u=0., penalty_v=0.,
               rank=1, free_directions=[3, 2], elapsed_seconds=.25,
               selection=bic_score(x, y, endpoint.coefficient, rank=1, direct_rrr=True).as_dict(),
               metrics=_metrics(endpoint.coefficient, data))
    plan = dict(plan_fingerprint="original", tasks=[task])
    result = dict(task=task, status="complete", execution_success=True, plan_fingerprint="original",
                  fingerprints=_fingerprints(data), files={"states.npz": sha(folder / "states.npz")}, outcomes=[row])
    put(folder / "result.json", result)
    put(folder / "status.json", dict(status="finished", result_sha256=sha(folder / "result.json")))
    manifest = dict(files={"sparse-smart-v2/src/sparse_smart_v2/rrr.py": "same",
                           "sparse-smart-v2/src/sparse_smart_v2/selection.py": "same"})
    put(old_root / "source-manifest.json", manifest)
    return old_root, plan, manifest, data, endpoint


def test_rrr_reuse_preserves_coefficient_score_and_records_origin(rrr_case):
    root, plan, manifest, data, endpoint = rrr_case
    descriptor, reason = reuse.rrr_descriptor(root, plan, {"group_id": "g"}, 1, manifest)
    assert reason == "reusable_rrr"
    row, arrays, history = reuse.reuse_rrr(descriptor, {"p": 3, "q": 2}, {"rank": 1, "init_penalty": .1},
                                         {"support_tolerance": 0.}, data, "new_id")
    assert row["candidate_id"] == "new_id" and row["init_penalty"] == .1
    assert row["reuse_provenance"]["originating_plan_fingerprint"] == "original"
    assert row["original_elapsed_seconds"] == .25 and history is None
    np.testing.assert_array_equal(arrays["coefficient"], endpoint.coefficient)
    assert row["selection"]["score"] == pytest.approx(bic_score(data["X"], data["Y"], endpoint.coefficient, rank=1, direct_rrr=True).score)


def test_rrr_missing_rank_is_explicit_not_rounded(rrr_case):
    root, plan, manifest, *_ = rrr_case
    assert reuse.rrr_descriptor(root, plan, {"group_id": "g"}, 2, manifest) == (None, "estimated_rank_absent_from_existing_grid")


def test_rrr_rejects_changed_training_or_saved_state(rrr_case):
    root, plan, manifest, data, _ = rrr_case
    descriptor, _ = reuse.rrr_descriptor(root, plan, {"group_id": "g"}, 1, manifest)
    changed = dict(data, X=data["X"] + 1)
    with pytest.raises(ValueError, match="data mismatch"):
        reuse.reuse_rrr(descriptor, {"p": 3, "q": 2}, {"rank": 1}, {"support_tolerance": 0.}, changed, "new")
    Path(descriptor["states_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="arrays changed"):
        reuse.reuse_rrr(descriptor, {"p": 3, "q": 2}, {"rank": 1}, {"support_tolerance": 0.}, data, "new")


def test_assessment_gate_requires_frozen_training_only_pair(tmp_path):
    assessment = tmp_path / "assessment"
    assessment.mkdir()
    plan = dict(plan_fingerprint="assessment", configuration={"cheap_bic_phase": "assessment"},
                cheap_bic={"split_fingerprint": "split"})
    with pytest.raises(FileNotFoundError):
        reuse.verify_assessment_gate(assessment, plan)
    pair = dict(phase="develop", evaluation_metrics_used=False, iterations=200,
                development_plan_fingerprint="development", source_files_fingerprint=digest({"file": "hash"}))
    pair["artifact_fingerprint"] = digest(pair)
    pair_path = tmp_path / "development/selection/selected-pair.json"
    put(pair_path, pair)
    put(tmp_path / "development/plan.json", dict(plan_fingerprint="development", cheap_bic={"split_fingerprint": "split"}))
    put(assessment / "source-manifest.json", dict(files={"file": "hash"}))
    gate = dict(pair_path=str(pair_path), pair_sha256=sha(pair_path), assessment_plan_fingerprint="assessment",
                status="released_after_pair_freeze", development_plan_fingerprint="development")
    put(assessment / "assessment-gate.json", gate)
    assert reuse.verify_assessment_gate(assessment, plan) == gate
    pair_path.write_text("{}")
    with pytest.raises(ValueError, match="pair changed"):
        reuse.verify_assessment_gate(assessment, plan)


@pytest.fixture
def launcher(tmp_path):
    path = SIM.parent / "hpc/discovery/submit_sparse_smart_v2_cheap_bic.py"
    spec = importlib.util.spec_from_file_location("_tested_cheap_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "pilot"
    (root / "campaign/attempts").mkdir(parents=True)
    put(root / "source-manifest.json", {"files": {}})
    put(root / "pilot.json", dict(accepted_update_cap=200, workers=100, account="mkolar_1314", partition="main",
                                  pool_time="12:00:00",
                                  source_manifest_sha256=sha(root / "source-manifest.json")))
    return module, root


def test_stage_dry_run_never_submits_and_has_exact_resources(launcher, monkeypatch):
    module, root = launcher
    monkeypatch.setattr(module.core.subprocess, "run", lambda *a, **k: pytest.fail("scheduler called"))
    record = module.submit_stage(root, "develop", ["101", "102"], dry_run=True)
    assert "--dependency=afterany:101:102" in record["command"]
    assert "--ntasks=1" in record["command"] and "--cpus-per-task=1" in record["command"]
    assert not any("array" in arg for arg in record["command"])
    assert not (root / "campaign/attempts/develop-attempt-0001/submission.json").exists()


def test_stage_refuses_uncertain_submission_without_retry(launcher, monkeypatch):
    module, root = launcher
    put(root / "campaign/attempts/prepare-attempt-0001/submission.json",
        dict(status="submission_uncertain", pilot_sha256=sha(root / "pilot.json"),
             source_manifest_sha256=sha(root / "source-manifest.json")))
    monkeypatch.setattr(module.core.subprocess, "run", lambda *a, **k: pytest.fail("scheduler repeated"))
    with pytest.raises(ValueError, match="uncertain stage submission"):
        module.submit_stage(root, "prepare")


@pytest.mark.parametrize("workers", [0, 99, 101, 100.0, True])
def test_launcher_rejects_any_worker_setting_other_than_integer_100(launcher, workers):
    module, root = launcher
    settings = module.core.read(root / "pilot.json")
    put(root / "pilot.json", dict(settings, workers=workers))
    with pytest.raises(ValueError, match="exactly 100 workers"):
        module.settings(root)


def test_completed_pool_submits_only_missing_followup_without_refitting(launcher, monkeypatch):
    from types import SimpleNamespace

    module, root = launcher
    phase = root / "development"
    groups = [dict(group_id=f"g{i}") for i in range(165)]
    tasks = [dict(task_id=i, group_id=f"g{i // 6}") for i in range(990)]
    plan = dict(n_groups=165, n_tasks=990, groups=groups, tasks=tasks, configuration={"iterations": 200})
    put(phase / "plan.json", plan)
    put(phase / "source-manifest.json", {"files": {}})
    (phase / "campaign/attempts").mkdir(parents=True)
    chunks = module.bic.chunk_plan(phase, plan, 990)
    tag = "chunk-0000-attempt-0001"
    filename = "campaign-" + tag + ".tsv"
    (phase / filename).write_text("".join(f"{i}\n" for i in range(990)))
    put(phase / "campaign/attempts" / tag / "submission.json", dict(
        kind="pool", status="submitted", job_id="123", tag=tag, attempt=1, chunk_id=0,
        plan_sha256=sha(phase / "plan.json"), source_manifest_sha256=sha(phase / "source-manifest.json"),
        chunk_plan_sha256=sha(phase / "campaign/chunks.json"), task_file=filename,
        task_file_sha256=sha(phase / filename), task_ids=list(range(990))))
    checked, scheduler_checks, submissions = [], [], []
    monkeypatch.setattr(module.bic, "load_inputs", lambda actual: plan)
    monkeypatch.setattr(module.bic, "ready", lambda actual, actual_plan: True)

    def completed_task(actual, actual_plan, task_id):
        assert actual == phase and actual_plan is plan
        checked.append(task_id)
        return True

    def scheduler_state(job_id):
        scheduler_checks.append(job_id)
        return "terminal"

    def submit(command, **kwargs):
        submissions.append(command)
        return SimpleNamespace(returncode=0, stdout="456\n", stderr="")

    monkeypatch.setattr(module.bic, "completed_task", completed_task)
    monkeypatch.setattr(module.bic.core, "scheduler_state", scheduler_state)
    monkeypatch.setattr(module.core.subprocess, "run", submit)
    result = module.launch_phase(root, "development")
    assert checked == list(range(990)) and scheduler_checks == ["123"]
    assert len(submissions) == 1 and "--job-name=sv2cheap-develop" in submissions[0]
    assert "--ntasks=1" in submissions[0]
    assert not any(arg.startswith("--dependency=") for arg in submissions[0])
    assert result["fits"]["actions"][-1]["action"] == "already_finished"
    assert result["followup"]["job_id"] == "456"
    assert len(chunks["chunks"]) == 1
    # The recovered stage ledger also prevents a duplicate follow-up.
    again = module.launch_phase(root, "development")
    assert again["followup"] == result["followup"]
    assert len(submissions) == 1


def test_completed_pool_without_ledger_cannot_release_followup(launcher, monkeypatch):
    module, root = launcher
    phase = root / "development"
    put(phase / "plan.json", dict(n_groups=165, n_tasks=990, configuration={"iterations": 200}))
    monkeypatch.setattr(module.bic, "orchestrate", lambda *a: dict(actions=[
        dict(kind="pool", chunk_id=0, action="already_finished", n_tasks=0)]))
    monkeypatch.setattr(module.core.subprocess, "run", lambda *a, **k: pytest.fail("scheduler called"))
    with pytest.raises(ValueError, match="no submission ledger"):
        module.launch_phase(root, "development")


def scheduler_variables():
    return {name: value for name, value in os.environ.items()
            if name.startswith(("SLURM_", "SBATCH_", "SRUN_"))}


@pytest.fixture
def inherited_scheduler_environment(monkeypatch):
    for name, value in {"SLURM_JOB_ID": "parent-preparation", "SLURM_MEM_PER_NODE": "16384",
                        "SLURM_MEM_PER_CPU": "4096", "SLURM_MEM_PER_GPU": "8192",
                        "SBATCH_CPUS_PER_TASK": "4", "SRUN_CPU_BIND": "cores"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CHEAP_BIC_UNRELATED_ENV", "retained")
    return scheduler_variables()


@pytest.mark.parametrize("fail", [False, True])
def test_stage_submission_clears_scheduler_environment_and_restores_on_error(
        launcher, monkeypatch, inherited_scheduler_environment, fail):
    from types import SimpleNamespace

    module, root = launcher

    def submit(command, **kwargs):
        assert not scheduler_variables()
        assert os.environ["CHEAP_BIC_UNRELATED_ENV"] == "retained"
        assert "--mem=16G" in command and "--cpus-per-task=1" in command
        if fail:
            raise OSError("fixture submission failure")
        return SimpleNamespace(returncode=0, stdout="789\n", stderr="")

    monkeypatch.setattr(module.core.subprocess, "run", submit)
    if fail:
        with pytest.raises(OSError, match="fixture submission failure"):
            module.submit_stage(root, "prepare")
    else:
        record = module.submit_stage(root, "prepare")
        assert "clear inherited SLURM_" in record["scheduler_environment_policy"]
    assert scheduler_variables() == inherited_scheduler_environment


@pytest.mark.parametrize("fail", [False, True])
def test_pool_orchestration_clears_scheduler_environment_and_restores_on_error(
        launcher, monkeypatch, inherited_scheduler_environment, fail):
    from types import SimpleNamespace

    module, root = launcher
    put(root / "development/plan.json", dict(n_groups=165, n_tasks=990, configuration={"iterations": 200}))
    calls = []

    def orchestrate(phase, args):
        assert not scheduler_variables()
        assert os.environ["CHEAP_BIC_UNRELATED_ENV"] == "retained"
        assert args.workers == 100 and args.mem == "4G"
        calls.append("pool")
        if fail:
            raise ValueError("fixture orchestration failure")
        return dict(actions=[dict(kind="pool", job_id="123")])

    def submit(command, **kwargs):
        assert not scheduler_variables()
        calls.append("followup")
        return SimpleNamespace(returncode=0, stdout="456\n", stderr="")

    monkeypatch.setattr(module.bic, "orchestrate", orchestrate)
    monkeypatch.setattr(module.core.subprocess, "run", submit)
    if fail:
        with pytest.raises(ValueError, match="fixture orchestration failure"):
            module.launch_phase(root, "development")
    else:
        module.launch_phase(root, "development")
    assert calls == (["pool"] if fail else ["pool", "followup"])
    assert scheduler_variables() == inherited_scheduler_environment


def test_scheduler_environment_restores_nested_scopes_and_exceptions(launcher, inherited_scheduler_environment):
    module, _ = launcher
    with pytest.raises(RuntimeError, match="nested fixture"):
        with module.clean_scheduler_environment():
            assert not scheduler_variables()
            os.environ["SLURM_JOB_ID"] = "temporary-inner-parent"
            with module.clean_scheduler_environment():
                assert not scheduler_variables()
                os.environ["SRUN_CPUS_PER_TASK"] = "temporary-inner-value"
            assert scheduler_variables() == {"SLURM_JOB_ID": "temporary-inner-parent"}
            raise RuntimeError("nested fixture")
    assert scheduler_variables() == inherited_scheduler_environment
    assert os.environ["CHEAP_BIC_UNRELATED_ENV"] == "retained"
