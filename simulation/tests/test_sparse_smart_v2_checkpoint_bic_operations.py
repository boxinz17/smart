"""Frozen snapshots and fake-scheduler tests; no SSH or production simulations."""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[2]
SIM, HPC = CODE / "simulation", CODE / "hpc/discovery"
sys.path.insert(0, str(SIM))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


launch = module("_checkpoint_launch_tests", HPC / "submit_sparse_smart_v2_checkpoint_bic.py")
builder = module("_checkpoint_builder_tests", HPC / "build_sparse_smart_v2_checkpoint_bic_snapshot.py")
stages = module("_checkpoint_stage_tests", SIM / "run_sparse_smart_v2_checkpoint_bic_campaign.py")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(launch.core.canonical(value))


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "literal space $dollar {braces}"
    source = root / "source/hpc/discovery"
    source.mkdir(parents=True)
    for name in ("sparse_smart_v2_bic_prepare.sbatch", "sparse_smart_v2_bic_pool.sbatch",
                 "sparse_smart_v2_bic_summary.sbatch", "sparse_smart_v2_bic_worker.sh",
                 "sparse_smart_v2_checkpoint_bic_stage.sbatch"):
        shutil.copyfile(HPC / name, source / name)
    put(root / "source-manifest.json", dict(schema_version=1, source_root=str(root / "source"),
        files={str(p.relative_to(root / "source")): launch.core.sha(p) for p in source.iterdir()}))
    config = dict(remote_root=str(root), accepted_update_cap=200, initializer_pair=[.003, .1],
        tied_penalties=[.001, .0025, .01], seed_ids=list(range(100)), models=[0, 1, 2],
        workers=100, max_tasks_per_chunk=2, pool_time="06:00:00", stage_time="06:00:00",
        mem_per_cpu="4G", stage_mem="16G", account="mkolar_1314", partition="main",
        source_manifest_sha256=launch.core.sha(root / "source-manifest.json"))
    put(root / "campaign.json", config)
    plan = dict(schema_version=1, method="SparseSMARTv2BIC", root=str(root),
        source_manifest_sha256=config["source_manifest_sha256"], n_groups=3, n_tasks=6,
        groups=[dict(group_id=g) for g in "abc"],
        tasks=[dict(task_id=i, group_id="abc"[i // 2]) for i in range(6)], canary_task_ids=[0, 4],
        configuration=dict(iterations=200, selection_rule="bic_checkpoint"))
    plan["plan_fingerprint"] = stages.fitting.digest(plan)
    put(root / "plan.json", plan)
    (root / "work-items.tsv").write_text("".join(f"{i}\n" for i in range(6)))
    put(root / "preparation.json", dict(status="complete", success=True,
        plan_sha256=launch.core.sha(root / "plan.json"), source_manifest_sha256=config["source_manifest_sha256"],
        n_groups=3, n_tasks=6))
    return root, plan


def fake_scheduler(monkeypatch, root, *, ambiguous=False):
    calls = []
    def run(command, **kwargs):
        assert command[0] == "sbatch"
        assert not any(k.startswith(("SLURM_", "SBATCH_", "SRUN_")) for k in os.environ)
        # Submission intent exists before the side effect.
        intentions = [launch.core.read(p) for p in (root / "campaign/attempts").glob("*/submission.json")]
        assert any(r["status"] == "submitting" and r["command"] == command for r in intentions)
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="ambiguous" if ambiguous else str(9000 + len(calls)), stderr="")
    monkeypatch.setattr(launch.core.subprocess, "run", run)
    return calls


def passed_canary(root, plan):
    for tid in plan["canary_task_ids"]:
        folder = root / "tasks" / f"{tid:06d}"
        result = dict(task=plan["tasks"][tid], group_id=plan["tasks"][tid]["group_id"],
            plan_fingerprint=plan["plan_fingerprint"], source_manifest_sha256=plan["source_manifest_sha256"],
            execution_success=True, status="complete", files={})
        put(folder / "result.json", result)
        put(folder / "status.json", dict(result, status="finished", result_sha256=launch.core.sha(folder / "result.json")))
        for name in ("process-exit-code.txt", "launcher-exit-code.txt"):
            (folder / name).write_text("0\n")
    put(root / "canary-audit.json", dict(success=True, task_ids=plan["canary_task_ids"],
        plan_fingerprint=plan["plan_fingerprint"], source_manifest_sha256=plan["source_manifest_sha256"]))


def test_canary_and_production_partition_is_exact_and_production_is_serial(campaign, monkeypatch):
    root, plan = campaign
    calls = fake_scheduler(monkeypatch, root)
    for name in ("SLURM_NTASKS", "SBATCH_MEM_PER_CPU", "SRUN_CPUS_PER_TASK"):
        monkeypatch.setenv(name, "inherited")
    result = launch.launch_phase(root, "canary")
    assert len(calls) == 2
    assert result["actions"][0]["task_ids"] == [0, 4]
    assert "--ntasks=100" in calls[0] and "--mem-per-cpu=4G" in calls[0] and "--time=06:00:00" in calls[0]
    assert "--dependency=afterany:9001" in calls[1]
    assert "--ntasks=1" in calls[1] and "--mem=16G" in calls[1]
    assert os.environ["SLURM_NTASKS"] == "inherited"
    with pytest.raises(FileNotFoundError):
        launch.launch_phase(root, "production")
    assert len(calls) == 2
    passed_canary(root, plan)
    production = launch.launch_phase(root, "production")
    assert [r["task_ids"] for r in production["actions"]] == [[1], [2, 3], [5]]
    assert not any(x.startswith("--dependency") for x in calls[2])
    assert "--dependency=afterany:9003" in calls[3]
    assert "--dependency=afterany:9004" in calls[4]
    assert "--dependency=afterany:9005" in calls[5]
    assert all(not any(x.startswith(("--array", "--nodes", "--ntasks-per-node")) for x in cmd) for cmd in calls)
    all_ids = [t for c in launch.core.read(root / "campaign/chunks.json")["chunks"] for t in c["task_ids"]]
    assert sorted(all_ids) == list(range(6))
    # Reentering a completed or interrupted orchestrator reuses exact ledger
    # identities rather than submitting the same tasks again.
    launch.launch_phase(root, "canary")
    launch.launch_phase(root, "production")
    assert len(calls) == 6


def test_ambiguous_submission_blocks_repetition(campaign, monkeypatch):
    root, _ = campaign
    calls = fake_scheduler(monkeypatch, root, ambiguous=True)
    with pytest.raises(ValueError, match="ambiguous"):
        launch.launch_phase(root, "canary")
    assert len(calls) == 1
    with pytest.raises(ValueError, match="uncertain scheduler"):
        launch.launch_phase(root, "canary")
    assert len(calls) == 1


def test_dry_run_does_not_submit_and_changed_chunks_are_refused(campaign, monkeypatch):
    root, _ = campaign
    monkeypatch.setattr(launch.core.subprocess, "run", lambda *a, **k: pytest.fail("scheduler called"))
    result = launch.launch_phase(root, "canary", dry_run=True)
    assert result["dry_run"] and not list((root / "campaign/attempts").iterdir())
    chunk = root / "campaign-chunk-0000.tsv"
    chunk.chmod(0o644)
    chunk.write_text("0\n")
    with pytest.raises(ValueError, match="Immutable"):
        launch.launch_phase(root, "canary", dry_run=True)


@pytest.mark.parametrize("damage", ["failed_audit", "wrong_identity", "nonzero_exit", "missing_result"])
def test_production_gate_rejects_failed_or_changed_canary(campaign, monkeypatch, damage):
    root, plan = campaign
    passed_canary(root, plan)
    if damage in ("failed_audit", "wrong_identity"):
        gate = launch.core.read(root / "canary-audit.json")
        gate["success" if damage == "failed_audit" else "plan_fingerprint"] = False
        put(root / "canary-audit.json", gate)
    elif damage == "nonzero_exit":
        (root / "tasks/000000/process-exit-code.txt").write_text("7\n")
    else:
        (root / "tasks/000000/result.json").unlink()
    calls = fake_scheduler(monkeypatch, root)
    with pytest.raises((ValueError, FileNotFoundError)):
        launch.launch_phase(root, "production")
    assert not calls


def test_prepare_stage_is_one_cpu_and_canary_does_not_launch_before_readiness(campaign, monkeypatch):
    root, _ = campaign
    calls = fake_scheduler(monkeypatch, root)
    (root / "preparation.json").unlink()
    with launch.submission_lock(root):
        stage = launch.submit_stage(root, "prepare")
    assert "--ntasks=1" in stage["command"] and "--mem=16G" in stage["command"]
    with pytest.raises(ValueError, match="preparation must complete"):
        launch.launch_phase(root, "canary")
    assert len(calls) == 1


def test_canary_audit_failure_never_submits_production(campaign, monkeypatch):
    root, _ = campaign
    monkeypatch.setattr(stages, "check_inputs", lambda root: None)
    monkeypatch.setitem(sys.modules, "summarize_sparse_smart_v2_checkpoint_bic",
                        SimpleNamespace(audit=lambda root, task_ids: dict(success=False, reason="bad retained state")))
    monkeypatch.setattr(stages, "launcher", lambda root: pytest.fail("production submitted"))
    with pytest.raises(ValueError, match="production remains unsubmitted"):
        stages.canary_audit(root)
    assert launch.core.read(root / "canary-audit.json")["success"] is False


def test_snapshot_uses_current_dirty_source_and_hashes_optional_baselines(tmp_path, monkeypatch):
    code = tmp_path / "repo"
    code.mkdir()
    (code / "module.py").write_text("CURRENT = True\n")
    baseline = tmp_path / "baseline.json"
    put(baseline, dict(schema_version=1, files={"module.py": "0" * 64}))
    reference = tmp_path / "reference.json"
    put(reference, dict(root=builder.EXHAUSTIVE_ROOT, reference_root="/scratch2/oldpaper",
                        selection_scope=dict(models=[0, 1, 2], seed_ids=list(range(100)))))
    comparisons = tmp_path / "comparisons.json"
    put(comparisons, dict(records=[], source_sha256="c" * 64))
    monkeypatch.setattr(builder, "CODE", code)
    monkeypatch.setattr(builder, "source_paths", lambda _: ["module.py"])
    monkeypatch.setattr(builder, "git_state", lambda: dict(commit="head", status_porcelain=" M module.py", diff_sha256="d" * 64))
    local = tmp_path / "frozen"
    receipt = builder.build(local, Path("/scratch2/new-checkpoint"), baseline_manifest=baseline,
                            reference_plan_path=reference, comparison_baselines=comparisons)
    assert (local / "source/module.py").read_text() == "CURRENT = True\n"
    config = json.loads((local / "campaign.json").read_text())
    assert config["workers"] == 100 and config["seed_ids"] == list(range(100))
    assert config["initializer_pair"] == [.003, .1] and config["pool_time"] == "06:00:00"
    assert config["comparison_baselines_sha256"] == builder.sha(local / "comparison-baselines.json")
    with tarfile.open(receipt["archive"]) as archive:
        names = archive.getnames()
        assert "comparison-baselines.json" in names and "campaign.json" in names
        assert not any(name.endswith((".npz", ".npy", ".pkl")) for name in names)
    with pytest.raises(ValueError, match="fresh"):
        builder.build(local, Path("/scratch2/new-checkpoint"))


def test_source_inventory_rejects_raw_data_and_snapshot_rejects_scratch1(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "REQUIRED_ADDITIONS", ())
    (tmp_path / "data.npz").write_bytes(b"raw")
    with pytest.raises(ValueError, match="raw data"):
        builder.source_paths(dict(schema_version=1, files={"data.npz": "a" * 64}), tmp_path)
    with pytest.raises(ValueError, match="below /scratch2"):
        builder.build(tmp_path / "new", Path("/scratch1/new"))


def test_rank_preparation_computes_once_per_dataset_and_reads_training_only(tmp_path, monkeypatch):
    root, old = tmp_path / "new", tmp_path / "old"
    root.mkdir()
    groups = [dict(group_id="a", dataset_id="d1"), dict(group_id="alias", dataset_id="d1"),
              dict(group_id="b", dataset_id="d2")]
    reference = dict(groups=groups, plan_fingerprint="old-plan")
    put(old / "plan.json", reference)
    prepared = []
    for group in groups:
        path = old / "groups" / group["group_id"] / "group.json"
        put(path, dict(group=group, plan_fingerprint="old-plan", fingerprints={"training_observed_input_fingerprint": group["dataset_id"]},
                       files={"data.npz": dict(path="/scratch2/" + group["dataset_id"] + ".npz", sha256="datahash")}))
        prepared.append(dict(group_id=group["group_id"], group_json_sha256=stages.sha(path)))
    put(old / "preparation.json", dict(success=True, status="complete", plan_fingerprint="old-plan",
                                      plan_sha256=stages.sha(old / "plan.json"), groups=prepared))
    put(root / "source-manifest.json", dict(files={}))
    monkeypatch.setattr(stages, "check_inputs", lambda _: ({}, {}, old, reference))
    monkeypatch.setattr(stages, "EXPECTED_DATASETS", 2)
    original_sha = stages.sha
    # Pytest's own temporary files also live below /scratch2 on Discovery.
    # Stub only the two nonexistent data inputs, preserving all metadata hashes.
    fake_data_paths = {Path("/scratch2/d1.npz"), Path("/scratch2/d2.npz")}
    monkeypatch.setattr(stages, "sha", lambda p: "datahash" if Path(p) in fake_data_paths else original_sha(p))
    accessed, rank_calls = [], []
    class TrainingArchive:
        def __init__(self, path): self.dataset = Path(path).stem
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __getitem__(self, key):
            assert key in ("X", "Y", "C0")
            accessed.append(key)
            return self.dataset
    monkeypatch.setattr(stages.np, "load", lambda path, **kwargs: TrainingArchive(path))
    def ranked(x, y, c):
        assert x == y == c
        rank_calls.append(x)
        return dict(selected_rank=5, training_observed_input_fingerprint=x)
    monkeypatch.setattr(stages, "rank_selection", ranked)
    plan = dict(groups=groups, n_datasets=2, n_groups=3, n_tasks=6, canary_task_ids=[0],
                plan_fingerprint="new-plan", selection_scope=dict(seed_ids=list(range(100))))
    monkeypatch.setattr(stages, "freeze_plan", lambda *args: plan)
    def fitting_prepare(_):
        put(root / "preparation.json", dict(success=True))
        for group in groups:
            put(root / "groups" / group["group_id"] / "group.json",
                dict(fingerprints={"training_observed_input_fingerprint": group["dataset_id"]}))
        return dict(success=True)
    monkeypatch.setattr(stages.fitting, "prepare", fitting_prepare)
    result = stages.prepare(root, submit=False)
    assert result["ranked_datasets"] == 2 and rank_calls == ["d1", "d2"]
    assert accessed == ["X", "Y", "C0"] * 2
    stages.prepare(root, submit=False)
    assert rank_calls == ["d1", "d2"]  # An explicit stage retry reuses frozen rank records.
