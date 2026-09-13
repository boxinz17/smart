"""Submission and resume fixtures; no actual Slurm or scientific computation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

HPC = Path(__file__).resolve().parents[3] / "hpc/discovery"
spec = importlib.util.spec_from_file_location("bic_launcher", HPC / "submit_paper_source_comparators.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(launcher.canonical(value))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / "literal space $dollar {braces}"
    source = root / "source/hpc/discovery"
    source.mkdir(parents=True)
    for name in ("paper_source_comparators_prepare.sbatch", "paper_source_comparators_pool.sbatch",
                 "paper_source_comparators_summary.sbatch", "paper_source_comparators_worker.sh"):
        shutil.copyfile(HPC / name, source / name)
    files = {str(p.relative_to(root / "source")): launcher.sha(p) for p in source.iterdir()}
    write(root / "source-manifest.json", dict(schema_version=1, source_root=str(root / "source"), files=files))
    plan = dict(schema_version=1, method="PaperSourceComparators", root=str(root),
                source_manifest_sha256=launcher.sha(root / "source-manifest.json"),
                n_groups=2, groups=[{"group_id": "a"}, {"group_id": "b"}], n_tasks=4,
                tasks=[dict(task_id=i, group_id="a" if i < 2 else "b") for i in range(4)])
    plan["plan_fingerprint"] = hashlib.sha256(launcher.canonical(plan).strip()).hexdigest()
    write(root / "plan.json", plan)
    (root / "work-items.tsv").write_text("0\n1\n2\n3\n")
    monkeypatch.setattr(launcher, "RUN_PREFIX", tmp_path)
    return root, plan


def args(root, *extra):
    return ["--run-dir", str(root), "--max-tasks-per-chunk", "2", *extra]


def marker(root, plan):
    write(root / "preparation.json", dict(status="complete", success=True,
          plan_sha256=launcher.sha(root / "plan.json"), source_manifest_sha256=plan["source_manifest_sha256"],
          n_groups=2, n_tasks=4))
    write(root / "canary.json", dict(success=True, plan_fingerprint=plan["plan_fingerprint"]))


def scheduler(monkeypatch, *, ambiguous=False, state="COMPLETED"):
    submissions = []
    def run(cmd, **kwargs):
        if cmd[0] == "sbatch":
            submissions.append(cmd)
            return SimpleNamespace(returncode=0, stdout="uncertain reply" if ambiguous else str(8000 + len(submissions)), stderr="")
        if cmd[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="RUNNING\n" if state == "RUNNING" else "", stderr="")
        if cmd[0] == "sacct":
            return SimpleNamespace(returncode=0, stdout=cmd[cmd.index("--jobs") + 1] + "|" + state + "|\n", stderr="")
        raise AssertionError(cmd)
    monkeypatch.setattr(launcher.core.subprocess, "run", run)
    return submissions


def test_dry_run_declares_serial_pools_and_separate_summary(campaign, monkeypatch):
    root, plan = campaign
    monkeypatch.setattr(launcher.core.subprocess, "run", lambda *a, **k: pytest.fail("scheduler reached during dry run"))
    assert launcher.main(args(root, "--dry-run")) == 0
    result = launcher.read(root / "campaign-submission-plan.json")
    prep, *tail = result["actions"]
    pools, summary = tail[:-1], tail[-1]
    assert prep["kind"] == "prepare" and "--ntasks=1" in prep["command"]
    assert len(pools) == 2 and all("--ntasks=100" in row["command"] for row in pools)
    assert "--dependency=afterok:PREPARE_JOB_ID" in pools[0]["command"]
    assert "--dependency=afterany:POOL_0000_JOB_ID" in pools[1]["command"]
    assert "--dependency=afterany:POOL_0000_JOB_ID:POOL_0001_JOB_ID" in summary["command"]
    assert all(not any(item.startswith(("--array", "--nodes", "--ntasks-per-node")) for item in row["command"]) for row in pools)
    assert [row["group_ids"] for row in launcher.read(root / "campaign/chunks.json")["chunks"]] == [["a"], ["b"]]
    assert not list((root / "campaign/attempts").iterdir())


def test_partial_submission_requires_explicit_no_summary(campaign, monkeypatch):
    root, plan = campaign
    calls = scheduler(monkeypatch)
    assert launcher.main(args(root, "--chunks", "0")) == 2
    assert calls == []
    assert launcher.main(args(root, "--chunks", "0", "--no-summary")) == 0
    assert len(calls) == 2


def test_ambiguous_submission_is_not_repeated(campaign, monkeypatch):
    root, plan = campaign
    calls = scheduler(monkeypatch, ambiguous=True)
    assert launcher.main(args(root)) == 2
    assert len(calls) == 1
    record = launcher.read(root / "campaign/attempts/prepare-attempt-0001/submission.json")
    assert record["status"] == "submission_uncertain"
    assert launcher.main(args(root, "--resume")) == 2
    assert len(calls) == 1


def test_active_jobs_and_summary_are_not_submitted_twice(campaign, monkeypatch):
    root, plan = campaign
    calls = scheduler(monkeypatch, state="RUNNING")
    assert launcher.main(args(root)) == 0
    assert len(calls) == 4
    assert "--dependency=afterany:8002:8003" in calls[-1]
    assert launcher.main(args(root, "--resume")) == 0
    assert len(calls) == 4


def test_resume_skips_bound_finished_outcomes_and_detects_corruption(campaign):
    root, plan = campaign
    folder = root / "tasks/000000"
    result = dict(task=plan["tasks"][0], group_id="a", plan_fingerprint=plan["plan_fingerprint"],
                  source_manifest_sha256=plan["source_manifest_sha256"], status="complete",
                  execution_success=True, files={})
    write(folder / "result.json", result)
    write(folder / "status.json", dict(result, status="finished", result_sha256=launcher.sha(folder / "result.json")))
    for name in ("process-exit-code.txt", "launcher-exit-code.txt"):
        (folder / name).write_text("0\n")
    assert launcher.completed_task(root, plan, 0)
    (folder / "result.json").write_text((folder / "result.json").read_text() + " ")
    with pytest.raises(ValueError, match="invalid status"):
        launcher.completed_task(root, plan, 0)


def test_canary_subset_keeps_full_chunks_and_resume_launches_only_missing(campaign, monkeypatch):
    root, plan = campaign
    marker(root, plan)
    calls = scheduler(monkeypatch)
    assert launcher.main(args(root, "--task-ids", "0", "2", "--no-summary", "--workers", "2")) == 0
    assert len(calls) == 2
    chunks = launcher.read(root / "campaign/chunks.json")["chunks"]
    assert [chunk["task_ids"] for chunk in chunks] == [[0, 1], [2, 3]]
    for tid in (0, 2):
        folder = root / "tasks" / f"{tid:06d}"
        result = dict(task=plan["tasks"][tid], group_id=plan["tasks"][tid]["group_id"],
                      plan_fingerprint=plan["plan_fingerprint"], source_manifest_sha256=plan["source_manifest_sha256"],
                      status="complete", execution_success=True, files={})
        write(folder / "result.json", result)
        write(folder / "status.json", dict(result, status="finished", result_sha256=launcher.sha(folder / "result.json")))
        for name in ("process-exit-code.txt", "launcher-exit-code.txt"):
            (folder / name).write_text("0\n")
    assert launcher.main(args(root, "--resume", "--no-summary")) == 0
    assert len(calls) == 4
    records = [launcher.read(path) for path in sorted((root / "campaign/attempts").glob("chunk-*-attempt-0002/submission.json"))]
    assert [record["task_ids"] for record in records] == [[1], [3]]
    assert [record["completed_tasks_skipped"] for record in records] == [1, 1]


def test_full_submission_waits_for_active_canary_subset(campaign, monkeypatch):
    root, plan = campaign
    marker(root, plan)
    calls = scheduler(monkeypatch, state="RUNNING")
    assert launcher.main(args(root, "--task-ids", "0", "--no-summary")) == 0
    assert len(calls) == 1
    assert launcher.main(args(root, "--resume", "--no-summary")) == 2
    assert len(calls) == 1


def test_changed_source_stops_before_submission(campaign, monkeypatch):
    root, plan = campaign
    calls = scheduler(monkeypatch)
    path = root / "source/hpc/discovery/paper_source_comparators_worker.sh"
    path.write_text(path.read_text() + "\n# changed\n")
    assert launcher.main(args(root)) == 2
    assert calls == []


def test_scripts_parse_and_preserve_one_cpu_srun_dispatch():
    for suffix in ("prepare.sbatch", "pool.sbatch", "summary.sbatch", "worker.sh"):
        script = HPC / ("paper_source_comparators_" + suffix)
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
    worker = (HPC / "paper_source_comparators_worker.sh").read_text()
    assert "srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1" in worker
    assert "paper_source_comparators_20260913.runtime fit" in worker and "'%06d'" in worker
    pool = (HPC / "paper_source_comparators_pool.sbatch").read_text()
    assert 'parallel --plain --quote --jobs "$workers"' in pool and "--halt never" in pool
    assert "summarize_sparse" not in pool and "rsync" not in pool
