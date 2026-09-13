"""Operational fixtures for retrospective scoring; never invoke Slurm or fits."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

HPC = Path(__file__).resolve().parents[2] / "hpc/discovery"
spec = importlib.util.spec_from_file_location(
    "retrospective_bic_launcher", HPC / "submit_sparse_smart_v2_retrospective_bic.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(launcher.core.canonical(value))


def fingerprint(plan):
    plan.pop("plan_fingerprint", None)
    plan["plan_fingerprint"] = hashlib.sha256(json.dumps(
        plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def refresh_source(root, plan):
    source = root / "source"
    files = {str(path.relative_to(source)): launcher.core.sha(path)
             for path in source.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    write(root / "source-manifest.json", dict(schema_version=1, source_root=str(source), files=files))
    plan["source_manifest_sha256"] = launcher.core.sha(root / "source-manifest.json")
    fingerprint(plan)
    write(root / "plan.json", plan)


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "literal space $dollar 'quote' $(touch INJECTED) {braces}"
    source = root / "source/hpc/discovery"
    source.mkdir(parents=True)
    for name in ("submit_sparse_smart_v2_retrospective_bic.py", "submit_sparse_smart_v2_campaign.py",
                 "sparse_smart_v2_retrospective_bic.sbatch", "sparse_smart_v2_retrospective_bic_worker.sh"):
        shutil.copyfile(HPC / name, source / name)
    ids = ["m0_e0_k0_s0", "m0_e0_k0_s1"]
    plan = dict(schema_version=1, method="SparseSMARTv2RetrospectiveBIC", root=str(root),
                n_cases=2, cases=[dict(case_id=cid) for cid in ids])
    (root / "work-items.tsv").write_text("".join(cid + "\n" for cid in ids))
    (root / "campaign/attempts").mkdir(parents=True)
    refresh_source(root, plan)
    return root, plan


def arguments(**overrides):
    return SimpleNamespace(**dict(dict(workers=32, mem="4G", time="24:00:00",
                                       account="mkolar_1314", partition="main", dry_run=False), **overrides))


def scheduler(monkeypatch, root, *, ambiguous_at=None, returncode=0):
    calls = []

    def run(command, **kwargs):
        assert command[0] == "sbatch"
        assert kwargs == dict(capture_output=True, text=True)
        kind = command[-2]
        record = launcher.core.read(root / f"campaign/attempts/{kind}-attempt-0001/submission.json")
        assert record["status"] == "submitting" and record["command"] == command
        calls.append(command)
        if len(calls) == ambiguous_at:
            return SimpleNamespace(returncode=returncode, stdout="uncertain reply", stderr="scheduler detail")
        return SimpleNamespace(returncode=0, stdout=f"{9100 + len(calls)};discovery\n", stderr="")

    monkeypatch.setattr(launcher.core.subprocess, "run", run)
    return calls


def test_dry_run_has_one_shared_32_worker_pool_and_separate_summary(campaign, monkeypatch):
    root, _ = campaign
    monkeypatch.setattr(launcher.core.subprocess, "run", lambda *a, **k: pytest.fail("Dry run reached scheduler"))
    result = launcher.submit(root, arguments(dry_run=True))
    prepare, pool, summary = result["actions"]
    assert [row["kind"] for row in result["actions"]] == ["prepare", "pool", "summary"]
    assert "--ntasks=1" in prepare["command"] and "--ntasks=1" in summary["command"]
    assert "--ntasks=32" in pool["command"] and "--mem-per-cpu=4G" in pool["command"]
    assert "--dependency=afterok:PREPARE_JOB_ID" in pool["command"]
    assert "--dependency=afterany:POOL_JOB_ID" in summary["command"]
    for row in result["actions"]:
        assert "--cpus-per-task=1" in row["command"]
        assert not any(arg.startswith(("--array", "--nodes", "--ntasks-per-node")) for arg in row["command"])
        assert "--chdir=" + str(root) in row["command"]
        assert row["command"][-3:] == [str(root), row["kind"], "32"]
    assert not list((root / "campaign/attempts").iterdir())
    assert (root / "submission-preview.json").is_file()
    assert not (root / "submission.json").exists()


def test_submitted_jobs_are_never_submitted_twice(campaign, monkeypatch):
    root, _ = campaign
    (root / "campaign/attempts").rmdir()
    calls = scheduler(monkeypatch, root)
    result = launcher.submit(root, arguments())
    assert [row["job_id"] for row in result["actions"]] == ["9101", "9102", "9103"]
    assert "--dependency=afterok:9101" in calls[1]
    assert "--dependency=afterany:9102" in calls[2]
    assert launcher.submit(root, arguments())["actions"] == result["actions"]
    assert len(calls) == 3


@pytest.mark.parametrize("override", [dict(mem="8G"), dict(time="12:00:00"),
                                     dict(account="different_account"), dict(partition="different_partition")])
def test_resubmission_cannot_silently_change_resource_requests(campaign, monkeypatch, override):
    root, _ = campaign
    calls = scheduler(monkeypatch, root)
    launcher.submit(root, arguments())
    with pytest.raises(ValueError, match="different inputs/resources"):
        launcher.submit(root, arguments(**override))
    assert len(calls) == 3


def test_command_line_rejects_outputs_outside_scratch2(campaign, monkeypatch):
    root, _ = campaign
    monkeypatch.setattr(launcher.core.subprocess, "run", lambda *a, **k: pytest.fail("Scheduler reached"))
    with pytest.raises(ValueError, match="New output must be on /scratch2"):
        launcher.main(["--run-dir", str(root), "--dry-run"])


@pytest.mark.parametrize("ambiguous_at,returncode", [(1, 0), (1, 1), (2, 0)])
def test_uncertain_submission_persists_evidence_and_cannot_repeat(campaign, monkeypatch, ambiguous_at, returncode):
    root, _ = campaign
    calls = scheduler(monkeypatch, root, ambiguous_at=ambiguous_at, returncode=returncode)
    with pytest.raises(ValueError, match="rejected or ambiguous"):
        launcher.submit(root, arguments())
    kind = "prepare" if ambiguous_at == 1 else "pool"
    folder = root / f"campaign/attempts/{kind}-attempt-0001"
    record = launcher.core.read(folder / "submission.json")
    assert record["status"] == "submission_uncertain" and record["returncode"] == returncode
    assert (folder / "submission.stdout").read_text() == "uncertain reply"
    assert (folder / "submission.stderr").read_text() == "scheduler detail"
    with pytest.raises(ValueError, match="Uncertain prior submission"):
        launcher.submit(root, arguments())
    assert len(calls) == ambiguous_at


@pytest.mark.parametrize("change", ["work", "source", "plan"])
def test_changed_inputs_fail_before_scheduler(campaign, monkeypatch, change):
    root, plan = campaign
    calls = scheduler(monkeypatch, root)
    if change == "work":
        (root / "work-items.tsv").write_text("m0_e0_k0_s1\nm0_e0_k0_s0\n")
    elif change == "source":
        with (root / "source/hpc/discovery/sparse_smart_v2_retrospective_bic_worker.sh").open("a") as stream:
            stream.write("\n# changed\n")
    else:
        plan["n_cases"] = 3
        write(root / "plan.json", plan)
    with pytest.raises(ValueError, match="mismatch|changed|Changed"):
        launcher.submit(root, arguments())
    assert calls == []


def test_resubmission_cannot_change_workers_or_rebind_plan(campaign, monkeypatch):
    root, plan = campaign
    calls = scheduler(monkeypatch, root)
    launcher.submit(root, arguments())
    with pytest.raises(ValueError, match="different inputs/resources"):
        launcher.submit(root, arguments(workers=64))
    plan["reviewed_note"] = "new plan identity"
    fingerprint(plan)
    write(root / "plan.json", plan)
    with pytest.raises(ValueError, match="different inputs/resources"):
        launcher.submit(root, arguments())
    assert len(calls) == 3


def executable(path, content):
    path.write_text(content)
    path.chmod(0o755)


def test_pool_continues_independent_cases_and_reports_failure_with_literal_paths(campaign, monkeypatch, tmp_path):
    root, plan = campaign
    # Replace only runtime and scientific code with small deterministic fixtures.
    source = root / "source"
    (source / "hpc/discovery/env.sh").write_text("module() { :; }\nexport PYTHONPATH=''\n")
    (source / "simulation").mkdir()
    (source / "simulation/run_sparse_smart_v2_retrospective_bic.py").write_text(
        "import sys\nprint('SCORING', sys.argv[-1], flush=True)\nsys.exit(7 if sys.argv[-1].endswith('s0') else 0)\n")
    refresh_source(root, plan)
    calls = scheduler(monkeypatch, root)
    launcher.submit(root, arguments(workers=2))
    assert len(calls) == 3
    monkeypatch.undo()
    binaries = tmp_path / "fake commands"
    binaries.mkdir()
    virtual = tmp_path / "fake venv"
    (virtual / "bin").mkdir(parents=True)
    (virtual / "bin/python").symlink_to(sys.executable)
    executable(binaries / "parallel", "#!" + sys.executable + "\n" + r'''
import json,os,pathlib,subprocess,sys
args=sys.argv[1:]
if '--version' in args:
    print('fixture parallel');sys.exit(0)
assert '--halt' in args and args[args.index('--halt')+1]=='never'
assert args[args.index('--jobs')+1]=='2'
start=args.index('bash');end=args.index('::::')
command=args[start:end];rows=pathlib.Path(args[end+1]).read_text().splitlines()
codes=[]
for row in rows:
    codes.append(subprocess.run([row if value=='{}' else value for value in command]).returncode)
pathlib.Path(args[args.index('--joblog')+1]).write_text(json.dumps(codes))
sys.exit(1 if any(codes) else 0)
''')
    executable(binaries / "srun", "#!" + sys.executable + "\n" + r'''
import subprocess,sys
args=sys.argv[1:]
assert args[:6]==['--exclusive','--exact','--nodes=1','--ntasks=1','--cpus-per-task=1','--kill-on-bad-exit=0']
sys.exit(subprocess.run(args[6:]).returncode)
''')
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
               VENV=str(virtual), SLURM_JOB_ID="9102", SLURM_NTASKS="2", SLURM_CPUS_PER_TASK="1")
    result = subprocess.run(["bash", str(source / "hpc/discovery/sparse_smart_v2_retrospective_bic.sbatch"),
                             str(root), "pool", "2"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    folder = root / "campaign/attempts/pool-attempt-0001"
    assert json.loads((folder / "parallel-joblog.tsv").read_text()) == [7, 0]
    assert (folder / "exit-code.txt").read_text().strip() == "1"
    for cid, code in [("m0_e0_k0_s0", 7), ("m0_e0_k0_s1", 0)]:
        assert (root / f"logs/cases/{cid}.exit-code.txt").read_text().strip() == str(code)
        assert "SCORING " + cid in (root / f"logs/cases/{cid}.log").read_text()
    assert not (tmp_path / "INJECTED").exists()
    assert not (root / "INJECTED").exists()


def test_shell_scripts_parse():
    for name in ("sparse_smart_v2_retrospective_bic.sbatch", "sparse_smart_v2_retrospective_bic_worker.sh"):
        subprocess.run(["bash", "-n", str(HPC / name)], check=True, capture_output=True)
