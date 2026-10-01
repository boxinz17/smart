"""Exercise frozen inputs, manual release gates and actual shell failure handling."""
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


def module(name):
    spec = importlib.util.spec_from_file_location(name, HPC / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


launcher = module("submit_sparse_smart_v2_retained_state_bic")
builder = module("build_sparse_smart_v2_retained_state_bic_snapshot")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(launcher.core.canonical(value))


def fingerprint(plan):
    plan.pop("plan_fingerprint", None)
    plan["plan_fingerprint"] = builder.digest(plan)


def refresh(root, plan):
    source = root / "source"
    files = {p.relative_to(source).as_posix(): launcher.core.sha(p)
             for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    write(root / "source-manifest.json", dict(schema_version=1, source_root=str(source), files=files))
    plan["source_manifest_sha256"] = launcher.core.sha(root / "source-manifest.json")
    fingerprint(plan)
    write(root / "plan.json", plan)


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "literal space $dollar 'quote' $(touch INJECTED) {braces}"
    source = root / "source/hpc/discovery"
    source.mkdir(parents=True)
    for name in ("submit_sparse_smart_v2_retained_state_bic.py", "submit_sparse_smart_v2_campaign.py",
                 "sparse_smart_v2_retained_state_bic.sbatch", "sparse_smart_v2_retained_state_bic_worker.sh"):
        shutil.copyfile(HPC / name, source / name)
    ids = ["m0_e0_k0_s0", "m1_e0_k0_s0", "m2_e0_k0_s0", "m0_e0_k0_s1", "m1_e0_k0_s1"]
    plan = dict(schema_version=1, method=launcher.METHOD, root=str(root), n_cases=len(ids),
        cases=[dict(case_id=cid, case=dict(model_id=int(cid[1]))) for cid in ids],
        preflight_case_ids=ids[:3], preflight_chart_change_case_ids=[], selection_arms=list(builder.ARMS))
    for name, rows in (("work-items.tsv", ids), ("preflight-work-items.tsv", ids[:3]),
                       ("production-work-items.tsv", ids[3:])):
        (root / name).write_text("".join(cid + "\n" for cid in rows))
    (root / "campaign/attempts").mkdir(parents=True)
    refresh(root, plan)
    return root, plan


def arguments(**overrides):
    return SimpleNamespace(**dict(dict(stage="prepare", workers=32, mem="4G", time="24:00:00",
        account="mkolar_1314", partition="main", dry_run=False), **overrides))


def stage_ok(root, stage):
    folder = root / ("campaign/attempts/" + stage + "-attempt-0001")
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("exit-code.txt", "source-integrity-exit-code.txt"):
        (folder / name).write_text("0\n")


def prepared(root, plan):
    rows = []
    for row in plan["cases"]:
        cid = row["case_id"]
        write(root / "inputs" / (cid + ".json"), dict(case_id=cid))
        rows.append(dict(case_id=cid, input_sha256=launcher.core.sha(root / "inputs" / (cid + ".json"))))
    write(root / "preparation.json", dict(success=True, status="complete", n_cases=plan["n_cases"],
        plan_sha256=launcher.core.sha(root / "plan.json"), plan_fingerprint=plan["plan_fingerprint"],
        source_manifest_sha256=plan["source_manifest_sha256"], cases=rows))
    stage_ok(root, "prepare")


def case_result(root, plan, cid, **overrides):
    write(root / "inputs" / (cid + ".json"), dict(case_id=cid))
    identity = dict(schema_version=1, method=launcher.METHOD, case_id=cid,
        plan_fingerprint=plan["plan_fingerprint"], source_manifest_sha256=plan["source_manifest_sha256"],
        input_sha256=launcher.core.sha(root / "inputs" / (cid + ".json")))
    result = dict(identity, success=True, inventory=dict(current_coverage_complete=True,
        all_retained_coverage_complete=True, chart_changing_candidates=1 if cid in plan["preflight_chart_change_case_ids"] else 0),
        historical_comparison=dict(passed=True), monotonicity=dict(passed=True))
    result.update(overrides)
    write(root / "cases" / (cid + ".json"), result)
    write(root / "cases" / (cid + ".status.json"), dict(identity, status="finished", success=True,
        result_sha256=launcher.core.sha(root / "cases" / (cid + ".json"))))
    (root / "logs/cases").mkdir(parents=True, exist_ok=True)
    for ending in (".exit-code.txt", ".process-exit-code.txt"):
        (root / "logs/cases" / (cid + ending)).write_text("0\n")
    return result


def released(root, plan):
    prepared(root, plan)
    for cid in plan["preflight_case_ids"]:
        case_result(root, plan, cid)
    launcher.record_preflight(root)
    stage_ok(root, "preflight")


def scheduler(monkeypatch, root, ambiguous_at=None):
    calls = []

    def run(command, **kwargs):
        assert command[0] == "sbatch" and kwargs == dict(capture_output=True, text=True)
        assert not any(k.startswith(("SLURM_", "SBATCH_", "SRUN_")) for k in os.environ)
        kind = command[-2]
        record = launcher.core.read(root / f"campaign/attempts/{kind}-attempt-0001/submission.json")
        assert record["status"] == "submitting" and record["command"] == command
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="uncertain reply" if len(calls) == ambiguous_at
                               else f"{9100 + len(calls)};discovery\n", stderr="")

    monkeypatch.setattr(launcher.core.subprocess, "run", run)
    return calls


def test_dry_run_only_prepares_and_never_calls_scheduler(campaign, monkeypatch):
    root, _ = campaign
    monkeypatch.setattr(launcher.core.subprocess, "run", lambda *a, **k: pytest.fail("scheduler reached"))
    result = launcher.submit(root, arguments(dry_run=True))
    assert [r["kind"] for r in result["actions"]] == ["prepare"]
    cmd = result["actions"][0]["command"]
    assert "--ntasks=1" in cmd and "--mem=16G" in cmd
    assert not list((root / "campaign/attempts").iterdir())


def test_preflight_and_production_require_successful_manual_gates(campaign, monkeypatch):
    root, plan = campaign
    calls = scheduler(monkeypatch, root)
    with pytest.raises(FileNotFoundError):
        launcher.submit(root, arguments(stage="preflight"))
    prepared(root, plan)
    result = launcher.submit(root, arguments(stage="preflight"))
    assert [r["kind"] for r in result["actions"]] == ["preflight"]
    with pytest.raises(FileNotFoundError):
        launcher.submit(root, arguments(stage="production"))
    assert len(calls) == 1


def test_production_resources_and_idempotency(campaign, monkeypatch):
    root, plan = campaign
    released(root, plan)
    monkeypatch.setenv("SLURM_NTASKS", "91")
    monkeypatch.setenv("SBATCH_MEM_PER_CPU", "100G")
    monkeypatch.setenv("SRUN_CPUS_PER_TASK", "17")
    calls = scheduler(monkeypatch, root)
    result = launcher.submit(root, arguments(stage="production"))
    pool, summary = result["actions"]
    assert pool["job_id"] == "9101" and summary["job_id"] == "9102"
    assert "--ntasks=32" in calls[0] and "--mem-per-cpu=4G" in calls[0] and "--time=24:00:00" in calls[0]
    assert "--ntasks=1" in calls[1] and "--mem=16G" in calls[1] and "--dependency=afterany:9101" in calls[1]
    for cmd in calls:
        assert "--cpus-per-task=1" in cmd
        assert not any(x.startswith(("--array", "--nodes", "--ntasks-per-node")) for x in cmd)
    assert os.environ["SLURM_NTASKS"] == "91"
    assert launcher.submit(root, arguments(stage="production"))["actions"] == result["actions"]
    assert len(calls) == 2


@pytest.mark.parametrize("override", [dict(workers=64), dict(mem="8G"), dict(time="12:00:00")])
def test_cannot_change_resources_after_submission(campaign, monkeypatch, override):
    root, _ = campaign
    calls = scheduler(monkeypatch, root)
    launcher.submit(root, arguments())
    with pytest.raises(ValueError, match="different inputs/resources"):
        launcher.submit(root, arguments(**override))
    assert len(calls) == 1


def test_uncertain_pool_never_repeated(campaign, monkeypatch):
    root, plan = campaign
    released(root, plan)
    calls = scheduler(monkeypatch, root, ambiguous_at=1)
    with pytest.raises(ValueError, match="rejected or ambiguous"):
        launcher.submit(root, arguments(stage="production"))
    with pytest.raises(ValueError, match="uncertain prior submission"):
        launcher.submit(root, arguments(stage="production"))
    assert len(calls) == 1
    assert launcher.core.read(root / "campaign/attempts/pool-attempt-0001/submission.json")["status"] == "submission_uncertain"


@pytest.mark.parametrize("field", ["current_coverage_complete", "all_retained_coverage_complete", "history", "monotonicity", "chart"])
def test_preflight_scientific_failures_block_release(campaign, field):
    root, plan = campaign
    if field == "chart":
        plan["preflight_chart_change_case_ids"] = plan["preflight_case_ids"][:1]
        refresh(root, plan)
    prepared(root, plan)
    for cid in plan["preflight_case_ids"]:
        result = case_result(root, plan, cid)
    cid = plan["preflight_case_ids"][0]
    result = case_result(root, plan, cid)
    if field in ("current_coverage_complete", "all_retained_coverage_complete"):
        result["inventory"][field] = False
    elif field == "chart":
        result["inventory"]["chart_changing_candidates"] = 0
    else:
        result["historical_comparison" if field == "history" else field]["passed"] = False
    case_result(root, plan, cid, **{k: result[k] for k in ("inventory", "historical_comparison", "monotonicity")})
    with pytest.raises(ValueError, match="failed|changing charts"):
        launcher.record_preflight(root)


def test_preflight_outputs_and_prepared_inputs_are_bound(campaign):
    root, plan = campaign
    released(root, plan)
    cid = plan["preflight_case_ids"][0]
    write(root / "inputs" / (cid + ".json"), dict(case_id=cid, changed=True))
    with pytest.raises(ValueError, match="prepared input differs"):
        launcher.preflight_ready(root, plan)


@pytest.mark.parametrize("change", ["source", "work", "plan"])
def test_changed_frozen_inputs_fail_before_submission(campaign, monkeypatch, change):
    root, plan = campaign
    calls = scheduler(monkeypatch, root)
    if change == "source":
        (root / "source/hpc/discovery/sparse_smart_v2_retained_state_bic_worker.sh").write_text("changed")
    elif change == "work":
        (root / "production-work-items.tsv").write_text("m0_e0_k0_s0\n")
    else:
        plan["n_cases"] += 1
        write(root / "plan.json", plan)
    with pytest.raises(ValueError, match="changed|mismatch"):
        launcher.submit(root, arguments())
    assert not calls


def test_snapshot_clones_cases_and_freezes_current_code_without_data(campaign, tmp_path, monkeypatch):
    old_root, old = campaign
    code = tmp_path / "current code"
    code.mkdir()
    required = ("simulation/old.py", "simulation/sparse_smart_v2_retained_state_scoring.py")
    for name in required:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# current bytes " + name)
    old_manifest = dict(schema_version=1, files={"simulation/old.py": "OLD_BYTES"})
    write(old_root / "source-manifest.json", old_manifest)
    old.update(method="SparseSMARTv2RetrospectiveBIC", root="/scratch2/user/old",
               source_campaign_root="/scratch2/user/fits", source_manifest_sha256=builder.sha(old_root / "source-manifest.json"))
    fingerprint(old)
    write(old_root / "plan.json", old)
    monkeypatch.setattr(builder, "CODE", code)
    monkeypatch.setattr(builder, "REQUIRED", ())
    monkeypatch.setattr(builder, "EXPECTED_CASES", 5)
    monkeypatch.setattr(builder, "git_state", lambda: dict(commit="head", status_porcelain=" M source", diff_sha256="dirty"))
    output = tmp_path / "new snapshot"
    result = builder.build(output, Path("/scratch2/user/new"), previous_plan=old_root / "plan.json",
                           preflight_case_ids=old["preflight_case_ids"])
    plan = launcher.core.read(output / "plan.json")
    manifest = launcher.core.read(output / "source-manifest.json")
    assert plan["cases"] == old["cases"] and plan["selection_arms"] == list(builder.ARMS)
    assert plan["previous_plan_sha256"] == builder.sha(old_root / "plan.json")
    assert plan["previous_retrospective_root"] == old["root"]
    assert plan["preflight_chart_change_case_ids"] == []
    assert "original_sample_no_chart_events" in plan["preflight_chart_coverage_reason"]
    assert set(manifest["files"]) == set(required)
    assert (output / "source/simulation/old.py").read_text().startswith("# current bytes")
    assert result["source_files"] == 2 and Path(result["archive"]).is_file()
    with pytest.raises(ValueError, match="fresh"):
        builder.build(output, Path("/scratch2/user/new"), previous_plan=old_root / "plan.json",
                      preflight_case_ids=old["preflight_case_ids"])


def test_storage_guard_rejects_scratch1():
    with pytest.raises(ValueError, match="below /scratch2"):
        launcher.main(["--run-dir", "/scratch1/user/new", "--dry-run"])


def executable(path, content):
    path.write_text(content)
    path.chmod(0o755)


def test_pool_continues_failures_and_verifies_source_with_literal_paths(campaign, monkeypatch, tmp_path):
    root, plan = campaign
    source = root / "source"
    (source / "hpc/discovery/env.sh").write_text("module() { :; }\nexport PYTHONPATH=''\n")
    (source / "simulation").mkdir()
    (source / "simulation/run_sparse_smart_v2_retained_state_bic.py").write_text(
        "import sys\nprint('SCORING', sys.argv[-1], flush=True)\nsys.exit(7 if sys.argv[-1].startswith('m0') else 0)\n")
    refresh(root, plan)
    released(root, plan)
    calls = scheduler(monkeypatch, root)
    launcher.submit(root, arguments(stage="production", workers=2))
    assert len(calls) == 2
    monkeypatch.undo()
    binaries = tmp_path / "fake commands"
    binaries.mkdir()
    virtual = tmp_path / "fake venv"
    (virtual / "bin").mkdir(parents=True)
    (virtual / "bin/python").symlink_to(sys.executable)
    executable(binaries / "parallel", "#!" + sys.executable + "\n" + r'''
import json,pathlib,subprocess,sys
args=sys.argv[1:]
if '--version' in args:
    print('fixture parallel');sys.exit(0)
assert args[args.index('--halt')+1]=='never' and args[args.index('--jobs')+1]=='2'
start=args.index('bash');end=args.index('::::')
command=args[start:end];rows=pathlib.Path(args[end+1]).read_text().splitlines()
assert rows==['m0_e0_k0_s1','m1_e0_k0_s1']
codes=[subprocess.run([row if value=='{}' else value for value in command]).returncode for row in rows]
pathlib.Path(args[args.index('--joblog')+1]).write_text(json.dumps(codes))
sys.exit(1 if any(codes) else 0)
''')
    executable(binaries / "srun", "#!" + sys.executable + "\n" + r'''
import os,subprocess,sys
args=sys.argv[1:]
assert args[:6]==['--exclusive','--exact','--nodes=1','--ntasks=1','--cpus-per-task=1','--kill-on-bad-exit=0']
sys.exit(subprocess.run(args[6:],env=dict(os.environ,SLURM_STEP_ID='0')).returncode)
''')
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"], VENV=str(virtual),
               SLURM_JOB_ID="9101", SLURM_NTASKS="2", SLURM_CPUS_PER_TASK="1")
    result = subprocess.run(["bash", str(source / "hpc/discovery/sparse_smart_v2_retained_state_bic.sbatch"),
                             str(root), "pool", "2"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    folder = root / "campaign/attempts/pool-attempt-0001"
    assert json.loads((folder / "parallel-joblog.tsv").read_text()) == [7, 0]
    assert (folder / "exit-code.txt").read_text().strip() == "1"
    assert (folder / "source-integrity-exit-code.txt").read_text().strip() == "0"
    for cid, code in [("m0_e0_k0_s1", 7), ("m1_e0_k0_s1", 0)]:
        for ending in (".exit-code.txt", ".process-exit-code.txt"):
            assert (root / f"logs/cases/{cid}{ending}").read_text().strip() == str(code)
        assert "SCORING " + cid in (root / f"logs/cases/{cid}.log").read_text()
    assert not (tmp_path / "INJECTED").exists() and not (root / "INJECTED").exists()


def test_shell_scripts_parse():
    for name in ("sparse_smart_v2_retained_state_bic.sbatch", "sparse_smart_v2_retained_state_bic_worker.sh"):
        subprocess.run(["bash", "-n", str(HPC / name)], check=True, capture_output=True)
