"""No-fit fixtures for frozen source, exact chunk ownership and fake Slurm."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aggregate_sparse_smart_cases as cases
import discovery_budget_study as study
import discovery_case_aggregation as launcher


def _plans(tmp_path, *, models=(1,), seeds=(0, 1), settings=None):
    roots = []
    for experiment in (0, 1):
        root = tmp_path / f"raw inputs ' $x ; `untouched` {experiment}"
        plan = study.make_plan(models=models, experiments=(experiment,), seed_ids=seeds,
                               setting_index=settings)
        study.write_plan(plan, root)
        roots.append(root)
    return roots


def _prepare(tmp_path, **kwargs):
    roots = _plans(tmp_path)
    output = tmp_path / "analysis ' $x ; `untouched`"
    path = launcher.prepare_launch(roots, output, cases_per_chunk=3, **kwargs)
    return roots, output, path, launcher.verify_launch(path)[0]


def _policy_inputs(tmp_path):
    root = tmp_path / "cancelled producer"
    plan = study.make_plan(models=(0,), experiments=(0,), seed_ids=(0,),
                           setting_index=0, tuning_task_size=1)
    study.write_plan(plan, root)
    policy = [dict(run_root=str(root), task_id=task["task_id"],
                   reason="Explicitly cancelled after the other candidates finished",
                   launcher_exit_code=137) for task in plan["work_items"][:3]]
    return root, policy


def test_chunks_exactly_own_cases_grouped_by_run_including_exclusions(tmp_path):
    roots, output, path, launch = _prepare(tmp_path)
    index = cases.load_index(output / "campaign-index.json")
    chunks = launcher.read_json(path.parent / "chunks.json")
    keys = [key for chunk in chunks for key in chunk["case_keys"]]
    ordered = sorted(index["cases"], key=lambda row: (row["run_root"], row["case_key"]))
    assert keys == [row["case_key"] for row in ordered]
    assert len(keys) == len(set(keys)) == 22
    assert launch["applicable_cases"] == 20
    assert launch["scope"] == {"models": [1]}
    assert launch["publication_mode"] == "compact"
    assert len(chunks) == 8 and max(len(row["case_keys"]) for row in chunks) == 3
    assert not list(output.rglob("BudgetStudy_result*"))
    assert not (output / "cases").exists()
    assert not any("raw inputs" in key for key in launch["source"]["files"])
    assert all(root.exists() for root in roots)


def test_model_scope_has_2400_cases_and_96_chunks_for_model_two(tmp_path):
    root = tmp_path / "all-models"
    study.write_plan(study.make_plan(), root)
    path = launcher.prepare_launch([root], tmp_path / "analysis", models=[1])
    launch, chunks = launcher.verify_launch(path)
    assert launch["case_count"] == 2400
    assert launch["applicable_cases"] == 2100
    assert len(chunks) == 96
    assert all(key.startswith("m1_") for chunk in chunks for key in chunk["case_keys"])


def test_dry_run_without_site_packages_scheduler_or_numerical_imports(tmp_path):
    roots = _plans(tmp_path, seeds=(0,), settings=0)
    output = tmp_path / "dry analysis"
    script = """
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class NoNumerical(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in {'numpy','scipy','sklearn','sparse_smart','smart'}, fullname
sys.meta_path.insert(0, NoNumerical())
import discovery_case_aggregation as launch
def forbidden(*args, **kwargs):
    raise AssertionError('No subprocess during dry run')
launch.subprocess.run = forbidden
raise SystemExit(launch.main(sys.argv[2:]))
"""
    result = subprocess.run([sys.executable, "-B", "-S", "-c", script, str(launcher.HERE),
        "prepare", "--run-roots", *map(str, roots), "--output-root", str(output), "--models", "1", "--dry-run"],
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    printed = json.loads(result.stdout)
    assert printed["no_audits_performed"] and printed["no_fits_performed"]
    assert not (output / "launcher/submission.json").exists()
    assert not (output / "cases").exists()
    assert printed["resources"]["pilot_requests_provisional"]


@pytest.mark.parametrize("evidence_kind", ["launcher", "slurm"])
def test_prepare_forwards_missing_policy_and_binds_normalized_index(tmp_path, monkeypatch, evidence_kind):
    root, policy = _policy_inputs(tmp_path)
    if evidence_kind == "slurm":
        policy = policy[:1]
        policy[0].pop("launcher_exit_code")
        policy[0]["cancellation_evidence"] = dict(
            path=str(tmp_path / "cancellation.json"), sha256="a" * 64,
            job_id="11859590", step_id="11859590.1620", task_id=policy[0]["task_id"])
    calls = []
    prepare_campaign = cases.prepare_campaign
    def prepare(*args, **kwargs):
        calls.append(deepcopy(kwargs))
        return prepare_campaign(*args, **kwargs)
    monkeypatch.setattr(cases, "prepare_campaign", prepare)
    path = launcher.prepare_launch([root], tmp_path / "analysis", models=[0],
                                   allowed_missing_tasks=policy)
    launch, _ = launcher.verify_launch(path)
    index = cases.load_index(path.parent.parent / "campaign-index.json")
    assert calls[0]["allowed_missing_tasks"] == policy
    assert index["schema_version"] == 3
    assert launch["allowed_missing_tasks"] == index["allowed_missing_tasks"]
    assert launch["allowed_missing_tasks"][0]["case_key"] == "m0_e0_s0_k0"
    assert launch["allowed_missing_tasks"][0]["grid_candidate_ids"] == [0]
    launch["allowed_missing_tasks"][0]["reason"] = "unbound replacement"
    launcher.atomic_json(launch, path)
    with pytest.raises(ValueError, match="policy differs"):
        launcher.verify_launch(path)


def test_missing_policy_dry_run_freezes_contents_without_raw_or_scheduler_changes(tmp_path, monkeypatch, capsys):
    root, policy = _policy_inputs(tmp_path)
    policy_path = tmp_path / "allowed missing.json"
    policy_path.write_text(json.dumps(policy))
    raw_before = {name: launcher.sha(root / name) for name in ("study-plan.json", "work-items.tsv")}
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not invoke a scheduler or numerical audit")
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    monkeypatch.setattr(cases, "aggregate_cases", forbidden)
    output = tmp_path / "policy dry run"
    assert launcher.main(["prepare", "--run-roots", str(root), "--output-root", str(output),
                          "--models", "0", "--allowed-missing-tasks", str(policy_path), "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out)
    path = Path(printed["launch"])
    launch, _ = launcher.verify_launch(path)
    assert launch["allowed_missing_tasks"][0]["task_id"] == policy[0]["task_id"]
    assert raw_before == {name: launcher.sha(root / name) for name in raw_before}
    assert not (output / "cases").exists()
    assert not (path.parent / "submission.json").exists()
    # The original policy file is an input, not a mutable runtime override.
    policy_path.write_text("[]\n")
    assert launcher.verify_launch(path)[0]["allowed_missing_tasks"] == launch["allowed_missing_tasks"]
    index_path = output / "campaign-index.json"
    index = launcher.read_json(index_path)
    index["allowed_missing_tasks"][0]["reason"] = "changed after preparation"
    launcher.atomic_json(index, index_path)
    with pytest.raises(ValueError, match="Campaign index changed"):
        launcher.verify_launch(path)


def test_sbatch_argv_one_cpu_afterany_and_literal_paths(tmp_path, monkeypatch):
    _, _, path, launch = _prepare(tmp_path, concurrency=16, memory="12G", time_limit="08:00:00")
    commands = []
    def sbatch(argv, **kwargs):
        commands.append(argv)
        assert kwargs == {"check": False, "text": True, "capture_output": True}
        assert launcher.read_json(path.parent / "submission.json")["stages"][
            "array" if len(commands) == 1 else "summary"]["state"] == "intent"
        return subprocess.CompletedProcess(argv, 0, f"{12000 + len(commands)};discovery\n", "")
    monkeypatch.setattr(launcher.subprocess, "run", sbatch)
    receipt = launcher.submit_launch(path)
    assert len(commands) == 2
    for command in commands:
        assert command[:5] == ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "--cpus-per-task=1"]
        assert "--mem=12G" in command and "--time=08:00:00" in command
        assert command[-3:] == [launch["source_root"], str(path), launcher.sha(path)]
    assert "--array=0-7%16" in commands[0]
    assert "--dependency=afterany:12001" in commands[1]
    assert not any("afterok" in arg for command in commands for arg in command)
    assert receipt["stages"]["array"]["job_id"] == "12001"
    assert receipt["stages"]["summary"]["job_id"] == "12002"
    with pytest.raises(ValueError, match="already attempted"):
        launcher.submit_launch(path)
    assert len(commands) == 2


@pytest.mark.parametrize("summary", [False, True])
def test_log_directory_percent_is_literal_but_filename_ids_expand(tmp_path, summary):
    roots = _plans(tmp_path, seeds=(0,), settings=0)
    output = tmp_path / "analysis %A %j %%"
    path = launcher.prepare_launch(roots, output)
    launch, _ = launcher.verify_launch(path)
    arguments = launcher.sbatch_arguments(path, launch, summary=summary,
                                         array_job_id="12345" if summary else None)
    literal_directory = str(tmp_path / "analysis %%A %%j %%%%" / "launcher/logs")
    filename = "summary-%j" if summary else "case-%A_%a"
    assert f"--output={literal_directory}/{filename}.out" in arguments
    assert f"--error={literal_directory}/{filename}.err" in arguments
    # Non-pattern arguments must retain the original, unescaped paths.
    assert arguments[-3:] == [launch["source_root"], str(path), launcher.sha(path)]


@pytest.mark.parametrize("reply", [(0, "unrecognized successful response"), (1, ""), OSError("network interrupted")])
def test_uncertain_submission_never_automatically_resubmits(tmp_path, monkeypatch, reply):
    _, _, path, _ = _prepare(tmp_path)
    commands = []
    def sbatch(argv, **kwargs):
        commands.append(argv)
        if isinstance(reply, Exception):
            raise reply
        return subprocess.CompletedProcess(argv, reply[0], reply[1], "example scheduler diagnostic")
    monkeypatch.setattr(launcher.subprocess, "run", sbatch)
    with pytest.raises(RuntimeError, match="reconcil"):
        launcher.submit_launch(path)
    receipt = launcher.read_json(path.parent / "submission.json")
    assert receipt["stages"]["array"]["state"] == "uncertain"
    assert "summary" not in receipt["stages"]
    with pytest.raises(ValueError, match="already attempted"):
        launcher.submit_launch(path)
    with pytest.raises(ValueError, match="known submitted array"):
        launcher.submit_launch(path, summary_only=True)
    assert len(commands) == 1


def test_recover_unattempted_summary_only_from_durable_known_array(tmp_path, monkeypatch):
    _, _, path, launch = _prepare(tmp_path)
    receipt = dict(schema_version=1, launch_sha256=launcher.sha(path), stages={
        "array": dict(state="submitted", job_id="12345")})
    launcher.atomic_json(receipt, path.parent / "submission.json")
    commands = []
    def sbatch(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, "12346\n", "")
    monkeypatch.setattr(launcher.subprocess, "run", sbatch)
    launcher.submit_launch(path, summary_only=True)
    assert len(commands) == 1 and "--dependency=afterany:12345" in commands[0]
    assert not any(arg.startswith("--array=") for arg in commands[0])
    with pytest.raises(ValueError, match="already attempted"):
        launcher.submit_launch(path, summary_only=True)


@pytest.mark.parametrize("name", ["source/simulation/aggregate_sparse_smart_cases.py", "chunks.json", "../campaign-index.json"])
def test_changed_frozen_source_or_metadata_rejected_before_submission(tmp_path, monkeypatch, name):
    _, _, path, _ = _prepare(tmp_path)
    target = path.parent / name
    target.chmod(0o644)
    target.write_text(target.read_text() + "\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject changed artifacts before scheduler access")
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="changed"):
        launcher.submit_launch(path)


def test_analysis_checkout_change_does_not_change_frozen_runtime(tmp_path):
    working = tmp_path / "working source"
    launcher.freeze_source(launcher.ROOT, working)
    roots = _plans(tmp_path)
    path = launcher.prepare_launch(roots, tmp_path / "analysis", source_root=working)
    target = working / "simulation/aggregate_sparse_smart_cases.py"
    target.chmod(0o644)
    target.write_text("raise RuntimeError('changed checkout is not the runtime')\n")
    launch, _ = launcher.verify_launch(path)
    frozen = Path(launch["source_root"]) / target.relative_to(working)
    assert frozen.read_text() != target.read_text()
    with pytest.raises(ValueError, match="frozen source"):
        launcher.verify_launch(path, launcher.sha(path), executing_source=working)


def test_source_symlinks_and_raw_output_overlap_rejected(tmp_path):
    roots, output, path, launch = _prepare(tmp_path)
    target = Path(launch["source_root"]) / "smart/smart/__init__.py"
    target.unlink()
    target.symlink_to(launcher.ROOT / "smart/smart/__init__.py")
    with pytest.raises(ValueError, match="symlink"):
        launcher.verify_launch(path)
    with pytest.raises(ValueError, match="separate from raw"):
        launcher.prepare_launch(roots, roots[0] / "analysis")
    with pytest.raises(ValueError, match="already prepared"):
        launcher.prepare_launch(roots, output)


def test_worker_is_one_process_sequential_case_api_and_preserves_failures(tmp_path, monkeypatch):
    _, _, path, launch = _prepare(tmp_path)
    chunks = launcher.read_json(path.parent / "chunks.json")
    calls = []
    def aggregate(index_path, *, workers, case_keys):
        calls.append((index_path, workers, case_keys))
        return [dict(case_key=key, scientific=dict(audit_passed=i != 1),
                     execution=dict(fit_success=True, launch_success=i != 2))
                for i, key in enumerate(case_keys)]
    monkeypatch.setattr(cases, "aggregate_cases", aggregate)
    status = launcher.run_chunk(path, 0, launcher.sha(path), executing_source=launch["source_root"])
    assert status == 1
    assert calls == [(launch["index"], 1, chunks[0]["case_keys"])]
    reports = list((path.parent / "reports").glob("chunk-0-*.json"))
    assert len(reports) == 1
    assert launcher.read_json(reports[0])["failed_cases"] == chunks[0]["case_keys"][1:]


@pytest.mark.parametrize("audit_passed,available_success,reported_ids,expected_status", [
    (True, True, "expected", 0),
    (True, True, "subset", 0),
    (False, True, "expected", 1),
    (True, False, "expected", 1),
    (True, True, [], 1),
    (True, True, ["unlisted_task"], 1),
    (True, True, "duplicate", 1),
    (True, True, None, 1),
])
def test_worker_only_accepts_audited_available_tasks_bound_to_frozen_policy(
        tmp_path, monkeypatch, audit_passed, available_success, reported_ids, expected_status):
    root, policy = _policy_inputs(tmp_path)
    path = launcher.prepare_launch([root], tmp_path / "analysis", allowed_missing_tasks=policy)
    launch, chunks = launcher.verify_launch(path)
    case_key = chunks[0]["case_keys"][0]
    ids = [item["task_id"] for item in policy]
    if reported_ids == "expected":
        reported_ids = ids
    elif reported_ids == "subset":
        reported_ids = ids[:1]
    elif reported_ids == "duplicate":
        reported_ids = ids * 2
    report = dict(case_key=case_key, scientific=dict(audit_passed=audit_passed),
                  execution=dict(fit_success=False, launch_success=False,
                                 available_tasks_success=available_success,
                                 allowed_missing_task_ids=reported_ids))
    monkeypatch.setattr(cases, "aggregate_cases", lambda *args, **kwargs: [report])
    assert launcher.run_chunk(path, 0, launcher.sha(path),
                              executing_source=launch["source_root"]) == expected_status
    saved = launcher.read_json(next((path.parent / "reports").glob("chunk-0-*.json")))
    assert saved["failed_cases"] == ([case_key] if expected_status else [])


def test_worker_does_not_allow_partial_execution_without_frozen_policy(tmp_path, monkeypatch):
    _, _, path, launch = _prepare(tmp_path)
    def aggregate(index_path, *, workers, case_keys):
        return [dict(case_key=key, scientific=dict(audit_passed=True),
                     execution=dict(fit_success=False, launch_success=False,
                                    available_tasks_success=True,
                                    allowed_missing_task_ids=[f"{key}_g0"])) for key in case_keys]
    monkeypatch.setattr(cases, "aggregate_cases", aggregate)
    assert launcher.run_chunk(path, 0, launcher.sha(path), executing_source=launch["source_root"]) == 1


def test_worker_refuses_incomplete_returned_case_coverage(tmp_path, monkeypatch):
    _, _, path, launch = _prepare(tmp_path)
    monkeypatch.setattr(cases, "aggregate_cases", lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match="incomplete"):
        launcher.run_chunk(path, 0, launcher.sha(path), executing_source=launch["source_root"])
    assert not list((path.parent / "reports").glob("*.json"))


@pytest.mark.parametrize("script", ["case_aggregation.sbatch", "case_summary.sbatch"])
def test_batch_validates_env_once_no_install_and_propagates_exit(tmp_path, script):
    source = tmp_path / "source ' $literal"
    env = source / "hpc/discovery/env.sh"
    env.parent.mkdir(parents=True)
    env.write_text('printf "validated\\n" >> "$CHECK_LOG"\n')
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in (("python3", "exit 0"), ("python", 'printf "worker\\n" >> "$CHECK_LOG"; exit 17'),
                       ("pip", "exit 99"), ("rsync", "exit 99"), ("cp", "exit 99")):
        binary = binaries / name
        binary.write_text("#!/bin/bash\n" + body + "\n")
        binary.chmod(0o755)
    check = tmp_path / "checks"
    environment = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"], CHECK_LOG=str(check),
                       SLURM_NTASKS="1", SLURM_CPUS_PER_TASK="1", SLURM_ARRAY_TASK_ID="0")
    result = subprocess.run(["bash", str(launcher.ROOT / "hpc/discovery" / script), str(source),
                             str(tmp_path / "launch ' $.json"), "f" * 64], env=environment, capture_output=True, text=True)
    assert result.returncode == 17, result.stderr
    assert check.read_text().splitlines() == ["validated", "worker"]
    environment["SLURM_CPUS_PER_TASK"] = "2"
    result = subprocess.run(["bash", str(launcher.ROOT / "hpc/discovery" / script), str(source), "unused", "f" * 64],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 2
    assert check.read_text().splitlines() == ["validated", "worker"]


def test_backslash_output_path_rejected_before_preparation(tmp_path):
    roots = _plans(tmp_path)
    output = tmp_path / r"analysis\%A"
    with pytest.raises(ValueError, match="backslash"):
        launcher.prepare_launch(roots, output)
    assert not output.exists()


def test_campaign_submission_convenience_and_input_validation(tmp_path):
    roots = _plans(tmp_path)
    path = tmp_path / "submission.json"
    launcher.atomic_json(dict(jobs=[dict(run_dir=str(root)) for root in roots]), path)
    assert launcher.resolve_run_roots(campaign_submission=path) == list(map(str, roots))
    with pytest.raises(ValueError, match="either"):
        launcher.resolve_run_roots(roots, path)
    for kwargs in (dict(concurrency=0), dict(cases_per_chunk=0), dict(memory="8G; touch anything"),
                   dict(time_limit="00:00:00"), dict(account="bad\n#SBATCH")):
        with pytest.raises(ValueError):
            launcher.prepare_launch(roots, tmp_path / "unused", **kwargs)
