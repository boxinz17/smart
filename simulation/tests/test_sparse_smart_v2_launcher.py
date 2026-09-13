"""Local launcher fixtures with fake scheduler tools; no jobs or fits are run."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


CODE = Path(__file__).resolve().parents[2]
HPC = CODE / "hpc" / "discovery"
SCRIPTS = (
    "submit_sparse_smart_v2_pilot.sh", "sparse_smart_v2_prepare.sbatch",
    "sparse_smart_v2_pool.sbatch", "sparse_smart_v2_worker.sh",
)


def executable(path, source, *, python=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((f"#!{sys.executable}\n" if python else "#!/usr/bin/env bash\n") + source)
    path.chmod(0o755)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def marker(root):
    plan = json.loads((root / "plan.json").read_text())
    value = {
        "status": "complete", "success": True,
        "plan_sha256": digest(root / "plan.json"),
        "source_manifest_sha256": digest(root / "source-manifest.json"),
        "n_cases": len(plan["cases"]), "n_tasks": len(plan["tasks"]),
    }
    (root / "preparation.json").write_text(json.dumps(value))


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "run with $literal {1} characters"
    scripts = root / "source" / "hpc" / "discovery"
    scripts.mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copyfile(HPC / name, scripts / name)
    (scripts / "env.sh").write_text(
        'export PYTHONPATH="$SMART_SOURCE_ROOT/smart:$SMART_SOURCE_ROOT/sparse-smart/src"\n'
        'export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n'
    )
    runner = root / "source" / "simulation" / "run_sparse_smart_v2_pilot.py"
    executable(runner, '''
import argparse, hashlib, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['prepare', 'fit'])
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--task', type=int)
args = parser.parse_args()
root = args.root
with (root / 'logs' / 'runner-calls.jsonl').open('a') as handle:
    handle.write(json.dumps({'mode': args.mode, 'task': args.task,
                            'pythonpath': os.environ.get('PYTHONPATH'),
                            'tmpdir': os.environ.get('TMPDIR'),
                            'mplconfigdir': os.environ.get('MPLCONFIGDIR')}) + '\\n')
if args.mode == 'prepare':
    plan = json.loads((root / 'plan.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (root / 'preparation.json').write_text(json.dumps({
        'status': 'complete', 'success': True,
        'plan_sha256': digest(root / 'plan.json'),
        'source_manifest_sha256': digest(root / 'source-manifest.json'),
        'n_cases': len(plan['cases']), 'n_tasks': len(plan['tasks'])}))
else:
    (root / 'tasks' / f'{args.task:05d}' / 'status.json').write_text(json.dumps({'task': args.task}))
    if os.environ.get('FAKE_FAIL_TASK') == str(args.task):
        raise SystemExit(7)
''')
    (root / "plan.json").write_text(json.dumps({"cases": [{"id": 0}], "tasks": [{"id": i} for i in range(3)]}))
    (root / "work-items.tsv").write_text("0\n1\n2\n")
    files = {str(path.relative_to(root / "source")): digest(path)
             for path in (root / "source").rglob("*") if path.is_file()}
    (root / "source-manifest.json").write_text(json.dumps({
        "schema_version": 1, "source_root": str(root / "source"), "files": files,
    }))
    marker(root)
    (root / "logs").mkdir()
    bin_dir = tmp_path / "bin"
    executable(bin_dir / "sbatch", '''
import os
from pathlib import Path
Path(os.environ['FAKE_SCHEDULER_CALLED']).write_text('unexpected submission')
raise SystemExit(99)
''')
    executable(bin_dir / "srun", '''
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['FAKE_SRUN_LOG']).open('a') as handle:
    handle.write(json.dumps(args) + '\\n')
command = args[args.index('bash'):]
environment = dict(os.environ, SLURM_STEP_ID=command[-1], SLURM_CPUS_PER_TASK='1')
raise SystemExit(subprocess.run(command, env=environment).returncode)
''')
    executable(bin_dir / "parallel", '''
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
if '--version' in args:
    print('GNU parallel fixture')
    raise SystemExit(0)
Path(os.environ['FAKE_PARALLEL_LOG']).write_text(json.dumps(args))
split = args.index('::::')
command = args[args.index('bash'):split]
rows = Path(args[split+1]).read_text().splitlines()
if os.environ.get('FAKE_MUTATE_TASK_FILE'):
    Path(args[split+1]).write_text(os.environ['FAKE_MUTATE_TASK_FILE'])
joblog = Path(args[args.index('--joblog')+1])
joblog.write_text('Seq\\tExitval\\n')
failed = 0
for sequence, row in enumerate(rows, 1):
    current = [value.replace('{1}', row) for value in command]
    result = subprocess.run(current, env=dict(os.environ, PARALLEL_SEQ=str(sequence)))
    with joblog.open('a') as handle:
        handle.write(f'{sequence}\\t{result.returncode}\\n')
    failed += result.returncode != 0
raise SystemExit(min(failed, 100))
''')
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text("module() { return 0; }\n")
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               SMART_V2_PLAN_PYTHON=sys.executable,
               VENV=str(Path(sys.executable).parent.parent), BASH_ENV=str(bash_env),
               FAKE_SCHEDULER_CALLED=str(tmp_path / "scheduler-called"),
               FAKE_SRUN_LOG=str(tmp_path / "srun.jsonl"),
               FAKE_PARALLEL_LOG=str(tmp_path / "parallel.json"),
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("SLURM_JOB_ID", None)
    return root, env


def run_script(root, env, name, *args):
    return subprocess.run(["bash", str(root / "source/hpc/discovery" / name), *map(str, args)],
                          env=env, capture_output=True, text=True, timeout=30)


def test_dry_run_has_separate_preparation_and_100_distributed_cpu_pool(campaign):
    root, env = campaign
    result = run_script(root, env, SCRIPTS[0], "--run-dir", root, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert not Path(env["FAKE_SCHEDULER_CALLED"]).exists()
    submission = json.loads((root / "submission-plan.json").read_text())
    prepare, pool = submission["prepare"]["command"], submission["pool"]["command"]
    assert "--ntasks=1" in prepare and "--mem=8G" in prepare and "--time=00:15:00" in prepare
    assert "--ntasks=100" in pool and "--cpus-per-task=1" in pool
    assert "--mem-per-cpu=4G" in pool and "--time=12:00:00" in pool
    assert "--dependency=afterok:PREPARE_JOB_ID" in pool
    assert pool[-2:] == [str(root), "100"]
    assert prepare[-1] == str(root)
    assert not any(arg.startswith(("--nodes", "--ntasks-per-node", "--array")) for arg in pool)
    assert not (root / "submission.lock").exists()


def test_explicit_resources_and_literal_root_survive_command_recording(campaign):
    root, env = campaign
    result = run_script(root, env, SCRIPTS[0], "--run-dir", root, "--workers", 100,
                        "--time", "04:00:00", "--mem", "8G", "--dry-run")
    assert result.returncode == 0, result.stderr
    pool = json.loads((root / "submission-plan.json").read_text())["pool"]["command"]
    assert "--time=04:00:00" in pool and "--mem-per-cpu=8G" in pool
    assert pool[-2] == str(root)
    assert not (root.parent / "literal").exists()


def test_actual_submission_rejects_local_root_without_scheduler_call(campaign):
    root, env = campaign
    if root.is_relative_to('/scratch2'):
        pytest.skip('An on-cluster temporary root already satisfies the storage constraint')
    result = run_script(root, env, SCRIPTS[0], "--run-dir", root)
    assert result.returncode != 0 and "/scratch2" in result.stderr
    assert not Path(env["FAKE_SCHEDULER_CALLED"]).exists()


@pytest.mark.parametrize("pool_failure", [False, True])
def test_fake_submission_records_real_dependency_ids_and_partial_failure(campaign, tmp_path, pool_failure):
    root, env = campaign
    # Adapt only the fixed storage prefix in this temporary script copy. This
    # exercises real submission control flow locally without providing a
    # production flag that could bypass the cluster storage constraint.
    script = root / 'source/hpc/discovery' / SCRIPTS[0]
    text = script.read_text()
    needle = 'case "$run_dir" in /scratch2/*)'
    assert text.count(needle) == 1
    script.write_text(text.replace(needle, f'case "$run_dir" in {shlex.quote(str(root.parent))}/*)'))
    log = tmp_path / 'submission-calls.jsonl'
    executable(tmp_path / 'bin/sbatch', '''
import json, os, sys
from pathlib import Path
log = Path(os.environ['FAKE_SUBMISSION_CALLS'])
calls = len(log.read_text().splitlines()) if log.exists() else 0
with log.open('a') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
if calls == 1 and os.environ['FAKE_POOL_FAILURE'] == '1':
    print('fixture pool rejection', file=sys.stderr)
    raise SystemExit(42)
print(f'{8101+calls};fixture-cluster')
''')
    env.update(FAKE_SUBMISSION_CALLS=str(log), FAKE_POOL_FAILURE=str(int(pool_failure)))
    result = run_script(root, env, SCRIPTS[0], '--run-dir', root)
    assert result.returncode == (42 if pool_failure else 0), result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands) == 2
    assert '--dependency=afterok:8101' in commands[1]
    record = json.loads((root / 'submission.json').read_text())
    assert record['prepare']['job_id'] == '8101'
    assert record['pool']['job_id'] == (None if pool_failure else '8102')
    assert (root / 'prepare-job-id.txt').read_text().strip() == '8101'
    assert (root / 'submission-exit-code.txt').read_text().strip() == str(result.returncode)
    if pool_failure:
        assert 'fixture pool rejection' in (root / 'logs/pool-submission.err').read_text()
        assert not (root / 'pool-job-id.txt').exists()
    else:
        assert (root / 'pool-job-id.txt').read_text().strip() == '8102'
    again = run_script(root, env, SCRIPTS[0], '--run-dir', root)
    assert again.returncode != 0
    assert len(log.read_text().splitlines()) == 2  # no implicit resubmission


@pytest.mark.parametrize("arguments", [("--workers", "0"), ("--workers", "true"),
                                       ("--mem", "-1G"), ("--time", "bad")])
def test_bad_resources_are_rejected(campaign, arguments):
    root, env = campaign
    result = run_script(root, env, SCRIPTS[0], "--run-dir", root, "--dry-run", *arguments)
    assert result.returncode != 0
    assert not Path(env["FAKE_SCHEDULER_CALLED"]).exists()


def test_work_items_are_validated_before_submission(campaign):
    root, env = campaign
    (root / "work-items.tsv").write_text("task\n0\n1\n")
    result = run_script(root, env, SCRIPTS[0], "--run-dir", root, "--dry-run")
    assert result.returncode != 0 and "headerless consecutive integer" in result.stderr


def test_pool_runs_remaining_tasks_after_failure_and_preserves_step_statuses(campaign):
    root, env = campaign
    env.update(SLURM_JOB_ID="123", SLURM_NTASKS="100", SLURM_CPUS_PER_TASK="1", FAKE_FAIL_TASK="1")
    result = run_script(root, env, "sparse_smart_v2_pool.sbatch", root, 100)
    assert result.returncode == 1, result.stderr
    parallel = json.loads(Path(env["FAKE_PARALLEL_LOG"]).read_text())
    assert parallel[parallel.index("--jobs")+1] == "100"
    assert parallel[parallel.index("--halt")+1] == "never"
    assert "--quote" in parallel and "--joblog" in parallel
    steps = [json.loads(line) for line in Path(env["FAKE_SRUN_LOG"]).read_text().splitlines()]
    assert len(steps) == 3
    for index, step in enumerate(steps):
        assert all(option in step for option in ("--exclusive", "--exact", "--nodes=1", "--ntasks=1", "--cpus-per-task=1"))
        assert step[-2:] == [str(root), str(index)]
        task = root / "tasks" / f"{index:05d}"
        expected = "7" if index == 1 else "0"
        assert (task / "process-exit-code.txt").read_text().strip() == expected
        assert (task / "launcher-exit-code.txt").read_text().strip() == expected
        assert (task / "process-status.tsv").exists() and (task / "launcher-status.tsv").exists()
        assert (task / "environment.txt").exists()
    assert (root / "pool-exit-code.txt").read_text().strip() == "1"
    assert (root / "source-integrity-exit-code.txt").read_text().strip() == "0"
    assert (root / "fit-status.txt").read_text().strip() == "failed"
    before = json.loads((root / "logs/source-integrity-before.json").read_text())
    after = json.loads((root / "logs/source-integrity-after.json").read_text())
    assert before["task_file"] == "work-items.tsv" and before["n_selected_tasks"] == 3
    assert before["task_file_sha256"] == digest(root / "work-items.tsv")
    assert after["task_selection_unchanged"] is True
    calls = [json.loads(line) for line in (root / "logs/runner-calls.jsonl").read_text().splitlines()]
    assert [call["task"] for call in calls] == [0, 1, 2]
    for call in calls:
        assert call["pythonpath"].split(":")[0] == str(root / "source/sparse-smart-v2/src")
        assert Path(call["tmpdir"]).is_relative_to(root)
        assert Path(call["mplconfigdir"]).is_relative_to(root)


def test_pool_dispatches_only_explicit_subset_and_records_its_hash(campaign):
    root, env = campaign
    subset = root / "retry-items.tsv"
    subset.write_text("2\n0\n")
    authority_hash = digest(root / "work-items.tsv")
    env.update(SLURM_JOB_ID="124", SLURM_NTASKS="100", SLURM_CPUS_PER_TASK="1")
    result = run_script(root, env, "sparse_smart_v2_pool.sbatch", root, 100, subset.name)
    assert result.returncode == 0, result.stderr
    parallel = json.loads(Path(env["FAKE_PARALLEL_LOG"]).read_text())
    assert parallel[parallel.index("::::") + 1] == str(subset)
    assert parallel[parallel.index("--jobs") + 1] == "100"
    steps = [json.loads(line) for line in Path(env["FAKE_SRUN_LOG"]).read_text().splitlines()]
    assert [step[-1] for step in steps] == ["2", "0"]
    assert not (root / "tasks/00001").exists()
    before = json.loads((root / "logs/source-integrity-before.json").read_text())
    after = json.loads((root / "logs/source-integrity-after.json").read_text())
    assert before["n_tasks"] == 3 and before["n_selected_tasks"] == 2
    assert before["task_file"] == subset.name and before["task_file_sha256"] == digest(subset)
    assert after["task_file_sha256"] == before["task_file_sha256"]
    assert after["task_selection_unchanged"] is True
    assert digest(root / "work-items.tsv") == authority_hash == before["work_items_sha256"]


@pytest.mark.parametrize("contents,message", [
    ("", "must not be empty"), ("\n", "canonical decimal"),
    ("0\n0\n", "distinct task IDs"), ("01\n", "canonical decimal"),
    ("+1\n", "canonical decimal"), ("-1\n", "canonical decimal"),
    (" 1\n", "canonical decimal"), ("1\t2\n", "canonical decimal"),
    ("task_id\n0\n", "canonical decimal"), ("0\r\n", "canonical decimal"),
    ("3\n", "out-of-range task ID"),
])
def test_pool_rejects_invalid_subset_before_dispatch(campaign, contents, message):
    root, env = campaign
    (root / "retry-items.tsv").write_bytes(contents.encode())
    env.update(SLURM_JOB_ID="124", SLURM_NTASKS="100", SLURM_CPUS_PER_TASK="1")
    result = run_script(root, env, "sparse_smart_v2_pool.sbatch", root, 100, "retry-items.tsv")
    assert result.returncode != 0 and message in result.stderr
    assert not Path(env["FAKE_PARALLEL_LOG"]).exists()
    assert not Path(env["FAKE_SRUN_LOG"]).exists()


@pytest.mark.parametrize("name", ["../retry-items.tsv", "sub/retry-items.tsv", "/tmp/retry-items.tsv",
                                   "./work-items.tsv", ".", "..", "", "retry items.tsv", "retry;false.tsv"])
def test_pool_rejects_unsafe_subset_paths(campaign, name):
    root, env = campaign
    env.update(SLURM_JOB_ID="124", SLURM_NTASKS="100", SLURM_CPUS_PER_TASK="1")
    result = run_script(root, env, "sparse_smart_v2_pool.sbatch", root, 100, name)
    assert result.returncode != 0 and "simple basename" in result.stderr
    assert not Path(env["FAKE_PARALLEL_LOG"]).exists()


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink_inside", "symlink_outside"])
def test_pool_requires_regular_subset_file_directly_in_run_root(campaign, kind):
    root, env = campaign
    subset = root / "retry-items.tsv"
    if kind == "directory":
        subset.mkdir()
    elif kind.startswith("symlink"):
        target = root / "work-items.tsv" if kind == "symlink_inside" else root.parent / "outside.tsv"
        if kind == "symlink_outside":
            target.write_text("0\n")
        subset.symlink_to(target)
    env.update(SLURM_JOB_ID="124", SLURM_NTASKS="100", SLURM_CPUS_PER_TASK="1")
    result = run_script(root, env, "sparse_smart_v2_pool.sbatch", root, 100, subset.name)
    assert result.returncode != 0 and "regular file directly within RUN_DIR" in result.stderr
    assert not Path(env["FAKE_PARALLEL_LOG"]).exists()


def test_pool_subset_cannot_bypass_full_work_item_authority(campaign):
    root, env = campaign
    (root / "retry-items.tsv").write_text("0\n")
    (root / "work-items.tsv").write_text("0\n1\n")
    env.update(SLURM_JOB_ID="124", SLURM_NTASKS="100", SLURM_CPUS_PER_TASK="1")
    result = run_script(root, env, "sparse_smart_v2_pool.sbatch", root, 100, "retry-items.tsv")
    assert result.returncode != 0 and "Work-item IDs do not match plan" in result.stderr
    assert not Path(env["FAKE_PARALLEL_LOG"]).exists()


def test_pool_detects_valid_subset_mutation_after_dispatch(campaign):
    root, env = campaign
    subset = root / "retry-items.tsv"
    subset.write_text("2\n0\n")
    original_hash = digest(subset)
    env.update(SLURM_JOB_ID="124", SLURM_NTASKS="100", SLURM_CPUS_PER_TASK="1",
               FAKE_MUTATE_TASK_FILE="0\n2\n")
    result = run_script(root, env, "sparse_smart_v2_pool.sbatch", root, 100, subset.name)
    assert result.returncode == 74, result.stderr
    before = json.loads((root / "logs/source-integrity-before.json").read_text())
    after = json.loads((root / "logs/source-integrity-after.json").read_text())
    assert before["task_file_sha256"] == original_hash != after["task_file_sha256"]
    assert after["n_selected_tasks"] == before["n_selected_tasks"] == 2
    assert after["task_selection_unchanged"] is False
    assert "Task selection changed" in (root / "logs/source-integrity-after.err").read_text()
    assert (root / "source-integrity-exit-code.txt").read_text().strip() == "1"
    assert (root / "fit-status.txt").read_text().strip() == "failed"
    assert (root / "pool-exit-code.txt").read_text().strip() == "74"


@pytest.mark.parametrize("change", ["missing_marker", "changed_plan", "changed_source"])
def test_pool_rejects_missing_or_stale_preparation_before_parallel(campaign, change):
    root, env = campaign
    if change == "missing_marker":
        (root / "preparation.json").unlink()
    elif change == "changed_plan":
        with (root / "plan.json").open("a") as handle:
            handle.write(" ")
    else:
        with (root / "source/simulation/run_sparse_smart_v2_pilot.py").open("a") as handle:
            handle.write("\n# changed\n")
    env.update(SLURM_JOB_ID="123", SLURM_NTASKS="100", SLURM_CPUS_PER_TASK="1")
    result = run_script(root, env, "sparse_smart_v2_pool.sbatch", root, 100)
    assert result.returncode != 0
    assert not Path(env["FAKE_PARALLEL_LOG"]).exists()
    assert (root / "fit-status.txt").read_text().strip() == "failed"


def test_wrong_allocation_and_noninteger_worker_id_are_rejected(campaign):
    root, env = campaign
    env.update(SLURM_JOB_ID="123", SLURM_NTASKS="1", SLURM_CPUS_PER_TASK="1")
    result = run_script(root, env, "sparse_smart_v2_pool.sbatch", root, 100)
    assert result.returncode != 0
    result = run_script(root, env, "sparse_smart_v2_worker.sh", "dispatch", root, "1;false")
    assert result.returncode != 0
    assert not Path(env["FAKE_SRUN_LOG"]).exists()


def test_preparation_runs_imports_and_focused_tests_before_ready_marker(campaign, tmp_path):
    root, env = campaign
    (root / "preparation.json").unlink()
    venv = tmp_path / "prepare-venv"
    executable(venv / "bin/python", f'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['FAKE_PREPARE_COMMANDS']).open('a') as handle:
    handle.write(json.dumps(args) + '\\n')
if args[0] == '-c' or args[:2] == ['-m', 'pytest']:
    raise SystemExit(0)
os.execv({sys.executable!r}, [{sys.executable!r}] + args)
''')
    env.update(VENV=str(venv), SLURM_JOB_ID="122", SLURM_NTASKS="1", SLURM_CPUS_PER_TASK="1",
               FAKE_PREPARE_COMMANDS=str(tmp_path / "prepare-commands.jsonl"))
    result = run_script(root, env, "sparse_smart_v2_prepare.sbatch", root)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in Path(env["FAKE_PREPARE_COMMANDS"]).read_text().splitlines()]
    assert commands[0][0] == "-c" and "numpy" in commands[0][1]
    assert commands[1][:2] == ["-m", "pytest"]
    assert "--import-mode=importlib" in commands[1]
    assert "no:cacheprovider" in commands[1]
    assert commands[2][-3:] == ["prepare", "--root", str(root)]
    assert json.loads((root / "preparation.json").read_text())["success"]
    assert (root / "preparation-exit-code.txt").read_text().strip() == "0"


def test_scripts_have_no_alternate_storage_or_installation_paths():
    text = "\n".join((HPC / name).read_text() for name in SCRIPTS)
    assert "/" + "scratch" + "1" not in text
    assert "--array" not in text
    assert "pip install" not in text
    assert "scancel" not in text
    assert "rsync" not in text
    assert "tar " not in text
    pool = (HPC / "sparse_smart_v2_pool.sbatch").read_text()
    assert pool.index("module load gcc/13.3.0 python/3.12.8") < pool.index(
        'verify_preparation > "$run_dir/logs/source-integrity-before.json"')
