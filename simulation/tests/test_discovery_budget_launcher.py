"""Exercise launcher boundaries with temporary sources and fake scheduler tools.

These tests never call the cluster, an estimator, or a real Slurm command.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
HPC = REPO / "hpc" / "discovery"
SCRIPTS = ("submit_budget_study.sh", "budget_pool.sbatch", "budget_worker.sh")


def executable(path, body, *, python=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((f"#!{sys.executable}\n" if python else "#!/usr/bin/env bash\n") + body)
    path.chmod(0o755)


@pytest.fixture
def checkout(tmp_path):
    """Enough source for the real stdlib planner; no scientific imports occur."""
    root = tmp_path / "checkout"
    for name in SCRIPTS:
        target = root / "hpc" / "discovery" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HPC / name, target)
    simulation = root / "simulation"
    simulation.mkdir()
    for name in (
        "discovery_budget_study.py", "run_restricted_rrr.py", "run_sparse_smart.py",
        "run_sparse_smart_tuned.py", "run_sparse_smart_external.py", "external_validation_data.py",
        "run_sparse_smart_budget_study.py", "batch_manifest.py", "sparse_smart_selection.py",
        "sparse_smart_provenance.py", "merge_sparse_smart_budget_shards.py",
    ):
        shutil.copyfile(REPO / "simulation" / name, simulation / name)
    seed = simulation / "data" / "random_seeds" / "experiment_seeds.csv"
    seed.parent.mkdir(parents=True)
    shutil.copyfile(REPO / "simulation" / "data" / "random_seeds" / seed.name, seed)
    for package in ("smart/smart", "sparse-smart/src/sparse_smart"):
        target = root / package
        target.mkdir(parents=True)
        (target / "__init__.py").write_text("# A source-content planning fixture, never imported.\n")
    (root / ".python-version").write_text("3.12\n")
    (root / "python-constraints.txt").write_text("numpy==2.5.3\n")
    (root / "results").mkdir()
    (root / "results" / "must-not-copy.json").write_text("{}\n")
    return root


def dry_run(checkout, tmp_path, arguments=(), *, special_path=False):
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "scheduler-was-called"
    executable(bin_dir / "sbatch", 'touch "$SCHEDULER_MARKER"\nexit 99\n')
    name = "runs with $literal {1} % text" if special_path else "runs"
    run_root, archive_root = tmp_path / name, tmp_path / "archive"
    environment = dict(os.environ, SMART_PLAN_PYTHON=sys.executable,
                       PATH=f"{bin_dir}:{os.environ['PATH']}", SCHEDULER_MARKER=str(marker))
    result = subprocess.run(
        ["bash", str(checkout / "hpc/discovery/submit_budget_study.sh"),
         "--run-root", str(run_root), "--archive-root", str(archive_root),
         "--dry-run", *arguments],
        env=environment, capture_output=True, text=True, timeout=30,
    )
    assert not marker.exists(), "A dry run attempted scheduler submission"
    roots = sorted(run_root.glob("*")) if run_root.exists() else []
    return result, roots, archive_root


def test_one_seed_dry_run_splits_full_grid_and_distributes_resources(checkout, tmp_path):
    result, roots, archive_root = dry_run(checkout, tmp_path, ("--workers", "32", "--seeds", "0"))
    assert result.returncode == 0, result.stderr
    assert len(roots) == 1
    run = roots[0]
    plan = json.loads((run / "study-plan.json").read_text())
    rows = [line.split("\t") for line in (run / "work-items.tsv").read_text().splitlines()]
    assert len(rows) == plan["expected_tasks"] == 3033
    assert plan["expected_cells"] == 72
    assert plan["tuning_task_size"] == 1
    assert len({row[0] for row in rows}) == 3033
    assert all(len(row) == 5 and row[0].startswith("m") for row in rows)
    assert rows[0] == ["m0_e0_s0_k0_g0", "0", "0", "0", "0"]
    assert plan["expected_applicable"] == 63
    assert plan["expected_inapplicable"] == 9
    assert plan["configuration"]["iteration_budgets"] == [500, 1000, 2000, 4000, 8000]
    assert plan["configuration"]["init_penalties"] == [.01, .03, .1]
    assert plan["configuration"]["penalties_u"] == plan["configuration"]["penalties_v"] == [.0025, .01, .04, .16]
    assert plan["configuration"]["validation_iterations"] == [10, 25, 50, 100, 150, 200]
    stored = next(line.removeprefix("budget_args=(").removesuffix(")")
                  for line in (run / "budget-job-config.sh").read_text().splitlines()
                  if line.startswith("budget_args=("))
    budget_args = shlex.split(stored)
    assert budget_args[budget_args.index("--validation-iterations")+1] == "10,25,50,100,150,200"
    assert "--tuning-task-size" not in budget_args
    command = shlex.split((run / "submission-command.txt").read_text())
    assert "--ntasks=32" in command and "--cpus-per-task=1" in command
    assert "--mem-per-cpu=8G" in command and "--time=24:00:00" in command
    assert not any(arg.startswith(("--nodes", "--ntasks-per-node", "--array", "--exclusive")) for arg in command)
    assert (run / "source/.python-version").read_text() == "3.12\n"
    assert not (run / "source/results/must-not-copy.json").exists()
    archived = archive_root / run.name
    assert (archived / "study-plan.json").read_bytes() == (run / "study-plan.json").read_bytes()
    assert (archived / "source/hpc/discovery/budget_worker.sh").exists()
    assert (archived / "source/simulation/merge_sparse_smart_budget_shards.py").exists()
    configs = sorted((run / "task-configs").glob("*.sh"))
    assert len(configs) == 48
    assert len(list((archived / "task-configs").glob("*.sh"))) == 48
    for path in configs:
        assert (archived / "task-configs" / path.name).read_bytes() == path.read_bytes()


def test_all_mode_preserves_whole_cell_work_items_for_full_seed_scope(checkout, tmp_path):
    result, roots, _ = dry_run(checkout, tmp_path,
        ("--workers", "32", "--tuning-task-size", "all"))
    assert result.returncode == 0, result.stderr
    run = roots[0]
    plan = json.loads((run / "study-plan.json").read_text())
    rows = [line.split("\t") for line in (run / "work-items.tsv").read_text().splitlines()]
    assert len(rows) == plan["expected_cells"] == 7200
    assert plan["expected_applicable"] == 6300 and plan["expected_inapplicable"] == 900
    assert all(len(row) == 5 and "_g" not in row[0] for row in rows)
    assert not list((run / "task-configs").glob("*.sh"))


def test_worker_choice_is_required_before_artifacts_or_submission(checkout, tmp_path):
    result, roots, archive_root = dry_run(checkout, tmp_path)
    assert result.returncode != 0
    assert "--workers is required" in result.stderr
    assert not roots and not (tmp_path / "runs").exists()
    assert not archive_root.exists()


def test_restricted_dry_run_caps_workers_and_preserves_literal_paths(checkout, tmp_path):
    arguments = ("--models", "0", "--experiments", "3", "--seeds", "1,3-5:2",
                 "--setting-index", "5", "--workers", "128", "--iteration-budgets", "500", "2000",
                 "--init-penalties", ".03", "--penalties-u", ".0025", "--penalties-v", ".01",
                 "--validation-iterations", "10,25", "50", "150",
                 "--stationarity-tol", "2e-6", "--time", "01:00:00", "--mem", "4G")
    result, roots, _ = dry_run(checkout, tmp_path, arguments, special_path=True)
    assert result.returncode == 0, result.stderr
    run = roots[0]
    plan = json.loads((run / "study-plan.json").read_text())
    assert plan["expected_cells"] == 3 and plan["seed_ids"] == [1, 3, 5]
    assert plan["setting_index"] == 5
    assert plan["configuration"]["stationarity_tol"] == 2e-6
    assert plan["configuration"]["iteration_budgets"] == [500, 2000]
    assert plan["configuration"]["validation_iterations"] == [10, 25, 50, 150]
    command = shlex.split((run / "submission-command.txt").read_text())
    assert "--ntasks=3" in command and f"--chdir={run}" in command
    assert f"--output={str(run).replace('%', '%%')}/logs/pool-%j.out" in command
    assert command[-1] == str(run)
    assert plan["expected_tasks"] == 3
    assert len(list((run / "task-configs").glob("*.sh"))) == 1


@pytest.mark.parametrize("task_size, expected_tasks", [(2, 24), (4, 12)])
def test_tuning_chunks_keep_initializer_and_u_fixed(checkout, tmp_path, task_size, expected_tasks):
    result, roots, _ = dry_run(checkout, tmp_path, (
        "--models", "0", "--experiments", "0", "--seeds", "0", "--setting-index", "0",
        "--workers", "32", "--tuning-task-size", str(task_size)))
    assert result.returncode == 0, result.stderr
    run = roots[0]
    plan = json.loads((run / "study-plan.json").read_text())
    assert plan["expected_cells"] == 1 and plan["expected_tasks"] == expected_tasks
    assert f"--ntasks={expected_tasks}" in shlex.split((run / "submission-command.txt").read_text())
    configs = list((run / "task-configs").glob("*.sh"))
    assert len(configs) == expected_tasks
    actual = set()
    for path in configs:
        arguments = shlex.split(path.read_text().strip().removeprefix("tuning_args=(").removesuffix(")"))
        assert arguments[::2] == ["--init-penalties", "--penalties-u", "--penalties-v"]
        assert float(arguments[1]) in (.01, .03, .1)
        vs = tuple(map(float, arguments[5].split(",")))
        assert len(vs) == task_size
        actual.update((float(arguments[1]), float(arguments[3]), v) for v in vs)
    assert actual == {(li, u, v) for li in (.01, .03, .1)
                      for u in (.0025, .01, .04, .16) for v in (.0025, .01, .04, .16)}


def test_dry_run_can_explicitly_disable_extra_validation(checkout, tmp_path):
    result, roots, _ = dry_run(checkout, tmp_path,
        ("--workers", "32", "--seeds", "0-2", "--validation-iterations", "none"))
    assert result.returncode == 0, result.stderr
    plan = json.loads((roots[0] / "study-plan.json").read_text())
    assert plan["expected_cells"] == 216 and plan["expected_applicable"] == 189
    assert plan["expected_tasks"] == 9099
    assert plan["configuration"]["validation_iterations"] == []
    assert "--validation-iterations none" in (roots[0] / "budget-job-config.sh").read_text()


@pytest.mark.parametrize("arguments", [
    ("--seeds", "100"), ("--models", "0,0"),
    ("--iteration-budgets", "2000,500"), ("--stationarity-tol", "nan"),
    ("--validation-iterations", "25,10"), ("--validation-iterations", "10,10"),
    ("--validation-iterations", "0"), ("--validation-iterations", "none,10"),
    ("--validation-iterations", "10.5"),
    ("--tuning-task-size", "0"), ("--tuning-task-size", "-1"),
    ("--tuning-task-size", "2.5"), ("--tuning-task-size", "all,1"),
    ("--models", "0,1", "--setting-index", "0"),
])
def test_bad_study_selection_is_rejected_without_submission(checkout, tmp_path, arguments):
    result, _, _ = dry_run(checkout, tmp_path, ("--workers", "32", *arguments))
    assert result.returncode != 0


@pytest.fixture
def worker_run(tmp_path):
    run, archive, bin_dir = tmp_path / "run with $literal {1} % text", tmp_path / "archive with spaces", tmp_path / "bin"
    source = run / "source"
    hpc = source / "hpc/discovery"
    hpc.mkdir(parents=True)
    for name in ("budget_worker.sh", "budget_pool.sbatch"):
        shutil.copyfile(HPC / name, hpc / name)
    (source / "simulation").mkdir()
    (run / "tasks").mkdir()
    archive.mkdir()
    (hpc / "env.sh").write_text("export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n")
    config = (f"run_dir={shlex.quote(str(run))}\narchive_dir={shlex.quote(str(archive))}\n"
              f"pool_workers=1\nexport VENV={shlex.quote(str(tmp_path / 'venv'))}\n"
              "budget_args=(--iteration-budgets 500 2000 --checkpoint-interval 250 "
              "--validation-iterations 10,25,50,100,150,200 "
              "--init-penalties .03 --penalties-u .0025 --penalties-v .01 --stationarity-tol 1e-6)\n")
    (run / "budget-job-config.sh").write_text(config)
    executable(bin_dir / "srun", """
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ['SRUN_CAPTURE']).write_text(json.dumps({'args': args, 'memory': os.environ.get('SLURM_MEM_PER_CPU')}))
if os.environ.get('SRUN_FAIL'):
    print('deliberate launch failure', file=sys.stderr)
    raise SystemExit(int(os.environ['SRUN_FAIL']))
raise SystemExit(subprocess.call(args[args.index('bash'):]))
""", python=True)
    executable(bin_dir / "python", """
import json, os, sys
from pathlib import Path
if sys.argv[1:] == ['--version']:
    print('Python fixture')
    raise SystemExit(0)
Path(os.environ['PYTHON_CAPTURE']).write_text(json.dumps(sys.argv[1:]))
print('fake runner called; no estimator imported')
raise SystemExit(int(os.environ.get('RUNNER_FAIL', '0')))
""", python=True)
    environment = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
                       SLURM_JOB_ID="123", SLURM_NTASKS="1", SLURM_MEM_PER_CPU="8192",
                       SRUN_CAPTURE=str(tmp_path / "srun.json"), PYTHON_CAPTURE=str(tmp_path / "python.json"))
    return run, archive, bin_dir, environment


def test_worker_launch_failure_is_preserved_and_archived(worker_run):
    run, archive, _, environment = worker_run
    environment["SRUN_FAIL"] = "17"
    task = "m0_e3_s1_k5"
    result = subprocess.run(["bash", str(run / "source/hpc/discovery/budget_worker.sh"),
                             "dispatch", str(run), task, "0", "3", "1", "5"],
                            env=environment, timeout=10)
    assert result.returncode == 17
    assert (run / "tasks" / task / "launcher-exit-code.txt").read_text().strip() == "17"
    assert (archive / "tasks" / task / "launcher-exit-code.txt").read_text().strip() == "17"
    assert not Path(environment["PYTHON_CAPTURE"]).exists()
    assert "deliberate launch failure" in (run / "logs/steps" / f"{task}.err").read_text()


def test_worker_runs_one_named_argument_cell_and_preserves_runner_failure(worker_run):
    run, archive, _, environment = worker_run
    environment["RUNNER_FAIL"] = "23"
    task = "m0_e3_s1_k5"
    result = subprocess.run(["bash", str(run / "source/hpc/discovery/budget_worker.sh"),
                             "dispatch", str(run), task, "0", "3", "1", "5"],
                            env=environment, timeout=10)
    assert result.returncode == 23
    launch = json.loads(Path(environment["SRUN_CAPTURE"]).read_text())
    assert all(arg in launch["args"] for arg in ("--exclusive", "--exact", "--nodes=1", "--ntasks=1", "--cpus-per-task=1"))
    assert launch["memory"] == "8192"
    invocation = json.loads(Path(environment["PYTHON_CAPTURE"]).read_text())
    for flag, value in (("--models", "0"), ("--experiments", "3"), ("--seed-ids", "1"),
                        ("--setting-index", "5"), ("--workers", "1"), ("--profile", "full"),
                        ("--validation-iterations", "10,25,50,100,150,200"),
                        ("--output-root", str(run / "tasks" / task / "results"))):
        assert invocation[invocation.index(flag)+1] == value
    for name in ("process-exit-code.txt", "exit-code.txt", "launcher-exit-code.txt"):
        assert (run / "tasks" / task / name).read_text().strip() == "23"
        assert (archive / "tasks" / task / name).read_text().strip() == "23"


def test_worker_shard_appends_only_tuning_overrides_and_preserves_source(worker_run):
    run, archive, _, environment = worker_run
    task = "m0_e3_s1_k5_g12"
    directory = run / "task-configs"
    directory.mkdir()
    (directory / "g12.sh").write_text(
        "tuning_args=(--init-penalties 3e-2 --penalties-u .16 --penalties-v .04,.16)\n")
    result = subprocess.run(["bash", str(run / "source/hpc/discovery/budget_worker.sh"),
                             "dispatch", str(run), task, "0", "3", "1", "5"],
                            env=environment, timeout=10)
    assert result.returncode == 0
    invocation = json.loads(Path(environment["PYTHON_CAPTURE"]).read_text())
    assert invocation[0] == str(run / "source/simulation/run_sparse_smart_budget_study.py")
    assert invocation[-6:] == ["--init-penalties", "3e-2", "--penalties-u", ".16",
                               "--penalties-v", ".04,.16"]
    assert invocation.count("--penalties-u") == invocation.count("--penalties-v") == 2
    assert invocation[invocation.index("--seed-file")+1] == str(
        run / "source/simulation/data/random_seeds/experiment_seeds.csv")
    assert invocation[invocation.index("--output-root")+1] == str(run / "tasks" / task / "results")
    assert (archive / "tasks" / task / "launcher-exit-code.txt").read_text().strip() == "0"


@pytest.mark.parametrize("task", ["m0_e3_s1_k5_g-1", "m0_e3_s1_k5_g00", "m0_e3_s1_k5_gx",
                                   "../m0_e3_s1_k5_g0", "m0_e3_s1_k5_g0/extra", "m1_e3_s1_k5_g0"])
def test_worker_rejects_invalid_shard_identity_before_dispatch(worker_run, task):
    run, _, _, environment = worker_run
    result = subprocess.run(["bash", str(run / "source/hpc/discovery/budget_worker.sh"),
                             "dispatch", str(run), task, "0", "3", "1", "5"],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "Task identity mismatch" in result.stderr
    assert not Path(environment["SRUN_CAPTURE"]).exists()
    assert not Path(environment["PYTHON_CAPTURE"]).exists()


@pytest.mark.parametrize("kind", ["missing", "symlink", "directory_symlink", "wrong_flags", "bad_value", "empty"])
def test_worker_rejects_missing_unsafe_or_wrong_tuning_configuration(worker_run, tmp_path, kind):
    run, _, _, environment = worker_run
    task = "m0_e3_s1_k5_g0"
    directory = run / "task-configs"
    directory.mkdir()
    path = directory / "g0.sh"
    valid = "tuning_args=(--init-penalties .03 --penalties-u .04 --penalties-v .01,.04)\n"
    if kind == "symlink":
        outside = tmp_path / "outside-task.sh"
        outside.write_text(valid)
        path.symlink_to(outside)
    elif kind == "directory_symlink":
        directory.rmdir()
        outside = tmp_path / "outside-configs"
        outside.mkdir()
        (outside / path.name).write_text(valid)
        directory.symlink_to(outside, target_is_directory=True)
    elif kind == "wrong_flags":
        path.write_text("tuning_args=(--models 2 --penalties-u .04 --penalties-v .01)\n")
    elif kind == "bad_value":
        path.write_text("tuning_args=(--init-penalties .03 --penalties-u .04 --penalties-v --models)\n")
    elif kind == "empty":
        path.write_text("tuning_args=()\n")
    result = subprocess.run(["bash", str(run / "source/hpc/discovery/budget_worker.sh"),
                             "dispatch", str(run), task, "0", "3", "1", "5"],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2, result.stderr
    assert not Path(environment["PYTHON_CAPTURE"]).exists()
    if kind in ("missing", "symlink", "directory_symlink"):
        assert not Path(environment["SRUN_CAPTURE"]).exists()
    else:
        assert (run / "tasks" / task / "process-exit-code.txt").read_text().strip() == "2"


@pytest.mark.parametrize("environment_status", [0, 9])
def test_pool_collects_after_worker_failure_and_merges_only_with_valid_runtime(worker_run, tmp_path, environment_status):
    run, archive, bin_dir, environment = worker_run
    (run / "study-plan.json").write_text("{}\n")
    (run / "work-items.tsv").write_text("m0_e3_s1_k5\t0\t3\t1\t5\n")
    # The fake dispatcher never launches children; its failure simulates a
    # failed work item. The fake Python only records controller-stage calls.
    executable(bin_dir / "parallel", 'if [[ "$*" == *--version* ]]; then echo fixture; exit 0; fi\nexit 3\n')
    executable(bin_dir / "python", """
import json, os, sys
with open(os.environ['CONTROLLER_CAPTURE'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
raise SystemExit(4 if 'aggregate' in sys.argv else 0)
""", python=True)
    venv_bin = tmp_path / "venv/bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(bin_dir / "python")
    bash_env = tmp_path / "bash-env"
    bash_env.write_text("module() { return 0; }\n")
    environment.update(BASH_ENV=str(bash_env), CONTROLLER_CAPTURE=str(tmp_path / "controller.jsonl"))
    if environment_status:
        (run / "source/hpc/discovery/env.sh").write_text(f"return {environment_status}\n")
    result = subprocess.run(["bash", str(run / "source/hpc/discovery/budget_pool.sbatch"), str(run)],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 3, result.stderr
    calls = [json.loads(line) for line in Path(environment["CONTROLLER_CAPTURE"]).read_text().splitlines()]
    aggregate_call = next(call for call in calls if "aggregate" in call)
    assert ("--merge-shards" in aggregate_call) == (environment_status == 0)
    assert any(call[0].endswith("summarize_sparse_smart_budget_study.py") for call in calls) == (environment_status == 0)
    status = (run / "stage-status.tsv").read_text()
    assert "workers\t3" in status and "aggregate\t4" in status
    assert f"environment\t{environment_status}" in status and f"summary\t{environment_status}" in status
    assert (archive / "pool-exit-code.txt").read_text().strip() == "3"
