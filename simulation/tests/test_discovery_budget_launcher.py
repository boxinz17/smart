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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import discovery_budget_study as study


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
    assert len(rows) == plan["expected_tasks"] == 6309
    assert plan["expected_cells"] == 72
    assert plan["tuning_task_size"] == 1
    assert len({row[0] for row in rows}) == 6309
    assert all(len(row) == 5 and row[0].startswith("m") for row in rows)
    assert rows[0] == ["m0_e0_s0_k0_g0", "0", "0", "0", "0"]
    assert plan["expected_applicable"] == 63
    assert plan["expected_inapplicable"] == 9
    assert plan["configuration"]["iteration_budgets"] == [500, 1000, 2000]
    assert plan["configuration"]["init_penalties"] == [.01, .03, .1, .3]
    assert plan["configuration"]["penalties_u"] == plan["configuration"]["penalties_v"] == [.0025, .01, .04, .16, .32]
    assert plan["configuration"]["validation_iterations"] == [1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200]
    stored = next(line.removeprefix("budget_args=(").removesuffix(")")
                  for line in (run / "budget-job-config.sh").read_text().splitlines()
                  if line.startswith("budget_args=("))
    budget_args = shlex.split(stored)
    assert budget_args[budget_args.index("--validation-iterations")+1] == "1,2,5,10,15,20,25,50,100,150,200"
    for flag, key, expected in (("--validation-interval", "validation_interval", 50),
                               ("--validation-patience", "validation_patience", 300),
                               ("--validation-min-iterations", "validation_min_iterations", 500),
                               ("--validation-min-relative-improvement", "validation_min_relative_improvement", .001),
                               ("--n-validation", "n_validation", 200)):
        assert plan["configuration"][key] == expected
        assert float(budget_args[budget_args.index(flag) + 1]) == expected
    assert "--no-validation-stop" not in budget_args
    assert "--tuning-task-size" not in budget_args
    command = shlex.split((run / "submission-command.txt").read_text())
    assert "--ntasks=32" in command and "--cpus-per-task=1" in command
    assert "--mem-per-cpu=8G" in command and "--time=24:00:00" in command
    assert not any(arg.startswith(("--nodes", "--ntasks-per-node", "--array", "--exclusive")) for arg in command)
    assert (run / "source/.python-version").read_text() == "3.12\n"
    assert not (run / "source/results/must-not-copy.json").exists()
    assert not archive_root.exists()
    assert (run / "source/hpc/discovery/budget_worker.sh").exists()
    assert (run / "source/simulation/merge_sparse_smart_budget_shards.py").exists()
    assert (run / "archive-status.txt").read_text().strip() == "deferred"
    assert (run / "postprocessing-status.txt").read_text().strip() == "deferred"
    configs = sorted((run / "task-configs").glob("*.sh"))
    assert len(configs) == 100


@pytest.mark.parametrize("n_validation", [100, 200])
def test_dry_run_disabled_stopping_and_validation_overrides_reach_plan_and_runner(checkout, tmp_path,
                                                                               n_validation):
    arguments = ("--models", "0", "--experiments", "0", "--setting-index", "0", "--seeds", "0",
                 "--workers", "1", "--init-penalties", ".03", "--penalties-u", ".01",
                 "--penalties-v", ".01", "--no-validation-stop", "--n-validation", str(n_validation),
                 "--validation-interval", "25", "--validation-patience", "150",
                 "--validation-min-iterations", "250", "--validation-min-relative-improvement", ".002")
    result, roots, _ = dry_run(checkout, tmp_path, arguments)
    assert result.returncode == 0, result.stderr
    assert len(roots) == 1
    config = json.loads((roots[0] / "study-plan.json").read_text())["configuration"]
    assert config["n_validation"] == n_validation and config["validation_patience"] is None
    assert config["validation_interval"] == 25 and config["validation_min_iterations"] == 250
    assert config["validation_min_relative_improvement"] == .002
    stored = next(line.removeprefix("budget_args=(").removesuffix(")")
                  for line in (roots[0] / "budget-job-config.sh").read_text().splitlines()
                  if line.startswith("budget_args=("))
    budget_args = shlex.split(stored)
    assert "--no-validation-stop" in budget_args
    for flag, expected in (("--n-validation", n_validation), ("--validation-interval", 25),
                           ("--validation-patience", 150), ("--validation-min-iterations", 250),
                           ("--validation-min-relative-improvement", .002)):
        assert float(budget_args[budget_args.index(flag) + 1]) == expected


@pytest.mark.parametrize("archive_kind", ["omitted", "environment", "symlink_loop", "file_parent"])
def test_submission_never_requires_or_resolves_archive_storage(checkout, tmp_path, archive_kind):
    bin_dir = tmp_path / "bin"
    capture = tmp_path / "sbatch.json"
    executable(bin_dir / "sbatch", """
import json, os, sys
from pathlib import Path
Path(os.environ['SBATCH_CAPTURE']).write_text(json.dumps(sys.argv[1:]))
print('12345')
""", python=True)
    run_root = tmp_path / "runs with $literal {1} % text"
    archive = tmp_path / "archive $literal {1} % text"
    extra = []
    environment = dict(os.environ, SMART_PLAN_PYTHON=sys.executable,
                       PATH=f"{bin_dir}:{os.environ['PATH']}", SBATCH_CAPTURE=str(capture))
    environment.pop("SMART_ARCHIVE_ROOT", None)
    if archive_kind == "symlink_loop":
        archive.symlink_to(archive.name)
        extra = ["--archive-root", str(archive)]
    elif archive_kind == "file_parent":
        archive.write_text("Unavailable archive storage must remain untouched.\n")
        extra = ["--archive-root", str(archive / "unavailable")]
    elif archive_kind == "environment":
        environment["SMART_ARCHIVE_ROOT"] = str(archive)
    result = subprocess.run(
        ["bash", str(checkout / "hpc/discovery/submit_budget_study.sh"),
         "--run-root", str(run_root), "--workers", "1", "--models", "0",
         "--experiments", "3", "--setting-index", "5", "--seeds", "1",
         "--tuning-task-size", "all", *extra],
        env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    run, = run_root.iterdir()
    submitted = json.loads(capture.read_text())
    assert submitted[-1] == str(run)
    assert "--ntasks=1" in submitted
    assert (run / "study-plan.json").is_file()
    assert (run / "source/simulation/run_sparse_smart_budget_study.py").is_file()
    assert (run / "archive-status.txt").read_text().strip() == "deferred"
    assert (run / "postprocessing-status.txt").read_text().strip() == "deferred"
    expected_archive = extra[-1] if extra else environment.get("SMART_ARCHIVE_ROOT", "")
    configuration = (run / "budget-job-config.sh").read_text()
    assert len([line for line in configuration.splitlines() if line.startswith("requested_archive_root=")]) == 1
    requested = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s" "$requested_archive_root"', "bash", str(run / "budget-job-config.sh")],
        env=environment, capture_output=True, text=True, check=True)
    assert requested.stdout == expected_archive
    assert not any(line.startswith("archive_dir=") for line in configuration.splitlines())
    assert "Archival: deferred" in (run / "manifest.txt").read_text()
    if archive_kind == "file_parent":
        assert archive.read_text() == "Unavailable archive storage must remain untouched.\n"
    elif archive_kind == "symlink_loop":
        assert archive.is_symlink() and os.readlink(archive) == archive.name
    else:
        assert not archive.exists()


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


@pytest.mark.parametrize("task_size, expected_tasks", [(2, 60), (4, 40), (5, 20)])
def test_tuning_chunks_keep_initializer_and_u_fixed(checkout, tmp_path, task_size, expected_tasks):
    result, roots, _ = dry_run(checkout, tmp_path, (
        "--models", "0", "--experiments", "0", "--seeds", "0", "--setting-index", "0",
        "--workers", "32", "--tuning-task-size", str(task_size)))
    assert result.returncode == 0, result.stderr
    run = roots[0]
    plan = json.loads((run / "study-plan.json").read_text())
    assert plan["expected_cells"] == 1 and plan["expected_tasks"] == expected_tasks
    assert f"--ntasks={min(32, expected_tasks)}" in shlex.split((run / "submission-command.txt").read_text())
    configs = list((run / "task-configs").glob("*.sh"))
    assert len(configs) == expected_tasks
    actual = set()
    for path in configs:
        arguments = shlex.split(path.read_text().strip().removeprefix("tuning_args=(").removesuffix(")"))
        assert arguments[::2] == ["--init-penalties", "--penalties-u", "--penalties-v"]
        assert float(arguments[1]) in (.01, .03, .1, .3)
        vs = tuple(map(float, arguments[5].split(",")))
        assert 1 <= len(vs) <= task_size
        actual.update((float(arguments[1]), float(arguments[3]), v) for v in vs)
    assert actual == {(li, u, v) for li in (.01, .03, .1, .3)
                      for u in (.0025, .01, .04, .16, .32) for v in (.0025, .01, .04, .16, .32)}


def test_dry_run_can_explicitly_disable_extra_validation(checkout, tmp_path):
    result, roots, _ = dry_run(checkout, tmp_path,
        ("--workers", "32", "--seeds", "0-2", "--validation-iterations", "none"))
    assert result.returncode == 0, result.stderr
    plan = json.loads((roots[0] / "study-plan.json").read_text())
    assert plan["expected_cells"] == 216 and plan["expected_applicable"] == 189
    assert plan["expected_tasks"] == 18927
    assert plan["configuration"]["validation_iterations"] == []
    assert "--validation-iterations none" in (roots[0] / "budget-job-config.sh").read_text()


@pytest.mark.parametrize("preset, setting, penalty_count", [
    ("source-rank-5", 2, 6), ("source-rank-7", 3, 5),
])
def test_targeted_preset_restricts_source_rank_and_expands_budget(
    checkout, tmp_path, preset, setting, penalty_count,
):
    result, roots, _ = dry_run(checkout, tmp_path, (
        "--workers", "150", "--tuning-preset", preset, "--models", "1", "--seeds", "0-2"))
    assert result.returncode == 0, result.stderr
    run = roots[0]
    plan = json.loads((run / "study-plan.json").read_text())
    assert plan["models"] == [1] and plan["experiments"] == [2]
    assert plan["setting_index"] == setting
    assert plan["expected_cells"] == plan["expected_applicable"] == 3
    assert plan["expected_tasks"] == 3 * 4 * penalty_count**2
    config = plan["configuration"]
    expected_penalties = [.0025, .01, .04, .16, .32]
    if preset == "source-rank-5":
        expected_penalties.insert(0, .001)
    assert config["penalties_u"] == config["penalties_v"] == expected_penalties
    assert config["init_penalties"] == [.01, .03, .1, .3]
    assert config["iteration_budgets"] == [500, 1000, 2000]
    assert config["checkpoint_interval"] == 250 and config["stationarity_tol"] == 1e-6
    assert config["validation_iterations"] == [1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200]
    assert "--ntasks=150" in shlex.split((run / "submission-command.txt").read_text())
    assert f"Tuning preset: {preset}\n" in (run / "manifest.txt").read_text()


@pytest.mark.parametrize("preset_first", [True, False])
def test_targeted_preset_preserves_explicit_overrides_in_either_order(checkout, tmp_path, preset_first):
    preset = ("--tuning-preset", "source-rank-5")
    overrides = ("--iteration-budgets", "500,2000", "--init-penalties", ".03",
                 "--penalties-u", ".01", "--penalties-v", ".04,.16",
                 "--checkpoint-interval", "100", "--validation-iterations", "none",
                 "--stationarity-tol", "2e-6", "--tuning-task-size", "all")
    ordered = (*preset, *overrides) if preset_first else (*overrides, *preset)
    result, roots, _ = dry_run(checkout, tmp_path, (
        "--workers", "150", "--models", "0", "--experiments", "2-2", "--setting-index", "2",
        "--seeds", "0", *ordered))
    assert result.returncode == 0, result.stderr
    run = roots[0]
    plan = json.loads((run / "study-plan.json").read_text())
    config = plan["configuration"]
    assert config["iteration_budgets"] == [500, 2000]
    assert config["init_penalties"] == [.03]
    assert config["penalties_u"] == [.01] and config["penalties_v"] == [.04, .16]
    assert config["validation_iterations"] == [] and config["checkpoint_interval"] == 100
    assert config["stationarity_tol"] == 2e-6
    assert plan["expected_cells"] == 1 and "tuning_task_size" not in plan
    assert len((run / "work-items.tsv").read_text().splitlines()) == 1
    assert "--ntasks=1" in shlex.split((run / "submission-command.txt").read_text())


@pytest.mark.parametrize("arguments, message", [
    (("--tuning-preset", "unknown"), "Invalid --tuning-preset"),
    (("--tuning-preset", "source-rank-5"), "exactly one model ID"),
    (("--tuning-preset", "source-rank-7", "--models", "0-2"), "exactly one model ID"),
    (("--tuning-preset", "source-rank-5", "--models", "0", "--experiments", "0"), "--experiments 2"),
    (("--tuning-preset", "source-rank-7", "--models", "0", "--experiments", "2,3"), "--experiments 2"),
    (("--tuning-preset", "source-rank-5", "--models", "0", "--setting-index", "3"), "--setting-index 2"),
    (("--tuning-preset", "source-rank-7", "--models", "0", "--setting-index", "2"), "--setting-index 3"),
])
def test_invalid_preset_scope_is_rejected_before_artifacts(checkout, tmp_path, arguments, message):
    result, roots, archive_root = dry_run(checkout, tmp_path, ("--workers", "32", *arguments))
    assert result.returncode != 0 and message in result.stderr
    assert not roots and not (tmp_path / "runs").exists()
    assert not archive_root.exists()


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
    # A stale legacy archive destination is deliberately unusable. None of the
    # execution paths may resolve or touch it, even when they fail.
    archive.symlink_to(archive.name)
    (hpc / "env.sh").write_text("export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n")
    config = (f"run_dir={shlex.quote(str(run))}\narchive_dir={shlex.quote(str(archive))}\n"
              f"pool_workers=1\nexport VENV={shlex.quote(str(tmp_path / 'venv'))}\n"
              "budget_args=(--iteration-budgets 500 2000 --checkpoint-interval 250 "
              "--validation-iterations 10,25,50,100,150,200 "
              "--init-penalties .03 --penalties-u .0025 --penalties-v .01 --stationarity-tol 1e-6)\n")
    (run / "budget-job-config.sh").write_text(config)
    executable(bin_dir / "srun", """
import json, os, signal, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ['SRUN_CAPTURE']).write_text(json.dumps({'args': args, 'memory': os.environ.get('SLURM_MEM_PER_CPU')}))
if os.environ.get('SRUN_FAIL'):
    print('deliberate launch failure', file=sys.stderr)
    raise SystemExit(int(os.environ['SRUN_FAIL']))
if os.environ.get('SRUN_TERM'):
    os.kill(os.getppid(), signal.SIGTERM)
    raise SystemExit(0)
raise SystemExit(subprocess.call(args[args.index('bash'):]))
""", python=True)
    executable(bin_dir / "python", """
import json, os, signal, sys
from pathlib import Path
if sys.argv[1:] == ['--version']:
    print('Python fixture')
    raise SystemExit(0)
Path(os.environ['PYTHON_CAPTURE']).write_text(json.dumps(sys.argv[1:]))
print('fake runner called; no estimator imported')
if os.environ.get('FAKE_RESULT_FIXTURE'):
    fixture = json.loads(Path(os.environ['FAKE_RESULT_FIXTURE']).read_text())
    root = Path(sys.argv[sys.argv.index('--output-root')+1])
    for relative, value in fixture.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value))
if os.environ.get('RUNNER_TERM'):
    os.kill(os.getppid(), signal.SIGTERM)
    raise SystemExit(0)
raise SystemExit(int(os.environ.get('RUNNER_FAIL', '0')))
""", python=True)
    for name in ("rsync", "cp", "pip"):
        executable(bin_dir / name, 'printf "%s\\n" "$0 $*" >> "$FORBIDDEN_CAPTURE"\nexit 91\n')
    environment = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
                       SLURM_JOB_ID="123", SLURM_NTASKS="1", SLURM_MEM_PER_CPU="8192",
                       SRUN_CAPTURE=str(tmp_path / "srun.json"), PYTHON_CAPTURE=str(tmp_path / "python.json"),
                       FORBIDDEN_CAPTURE=str(tmp_path / "forbidden-calls.txt"))
    return run, archive, bin_dir, environment


def assert_archive_untouched(archive, environment):
    assert archive.is_symlink() and os.readlink(archive) == archive.name
    assert not Path(environment["FORBIDDEN_CAPTURE"]).exists()


def test_worker_launch_failure_is_preserved_with_archive_deferred(worker_run):
    run, archive, _, environment = worker_run
    environment["SRUN_FAIL"] = "17"
    task = "m0_e3_s1_k5"
    result = subprocess.run(["bash", str(run / "source/hpc/discovery/budget_worker.sh"),
                             "dispatch", str(run), task, "0", "3", "1", "5"],
                            env=environment, timeout=10)
    assert result.returncode == 17
    assert (run / "tasks" / task / "launcher-exit-code.txt").read_text().strip() == "17"
    assert (run / "tasks" / task / "archive-status.txt").read_text().strip() == "deferred"
    assert not (run / "tasks" / task / "process-exit-code.txt").exists()
    assert_archive_untouched(archive, environment)
    assert not Path(environment["PYTHON_CAPTURE"]).exists()
    assert "deliberate launch failure" in (run / "logs/steps" / f"{task}.err").read_text()


@pytest.mark.parametrize("runner_status", [0, 23])
def test_worker_runs_one_named_argument_cell_and_preserves_runner_outcome(worker_run, runner_status):
    run, archive, _, environment = worker_run
    environment["RUNNER_FAIL"] = str(runner_status)
    task = "m0_e3_s1_k5"
    result = subprocess.run(["bash", str(run / "source/hpc/discovery/budget_worker.sh"),
                             "dispatch", str(run), task, "0", "3", "1", "5"],
                            env=environment, timeout=10)
    assert result.returncode == runner_status
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
        assert (run / "tasks" / task / name).read_text().strip() == str(runner_status)
    task_dir = run / "tasks" / task
    assert (task_dir / "archive-status.txt").read_text().strip() == "deferred"
    assert (task_dir / "environment.txt").is_file()
    assert (task_dir / "started.lock").is_dir()
    assert "fake runner called" in (task_dir / "slurm.out").read_text()
    assert_archive_untouched(archive, environment)


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
    assert (run / "tasks" / task / "launcher-exit-code.txt").read_text().strip() == "0"
    assert (run / "tasks" / task / "archive-status.txt").read_text().strip() == "deferred"
    assert_archive_untouched(archive, environment)


def test_worker_runs_without_any_archive_configuration(worker_run):
    run, archive, _, environment = worker_run
    config = run / "budget-job-config.sh"
    config.write_text("\n".join(line for line in config.read_text().splitlines()
                               if not line.startswith("archive_dir=")) + "\n")
    task = "m0_e3_s1_k5"
    result = subprocess.run(["bash", str(run / "source/hpc/discovery/budget_worker.sh"),
                             "dispatch", str(run), task, "0", "3", "1", "5"],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (run / "tasks" / task / "archive-status.txt").read_text().strip() == "deferred"
    assert_archive_untouched(archive, environment)


@pytest.mark.parametrize("interruption", ["RUNNER_TERM", "SRUN_TERM"])
def test_worker_handled_term_preserves_signal_status_without_archival(worker_run, interruption):
    run, archive, _, environment = worker_run
    environment[interruption] = "1"
    task = "m0_e3_s1_k5"
    result = subprocess.run(["bash", str(run / "source/hpc/discovery/budget_worker.sh"),
                             "dispatch", str(run), task, "0", "3", "1", "5"],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 143, result.stderr
    task_dir = run / "tasks" / task
    assert (task_dir / "launcher-exit-code.txt").read_text().strip() == "143"
    assert (task_dir / "archive-status.txt").read_text().strip() == "deferred"
    if interruption == "RUNNER_TERM":
        assert (task_dir / "process-exit-code.txt").read_text().strip() == "143"
        assert (task_dir / "exit-code.txt").read_text().strip() == "143"
    else:
        assert not (task_dir / "process-exit-code.txt").exists()
    assert_archive_untouched(archive, environment)


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


@pytest.mark.parametrize("worker_status, handled_term", [(0, False), (3, False), (0, True)])
def test_pool_exits_after_fitting_without_controller_python_or_archival(
    worker_run, tmp_path, worker_status, handled_term,
):
    run, archive, bin_dir, environment = worker_run
    config = run / "budget-job-config.sh"
    config.write_text("\n".join(line for line in config.read_text().splitlines()
                               if not line.startswith("archive_dir=")) + "\n")
    (run / "study-plan.json").write_text("{}\n")
    (run / "work-items.tsv").write_text("m0_e3_s1_k5\t0\t3\t1\t5\n")
    # The fake dispatcher never launches children. It captures the exact queue
    # contract and simulates its outcome, including a handled controller TERM.
    executable(bin_dir / "parallel", """
import json, os, signal, sys
from pathlib import Path
args = sys.argv[1:]
if '--version' in args:
    print('GNU Parallel fixture')
    raise SystemExit(0)
Path(os.environ['PARALLEL_CAPTURE']).write_text(json.dumps(args))
joblog = Path(args[args.index('--joblog')+1])
joblog.write_text('Seq\\tExitval\\n1\\t' + os.environ['WORKER_STATUS'] + '\\n')
if os.environ.get('PARALLEL_TERM'):
    os.kill(os.getppid(), signal.SIGTERM)
raise SystemExit(int(os.environ['WORKER_STATUS']))
""", python=True)
    executable(bin_dir / "python", 'printf "controller Python was called\\n" >> "$FORBIDDEN_CAPTURE"\nexit 92\n')
    venv_bin = tmp_path / "venv/bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(bin_dir / "python")
    bash_env = tmp_path / "bash-env"
    bash_env.write_text("module() { return 0; }\n")
    environment.update(BASH_ENV=str(bash_env), PARALLEL_CAPTURE=str(tmp_path / "parallel.json"),
                       WORKER_STATUS=str(worker_status))
    if handled_term:
        environment["PARALLEL_TERM"] = "1"
    # Even a broken scientific environment must not be sourced by a fitting-only
    # controller after its workers terminate.
    (run / "source/hpc/discovery/env.sh").write_text(
        'printf "controller sourced scientific environment\\n" >> "$FORBIDDEN_CAPTURE"\nreturn 9\n')
    result = subprocess.run(["bash", str(run / "source/hpc/discovery/budget_pool.sbatch"), str(run)],
                            env=environment, capture_output=True, text=True, timeout=10)
    expected_status = 143 if handled_term else worker_status
    assert result.returncode == expected_status, result.stderr
    arguments = json.loads(Path(environment["PARALLEL_CAPTURE"]).read_text())
    assert arguments[arguments.index("--jobs")+1] == "1"
    assert arguments[arguments.index("--halt")+1] == "never"
    assert "--plain" in arguments and "--quote" in arguments
    assert arguments[arguments.index("--colsep")+1] == "\\t"
    assert arguments[arguments.index("--joblog")+1] == str(run / "logs/parallel-joblog.tsv")
    assert arguments[-1] == str(run / "work-items.tsv")
    assert arguments[arguments.index("bash"):] == [
        "bash", "./source/hpc/discovery/budget_worker.sh", "dispatch", ".",
        "{1}", "{2}", "{3}", "{4}", "{5}", "::::", str(run / "work-items.tsv")]
    status_rows = (run / "stage-status.tsv").read_text().splitlines()
    assert status_rows[0] == "stage\texit_code"
    if not handled_term:
        assert status_rows[1:] == [f"workers\t{worker_status}"]
    else:
        assert all(row.startswith("workers\t") for row in status_rows[1:])
    assert (run / "pool-exit-code.txt").read_text().strip() == str(expected_status)
    assert (run / "pipeline-mode.txt").read_text().strip() == "fit-only"
    assert (run / "fit-status.txt").read_text().strip() == ("finished" if not expected_status else "failed")
    for name in ("archive-status.txt", "postprocessing-status.txt"):
        assert (run / name).read_text().strip() == "deferred"
    assert (run / "logs/parallel-joblog.tsv").is_file()
    assert not (run / "results").exists()
    assert not (run / "logs/aggregate.log").exists()
    assert not (run / "logs/summary.log").exists()
    assert_archive_untouched(archive, environment)


def test_fitting_only_worker_preserves_collector_compatible_primary_artifacts(
    checkout, worker_run, tmp_path,
):
    fixture_run, archive, _, environment = worker_run
    prepared, roots, archive_root = dry_run(checkout, tmp_path / "prepare", (
        "--workers", "1", "--models", "0", "--experiments", "2",
        "--setting-index", "0", "--seeds", "1"))
    assert prepared.returncode == 0, prepared.stderr
    run, = roots
    plan = study.read_json(run / "study-plan.json")
    task, = study.planned_tasks(plan)
    setting = task["simulation_setting"]
    identity = dict(schema_version=1, method=study.METHOD, model="model1", experiment="exp3",
        rd_seed_id=task["seed"], random_seed=task["random_seed"], setting=setting,
        configuration=study.resolved_configuration(setting, plan["configuration"]),
        generator_arguments=dict(n=setting["n"], p=setting["p"], q=setting["q"], sigma0=setting["sigma0"],
                                 sigma=.5, r_star=5, r0_star=10, random_seed=task["random_seed"]))
    provenance = plan["source"]["implementation"]["false"]
    record = dict(identity, configuration_fingerprint=study.digest(identity),
        implementation_fingerprint_scheme=study.SCHEME,
        implementation_manifest=provenance["manifest"], implementation_fingerprint=provenance["fingerprint"],
        applicable=False, status="inapplicable", success=False, failure_reason=task["inapplicability_reason"])
    manifest = dict(schema_version=1, method=study.METHOD, models=[0], experiments=[2], seed_ids=[1],
        profile=plan["profile"], setting_index=0, expected_cells=1, configuration=plan["configuration"],
        seed_file_sha256=plan["seed_file_sha256"], attempt_status="completed", errors=[],
        cells=[dict(model="model1", experiment="exp3", setting=setting["suffix"], seed_id=1)])
    fixture = tmp_path / "fake-runner-output.json"
    fixture.write_text(json.dumps({str(study.relative_result(task)): record, "budget_study_manifest.json": manifest}))
    environment["FAKE_RESULT_FIXTURE"] = str(fixture)
    shutil.copyfile(fixture_run / "source/hpc/discovery/env.sh", run / "source/hpc/discovery/env.sh")
    result = subprocess.run(["bash", str(run / "source/hpc/discovery/budget_worker.sh"),
                             "dispatch", str(run), task["task_id"], "0", "2", "1", "0"],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    errors = []
    collected, raw, artifact = study._read_task(run, task, plan, errors)
    assert not errors
    assert collected == json.loads(raw) == record
    assert artifact["task_id"] == task["task_id"]
    assert artifact["cell_manifest"]["attempt_status"] == "completed"
    assert (run / "tasks" / task["task_id"] / "archive-status.txt").read_text().strip() == "deferred"
    assert not (run / "results").exists(), "Fitting unexpectedly created an aggregate"
    assert not archive_root.exists()
    assert_archive_untouched(archive, environment)
