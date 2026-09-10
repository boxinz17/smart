# SMART simulations on USC Discovery

Connect with `ssh discovery`. The deployment is under
`$HOME/projects/smart`; the Python environment is `$HOME/envs/smart`.
Use Slurm for environment installation and, when authorized, simulations.
Packages are installed and both array and GNU Parallel pool launchers are available. Simulations,
tests, smoke checks, and comparisons remain stopped until requested.

The configuration was checked for account `mkolar_1314` and partition `main`.
Python jobs load `gcc/13.3.0` and `python/3.12.8`. Installation additionally
loads `r/4.4.3` to prepare dependencies for later competing-method runs.
`VENV`, `R_LIBS_USER`, `SMART_ACCOUNT`, `SMART_PARTITION`, and `SMART_R_MODULE`
override the corresponding defaults. The source snapshot supplies `smart`,
`bi_smart`, and `sparse_smart`; the environment supplies their dependencies.

Local development, GitHub CI, and Discovery share the repository-root
`.python-version` and `python-constraints.txt`. Python patch releases within
the declared minor are allowed, so Discovery keeps its `python/3.12.8` module.
The exact NumPy, SciPy, scikit-learn, joblib, and threadpoolctl versions come
from the shared constraints. `env.sh` checks the active environment against
these files before a worker starts and fails on a mismatch; it never installs
or updates packages. The check uses each job's saved source snapshot.
`hpc/discovery/python-constraints.txt` is a compatibility include of the
root file, not a separate set of pins.

## Install packages

The bootstrap runs on a compute node and installs packages without running
the methods or their tests. Create the Slurm log directory before submission;
Slurm opens its log files before the script starts.

```bash
cd "$HOME/projects/smart"
mkdir -p "$HOME/smart-results/setup"
sbatch --parsable \
  --output="$HOME/smart-results/setup/bootstrap-%j.out" \
  --error="$HOME/smart-results/setup/bootstrap-%j.err" \
  hpc/discovery/bootstrap.sbatch
```

The bootstrap verifies the module's Python minor before creating the virtual
environment. It then installs all three Python packages in editable mode, the
dependencies constrained by the repository-root `python-constraints.txt`, and
`pdfplumber` for later paper-figure work, using the virtual environment's pip.
After installation, it checks package compatibility and the shared runtime
pins. It also installs R packages `rrpack`, `jsonlite`, `MASS`, and `reticulate`
with their dependencies for the subsequent full comparison. Installed package
versions are recorded in the setup output directory. Use the job logs and
Slurm status to confirm the installation outcome; installation does not
establish that the methods run correctly.

Installation job `11851795` completed successfully. Its full log and Python/R
version records are in `$HOME/smart-results/setup/`; `r-package-versions.csv`
also records the installed R versions here. This installation submitted no
simulation, test-suite, or comparison jobs.

`smoke.sbatch` remains available for a separately authorized manual check. It
runs tests and small Python simulations, so **do not submit it as part of
installation**.

`env.sh` activates and validates the Python environment. From the checkout
root, the same metadata-only check can be run directly:

```bash
"${VENV:-$HOME/envs/smart}/bin/python" environment/check_runtime.py --root "$PWD"
```

When R use is authorized later,
activate its module and private package library in the relevant Slurm job:

```bash
source hpc/discovery/env.sh
module load "${SMART_R_MODULE:-r/4.4.3}"
export R_LIBS_USER="${R_LIBS_USER:-$VENV/R/library}"
```

## Submit simulations later

The commands below are for future use after simulation runs are authorized.
Initial runs will use the new methods and compare their saved results with
the paper's existing curves. Competing-method reruns are deferred.

`--mode pool` submits one job using GNU Parallel to dispatch single-CPU work
items dynamically as worker slots become available. The job
requests `--ntasks=WORKERS --cpus-per-task=1 --mem-per-cpu=MEMORY`, with no
fixed node count or tasks-per-node requirement. Slurm can place the workers
across any number of suitable nodes; it grants the full allocation before
starting the job. For example, 128 workers at 4 GB each request 128 CPUs and
512 GB total memory, distributed across the allocated nodes.
Pool jobs load Discovery's `parallel/20240522` module and reuse the installed
Python environment; changing the dispatcher does not require reinstalling
packages. `SMART_PARALLEL_MODULE` overrides the saved pool module selection.

This previews a single job with 128 workers and 600 work items (100 existing
seeds times six source-noise settings):

```bash
bash hpc/discovery/submit.sh run_sparse_smart_external.py 0 3 \
  --mode pool --seeds 0-99 --settings all --workers 128 --dry-run
```

`pool.sbatch` passes the rows of `work-items.tsv` to GNU Parallel, with at most
`WORKERS` items active. Each item starts a separate
`srun --exclusive --exact -N1 -n1 -c1` step through `run_worker.sh`, with a
private working directory and logs. GNU Parallel waits 0.2 seconds between
launches to reduce Slurm launch pressure. As an item finishes, the next row
can use the freed slot; the Slurm allocation remains fixed for the entire job.
The job's wall-time limit covers the whole queue. A failed item does not stop
the remaining rows, but the overall job reports a nonzero exit status. There
is no automatic retry or resume.

Without `--settings`, one work item handles a whole seed. `--workers` defaults
to 128 but is capped by the number of work items, so 100 seeds without splitting
settings request at most 100 workers. `--settings all`, a list, or a range
(such as `--settings 0,2-4`) splits settings while keeping the existing seeds
and runner arguments. The supported grids have 5, 6, 7, and 6 settings for
experiment IDs 0, 1, 2, and 3. Do not combine `--settings` with a runner's
`--setting-index`. Seed IDs remain 0-99; no new random seeds are invented.

`--mode array` remains the default and preserves one separately scheduled job
per seed. Its entry point is `simulation.sbatch`; `--concurrency` controls the
number of running array jobs. `--workers` and `--settings` apply only to pool
mode. Existing array examples:

```bash
cd "$HOME/projects/smart"

# One actual restricted-RRR cell on the first existing seed.
bash hpc/discovery/submit.sh run_restricted_rrr.py 0 0 -- --setting-index 0

# Inspect a five-seed submission without launching it.
bash hpc/discovery/submit.sh run_sparse_smart_external.py 0 3 \
  --seeds 0-4 --concurrency 5 --dry-run -- --iterations 500

# Launch the same five-seed experiment.
bash hpc/discovery/submit.sh run_sparse_smart_external.py 0 3 \
  --seeds 0-4 --concurrency 5 -- --iterations 500
```

The default is **one seed**, one CPU per worker, 4 GB per worker, and two hours.
Explicitly select larger seed ranges when ready; the launcher never submits a
full sweep automatically. In array mode the default concurrency cap is **100**. `--seeds 0-99` requests
100 tasks with up to 100 running simultaneously: at most 100 CPUs and 400 GB
RAM at the default resources. Discovery's `normal` QoS currently limits each
user to 100 running jobs across submissions, so other jobs and cluster capacity
can reduce the number that starts. A pool consumes one job slot, with its
worker CPUs and memory still subject to resource limits. Use `--concurrency`
to lower a particular array's cap. Comma-separated lists and stepped ranges such as
`0,2,4-10:2` are accepted; duplicate seeds are rejected. Use `--time` and
`--mem` to adjust resources after inspecting completed jobs.

Model IDs are 0–2 and experiment IDs are 0–3. Each task runs its experiment's
entire setting grid unless the driver receives `--setting-index` or pool mode
splits it with `--settings`.
The accepted per-seed drivers are `run_restricted_rrr.py`,
`run_restricted_rrr_gauss_newton.py`, `run_sparse_smart.py`,
`run_sparse_smart_tuned.py`, and `run_sparse_smart_external.py`. The launcher
currently rejects legacy SMART/SMARTCV and R baseline reruns. Drivers that
submit their own grids, such as `run_sparse_smart_external_grid.py`, use a
different interface and are rejected.

Put runner arguments after `--`. Arguments are passed literally without shell
evaluation. `--output-root` and `--seed-file` overrides are rejected to retain
task isolation and the archived seed provenance. Any other file arguments must
refer to files inside the source snapshot. BLAS/OpenMP threads are capped at
one to match the allocation.

## Compare with saved paper results later

`simulation/paper_reference/v1_simulation_curves.csv` contains the existing
paper curves; its companion README and provenance file document their
extraction. These are approximate figure aggregates from 100 repetitions,
not paired per-seed observations. Comparisons must retain differences in
sample budgets, tuning procedures, and parameter meanings.

The external-validation summarizers also verify the hashes of the original
paper PDFs. After retrieving the archived results, run those summarizers from
the local manuscript repository layout where the PDFs and provenance paths
are available. The software-only Discovery checkout does not supply that
layout. No summarizer or comparison is part of package installation.

## Results and provenance

Each submission creates a unique directory under
`/scratch1/$USER/smart/runs/` (override with `SMART_RUN_ROOT` or `--run-root`).
It snapshots the current source, including uncommitted edits, and records Git
HEAD/status and literal runner arguments. Generated results, environments, and
single-cell data are excluded. A dry run creates this reviewable snapshot but
does not call `sbatch`.

Each work item receives a private source copy and simulation working directory.
This preserves runner-relative paths and keeps output files from different
tasks or submissions separate. New runs do not automatically reuse older results. Package imports
are resolved from the saved source, so editing the working repository after
submission does not change queued runs.

The source and submission metadata are copied to `$HOME/smart-results/`
(override with `SMART_ARCHIVE_ROOT` or `--archive-root`). On normal completion
or a handled process failure, each task copies its logs, environment record,
exit code, and partial or complete results into:

```text
$HOME/smart-results/<run-name>/source/
$HOME/smart-results/<run-name>/work-items.tsv
$HOME/smart-results/<run-name>/tasks/<task-id>/result/
$HOME/smart-results/<run-name>/tasks/<task-id>/slurm.out
$HOME/smart-results/<run-name>/tasks/<task-id>/slurm.err
$HOME/smart-results/<run-name>/tasks/<task-id>/exit-code.txt
$HOME/smart-results/<run-name>/tasks/<task-id>/archive-status.txt
$HOME/smart-results/<run-name>/logs/parallel-joblog.tsv
$HOME/smart-results/<run-name>/logs/steps/<task-id>.out
$HOME/smart-results/<run-name>/logs/steps/<task-id>.err
```

Unsplit task IDs are seed IDs, preserving `tasks/0/` etc. Split IDs include both
indices, for example `tasks/s7_k2/`. Controller logs are under `logs/`. Pool mode
also records GNU Parallel's item timings and exit statuses in
`logs/parallel-joblog.tsv`, and Slurm step output in
`logs/steps/<task-id>.out` and `.err`; the runner's own output remains in the
task directory's `slurm.out` and `slurm.err`. Step logs are archived by the
pool job even if a step fails before the runner starts.
Gather results from the task directories into
a fresh result tree before using the existing summarizers; the task-directory
layout is an execution detail, not the input layout expected by those scripts.

Scratch is temporary; home is the initial durable archive because no group
project directory was available during setup. Move larger campaigns to an
allocated project directory and adjust both roots to fit available quotas.
Abrupt node loss or SIGKILL can prevent the exit trap from archiving. If that
happens, recover the task's `source/simulation/result/` from the run directory.
No output or scratch directories are automatically deleted.

The Discovery checkout was copied from the local working tree, including
uncommitted changes, onto branch `codex/discovery-setup`. Its public GitHub
fetch URL uses HTTPS. The local development snapshot may be ahead of the public
repository; review local/remote changes before merging or pulling updates.
Each submission's archived source is the record of exactly what ran.

## Monitor and check outcomes

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
tail -f /scratch1/$USER/smart/runs/RUN_NAME/tasks/0/slurm.out
# Cancel a particular submitted array if needed:
scancel JOB_ID
```

Slurm `COMPLETED` means the process exited successfully. Some modern runners
also write scientific statuses such as `failed` or `inapplicable` while exiting
zero. Check saved status/success fields and finite coefficient errors before
combining results. A short iteration limit verifies execution, not convergence.
`exit-code.txt` records the simulation process status; `archive-status.txt`
separately records whether copying results succeeded. An archive failure also
makes the Slurm task fail even if the simulation process succeeded.

The existing `simulation/submit_*.sh` files are older cluster templates. Use
this directory's launcher for Discovery.

USC references: [Getting started with Discovery](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/getting-started-discovery),
[running Slurm jobs](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs),
and [Python on CARC](https://www.carc.usc.edu/user-guides/advanced-hpc-programming/programming-languages/python).
The pool dispatch pattern follows the
[RCC GNU Parallel guide](https://docs.rcc.uchicago.edu/slurm/sbatch/#gnu-parallel).
