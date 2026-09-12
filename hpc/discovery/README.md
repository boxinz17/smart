# SMART simulations on USC Discovery

Connect with `ssh discovery`. The deployment is under
`$HOME/projects/smart`; the Python environment is `$HOME/envs/smart`.
Use Slurm for environment installation and, when authorized, simulations.
Packages are installed and both array and GNU Parallel pool launchers are available.
The dedicated budget-study launcher below supports the full paper grid for
SparseSMART. No launcher reinstalls packages or fits competing methods.

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

Discovery workers and the final merge/audit use the fixed OpenBLAS kernel
`Haswell`, exported by `env.sh` before its first Python invocation. This
overrides inherited `OPENBLAS_CORETYPE` values so different node generations
follow the same numerical policy. Before importing NumPy or SciPy, the runtime
check requires AVX2 and FMA CPU flags in `/proc/cpuinfo`. It then verifies that
both packages loaded OpenBLAS with the Haswell architecture and one thread;
an unsupported CPU, missing backend, or silent kernel fallback fails the job.
NumPy SIMD dispatch is also fixed: the pinned NumPy 2.5.3 Linux wheel must
report the compiled baseline `X86_V2` and exactly the audited optional groups
`X86_V3,X86_V4,AVX512_ICL,AVX512_SPR`. `env.sh` forces all four groups off with
`NPY_DISABLE_CPU_FEATURES` before Python starts. The checker rejects a different
baseline/dispatch build, an unavailable baseline, or an enabled optional group,
and records the baseline, dispatch groups, and active CPU feature flags.

The runtime report records the requested numerical environment and active
library paths, versions, architectures, and thread counts. Budget-study worker
reports remain in `tasks/<task-id>/slurm.out`. Historical combined jobs also
have a postprocessing controller report in `logs/environment.txt`; the fit-only
launcher does not run that stage or copy reports to archive storage. Fixed package versions alone
do not establish matching floating-point results across node types; use a
cross-node data-fingerprint check before a new campaign. Historical runs keep
their original saved environment and provenance. Use a fresh results root
after changing the numerical policy.

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

## Full paper grid budget study

Use `submit_budget_study.sh` for `run_sparse_smart_budget_study.py`, whose named
arguments and batch manifests differ from the per-seed runners supported by
the generic `submit.sh` below. This launcher is **fit-only**: it saves primary
results and execution records, then releases its allocation when fitting ends.
Aggregation, scientific summaries, and archival are separate, deferred work.
The expanded grid is provisional: first use
three-seed probes in settings sensitive to the previous grid boundaries, then
reassess coverage and runtime before a 100-seed campaign. To preview the expanded
grid across every model, experiment, and setting on Discovery:

```bash
cd "$HOME/projects/smart"
bash hpc/discovery/submit_budget_study.sh --workers 32 \
  --tuning-preset expanded --seeds 0-2 --tuning-task-size 1 \
  --time 24:00:00 --mem 8G --stationarity-tol 1e-6 --dry-run
```

The dedicated launcher's defaults are **all three models, all four paper
experiments, all settings, and saved seed IDs 0–99**. Its plan contains 7,200
cases: 6,300 applicable cases and 900 explicit inapplicable records. The current
estimator cannot fit target rank 11 with source rank 10, or target rank 5 with
source rank 0/3. These cases remain visible in the report; they are not silently
dropped or fitted using a different method.

Each applicable case tunes 100 combinations: the initialization penalties
`{0.01, 0.03, 0.1, 0.3}` crossed with the left/right penalties
`{0.0025, 0.01, 0.04, 0.16, 0.32}`, using all paper-grid
training rows and 200 independent validation rows. The three-seed pilot above
contains 216 cases, including 189 applicable cases and 27 explicit exclusions,
for **18,900 continuous trajectories in 18,927 work items** (including the 27
inapplicable records). The full 100-seed scope has 630,000 trajectories in
630,900 work items. Split a full campaign into smaller submissions or use larger
tuning chunks to respect cluster Slurm-step limits; the launcher does not
automatically batch that campaign. It compares budgets
**500, 1,000, and 2,000**, retaining
regular checkpoints every 250 updates and evaluating validation every 50. Additional validation checks at
**1, 2, 5, 10, 15, 20, 25, 50, 100, 150, and 200** capture useful iterates before the first regular
checkpoint. Extra validation checks retain factors when they improve the
validation best; regular checkpoints and budget endpoints retain their states.
This separates validation frequency from full-state retention. The optimizer checks constrained stationarity
every iteration and stops early at tolerance `1e-6`, including its proximal
uncertainty allowance. The maximum budget is a limit, not a convergence claim;
validation selection and optimization convergence are reported separately.
Validation stopping is enabled by default: after at least 500 accepted updates,
stop at an evaluation when 300 iterations have elapsed without a cumulative
0.1% decrease from the last significant best MSE. Every actual improvement
remains eligible for selection, including improvements smaller than 0.1%.
The model retains the best observed iterate and records `validation_stop` as
the terminal reason. This completes the declared policy without claiming
convergence or coverage of a larger unattained iteration cap. Such case results
use `policy_complete`; their cap records keep `coverage_complete=false` where
appropriate. Full-budget comparison summaries therefore remain conservative.

Use `--validation-interval`, `--validation-patience`,
`--validation-min-iterations`, and `--validation-min-relative-improvement` to
configure the rule. `--no-validation-stop` creates a fixed-budget control;
stationarity stopping remains active. `--n-validation` changes the independent
validation sample size without changing training observations. Validation
sizes, schedules, and stopping policy all enter the saved identity; use a fresh
output root after changing them. Historical plans retain their recorded settings.
`--iteration-budgets`, `--checkpoint-interval`, `--validation-iterations`, `--stationarity-tol`,
`--init-penalties`, `--penalties-u`, and `--penalties-v` make these choices explicit.
Additional validation iterations must be positive, unique, and increasing.
They are combined with regular checkpoints, cap endpoints, and initialization;
entries above the maximum budget are ignored during fitting. Use
`--validation-iterations none` to disable only the additional checks. The
declared schedule remains part of the saved configuration and provenance.

The expanded refinement grid and denser early validation address two separate
uncertainties in the previous pilot: upper-bound penalty selections and
selections at the first available positive checkpoint. Initialization is also
tuned over `0.01, 0.03, 0.1, 0.3` by default, giving 100 combinations with the
5-by-5 refinement grid. Task splitting covers
all three tuning dimensions. Compare initializer-selected cases separately: their
refinement penalties can tie. Use the pilot to reassess grid boundaries and
runtime before committing to 100 seeds; final performance should be evaluated
separately from the validation data used for tuning.

Two optional presets prepare targeted source-rank probes. Each requires an
explicit **single model** and selects experiment 2 (vary the fitted source
rank), with setting index 2 for rank 5 or index 3 for rank 7. Both use the same
500/1,000/2,000 caps and validation stopping. `source-rank-5` adds `0.001` to both
U/V grids, giving 144 combinations per case; `source-rank-7` retains the expanded
100-combination grid. Longer caps require an explicit `--iteration-budgets` override.

```bash
# Source-rank-5 probe for Model 1 (model ID 0): 3 cases, 432 work items.
bash hpc/discovery/submit_budget_study.sh --workers 150 \
  --tuning-preset source-rank-5 --models 0 --seeds 0-2 --dry-run

# Source-rank-7 probe for Model 1: 3 cases, 300 work items.
bash hpc/discovery/submit_budget_study.sh --workers 150 \
  --tuning-preset source-rank-7 --models 0 --seeds 0-2 --dry-run

# Expanded grid for one boundary-sensitive case, without a longer budget.
bash hpc/discovery/submit_budget_study.sh --workers 150 \
  --tuning-preset expanded --models 0 --experiments 3 --setting-index 5 \
  --seeds 0-2 --dry-run
```

Choose model ID 1 or 2 instead as needed. Explicit tuning flags override preset
defaults regardless of their command-line order. Explicit experiment or setting
flags that conflict with a targeted preset are rejected before any artifacts are
created. `expanded` leaves the chosen case scope unchanged; all presets still
default to seed IDs 0–99 unless `--seeds` restricts them. Increasing the iteration
cap alone does not repair trajectories that terminate with numerical stagnation.

`--dry-run` creates a source snapshot, `study-plan.json`, `work-items.tsv`,
shared tuning-subset overrides in `task-configs/`,
and submission metadata in the run directory, but does not submit a job or fit
anything. Remove `--dry-run` to submit that scope. Use `--models`,
`--experiments`, `--seeds`, and `--setting-index` to request a smaller scope;
`--setting-index` requires one model and one experiment. Indices are zero-based.
`--archive-root` and `SMART_ARCHIVE_ROOT` are accepted as deprecated compatibility
metadata only: the fit-only launcher does not resolve, create, validate, or write
that destination. Archive storage is not required for preparation or execution.

`--workers N` is required: choose the concurrency for each submission based on
cluster availability, for example 16, 32, 64, or 128. It is independent of the
number of seeds, cases, and tuning combinations. `budget_pool.sbatch` uses GNU
Parallel to dispatch one single-CPU Slurm step per planned work item. By
default, `--tuning-task-size 1` gives each penalty combination its own work
item, so workers can start another combination as soon as one trajectory
finishes. One seed has 72 data cases but **6,309 work items**: 63 applicable
cases times 100 combinations, plus nine inapplicable records. Workers use
`--workers 1` internally. This is one Slurm job, not `N` separate batch jobs.
Slurm distributes its requested CPU slots across suitable nodes, with no fixed
node count. For example, 32 workers at the default 8 GB per CPU request
**32 CPUs and 256 GB of aggregate memory**. The 24-hour limit covers the entire
fitting queue, not each case. These resource choices are adjustable;
the small pilot does not predict full-grid runtime. Slurm grants the requested
allocation before starting the pool; it stays fixed while the queue drains.

Use `--tuning-task-size N` to group up to `N` combinations in each work item.
Each group fixes the initializer and U penalty and takes a consecutive chunk
of V penalties, with a shorter final chunk when needed. Thus sizes 2, 4, and 5
give 60, 40, and 20 fitting work items per applicable case with the default grid.
`--tuning-task-size all` restores one work item per data case, running all
100 combinations sequentially. Small
groups reduce the long tail from uneven trajectory runtimes; larger groups
reduce repeated initialization, data generation, and Slurm-step overhead.
The concurrency remains controlled solely by `--workers`.

Every worker has a separate `tasks/<task-id>/results` manifest/output root.
Split task IDs such as `m0_e0_s0_k0_g0` identify their original data case and
first global grid index. `task-configs/g0.sh` and similar files hold the tuning
overrides shared by every data case; the default grid needs only 100 such files.
These overrides and the source snapshot remain in the run directory. Workers
validate the numerical environment before fitting and preserve the full result
JSON, including every currently retained trajectory, factor checkpoint,
validation record and numerical diagnostic. Scientific runner arguments and
source/seed fingerprints are unchanged by the fit-only execution boundary.

After the last task terminates, the controller writes its fitting outcome and
exits. It performs no post-fitting environment inventory, aggregation, summary,
or recursive archival. Neither worker exit trap copies task directories.
`process-exit-code.txt`, `exit-code.txt`, and `launcher-exit-code.txt` preserve
the task outcomes; task/step logs, `logs/parallel-joblog.tsv`,
`stage-status.tsv`, and `pool-exit-code.txt` preserve batch execution evidence.
Run-level `postprocessing-status.txt` and `archive-status.txt` say `deferred`;
task-level `archive-status.txt` also says `deferred`. A successful Slurm job
means **fits finished; aggregation pending**, not a completed scientific campaign.
Real fit and launch failures remain nonzero; numerical stagnation and incomplete
tuning coverage remain visible in the scientific JSONs.

Keep the raw task directories and their manifests/status records for later
aggregation. The existing collector can merge the tuning shards, retain their
original validation references and deterministic candidate order, and report
missing or failed cases without fitting again. It is no longer invoked by the
launcher. Separate case-parallel aggregation is now implemented locally; its
Slurm launcher is described in [CASE_AGGREGATION.md](CASE_AGGREGATION.md).
Archival and cleanup remain separate decisions; see the [rollout plan](PIPELINE_RESTRUCTURE_PLAN.md).

Checkpoint states are held in memory until that task's trajectories finish and
are serialized into its final result JSON; an interrupted task must restart.
The launcher does not automatically resubmit a timed-out queue or delete raw
results. Provision primary storage and file quota for all retained results until
separate aggregation/archival occurs. Existing completed-job archives remain
unchanged. The generic launchers below retain their own archival behavior.

See the [budget-study guide](../../simulation/SPARSE_SMART_V05_BUDGET_STUDY.md)
for selection, early stopping, and interpretation details.

## Other per-seed simulations

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

The external-validation summarizers verify the committed CSV and provenance.
They also verify original paper PDFs when those files are available; absent
PDFs are recorded as unavailable, so the software-only Discovery checkout can
use the saved curves. See the [paper-reference guide](../../simulation/paper_reference/README.md).
No summarizer or comparison is part of package installation.

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

## Independent SparseSMART case aggregation

The local phase-two case runner, compact campaign reducer and dedicated Slurm
array launcher are documented in [CASE_AGGREGATION.md](CASE_AGGREGATION.md).
They audit existing outputs in place, with model-scoped indexes, selected-factor
publication and verified per-case restart support. The default compact mode
keeps all other factors in the raw files rather than writing a full duplicate.

`submit_case_aggregation.sh --models 1` selects Model II (zero-based ID 1).
The launcher defaults to metadata/code preparation only; `--dry-run` also prints
the request and `--submit` explicitly submits. Each array element uses one CPU
for a sequential chunk of cases; the concurrency cap is configurable. One
summary job follows the array with `afterany`, so failed or missing cases stay
visible. No fits, per-worker source copies, raw-data copies or archive operations
are part of this stage. Default memory/time requests require a Discovery pilot.
