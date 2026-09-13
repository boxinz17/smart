# SparseSMART v2 pilot on Discovery

This launcher submits a one-CPU preparation job followed by a fit-only pool
with 100 CPUs. GNU Parallel schedules one exclusive single-CPU `srun` step per
planned tuning configuration. Slurm may place the allocation across multiple
nodes; the main submission does not prescribe a node count. Jobs are ordinary
batch jobs with an `afterok` dependency, not job arrays.

## Frozen run inputs

The run directory must already exist beneath `/scratch2`, for example
`/scratch2/mkolar/smart/runs/v2-pilot-TIMESTAMP`. Deployment supplies:

```text
RUN_DIR/
  source/                    frozen code tree, including sparse-smart-v2/
  source-manifest.json        source-relative paths and SHA-256 hashes
  plan.json                  scientific settings, cases, and task table
  work-items.tsv             one integer task ID per line, no header
```

`source/` is the repository's **code tree**: the runner is at
`source/simulation/run_sparse_smart_v2_pilot.py`. Freeze that tree read-only
after writing its manifest. `plan.json` is the authoritative scientific
configuration; the launcher does not change initializer dimension, free-row
counts, penalties, margins, support caps, seeds, or validation settings.

The existing Python environment is `/home1/mkolar/envs/smart`, overridable
through `VENV`. Workers import both sparse packages from the frozen source
tree. They do not install packages or modify the shared environment.

## Inspect and submit

From the frozen run directory or its source checkout:

```bash
RUN_DIR=/scratch2/mkolar/smart/runs/v2-pilot-TIMESTAMP
bash "$RUN_DIR/source/hpc/discovery/submit_sparse_smart_v2_pilot.sh" \
  --run-dir "$RUN_DIR" --workers 100 --time 12:00:00 --mem 4G --dry-run
```

A dry run writes `submission-plan.json` and prints both quoted commands. It
never calls Slurm. For local launcher tests, dry runs also accept a local
absolute directory. Actual submissions require a resolved `/scratch2` path.
Remove `--dry-run` to submit the two jobs:

```bash
bash "$RUN_DIR/source/hpc/discovery/submit_sparse_smart_v2_pilot.sh" \
  --run-dir "$RUN_DIR" --workers 100 --time 12:00:00 --mem 4G
```

Defaults are account `mkolar_1314`, partition `main`, 100 tasks with one CPU
each, 4G memory **per CPU**, and 12 hours for the main pool. `--account`,
`--partition`, `--workers`, `--mem`, and `--time` override those resources.
Preparation uses one CPU, 8G total memory, and 15 minutes. It loads the shared
runtime, checks numerical imports, runs the v2 package and runner/launcher
tests, then executes `prepare --root RUN_DIR`. Pytest uses importlib mode and
a temporary directory under the run root; it does not create a source cache.
The atomic `preparation.json` marker appears only after successful preparation.

The pool starts only after preparation succeeds. It verifies the marker
against the plan and manifest, checks all frozen source hashes, and runs GNU
Parallel with `--jobs 100 --halt never`. Each item starts:

```text
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 ...
```

The one-node constraint belongs to each one-CPU step, not to the entire pool.
The module environment includes Python and GNU Parallel before source checks.
Each worker then loads the normal scientific environment and prepends
`source/sparse-smart-v2/src` to its import path. Temporary and matplotlib
cache directories stay beneath the run directory.

## Status and failure handling

Submission prints both job IDs and preserves them in `prepare-job-id.txt`,
`pool-job-id.txt`, and `submission.json`, together with the exact commands.
An exclusive `submission.lock` prevents an accidental second submission of
the same run. If preparation was submitted but the pool submission failed,
its recorded job ID remains available; the launcher does not cancel it or
any other job automatically.

Preparation and pool stdout/stderr are in `logs/`. The pool retains
`logs/parallel-joblog.tsv`, checks frozen source hashes again when it exits,
and writes `pool-exit-code.txt`, `pool-process-status.tsv`, and
`fit-status.txt`. A source-integrity failure makes the pool unsuccessful.
GNU Parallel continues the remaining planned tasks after an individual fit
fails and returns a nonzero aggregate exit status.

Each zero-padded task directory, `tasks/00000/`, retains runner artifacts and
separate scheduler-process records:

```text
result.json, history.json.gz, states.npz, status.json
live-checkpoint.json, live-checkpoint-{0,1}.npz, live-history-{0,1}.json.gz
launcher-exit-code.txt, launcher-status.tsv
process-exit-code.txt, process-status.tsv, environment.txt, slurm.out, slurm.err
```

The live checkpoint pointer identifies one of two alternating slots by filename
and hash, so an interrupted write leaves the previous checkpoint available.
The runner owns its fit lock and scientific status. The worker records do not
overwrite that status. A recorded ineligible initializer or unsuccessful
optimization is distinct from a process failure: it produces an explicit
scientific failure record, while an execution error returns a nonzero process
exit code. The pipeline performs fitting only after preparation;
it does not copy, archive, aggregate, or summarize the completed campaign.
Existing runs and jobs remain untouched.

## Adaptive anchor diagnostic

New pilot plans explicitly enable `adaptive_anchors=True` with at most 16
switches. The original free rows, penalties, support caps, anchor floor,
source frames, seeds, and iteration/validation settings stay fixed. The
package default remains fixed anchors. New code is deployed to a fresh run
root, preserving the earlier fixed-chart runs for comparison.

`result.json` records `anchor_switches` and numerical work. The states archive
has `states_schema_version=2`: terminal and selected states have separate
`terminal_anchors_*`/`terminal_center_*` and
`selected_anchors_*`/`selected_center_*` arrays. Each checkpoint stores the
same separate geometry under `checkpoint_ITERATION_` prefixes. Unprefixed
`anchors_*`/`center_*` arrays describe only the archive's terminal state.
Live checkpoint slots use the same convention. Consumers must use the chart
belonging to each saved state; a later chart cannot reconstruct an earlier
validation winner safely.

The current continuation rule requires a strict anchor-margin gain above
numerical slack and the unchanged floor, with no percentage buffer. It uses
two bounded search starts (current anchors and restricted pivoted QR) and can
recenter a Cayley rotation near its limit. Events retain separate actions and exact center hashes.

For a targeted repair, the pool accepts an optional third argument containing
a task-list basename within the run root:

```bash
bash "$RUN_DIR/source/hpc/discovery/sparse_smart_v2_pool.sbatch" \
  "$RUN_DIR" 100 retry-items.tsv
```

Invoke it inside the matching Slurm allocation. The subset must contain
distinct canonical task IDs from the full plan. The full `work-items.tsv`
remains authoritative; pre/post checks verify both lists and the selected
list's hash. A fresh source snapshot and plan keep reruns distinct from the
earlier results. The default two-argument invocation still fits the full plan.
