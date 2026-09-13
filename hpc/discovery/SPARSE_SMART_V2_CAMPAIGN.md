# SparseSMARTv2 paper campaign on Discovery

The campaign helper operates on an existing frozen run root beneath `/scratch2`.
It never changes the scientific plan, runs fitting on the login node, or starts
an audit or aggregation job. Root deployment supplies `source/`,
`source-manifest.json`, `plan.json`, and the full immutable `work-items.tsv`.
`source/` is the repository's code tree. The runtime remains
`/home1/mkolar/envs/smart`; no package installation occurs in jobs.

Use one root for each model/experiment pair. The selected settings and seed IDs
must already be explicit in each runner plan. For Models I–III and seed IDs
0–29, Experiment 1 has 150 cases and 18,000 fits per model. Experiment 2 has
180 cases and 21,600 fits per model when all six ranks are explicitly admitted,
or 150 cases and 18,000 fits with setting indices 0–4. The rank-11 policy is a
scientific plan decision; this launcher does not expand or omit it. Across six
roots these choices give 990 cases/118,800 fits or 900 cases/108,000 fits.

Each case's 120 tuning tasks stay in the same chunk. The default maximum of
10,000 tasks produces chunks of 9,960 and 8,040 for an 18,000-fit root; a
21,600-fit root adds a third chunk of 1,680 after two chunks of 9,960. Preparation
runs once per root and shares all case data, source frames and initializer
artifacts across the chunks. Case inputs are never copied into chunk folders.

## Preview and prepare

```bash
RUN_DIR=/scratch2/mkolar/smart/runs/PAPER_MODEL_EXPERIMENT_TIMESTAMP
/home1/mkolar/envs/smart/bin/python \
  "$RUN_DIR/source/hpc/discovery/submit_sparse_smart_v2_campaign.py" \
  --run-dir "$RUN_DIR" --prepare-only --dry-run
```

A dry run creates or verifies immutable chunk definitions and writes
`campaign-submission-plan.json`. It does not contact Slurm. It permits local
absolute roots for testing. All actual submissions require a resolved
`/scratch2` root.

Remove `--dry-run` to submit preparation only. Defaults are one CPU, 16G, and
one hour; use `--prepare-mem` and `--prepare-time` to change them. Preparation
checks the exact submitted input hashes, numerical imports, package tests,
runner/launcher tests, and the campaign auditor tests. It then calls the
runner's `prepare --root ROOT`. The runner's atomic `preparation.json` marker
appears only after every case has been checked. Failed or interrupted
preparation cannot release dependent fit allocations. A subsequent preparation
attempt reuses cases the runner has already prepared and verified.

After the separate scientific gate has passed, inspect and submit fit chunks:

```bash
/home1/mkolar/envs/smart/bin/python \
  "$RUN_DIR/source/hpc/discovery/submit_sparse_smart_v2_campaign.py" \
  --run-dir "$RUN_DIR" --workers 100 --time 12:00:00 --mem 4G --dry-run
```

Remove `--dry-run` for actual submission. An existing verified preparation
marker makes this a fit-only submission. Without that marker, the full
invocation submits preparation and gives every chunk an `afterok` dependency
on that root's preparation job. Chunks are independent; there are no artificial
dependencies between chunks, models or experiments. Use `--chunks 0 1` to select
specific chunk indices. `--prepare-only` never submits fit pools.

Every fit allocation requests `--ntasks=100 --cpus-per-task=1` by default, with
4G per CPU and 12 hours. GNU Parallel runs 100 workers, each through an exclusive
one-CPU `srun --exact --nodes=1 --ntasks=1 --cpus-per-task=1` step. Slurm may place
the allocation across several nodes. There are no job arrays. `--workers`,
`--time`, `--mem`, `--account`, and `--partition` are explicit operational
controls. Concurrent roots and chunks each request their own allocation; the
100-worker default is per allocation, not a campaign-wide cap.

## Durable records and resume

`campaign/chunks.json` binds the full plan/source hashes and the case-aligned
partition. `campaign-chunk-NNNN.tsv` files are immutable. Each actual submission
has a separate directory, for example:

```text
campaign/attempts/chunk-0000-attempt-0001/
  submission.json
  submission.stdout, submission.stderr
  slurm-JOBID.out, slurm-JOBID.err
  parallel-joblog.tsv
  source-integrity-before.json, source-integrity-after.json
  source-integrity-after.err, source-integrity-exit-code.txt
  pool-process-status.tsv, pool-exit-code.txt, fit-status.txt
  tmp/, matplotlib/
```

The exact command, selected IDs and task-file hash are persisted before calling
`sbatch`; the returned job ID is then attached to that record. A process-level
submission lock prevents simultaneous helper calls from racing. Pool checks
bind their selected file to the exact record and immutable chunk plan. Source
and task-list integrity checks run before and after fitting. Preparation
attempts have separate `prepare-attempt-NNNN` directories and status files.
Per-task runner, process, and step artifacts remain in the usual task folders.

To continue after a wall-time limit or execution failure:

```bash
/home1/mkolar/envs/smart/bin/python \
  "$RUN_DIR/source/hpc/discovery/submit_sparse_smart_v2_campaign.py" \
  --run-dir "$RUN_DIR" --chunks 0 --resume
```

The helper checks `squeue` and `sacct` before retrying a submitted chunk. Active
jobs are left running. A missing or inconclusive accounting result blocks a
new submission; an ambiguous `sbatch` response likewise requires reconciliation
of its saved record. Dry runs never query the scheduler and leave existing
submissions unqueried rather than assuming they have ended.

Finished tasks are skipped only after their plan/source identities, result and
artifact hashes, status fields, and zero process/launcher exit codes are checked.
This includes scientific fit failures and strict initializer exclusions: resume
does not silently retry them. Unfinished tasks receive a new immutable attempt
list; old attempt lists and logs remain available. The current runner refits
unfinished tasks from their shared initializer, rather than claiming to resume
an arbitrary saved optimization state. Corrupt finished artifacts or mismatched
identities stop submission for inspection.

If preparation fails, existing pools with an unsatisfied dependency can remain
pending. Inspect those campaign jobs before another attempt: the helper neither
cancels jobs nor rewires existing dependencies. Audit and aggregation remain
separate jobs after fitting, with their own resource and provenance records.
