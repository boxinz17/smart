# SparseSMARTv2 paper campaign on Discovery

The campaign helper operates on an existing frozen run root beneath `/scratch2`.
It never changes the scientific plan, runs fitting on the login node, or starts
an audit or aggregation job. Root deployment supplies `source/`,
`source-manifest.json`, `plan.json`, and the full immutable `work-items.tsv`.
`source/` is the repository's code tree. The runtime remains
`/home1/mkolar/envs/smart`; no package installation occurs in jobs.

Use one root for each model/experiment pair. The selected settings and seed IDs
must already be explicit in each runner plan. New plans deduplicate equivalent
RRR endpoints: the six-initializer, twenty-U/V grid has 114 iterative tasks and
one direct RRR task per case. The frozen plan records the original grid indices
represented by the retained RRR task. Old 120-task plans remain executable.

For the 100-seed campaign (rows 0–99), Experiments 1 and 3 have 500 cases and
57,500 tasks per model; Experiment 3 uses supported source counts 5,7,10,15,20.
Experiments 2 and 4 have 600 cases and 69,000 tasks per model. Across Models
I–III this gives 6,600 cases and 759,000 planned outcomes before initializer
exclusions. The rank-11 policy is declared in the plan; this launcher does not
expand or omit settings itself.

For the upcoming Models I–III, Experiments 1–4 campaign, the approved rank-11
setting uses `plan --rank11-policy expand`: fitted target rank 11, supplied
source rank 11, initializer dimension 11, and 11 unpenalized directions on each
side. Other fitted ranks retain their existing source-rank settings. These are
estimator dimensions; the data-generating target/source ranks remain 5/10.
Existing frozen plans retain their recorded settings.

Each case's actual planned tasks stay in the same chunk. With 115 tasks per
case, the default maximum of 10,000 tasks gives 86 cases/9,890 tasks in a full
chunk. The 100-seed roots need six or seven chunks, 78 fitting allocations in
total. Preparation
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
