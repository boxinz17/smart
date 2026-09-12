# Independent case aggregation

Case aggregation reads saved raw results in place. It never fits estimators,
alters frozen fitting snapshots, archives worker directories, or deletes raw
results. The separate Slurm launcher submits analysis jobs only when explicitly
invoked. Local validation does not establish Discovery memory or filesystem
performance; deployment still starts with a pilot.

## Model scope and publication modes

A case is one model/experiment/setting/seed across its entire planned tuning
grid. Model II has 2,100 applicable cases and 300 explicit inapplicable records
in the full campaign. Each applicable case combines 100 or 144 tuning shards.

Preparation reads plans and work tables, not scientific results. Model IDs are
zero-based: `--models 1` means **Model II**. This filters the declared index,
not just execution: other models in mixed general batches do not become missing
Model II results. Every explicitly requested model must occur in the supplied
plans. Coverage always describes the supplied roster; selecting Model II alone
does not assert that the caller supplied every setting or all 100 seeds.

```bash
python simulation/aggregate_sparse_smart_cases.py prepare \
  --run-roots /absolute/raw/batch-one /absolute/raw/batch-two \
  --models 1 --publication-mode compact \
  --output-root /absolute/analysis/model-II-v1

python simulation/aggregate_sparse_smart_cases.py run \
  --index /absolute/analysis/model-II-v1/campaign-index.json \
  --workers 4 --dry-run
```

Version-two indexes bind scope and publication mode to their fingerprint.
Duplicate or overlapping selected cases are rejected, including cases with
different tuning configurations. The output must be separate from all raw run
trees. A changed roster, scope, plan, or publication mode requires a fresh output
root. Legacy version-one full indexes remain readable.

The prepare CLI and Slurm launcher default to **compact**. The Python
`prepare_campaign(..., models=None, publication_mode="full")` API retains full
mode by default for compatibility.

## What each case writes

```text
model-II-v1/
  campaign-index.json
  cases/<case-key>/
    .lock
    current.json
    generations/<content-identity>/
      audit.json
      receipt.json
      selected-factors.json   # compact mode
      # merged.json           # full mode instead of selected-factors.json
  summary/campaign-summary.json
```

Both modes perform the same shard and merged numerical audits and use the same
validation selection rule. By default aggregation verifies exact shard coverage, manifests,
identities, configuration, source provenance and execution markers. It regenerates
data to check saved factors and validation choices, without fitting an estimator.

- **Compact:** selected-factors.json stores each distinct winning
  (grid candidate, factor key) once across all requested budgets. It includes
  initialization when selected and earlier usable winners after a failed
  extension. Every factor has its original raw path/SHA256, task identity and
  local grid ID. Per-budget selections reference these factors.
- **Full:** merged.json retains all original trajectories and factors using the
  existing full scientific format. This duplicates substantial raw factor data.
- **Both:** audit.json contains numerical/coverage/convergence outcomes, selected
  penalties and iterations, scalar tuning histories and metrics. receipt.json
  binds original fit provenance, analysis code/runtime, input hashes and output
  hashes. An invalid input produces an audit/receipt without a scientific
  artifact. An audited inapplicable case produces an empty factor collection
  in compact mode.

Raw results remain the authoritative full trajectory collection in compact
mode. Selected factors support rechecking chosen estimates; they do not replace
all rejected candidates. Keep raw files and logs until scientific checks and
the retention decision are complete. No archival repair, automatic refit or
cleanup is implemented.

Compact output reduces disk writes, **not current peak RAM**: the merger still
constructs full records in memory for numerical auditing. Each fresh scientific
JSON is read twice: the first read supplies both parsing and a content hash; a
final complete rehash detects changes during auditing. Timestamp checks never
replace that final verification.

## Independent execution and restart

```bash
python simulation/aggregate_sparse_smart_cases.py case \
  --index /absolute/analysis/model-II-v1/campaign-index.json \
  --case-key m1_e2_s0_k5

python simulation/aggregate_sparse_smart_cases.py run \
  --index /absolute/analysis/model-II-v1/campaign-index.json \
  --workers 4 --case-keys m1_e2_s0_k5 m1_e2_s1_k5
```

Only process terminal producers with complete task sets, except for explicitly
permitted cancellations under the frozen policy below. Other missing or invalid
inputs yield diagnostics; changes during auditing prevent publication.
Independent cases have separate output directories and advisory locks.
Publications use same-filesystem atomic rename into immutable generations,
without a second storage copy. Restart verifies inputs and published hashes,
then reuses unchanged generations without repeating numerical audits.
Input or analysis changes create a new generation. Corrupt publications fail
closed rather than being silently overwritten.

The current pointer is withdrawn during a locked attempt and restored only
after successful verification/publication. Interrupted cases may temporarily
appear missing; rerunning safely restores a valid generation. Slurm chunks
retain the case as the restart unit and continue independent cases after errors.
The actual Discovery filesystem still needs a locking/rename smoke check.

## Discovery array launcher

Run preparation from the analysis checkout on Discovery. The input campaign
submission supplies explicit raw run roots; it does not scan storage for results.
The following is a full Model II **dry-run**, not a submission:

```bash
bash hpc/discovery/submit_case_aggregation.sh \
  --campaign-submission /absolute/campaign/submission.json \
  --output-root /scratch1/USERNAME/smart/analysis/model-II-v1 \
  --models 1 --publication-mode compact \
  --cases-per-chunk 25 --concurrency 4 \
  --mem 8G --time 04:00:00 --dry-run
```

The paths above are placeholders. Raw results stay in their existing locations.
Only a curated analysis code/runtime snapshot is copied, once under
`launcher/source`; no worker directory or raw scientific file is staged.
Source hashes, the exact scoped index and disjoint chunk roster are frozen.
The default 25 cases per chunk gives 96 array elements for all 2,400 Model II
records, including the cheap inapplicable records. Cases are grouped by producer
run root to reuse plan/source metadata in each worker.

Each array element requests **one CPU** and processes its cases sequentially
in one Python process. `--concurrency` limits simultaneously running elements;
it can be selected as 4, 16, 32, etc. at preparation. The default 8 GiB and
four-hour request per element are **provisional pilot settings**, not measured
production requirements. Use a separate pilot output root with a small explicit
set of completed producer runs, and/or single-case commands to benchmark
representative ordinary and 16,000-iteration cases first.

Unless `--submit` is supplied, preparation does no scheduling, numerical audit,
environment activation or installation. It writes metadata and the code snapshot
and prints the exact array request. To submit an already prepared plan:

```bash
python3 -B -S simulation/discovery_case_aggregation.py submit \
  --launch /scratch1/USERNAME/smart/analysis/model-II-v1/launcher/launch.json
```

Analysis output paths may contain spaces and percent signs; backslashes are
rejected because Slurm treats them specially in log filename patterns.

Alternatively use `--submit` instead of `--dry-run` during fresh preparation.
The launcher submits an array and a one-CPU summary with an
`afterany:<array-job-id>` dependency. Thus failed cases still appear in the
summary's coverage report. Every job verifies the frozen source and loads the
existing shared environment; no packages are installed.

`launcher/submission.json` records intent before each scheduler call, exact
arguments and returned job IDs. Repeated or ambiguous attempts are rejected to
avoid duplicate jobs. If the array ID is known but summary submission was never
attempted, the `submit-summary --launch ...` command submits only the summary.
Uncertain scheduler outcomes require reconciliation with Slurm and the recorded
receipt; do not delete the receipt and retry blindly. There is no automatic
requeue, cancellation or resubmission. Case/chunk commands remain explicitly
rerunnable inside an appropriate allocation, reusing prior valid case outputs.

## Explicitly permitted cancelled tasks

The Model I exception concerns one seed with three intentionally cancelled
one-candidate tasks. Its available search contains 97 of the original 100
candidates. To aggregate that seed, supply a JSON list using
`--allowed-missing-tasks /absolute/allowed-missing-tasks.json` during preparation.
Each entry must contain the original absolute `run_root`, exact planned `task_id`,
a nonempty cancellation `reason`, and `launcher_exit_code: 137`. The ordinary
case prepare CLI and the Slurm launcher accept the same option; the Python
preparation API accepts the corresponding `allowed_missing_tasks` list.

Whole-job cancellation can terminate a worker before it writes any exit marker.
For that case, replace `launcher_exit_code` with `cancellation_evidence`, an
object containing an absolute JSON `path`, its `sha256`, and the exact `job_id`,
`step_id` (for example `11859590.1620`), and `task_id`. The referenced evidence
must have `schema_version: 1`, `kind: "slurm_cancelled_task"`, the original
`run_root`, the same three IDs, and `environment_sha256` for the task's original
`environment.txt`. Its `sacct` object contains `job` and `step` records with
matching `JobIDRaw`, a `CANCELLED` state (optionally `CANCELLED by UID`), and the
recorded `ExitCode`. Preserve additional accounting timestamps and the capture
command in that file for provenance.

This mode requires all three original task exit markers to be absent. It checks
the saved environment's job, step, and task identities, and fingerprints the
evidence and environment on every audit or restart. The campaign summary uses
those audited receipts without reopening raw inputs. Never create or alter
original exit markers to satisfy a missing-task policy. The same missing-result,
manifest, and coverage checks described below apply to both evidence modes.

```bash
bash hpc/discovery/submit_case_aggregation.sh \
  --campaign-submission /absolute/campaign/submission.json \
  --output-root /scratch1/USERNAME/smart/analysis/model-I-exceptions-v1 \
  --models 0 --publication-mode compact \
  --allowed-missing-tasks /absolute/allowed-missing-tasks.json \
  --cases-per-chunk 25 --concurrency 32 --dry-run
```

Preparation checks each policy entry against the original plan and derives its
case key and global candidate IDs. Version-three indexes include this normalized
policy in their fingerprint, and the launch manifest binds the entire index by
hash. The launcher also verifies its recorded policy against the frozen index.
Changing the original policy file after preparation does not change a prepared
analysis; a policy change requires a fresh output root. The policy does not
alter fitting plans, raw results, or an existing frozen Model II analysis.

Permission applies only when the named task has no result and its recorded
cancellation evidence matches. A valid restored result is audited and used.
Unlisted missing tasks, malformed results, unrelated execution failures, or
failed scientific audits remain failures. Tasks can own multiple candidates;
missing candidate IDs always come from the plan rather than from counting task
names.

The original grid, candidate IDs, and tie order remain intact. Available shards
receive the usual numerical audit. Missing candidates carry no invented fit,
score, or factor state and remain unresolved at every budget. Selection uses
the existing validation rule among available candidates; the case is marked
partial and never claims complete coverage of the original grid. Original
execution outcomes and the exact exception remain visible.

For the affected Model I setting, an available-result report can retain all
100 seeds when their available fits pass their audits, including the one
97-candidate search. The `policy_available` statistics retain that seed, while
`policy_sensitivity_excluding_missing` reports the sensitivity result excluding
seeds with permitted missing candidates. Existing `complete_grid` statistics
still require full budget coverage. Report the permitted cancellation alongside
both aggregates. The exception does not establish which candidate would have won the full
search or that the missing candidates have negligible impact. Other incomplete
budget coverage remains separately identified.

## Campaign summary and interpretation

```bash
python simulation/summarize_sparse_smart_campaign.py \
  --index /absolute/analysis/model-II-v1/campaign-index.json \
  --output-root /absolute/analysis/model-II-v1/summary
```

The reducer reads only indexes, current pointers, receipts and compact audits.
It does not traverse raw task directories, read or stat factor artifacts,
regenerate data, or fit models. Factor-file integrity is checked by case
publication/restart, not on every report regeneration.

Default reports retain per-seed choices, budget metrics, coverage, convergence
and audit/receipt references with hashes. Detailed histories stay in case audits
instead of being copied again into the campaign report. `--include-histories`
is an explicit opt-in to the larger report.

Means and standard errors are computed from individual seeds within each
model/experiment/setting/configuration. Different grids and budgets retain their
identity. Missing or invalid cases remain in expected counts; complete-grid
statistics and incomplete-grid available winners stay separate. Validation MSE
uses the tuning sample and is not an independent test estimate.

The confirmed legacy patterns process=0, worker=0 or 74, launcher=74 and
archive=failed are classified as archive-only problems. Other missing,
malformed or nonzero fitting/launch outcomes remain failures unless they meet
the exact frozen cancelled-task policy above. Original markers
are preserved, and scientific content must still pass its audit. No archive copy
is required. Model II had no known legacy archive errors in the inspected campaign.

Case/chunk exit zero means the audit and execution accounting passed, including
any explicitly permitted cancellation; it does not
require convergence or complete iteration-budget coverage. Without a missing-task
policy, summary exit zero requires complete primary coverage. With a policy,
summary exit zero requires `allowed_missing_policy_complete`: every expected
seed has an audited available result with acceptable execution accounting under
the frozen policy. `primary_complete` still reports full-grid primary coverage
separately. Exit one reports incomplete coverage and exit two denotes invalid
invocation/index.

## Performance curves from completed summaries

The curve exporter reads one saved campaign summary and the versioned v1 paper
reference bundle. It does not rerun aggregation, open raw fitting shards, fit
estimators, or submit jobs. It can run locally after transferring just the
compact summary. A summary job with exit one can still have a successfully
written, scientifically audited report; inspect its coverage and audit flags
rather than treating the scheduler label as evidence that no results exist.

```bash
python simulation/export_sparse_smart_campaign_curves.py \
  --summary /absolute/analysis/model-II-v1/summary/campaign-summary.json \
  --output-root /absolute/analysis/model-II-v1/curves
```

Add `--manuscript-root /absolute/manuscript` to verify available source PDF
hashes, or `--no-plots` to export tables without Matplotlib. Outputs include
per-seed selected penalties/iterations, all-budget statistics, configured-cap
performance curves, descriptive paper comparisons, PNG/PDF figures and a
manifest binding the input and output hashes.

The available-results curve pools individually audited, execution-usable seeds
with both complete and partial budget coverage. An explicit version-three
cancelled-task policy is respected. Complete-grid-only statistics and exclusion
of permitted missing-task cases are separate sensitivity analyses; computational
failure can make these subsets unrepresentative of all seeds. The maximum cap
comes from the case configuration, not from choosing the cap with lowest true
coefficient error. Selection of tuning parameters and checkpoints remains based
on validation MSE. The displayed error is the unsquared normalized Frobenius
coefficient error, with Monte Carlo standard errors across seeds.

Published means and standard errors are digitized figure aggregates, not paired
replicate data. Tuning protocols differ: SparseSMART uses an additional
independent validation sample. In Model II/III Experiment 2, the checked-in R
baseline runners use n=200 while the Python SparseSMART runner uses a
model-specific sample size. Keep those reference overlays descriptive and
qualify sample-size and rank-selection differences before making claims about
matched experimental performance.

## Validation

Tests use saved synthetic records and fake scheduler commands. They compare
full/compact audited metrics, reconstruct selected-factor metrics, verify
deduplication and raw preservation, check two-read integrity and changed inputs,
exercise scoped indexes and restart/corruption behavior, and ensure the reducer
never opens factor artifacts. Launcher tests exercise exact ownership,
one-CPU requests, literal paths, dry-run isolation, source changes, dependencies
and durable submission failure records. No scientific simulations or Discovery
jobs are needed to run these tests.
