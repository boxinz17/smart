# SparseSMART v0.5: continuous iteration-budget study

This guide describes the current source workflow. The
[historical results section](#historical-implementation-checks-and-targeted-results)
preserves the original v0.5 verification snapshot, recorded in repository
commit `d4b6e7b` on **2026-09-09**. The saved
[v0.5.0 wheel](../sparse-smart/dist/sparse_smart-0.5.0-py3-none-any.whl) is archival;
it does not contain the later audit fixes. Use the editable source installation
in the [environment guide](../environment/README.md) for current runs.

The budget study uses one continuous solver trajectory per parameter setting.
It captures regular checkpoints every 250 updates, validates additionally at
iterations 1, 2, 5, 10, 15, 20, 25, 50, 100, 150, and 200, and compares maximum budgets 500,
1,000, 2,000, 4,000, and 8,000. The existing independent-budget runners retain
their previous behavior. This study has separate result files and summaries.

The default study is 45 cells: three model sizes, fitted source ranks 5 and 7
and source noise 0.5, each on saved seed IDs 0–4. Each cell uses 100 combinations:
initialization penalties {0.01, 0.03, 0.1, 0.3} crossed with left/right penalties
{0.0025, 0.01, 0.04, 0.16, 0.32}, projected initialization, and the practical anchor
solver. There are 4,500 trajectories, each with a maximum of 8,000 updates. All original training
rows and 100 independent validation rows are used; the training size is 200,
300, or 500 according to the paper-grid setting. No competing method is fitted.

At every checkpoint, the artifact records endpoint and best-so-far validation
MSE and relative `selection_score`, diagnostic coefficient RMSE, objective,
movement, and stationarity components. Compact ambient factors make both selected and endpoint scores
independently reproducible. Coefficient truth is read for reporting after
selection; it is never supplied to the tuner.

Extra validation times are separate from regular state checkpoints. At an
extra time, the trajectory records validation metadata and retains factors if
the validation best improves; regular checkpoints, attained cap endpoints,
and successful stationary endpoints retain their states. Failed terminal
endpoints retain diagnostic records. Thus an early selected
iterate is recoverable without storing a full state for every validation
check. The summary distinguishes factor-audited validation decisions from
checks whose saved evidence is metadata only.

`--validation-iterations` accepts a comma-separated or space-separated list
of positive, unique, increasing iterations, or `none` to disable the extra
checks. The actual validation schedule is the sorted union of those times
within the maximum budget, initialization, regular checkpoints, and budget
caps. The declared extra times, effective schedule, and state-retention policy
are recorded in result provenance. Historical records lacking this field keep
their original periodic-only schedule and identity; they are not retroactively
treated as densely validated.

This follow-up pilot expands each refinement penalty grid upward to `0.32` and tunes
initialization over `0.01, 0.03, 0.1, 0.3` by default. It is intended to resolve early
validation minima, upper-bound penalty selections, and sensitivity to
initialization regularization. `--init-penalties` controls the initialization
grid, and the distributed
launcher splits their full Cartesian product with the U/V grids across tasks.
Report initializer-selected cases separately because refinement
penalties can tie there. A wider search should be evaluated on observations
separate from the validation sample used to select the model.

Selection compares each prediction `P` directly with the incumbent using
`mean((P-P_incumbent)*((P-Y)+(P_incumbent-Y)))`, with extended-precision
products and accumulation where available. A negative difference replaces the
incumbent; an exactly zero computed difference retains the earlier iterate,
grid candidate, or budget. Absolute MSE and relative `selection_score` are
reporting quantities and do not break pairwise ties. The relative score is
`mean(2*(P0-Y)*(P-P0) + (P-P0)**2)`, using one fixed initializer prediction
`P0` per cell; scores from different cells need not share a reference.

Current records carry `selection_rule="pairwise-validation-loss-v1"`,
`selection_comparison` incumbent records, per-budget candidate comparisons,
and `validation_comparisons` between selected cap predictions. The summary
regenerates validation observations, reconstructs the canonical predictions
from saved factors, verifies the reported absolute and relative scores using
`validation_reference_prediction`, and checks/replays the pairwise decisions.
Historical artifacts without the marker remain readable under their original
rule: relative score with absolute-MSE tie-breaking when `selection_score`
exists, otherwise absolute MSE. They are not silently converted to the current
selection rule or pooled with it.

## Eligibility, coverage, and stabilization

A completed positive checkpoint remains eligible after later numerical
failure. It is still marked as a fallback when the requested cap was not
reached. For example, failure at update 400 retains the completed 250-update
prefix at cap 500, with `budget_reached=False`. The failed partial endpoint
cannot replace it. A stationary early endpoint does cover later caps.

The summary compares cumulative eligible validation minima, especially 2k–4k,
4k–8k, and 2k–8k. By default, a material gain must exceed
`max(0.0001, 0.001 * baseline_validation_MSE)`; both thresholds are configurable.
Current gains use the direct pairwise loss difference between the selected
predictions at the two caps, independently checked against saved factors.
Historical records retain their original score-difference calculation. The
threshold's baseline remains absolute MSE.
This is a practical comparison threshold, not a significance test. Missing
cells and incomplete candidate extensions cannot establish a validation
plateau. Selection near the cap is reported separately.

Optimization diagnostics remain separate: objective change, parameter
movement, constrained mapping displacement, inner-solve uncertainty, strict
stationarity, and precision limitation. Interval objective changes sum the
stable per-step differences, avoiding subtraction of large objective totals.
The stationarity tolerance defaults to 1e-6 and is configurable with
`--stationarity-tol`. The optimizer checks the constrained residual, including
the proximal-solve uncertainty allowance, every iteration. It stops early
when that residual meets the tolerance, retaining the terminal state even
between scheduled checkpoints. That stationary endpoint covers later caps;
an earlier validation-selected iterate is still distinguished from the
stationary terminal iterate. Statistical stabilization does not automatically
imply numerical convergence. A small validation gain or parameter step alone
does not trigger early stopping.

The maximum budget of 8,000 allows longer trajectories than the 2,000-update
Discovery pilot; it is not a guarantee that every setting converges. Keep the
500/1,000/2,000/4,000/8,000 cap comparisons to measure the benefit of the extra
updates. Do not loosen the stationarity tolerance solely to relabel a
budget-limited or precision-limited fit as converged.

## Commands

From `code/simulation`, use the current source package and the shared pinned
environment created by the [environment guide](../environment/README.md).
The examples use output roots separate from the historical runs linked below;
choose another fresh root whenever the source or configuration changes:

```sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=../sparse-smart/src:.
export MPLCONFIGDIR=/private/tmp/sparse-smart-v05-mpl
export XDG_CACHE_HOME=/private/tmp/sparse-smart-v05-cache
../.venv/bin/python ../environment/check_runtime.py
```

Inspect the complete five-seed study without fitting:

```sh
../.venv/bin/python \
  run_sparse_smart_budget_study.py --seed-count 5 --workers 3 \
  --validation-iterations 1 2 5 10 15 20 25 50 100 150 200 \
  --output-root result/sparse_smart_budget_study_current --dry-run
```

Remove `--dry-run` to execute those 45 cells. The full five-seed study was not
launched during implementation; the software is checked with targeted cases.

Summarize the completed study with independent factor/data checks:

```sh
../.venv/bin/python \
  summarize_sparse_smart_budget_study.py \
  --result-root result/sparse_smart_budget_study_current \
  --output-root result/sparse_smart_budget_study_current/summary \
  --absolute-threshold 0.0001 --relative-threshold 0.001
```

Each cell is saved atomically after its trajectories finish. Rerunning an
identical study verifies configuration, code, and regenerated data fingerprints
before skipping existing cells. The runner rejects unrelated result roots and
incompatible study manifests. Current implementation fingerprints use
`sparse-smart-source-content-v1`: logical file names and content hashes form
`implementation_manifest`, while absolute source locations are separate
provenance. Identical source copies in different worker directories therefore
have the same implementation digest. The saved configuration/identity payload
is checked against its stored digest before reuse. Legacy records with no
supported implementation scheme cannot be resumed; use a fresh output root
to recompute them. They remain readable by historical summaries.

Progress is written to a unique file in `budget_study_manifest_attempts/`.
`budget_study_manifest.json` is published only when all requested tasks return
without task exceptions; recorded numerical failures or inapplicable settings
do not by themselves make the batch incomplete. Failed or interrupted attempts
retain their progress/errors without overwriting a prior completed manifest.
Resume and manifest-scope summaries prefer the canonical manifest, or the
latest attempt when no canonical manifest exists. Checkpoint capture preserves
the continuous in-memory trajectory; it is not a process-restart/resume protocol for a cell
whose process was interrupted.

For a single Model III source-rank case:

```sh
../.venv/bin/python \
  run_sparse_smart_budget_study.py --models 2 --experiments 2 \
  --setting-index 3 --seed-ids 3 --workers 1 \
  --output-root result/sparse_smart_budget_smoke_current/model3_rs7
```

For Model I with source noise 0.5:

```sh
../.venv/bin/python \
  run_sparse_smart_budget_study.py --models 0 --experiments 3 \
  --setting-index 5 --seed-ids 0 --workers 1 \
  --output-root result/sparse_smart_budget_smoke_current/model1_noise05
```

These cases validate the workflow; they cannot settle the budget for the full
five-seed study or establish readiness of a final 100-seed experiment.

Use `--manifest-scope` when summarizing a deliberately restricted study; it
audits every cell declared in that study's manifest, including missing cells:

```sh
../.venv/bin/python \
  summarize_sparse_smart_budget_study.py \
  --result-root result/sparse_smart_budget_smoke_current/model3_rs7 \
  --output-root result/sparse_smart_budget_smoke_current/model3_rs7/summary \
  --manifest-scope
```

The same flag supports `--profile full` studies, including structurally
inapplicable target/source-rank combinations. Their records still undergo
identity, configuration, and regenerated-data fingerprint checks. They count
toward the declared scope and the audit's `expected_inapplicable_cells`,
`recorded_inapplicable_cells`, and `missing_inapplicable_cells`, but are excluded
from validation-gain denominators. A recorded inapplicable cell is neither a
missing record nor an unresolved optimization cap. A missing inapplicable
record is reported explicitly. An entirely inapplicable scope has no budget
gain comparisons and requires no fits.

## Full paper grid on Discovery

Use the dedicated [Discovery budget launcher](../hpc/discovery/README.md#full-paper-grid-budget-study)
for a distributed study. The Python runner's `--workers` option alone creates
processes on one node; it does not distribute work across Slurm nodes.

The full grid contains three models and four experiments, with 5 sample-size,
6 fitted-target-rank, 7 fitted-source-rank, and 6 source-noise settings per
model. With saved seed IDs 0–99, it declares 7,200 cases. The current estimator
requires `1 <= target_rank <= source_rank`, so 900 cases are explicitly
inapplicable: target rank 11 with source rank 10, and source ranks 0/3 with
target rank 5. These constraints also occur in the initializer and source
chart. Fitting those cases would require a declared method extension, not
simply removal of the runner checks.

The remaining 6,300 cases each use 100 penalty combinations by default:
630,000 continuous trajectories, each capped at 8,000 updates with stationarity
stopping enabled. The five budgets are views of each trajectory, not five
independent fits. Each case uses its paper-grid training sample size and 100
independent validation observations. Fitted rank changes do not change the
generator's true ranks, which remain 5 and 10.

Start with saved seed IDs 0–2 and 32 workers to measure the expanded tuning
grid and early validation schedule before launching the full scope:

```bash
bash hpc/discovery/submit_budget_study.sh --workers 32 --seeds 0-2 \
  --tuning-task-size 1 \
  --validation-iterations 1,2,5,10,15,20,25,50,100,150,200 \
  --init-penalties .01,.03,.1,.3 --penalties-u .0025,.01,.04,.16,.32 \
  --penalties-v .0025,.01,.04,.16,.32 --dry-run
```

Run this from the repository root on Discovery. The preview declares 216
cases, with 189 applicable cases, 27 exclusions, and 18,900 trajectories.
The default `--tuning-task-size 1` creates 18,927 GNU Parallel work items:
one per trajectory plus one per inapplicable case. One seed therefore creates
6,309 work items; the full 100-seed scope creates 630,900. Each submission uses
one Slurm allocation with at most `--workers` single-CPU steps active at once.
`--dry-run` creates a source snapshot and plan but submits no job. Choose
the worker count and whole-pool wall time for the actual submission based on
cluster availability and pilot measurements.

Use targeted follow-up presets for source-rank settings that need a lower
penalty probe or a longer trajectory. Each requires exactly one model ID;
the launcher selects experiment 2 and the corresponding original grid index:

```bash
bash hpc/discovery/submit_budget_study.sh --workers 32 --models 0 --seeds 0-2 \
  --tuning-preset source-rank-5 --dry-run

bash hpc/discovery/submit_budget_study.sh --workers 32 --models 0 --seeds 0-2 \
  --tuning-preset source-rank-7 --dry-run
```

`source-rank-5` selects index 2 and adds `0.001` to both refinement grids,
giving 144 combinations per case. `source-rank-7` selects index 3 and retains
the 100-combination grid. Both add a 16,000-update cap while preserving the
earlier caps, dense validation checks, regular 250-update checkpoints, and
certified stationarity stopping. Their single-setting scope keeps these
exploratory runs separate from the general `expanded` preset, which remains
capped at 8,000. Explicit `--init-penalties`, `--penalties-u`, `--penalties-v`,
`--iteration-budgets`, and `--validation-iterations` overrides remain available.
The expanded grid and targeted presets are follow-up choices, not a settled
configuration for a final 100-seed campaign.

`--tuning-task-size N` combines up to N penalty combinations in each work item,
holding initializer and U fixed while grouping consecutive V values. With the
default four-by-five-by-five grid, sizes 1, 2, and 5 produce 100, 60, and 20 fitting
work items per applicable case. `--tuning-task-size all` restores the original
whole-case work item. Smaller groups reduce the scheduling tail for uneven
runtimes, at the cost of repeated data generation/initialization and more
Slurm steps. All groups use the same saved data seed and validation schedule.
For the full 100-seed scope, use separate seed batches or larger tuning chunks
to respect the cluster's per-job step limit; limiting concurrent workers alone
does not limit the total number of steps in a job.

The distributed launcher creates one manifest/output root per work item and
archives the shared tuning overrides in `task-configs/g<first-grid-id>.sh`.
After verifying the runtime, it merges
the saved subsets into one result per original data case, comparing saved
predictions in the original grid order before `--manifest-scope` auditing.
The merge preserves selection tie-breaking and records incomplete candidate
coverage; it performs no estimator fitting. If runtime verification fails,
collection reports metadata and missing cases without scientifically merging
the subsets. It retains
inapplicable, missing, and failed cases explicitly. Do not point independent
driver processes at one shared output root. The launcher and audit perform no
competing-method fits. Checkpoints remain in-memory during a task's fitting;
an interrupted task must restart, while completed compatible records can be
verified and reused by the runner. Plan storage for the full checkpoint
artifacts and their durable archives; the one-cell pilot does not establish
the full grid's wall time or output volume.

## Historical implementation checks and targeted results

The counts and results below describe the original v0.5 implementation checks,
as recorded in the **2026-09-09** repository snapshot (`d4b6e7b`).
They are retained as historical evidence and do not report the later audit
fixes or establish the current source's numerical trajectories.

The package suite passed 322 tests from source and from an isolated wheel
installation. The SparseSMART simulation suites passed 314 tests, including
continuous-budget runner and summary checks. Tests cover prefix equivalence,
failure fallback, verified stationarity, missing checkpoints, factor rescoring,
and exclusion of incomplete extensions from plateau evidence.

Two real saved-seed cases were run through the nine-candidate grid. Their
summaries regenerated the data and independently rescored every saved factor
state (297 for Model I and 189 for Model III).

| Case | MSE at cap 2,000 | MSE at cap 4,000 | MSE at cap 8,000 | Selected update | Candidates reaching 8,000 |
|---|---:|---:|---:|---:|---:|
| Model I, source noise 0.5, seed ID 0 | 0.290933 | 0.287739 | 0.287728 | 4,250 | 9/9 |
| Model III, fitted source rank 7, seed ID 3 | 0.253696 | 0.253696* | 0.253696* | 1,500 | 2/9 |

The Model I gain from 2,000 to 4,000 is material under the declared threshold;
the additional gain from 4,000 to 8,000 is below it. The selected checkpoint
is update 4,250, while the endpoint still fails the strict stationarity test.
See the [Model I audited report](result/sparse_smart_v05_budget_smoke/model1_noise05/summary/report.md).

\* Model III retains the successful earlier winner after later failures. Only
6/9 candidates reached 4,000 and 2/9 reached 8,000. Seven trajectories ended
with `numerical_stagnation`, between updates 2,236 and 7,957. At those endpoints,
the recorded inner-solve uncertainty exceeds the 1e-6 stationarity tolerance;
precision limitation remains distinct from convergence. The summary therefore
marks comparisons beyond 2,000 as unresolved. The repeated retained MSE does
not establish a plateau. See the [Model III audited report](result/sparse_smart_v05_budget_smoke/model3_rs7/summary/report.md).

These are two individual-seed checks, not the full 45-cell study. They validate
the study infrastructure and checkpoint retention but do not settle a common
iteration budget or establish readiness for 100 seeds.

The saved [v0.5.0 wheel](../sparse-smart/dist/sparse_smart-0.5.0-py3-none-any.whl)
is archival. Use the current source and pinned environment above to run the
repaired implementation; the archived wheel does not contain subsequent audit
fixes, and the historical counts above do not certify current source or a
newly rebuilt wheel.
