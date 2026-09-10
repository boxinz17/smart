# SparseSMART v0.5: continuous iteration-budget study

The budget study uses one continuous solver trajectory per parameter setting.
It captures checkpoints every 250 updates and compares maximum budgets 500,
1,000, 2,000, 4,000, and 8,000. The existing independent-budget runners retain
their previous behavior. This study has separate result files and summaries.

The default study is 45 cells: three model sizes, fitted source ranks 5 and 7
and source noise 0.5, each on saved seed IDs 0–4. Each cell uses the nine
left/right penalty combinations from {0.0025, 0.01, 0.04}, initializer penalty
0.03, projected initialization, and the practical anchor solver. There are
405 trajectories, each with a maximum of 8,000 updates. All original training
rows and 100 independent validation rows are used; the training size is 200,
300, or 500 according to the paper-grid setting. No competing method is fitted.

At every checkpoint, the artifact records endpoint and best-so-far validation
MSE, diagnostic coefficient RMSE, objective, movement, and stationarity
components. Compact ambient factors make both selected and endpoint scores
independently reproducible. Coefficient truth is read for reporting after
selection; it is never supplied to the tuner.

## Eligibility, coverage, and stabilization

A completed positive checkpoint remains eligible after later numerical
failure. It is still marked as a fallback when the requested cap was not
reached. For example, failure at update 400 retains the completed 250-update
prefix at cap 500, with `budget_reached=False`. The failed partial endpoint
cannot replace it. A stationary early endpoint does cover later caps.

The summary compares cumulative eligible validation minima, especially 2k–4k,
4k–8k, and 2k–8k. By default, a material gain must exceed
`max(0.0001, 0.001 * baseline_validation_MSE)`; both thresholds are configurable.
This is a practical comparison threshold, not a significance test. Missing
cells and incomplete candidate extensions cannot establish a validation
plateau. Selection near the cap is reported separately.

Optimization diagnostics remain separate: objective change, parameter
movement, constrained mapping displacement, inner-solve uncertainty, strict
stationarity, and precision limitation. Interval objective changes sum the
stable per-step differences, avoiding subtraction of large objective totals.
The stationarity tolerance stays at 1e-6. Statistical stabilization does not
automatically imply numerical convergence.

## Commands

From `code/simulation`, use the v0.5 source package and the shared pinned
environment created by the [environment guide](../environment/README.md):

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
  --output-root result/sparse_smart_v05_budget_study --dry-run
```

Remove `--dry-run` to execute those 45 cells. The full five-seed study was not
launched during implementation; the software is checked with targeted cases.

Summarize the completed study with independent factor/data checks:

```sh
../.venv/bin/python \
  summarize_sparse_smart_budget_study.py \
  --result-root result/sparse_smart_v05_budget_study \
  --output-root result/sparse_smart_v05_budget_study/summary \
  --absolute-threshold 0.0001 --relative-threshold 0.001
```

Each cell is saved atomically after its trajectories finish. Rerunning an
identical study verifies configuration, code, and regenerated data fingerprints
before skipping existing cells. The runner rejects unrelated result roots and
incompatible study manifests. Checkpoint capture preserves the continuous
in-memory trajectory; it is not a process-restart/resume protocol for a cell
whose process was interrupted.

For a single Model III source-rank case:

```sh
../.venv/bin/python \
  run_sparse_smart_budget_study.py --models 2 --experiments 2 \
  --setting-index 3 --seed-ids 3 --workers 1 \
  --output-root result/sparse_smart_v05_budget_smoke/model3_rs7
```

For Model I with source noise 0.5:

```sh
../.venv/bin/python \
  run_sparse_smart_budget_study.py --models 0 --experiments 3 \
  --setting-index 5 --seed-ids 0 --workers 1 \
  --output-root result/sparse_smart_v05_budget_smoke/model1_noise05
```

These cases validate the workflow; they cannot settle the budget for the full
five-seed study or establish readiness of a final 100-seed experiment.

Use `--manifest-scope` when summarizing a deliberately restricted study; it
audits every cell declared in that study's manifest, including missing cells:

```sh
../.venv/bin/python \
  summarize_sparse_smart_budget_study.py \
  --result-root result/sparse_smart_v05_budget_smoke/model3_rs7 \
  --output-root result/sparse_smart_v05_budget_smoke/model3_rs7/summary \
  --manifest-scope
```

## Implementation checks and targeted results

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

The installable package is [the v0.5.0 wheel](../sparse-smart/dist/sparse_smart-0.5.0-py3-none-any.whl).
