# SparseSMART v0.4.0: checkpoint and solver regression

**Archival report.** This document preserves the v0.4.0 regression, test
counts, commands, and wheel recorded in the September 9, 2026 repository
snapshot. Its numerical results have not been regenerated for later fixes.
In particular, the warm-started line-search rule below is historical; the
current solvers reset the trial inverse step at every iteration. For current
setup and behavior, see the [environment guide](../environment/README.md),
[package README](../sparse-smart/README.md),
[algorithm documentation](../sparse-smart/docs/algorithm.md), and
[budget-study guide](SPARSE_SMART_V05_BUDGET_STUDY.md).

The v0.4.0 package and simulation runners implemented the three changes
recorded below. Historical v0.3 numerical artifacts were unchanged.

1. `SparseSMARTTuner(iterations=2000, iteration_budgets=(500, 2000), ...)`
   retains the best successful fitted model at each budget in `checkpoints_`.
   Validation selects among successful fits across all budgets. A later failure
   cannot disqualify an earlier checkpoint; failed partial fits remain excluded.
   Equal scores prefer the earlier budget and then the earlier grid position.
2. The practical anchor solver records mapping displacement and inner proximal
   uncertainty separately. When their uncertainty interval straddles the
   stationarity tolerance, it attempts tighter diagnostic solves. Floating-point
   allowances remain in the residual; numerical stagnation and precision limits
   do not imply convergence. Objective changes use the quadratic difference
   identity and entrywise L1 differences, preserving the sufficient-decrease
   condition without subtracting two large objective values.
3. Each practical line search starts from half the previous accepted inverse
   step, bounded below by the configured initial inverse step. Feasibility and
   sufficient-decrease checks still apply. A resolution-sized inherited trial
   gets one retry from the initial inverse step before declaring stagnation.
   The diagnostic reference step stays fixed. The prescribed chart solver's
   iteration rule is unchanged.

Budgets are independent deterministic fits from the original initializer on
the same training and validation observations. They are not resumed internal
solver states. The default `iteration_budgets=None` retains single-budget
behavior. JSON artifacts record candidate budgets, selected budget/candidate,
per-budget outcomes, and continuation diagnostics; the validators cross-check
these fields while accepting historical records that omit them.

## Targeted regression

Only Model III, Experiment 3, fitted source rank 7, seed ID 3 was rerun.
The saved random seed is 816464123. This paper-grid setting uses 500 training
observations and 100 separate validation observations, with p=300 and q=200.
The fitted target rank is 5; the generating target/source ranks remain 5/10.
No competing method was fitted.

| Run | Successful candidate fits | Failed candidates | Selected validation MSE | Coefficient RMSE |
| --- | ---: | ---: | ---: | ---: |
| Saved v0.3, 500 updates | 9 | 0 | 0.253697992 | 0.002557362 |
| Saved v0.3, 2,000 updates | 6 | 3 | 0.255828632 | 0.003588272 |
| v0.4, budgets 500 and 2,000 | 18 | 0 | 0.253696046 | 0.002555649 |

The new winner uses penalties (0.04, 0.01), budget 2,000, and selected iteration
1,390. Its fit completed all 2,000 updates. All three parameter choices that
previously stopped with numerical stagnation completed at the longer budget.
The complete two-budget search took 135.91 seconds on this local run; this is
one timing observation, not a controlled performance benchmark.

The new 500-update checkpoint's best validation MSE is 0.254565005, worse than
the old 500-update score. Warm-started searches change the optimization path;
fewer rejected trials do not guarantee better results at a fixed update count.
Within the new run, all nine 500-update validation histories exactly match
the corresponding first 501 entries of the 2,000-update histories.

Strict convergence remains unresolved. The winning fit's terminal residual is
8.1427e-6: mapping displacement 3.6041e-6 plus proximal uncertainty 4.5386e-6,
against tolerance 1e-6. It is marked `max_iterations`,
`optimization_converged=False`, and `precision_limited=True`. Its
validation-selected state at iteration 1,390 has a larger residual than the
terminal state. The three formerly stalled choices have precision-limited
terminal diagnostics. This regression establishes the software fixes, not
readiness of every experiment for a final 100-seed run.

## Validation and reproduction

These are the checks recorded for the v0.4.0 snapshot, not current test counts
or a current-source wheel verification:

- 281 package tests passed from source and from an isolated installation of
  the v0.4.0 wheel.
- 252 SparseSMART simulation runner and summary tests passed.
- An independent audit regenerated the data and reproduced the selected
  validation MSE and coefficient error exactly. Training, validation, truth,
  and combined input fingerprints match both historical runs.
- All 2,000 accepted winner steps satisfy the stable sufficient-decrease
  condition and warm-start recurrence. All 2,001 history rows satisfy the
  residual decomposition, fixed diagnostic inverse step 20, and anchor floors.
- Synthetic failure tests separately verify earlier-checkpoint retention when
  a later fit fails, including exclusion of a lower-scoring failed partial fit.

The historical invocation below requires the matching v0.4.0 source package,
simulation runner, and recorded environment, from `code/simulation`. A current
source installation follows different optimization paths. Use a fresh output
root for any new run; the saved root below identifies the historical artifact.

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/sparse-smart-v04-mpl \
XDG_CACHE_HOME=/private/tmp/sparse-smart-v04-cache \
PYTHONPATH=../sparse-smart/src:. \
/Users/mladen.kolar/miniforge3/envs/smart-boxinj/bin/python \
run_sparse_smart_external.py 2 2 3 --setting-index 3 \
  --iterations 2000 --iteration-budgets 500 2000 \
  --init-penalties 0.03 \
  --penalties-u 0.0025,0.01,0.04 --penalties-v 0.0025,0.01,0.04 \
  --inverse-step 20 --validation-size 100 --validation-seed-tag 1397970481 \
  --spectral-step projected --initialization-spectrum projected \
  --refinement-solver anchor_projected --stationarity-tol 1e-6 \
  --output-root result/sparse_smart_v04_checkpoint_regression
```

The saved implementation fingerprint is
`9dbdb0c6d8d915abc2ec4e292a188c2e35405b1573dc9e873723bd63f27dae56`.

[Raw regression result](result/sparse_smart_v04_checkpoint_regression/model3/exp3/SparseSMARTExternal_result_model3_exp3_rs=7_rd_seed_id=3.json)
and [archival v0.4.0 wheel](../sparse-smart/dist/sparse_smart-0.4.0-py3-none-any.whl).
This wheel preserves the earlier implementation and does not include subsequent
source changes.
