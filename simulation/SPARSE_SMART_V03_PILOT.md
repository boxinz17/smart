# SparseSMART 0.3 uniform five-seed pilot

This pilot evaluates SparseSMART 0.3 across all three saved simulation models
and all four paper experiment grids. The main run uses the same implementation,
penalty grid, and 500-update limit throughout. A separate 2000-update run checks
the difficult source-rank and high-source-noise settings. Only SparseSMART is
fitted; comparison-method results are read from the paper's saved figures.

The main 500-update run finished on September 9, 2026, at 22:19:39 UTC, with
315 successful fits, 45 structurally inapplicable cells, and no missing or
failed cells. All 2835 candidate fits succeeded. The manifest records 66
verified checkpoint reuses and 294 newly written records. The separate
2000-update run also completed all 45 cells. Its nine-candidate searches
contain 402 successful candidates and three failed candidates; every cell
retains a successful selected fit. The combined independent audit is complete.

## Completed 500-update results

Each entry below is mean coefficient error ± standard error over five seeds,
all successful. Paper SMART values are approximate means extracted from the
saved figure curves, using the paper's different tuning procedure and data
budget.

| Model | Default source noise .01 | Paper SMART, .01 | Source noise .5 | Paper SMART, .5 |
|---|---:|---:|---:|---:|
| I | .005696 ± .000431 | .007155 | .034055 ± .001489 | .021689 |
| II | .004482 ± .000148 | .004737 | .024461 ± .000431 | .013147 |
| III | .002378 ± .000289 | .003275 | .015783 ± .000932 | .007920 |

The default-setting means are numerically below the paper SMART references;
the high-source-noise means remain above them. This is descriptive evidence,
not an equal-data-budget superiority comparison. In the high-noise cells,
validation selects iterations 496–500, so the iteration limit remains relevant.
Across the entire main grid, six winning optimization trajectories reach the
stationarity criterion and 309 end at the iteration limit; none of the retained
validation-selected iterates is certified stationary.

The [main summary audit](result/sparse_smart_v03_pilot_500/grid_summary/validation_audit.json)
checked all 360 records against 150 regenerated datasets, verified coefficient
errors and paper PDF hashes, and checked 15 repeated-default groups across
experiments. See the
[full main report](result/sparse_smart_v03_pilot_500/grid_summary/comparison.md)
for all settings, candidate diagnostics, and paper references.

## Completed iteration-budget comparison

The longer run finished on September 9, 2026, at 22:52:46 UTC. All 45 paired
cells succeed at both budgets. Selected validation MSE improves in 44 pairs
and worsens in one; every longer-run winner selects an iterate after 500.
The table reports five-seed mean coefficient errors for descriptive evaluation,
not for choosing the budget or tuning a fit.

| Model | Source rank 5: 500 → 2000 | Source rank 7: 500 → 2000 | Source noise .5: 500 → 2000 |
|---|---:|---:|---:|
| I | .054909 → .036982 | .049070 → .034195 | .034055 → .028136 |
| II | .040193 → .030292 | .033663 → .026296 | .024461 → .019075 |
| III | .016723 → .013027 | .012306 → .010948 | .015783 → .011323 |

The three failed candidates occur in Model III, source rank 7, seed 3. They
stop with numerical stagnation at iterations 1416, 1334, and 1657, with
constrained residuals above the `1e-6` tolerance. Six other candidates succeed
in that cell. Its 500-update winning candidate loses eligibility after its
continuation fails, so the selected validation MSE rises from .253698 to
.255829. Failed partial fits remain excluded, and the shorter completed fit
is not automatically retained as a fallback. Successful cells therefore do
not imply that all candidate fits succeeded.

All 45 longer-run winning trajectories end at 2000 updates without reaching
stationarity; none of the retained iterates is certified stationary. All 45
selected iterates fall within the last 10% of that budget (iterations
1800–2000), and 15 select iteration 2000 itself. The
[paired report](result/sparse_smart_v03_pilot_2000/iteration_summary/report.md)
includes standard errors, validation changes, candidate eligibility, and
coefficient changes that measure stability without using truth.

The package is ready for reproducible exploratory simulations under this
recorded protocol. For the designated difficult cells, 500 updates are not
an adequate stability check; 2000 updates usually improve held-out selection
but do not establish convergence or a stable final budget. Near-budget
selections also occur outside the follow-up: 38 of the other 270 main fits
select iterations 450 or later, including 22 at source noise .05/.1.
The 45-cell extension therefore does not establish whole-grid iteration
stability. Assess remaining budget and penalty-grid sensitivity before
treating a larger repetition count as a definitive performance comparison.

## Independent verification

The [combined audit](result/sparse_smart_v03_pilot_500/regenerated_data_audit.json)
checked all 405 records against 150 regenerated datasets and their historical
shared inputs. The maximum discrepancy in independently recomputed validation
MSE is `2.22e-16`; recomputed coefficient errors agree exactly. All 72 main
summary means and standard errors agree exactly. All 405 longer-candidate
validation prefixes match their 500-update counterparts exactly.

The audit also verifies the current implementation fingerprints, the seed
file, 15 repeated-default groups, unchanged result-file hashes, and original
paper PDF hashes. Its 360 successful coefficient checks comprise 315 main
fits and 45 longer fits; the other 45 records are structurally inapplicable.
No estimators were fitted during verification.

## Data and settings

The seed IDs are 0–4 from
[`data/random_seeds/experiment_seeds.csv`](data/random_seeds/experiment_seeds.csv).
Each cell preserves the original generator's training observations, observed
source, and true coefficient. An independent stream generates 100 additional
tuning observations conditional on that coefficient. All candidates fit every
training row; the tuning observations select penalties and the retained
iterate, including the initializer at iteration zero. There is no refit on the
combined training and tuning sample. The true coefficient is used to generate
responses and evaluate coefficient error after selection, never to tune a fit.

| Model | Dimensions `(p, q)` | Experiment 1 training sizes | Training size in experiments 2–4 |
|---|---|---|---|
| I | `(100, 50)` | 200, 400, 600, 800, 1000 | 200 |
| II | `(150, 100)` | 300, 500, 700, 1000, 1200 | 300 |
| III | `(300, 200)` | 500, 700, 1000, 1200, 1500 | 500 |

Experiment 2 varies fitted target rank over `1,3,5,7,9,11`; experiment 3 varies
fitted source rank over `0,3,5,7,10,15,20`; experiment 4 varies source noise
over `0,.01,.02,.05,.1,.5`. The generating target and source ranks remain 5
and 10 throughout. Other settings use fitted ranks 5/10 and source noise .01.
The legacy scripts use the Model II/III training sizes listed here, whereas
the manuscript prose states a common default of 200.

The main grid has 360 nominal cells: 315 applicable and 45 structurally
inapplicable. SparseSMART requires `1 <= fitted_rank <= source_rank`, so rank
11 with source rank 10 and source ranks 0/3 with fitted rank 5 are recorded as
inapplicable. No rank is clamped and no missing or failed fit is assigned zero
error. Each applicable cell searches nine candidates: initializer penalty .03
and left/right penalties independently in `(.0025,.01,.04)`.

The 2000-update follow-up contains 45 cells: all three models and all five
seeds at source ranks 5/7 in experiment 3 and source noise .5 in experiment 4.
It retains all nine penalty candidates, the same data, and the same selection
rule. It is a matched iteration-budget comparison, separate from the uniform
500-update paper-grid report. It is also separate from the earlier single
candidate diagnostic in `result/sparse_smart_repairs_longer`.
Selections near 500 updates also occur outside these 45 designated cells,
including intermediate source-noise levels. The follow-up therefore does not
establish iteration stability across the entire grid.

## Solver and reproducibility

Both runs use projected initialization and anchor-constrained refinement in
`H=Z D^{-1}` coordinates, with full complementary support caps and projected
spectral steps. The singular-value weights remain in the L1 penalty. Source
noise above zero uses the explicitly labeled empirical source mode. These
settings do not certify the manuscript's statistical assumptions or global
optimality. Reaching the iteration budget does not establish stationarity.

Independent tuning uses `SeedSequence([existing_seed, 1397970481])`, PCG64,
AR(1) design covariance with correlation .5, and target-noise standard
deviation .5. The inherited `validation_fraction` and `split_seed` fields are
inactive because explicit tuning observations are supplied; they do not
reduce the training sample.

Use the existing `smart-boxinj` environment from `code/simulation/`:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=../sparse-smart/src:.
export MPLCONFIGDIR=/private/tmp/sparse-smart-pilot-mpl
export XDG_CACHE_HOME=/private/tmp/sparse-smart-pilot-cache
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

The explicit single-cell command, useful for inspecting the frozen options, is:

```bash
python run_sparse_smart_external.py 0 3 0 --setting-index 5 \
  --initialization-spectrum projected --refinement-solver anchor_projected \
  --iterations 500 --validation-size 100 \
  --output-root result/sparse_smart_v03_pilot_500 --dry-run
```

Model, experiment, seed, and setting indices are zero-based. The batch commands
are:

```bash
python run_sparse_smart_external_grid.py \
  --seed-count 5 --workers 3 --iterations 500 --profile full \
  --initialization-spectrum projected --refinement-solver anchor_projected \
  --reuse-root result/sparse_smart_repairs \
  --output-root result/sparse_smart_v03_pilot_500
python run_sparse_smart_external_grid.py \
  --seed-count 5 --workers 3 --iterations 2000 --profile difficult \
  --initialization-spectrum projected --refinement-solver anchor_projected \
  --output-root result/sparse_smart_v03_pilot_2000
```

Add `--dry-run` to inspect either batch without fitting. The actual pilot
launches both batches concurrently, with three worker processes per batch and
one numerical-library thread per process. Running the commands sequentially
preserves the same experiment.

Each output root contains `expanded_pilot_manifest.json`, including process
identifiers, the batch-driver hash, configurations, and per-cell outcomes.
Existing repair records may be reused only when the current runner verifies
exact data, configuration, and implementation fingerprints. Historical outputs
remain in their original directories; older successful fits are not mixed
into this uniform implementation's results.

After the batches finish, generate the main paper-grid report and the matched
iteration comparison:

```bash
python summarize_sparse_smart_external_grid.py \
  --result-root result/sparse_smart_v03_pilot_500 \
  --output-dir result/sparse_smart_v03_pilot_500/grid_summary
python summarize_sparse_smart_iteration_pilot.py \
  --base-root result/sparse_smart_v03_pilot_500 \
  --extended-root result/sparse_smart_v03_pilot_2000 \
  --output-root result/sparse_smart_v03_pilot_2000/iteration_summary \
  --seed-ids 0 1 2 3 4
```

## Reading the results

The main performance measure is
`norm(C_hat-C_star, fro) / sqrt(p*q)`. Means and standard errors summarize
five saved seeds per applicable setting; if any fits fail, reported means are
conditional on success and must retain their success denominators. Tuning MSE
is the score used to choose a candidate and iterate. It is not an independent
test-set estimate. Penalty-grid boundaries, selected iterations near the
budget, failures, and stationarity diagnostics remain relevant when reading
the curves.

The paper's plotted curves summarize 100 repetitions under its tuning
procedures; this pilot uses five seeds and 100 additional tuning observations.
The paper reference CSV contains values extracted from PDF vector curves,
not the underlying replicate results. Comparisons are descriptive: neither
an equal-total-data-budget comparison nor paired inference is available.
Paper figures and their provenance are checked using the saved PDF hashes.

Experiment 3 is **SparseSMART source-rank sensitivity on the paper grid**.
The paper's source-rank parameter counts leading source directions left
unpenalized, while SparseSMART's source rank determines its reduced initializer
and source-coordinate construction. These are different operations. In the
rank sweeps, fixed-rank SMART is a closer reference than automatic SMART,
which selects its own ranks.

The independent audit regenerates the training, tuning, and truth arrays;
checks equality to historical shared inputs; recomputes successful validation
prediction MSEs and coefficient errors; and compares the recorded code
fingerprint with the current frozen implementation. Inapplicable records have
their own expected fingerprint because their runner does not load the
estimator API. Repeated default settings across the four experiments are
checked for matching outcomes. Each longer candidate's shared validation
history is compared with its 500-update counterpart before interpreting an
iteration-budget difference.

The exact [independent audit script](result/sparse_smart_v03_pilot_500/audit_sparse_smart_v03_pilot.py)
is preserved beside the records. From `code/simulation/`, rerun it without
fitting any estimator:

```bash
python result/sparse_smart_v03_pilot_500/audit_sparse_smart_v03_pilot.py \
  --mode both \
  --main-manifest result/sparse_smart_v03_pilot_500/expanded_pilot_manifest.json \
  --extended-manifest result/sparse_smart_v03_pilot_2000/expanded_pilot_manifest.json \
  --wait
```

It checks arriving records but writes its final audit only after both manifests
finish and all 405 records pass. The audit records the script's SHA256 hash,
the command arguments, and every checked result-file hash.

## Output locations

- Main 500-update records: [`result/sparse_smart_v03_pilot_500`](result/sparse_smart_v03_pilot_500).
- Batch manifests: [500 updates](result/sparse_smart_v03_pilot_500/expanded_pilot_manifest.json)
  and [2000 updates](result/sparse_smart_v03_pilot_2000/expanded_pilot_manifest.json).
- Main comparison: [report](result/sparse_smart_v03_pilot_500/grid_summary/comparison.md),
  [CSV](result/sparse_smart_v03_pilot_500/grid_summary/comparison.csv), and
  [summary audit](result/sparse_smart_v03_pilot_500/grid_summary/validation_audit.json).
- Main plots: [Model I](result/sparse_smart_v03_pilot_500/grid_summary/comparison_model1.png),
  [Model II](result/sparse_smart_v03_pilot_500/grid_summary/comparison_model2.png), and
  [Model III](result/sparse_smart_v03_pilot_500/grid_summary/comparison_model3.png).
- Independent combined data audit:
  [`regenerated_data_audit.json`](result/sparse_smart_v03_pilot_500/regenerated_data_audit.json).
- Longer 2000-update records: [`result/sparse_smart_v03_pilot_2000`](result/sparse_smart_v03_pilot_2000).
- Matched iteration comparison: [report](result/sparse_smart_v03_pilot_2000/iteration_summary/report.md)
  and [audit](result/sparse_smart_v03_pilot_2000/iteration_summary/audit.json).

Both batch manifests, the main report and plots, the combined independent
audit, and the matched iteration report are finalized.
