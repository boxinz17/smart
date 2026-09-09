# SparseSMART across the saved paper simulation grid

This pilot runs SparseSMART only, using existing seed IDs 0–4, the original
training data for each setting, and 100 additional independent tuning
observations. It extends the completed Model I source-noise pilot to all three
models and four experiments. Earlier results are verified and reused.

| Model | Dimensions (p, q) | Experiment 1 training sizes | Training size for experiments 2–4 |
|---|---|---|---|
| I | (100, 50) | 200, 400, 600, 800, 1000 | 200 |
| II | (150, 100) | 300, 500, 700, 1000, 1200 | 300 |
| III | (300, 200) | 500, 700, 1000, 1200, 1500 | 500 |

The remaining sweeps use fitted target ranks `1,3,5,7,9,11`, source ranks
`0,3,5,7,10,15,20`, and source noise `0,.01,.02,.05,.1,.5`. All data-generating
target/source ranks remain 5/10. The manuscript says the default sample size
is 200 for all models; this pilot follows the saved simulation scripts, whose
actual sizes are shown above.

All candidates fit the full training sample. The separate 100 observations
select penalties and the retained iterate; they are not added to training.
The [external-validation design](SPARSE_SMART_EXTERNAL.md) describes the
independent random stream and exact preservation of the original observations.
The inherited internal-split options in the saved configuration are inactive.

The fixed tuning configuration is unchanged: initializer penalty 0.03,
left/right penalties independently in `(0.0025,0.01,0.04)`, nine candidates,
500 refinement updates, projected spectral steps, full complement support
caps, and empirical source mode. Rank dimensions are specified by the grid;
the validation search does not choose rank across cells.

From `code/simulation/`, with `smart` and `sparse-smart` installed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_sparse_smart_external_grid.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python summarize_sparse_smart_external_grid.py
```

The batch uses three separate processes to isolate the legacy generator's
global random state. `--dry-run` displays its configuration. For a smaller
selection, use, for example, `--models 1 2 --experiments 0 3 --seed-count 5`.
Model/experiment IDs are zero-based. Per-cell records remain in
`result/sparse_smart_external/model*/exp*/`; the batch manifest is
`result/sparse_smart_external/expanded_pilot_manifest.json`. The new report and
plots are in `result/sparse_smart_external/grid_summary/`.

There are 360 nominal cells, of which 315 satisfy the package's structural
rank restriction. It requires `1 <= fitted_rank <= source_rank`, so the 45
cells with fitted rank 11 at source rank 10, or source rank 0/3 at fitted rank
5, are recorded as inapplicable. Dimensions are never silently clamped.
Applicable cells can still fail initialization or optimization; the report
retains those failures and denominators. Missing, inapplicable, and failed
cells are distinct, and none is plotted as zero error.

Overfitting the true target rank can violate the initializer's declared
singular-value floor (0.05) or minimum gap (0.01). For example, the Model III,
rank-9, seed-0 smoke check had a converged Lasso solution but additional
singular values below these margins. All nine refinement-penalty candidates
share the same initializer, so changing those penalties cannot resolve that
particular rejection. Source ranks below the true source rank can also omit
target directions from the reduced initializer.

Experiment 3 is **SparseSMART source-rank sensitivity on the paper grid**.
The paper's source truncation parameter controls which leading source
directions are left unpenalized. SparseSMART's `source_rank` instead determines
its reduced initializer and source-coordinate construction. Equal numerical
values do not make those parameters mathematically identical.

Paper curves are read from the saved figure references; no competing method
is rerun. They summarize 100 repetitions under the paper's tuning procedures,
whereas this pilot uses five seeds and 100 additional tuning observations.
Comparisons are descriptive and do not have equal total data budgets. In the
rank sweeps, the fixed-rank SMART curve is the closer reference; automatic
SMART selects its own ranks and appears as a horizontal reference.

The grid summary checks stored data fingerprints against regenerated original
training and independent tuning samples, and checks the repeated default
setting across the four experiments for identical inputs and selected fits.
Coefficient error, not the reused tuning score, is the reported performance
measure. Reaching the iteration budget is not a convergence certificate.

A reproduced Model II failure at source noise 0.5, seed 0, with penalties
0.0025/0.0025 reaches the left-anchor floor after 122 updates. Its smallest
left-anchor singular value is 0.00500000000002639 against the bound 0.005;
backtracking reduces the accepted step norm to about 2e-13. The Cayley bounds
are slack. This identifies an anchor-boundary stall that remains after the
spectral-gap projection change. The algorithm's convergence criterion does not
pass; this diagnostic is not a KKT analysis of the full constrained problem.
The attribution is verified for this one candidate, while the other eight
saved candidates report the broader active-constraint-stall termination.
See the [diagnostic record](result/sparse_smart_external/constraint_diagnostics/model2_noise05_seed0.json).

The completed five-seed grid has no missing records:

| Model | Successful fits | Failed fits | Inapplicable |
|---|---:|---:|---:|
| I | 87 | 18 | 15 |
| II | 84 | 21 | 15 |
| III | 84 | 21 | 15 |
| Total | 255 | 60 | 45 |

Of the 60 failed fits, 58 fail the initializer's spectral margins in the
target/source rank sweeps. Two fail with active-constraint stalls at source
noise 0.5 (seed 0 in Models II and III). None of the successful optimization
runs meets the stationarity tolerance within 500 updates; validation can
select an earlier iterate. The 15 repeated-default groups match across their
four experiments, and all 360 data records and successful coefficient errors
were checked against 150 regenerated data-generating settings.

At source noise 0.5, SparseSMART's mean coefficient errors are 0.038196 (5/5
successful fits), 0.029905 (4/5), and 0.020790 (4/5) for Models I–III. The last
two means are conditional on success. The paper's automatic SMART references
are 0.021689, 0.013147, and 0.007920, respectively, under its different tuning
data budget. At source noise 0.01 and throughout the sample-size sweeps, the
new five-seed means are numerically below the corresponding paper SMART
references; this is descriptive evidence, not an equal-budget superiority
claim. The expanded grid identifies rank-initialization and anchor-constraint
handling as remaining limitations.

See the [full comparison](result/sparse_smart_external/grid_summary/comparison.md),
[CSV](result/sparse_smart_external/grid_summary/comparison.csv),
[audit](result/sparse_smart_external/grid_summary/validation_audit.json), and
plots for [Model I](result/sparse_smart_external/grid_summary/comparison_model1.png),
[Model II](result/sparse_smart_external/grid_summary/comparison_model2.png), and
[Model III](result/sparse_smart_external/grid_summary/comparison_model3.png).

## Subsequent solver repair

The results above describe the original pilot. SparseSMART 0.3 adds feasible
initialization and anchor-aware refinement; its separate targeted regression is
documented in [SPARSE_SMART_REPAIRS.md](SPARSE_SMART_REPAIRS.md). Historical result
files are preserved and must not be pooled with the new solver's results.
