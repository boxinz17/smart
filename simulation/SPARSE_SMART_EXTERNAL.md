# SparseSMART with 200 training and 100 independent tuning observations

This pilot fits SparseSMART only. For each existing Model I seed, the original
200-observation training experiment is unchanged. A separate random stream
generates 100 tuning observations with the same true coefficient, AR(1) design
covariance (rho 0.5), and target noise standard deviation (0.5).

Every candidate fits all 200 training rows. The 100 tuning rows select penalties
and the retained refinement iterate; they are never added to the fitting
sample. The known coefficient generates the simulated responses and measures
coefficient error after selection. It is not supplied to the estimator or used
to choose parameters.

The legacy generator draws its coefficient after drawing the design, so simply
increasing its sample size to 300 would change the experiment. The new helper
instead calls it once at n=200, then uses PCG64 with
`SeedSequence([existing_seed, 1397970481])` for the tuning data. Stored hashes
identify the training observations/source, independent tuning observations,
truth, implementation, and configuration separately.

Install the packages from the `code/` software repository root if needed:

```sh
python -m pip install -e ./smart -e ./sparse-smart
```

From `code/simulation/`, run the six source-noise levels for seeds 0–4:

```sh
for seed in 0 1 2 3 4; do
    python run_sparse_smart_external.py 0 3 "$seed" --validation-size 100
done
python summarize_sparse_smart_external.py
```

The three positional arguments are model ID, experiment ID, and existing seed
ID. Here `0 3` selects Model I's source-noise experiment, whose training sample
size is 200. `--setting-index 5` selects only source noise 0.5; `--dry-run` shows
the complete configuration. Other experiments retain their own training sample
sizes. `--validation-size` controls only the additional tuning sample size.

Defaults match the previous pilot: initializer penalty 0.03; left and right
penalties chosen independently from (0.0025, 0.01, 0.04); nine candidates; up to
500 refinement updates; projected spectral steps; and the full complement as
the support cap. The initializer and each accepted iterate are scored on the
same tuning sample. Successful completion at the iteration limit is not a
stationarity certificate. Positive source noise uses the explicitly labeled
empirical source mode; these experiments do not claim theorem certification.

Outputs are separate from the earlier 160/40 pilot, under
`result/sparse_smart_external/model1/exp4/`. Resumption checks the configuration,
training data, tuning data, truth, and implementation. Use a new output root for
changed experiments. The earlier runner and saved results remain available.

This restores the paper's 200-observation fitting sample but uses 100 additional
observations for tuning. Consequently, paper curves provide context under a
different total data budget and tuning procedure. Only SparseSMART is rerun;
the comparison does not establish superiority under an equal data budget.

The completed pilot has 30 successful cells (six source-noise levels and five
seeds) and 270 successful candidate fits. All 30 training/source and truth
fingerprints match the earlier saved experiments. At source noise 0.5, mean
coefficient error is 0.038196 (SE 0.003302), compared with 0.045070 in the earlier
160/40 pilot. All five high-noise searches choose penalties 0.0025 on both
sides and iterates 499–500. The iteration limit and penalty-grid boundary remain
relevant limitations; the added data does not establish convergence.

See the [comparison report](result/sparse_smart_external/summary/comparison.md),
[plot](result/sparse_smart_external/summary/comparison.png), and
[audit](result/sparse_smart_external/summary/validation_audit.json). The report's
inherited `validation_fraction` and `split_seed` configuration fields are
inactive when explicit tuning data is supplied; they do not reduce the 200-row
training sample.

For all three models and all four saved experiment grids, see the
[expanded pilot guide](SPARSE_SMART_EXTERNAL_GRID.md).
