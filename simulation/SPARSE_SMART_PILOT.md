# SparseSMART pilot using existing paper comparisons

This documents the archived fixed-parameter 0.1.1 pilot. Use that package version
to reproduce its implementation fingerprints. For the projected solver and
target-validation tuning in 0.2, see [SPARSE_SMART_TUNED.md](SPARSE_SMART_TUNED.md).

This workflow runs only the new `sparse_smart` estimator. It reuses
`data/random_seeds/experiment_seeds.csv` and the existing Python data generator.
Comparison curves are read from `v1/fig_smart/simulation_model*.pdf`; no other
method is fitted. See [paper_reference/README.md](paper_reference/README.md)
for the reproducible vector extraction and its precision limits.

## Run

Install the packages from the software repository root (`code/`):

```sh
python -m pip install -e ./smart -e ./sparse-smart
```

From `code/simulation/`, run one seed on the full sample-size or source-noise grid:

```sh
python run_sparse_smart.py 0 0 0
python run_sparse_smart.py 0 3 0
```

The three positional arguments are model ID (0-2), experiment ID (0-3), and
existing seed ID (0-99). `--setting-index` selects one grid cell. `--dry-run`
prints resolved inputs without fitting. The data-generating ranks remain 5
and 10 while fitted ranks vary in Experiments 2/3. Inapplicable combinations
are recorded rather than clamped.

The first pilot uses Model I, seeds 0-4, Experiments 1 and 4:

```sh
for seed in 0 1 2 3 4; do
    python run_sparse_smart.py 0 0 "$seed"
    python run_sparse_smart.py 0 3 "$seed"
done
python summarize_sparse_smart.py --seed-count 5
```

It uses n=200,400,600,800,1000 at sigma0=0.01, and sigma0=0,0.01,0.02,0.05,0.1,0.5
at n=200. There are 55 saved fits, with the n=200/sigma0=0.01 cell repeated in
both sweeps by construction. The reported metric is normalized Frobenius
coefficient error, matching the paper. Paper curves summarize 100 repetitions;
this five-seed pilot is exploratory and is not a paired comparison against
raw baseline replicates.

## Fixed practical settings

Defaults are 50 accepted updates, initializer penalty 0.03, refinement penalty
0.01, starting inverse step size 20, and at most `5*fitted_rank` nonanchor
coordinates on each side (capped by the coordinate count). Margins are
`d_lower=0.05, d_upper=12, gap=0.01, anchor_min=0.005, trial_radius=1`.
The supplied nominal ranks and sparsity budget use the simulation configuration.
The coefficient truth is used only for reporting initial/final errors; neither
iteration count nor hyperparameters are selected from those errors.

Noisy runs use the explicitly labeled empirical option
`enforce_source_accuracy=False`. The original V1 positive-noise grid lies
outside the package's conservative source-accuracy condition. Each record keeps
the actual sigma0, declared clean-spectrum gap 1, calculated eta0, and whether
the source condition failed. This allows a practical experiment without
misstating a bound or theorem guarantee. `--strict-source-check` enforces the
original condition and records rejected runs. Package defaults remain strict.
All spectral, anchor, support, trust, and descent checks remain active.

## Outputs and resumption

Results are written atomically under `result/sparse_smart_pilot/model1/exp*/`.
The file format is JSON and stores exact configuration/data/implementation
fingerprints, initial and final coefficients/errors, objective histories,
source diagnostics, statuses, and runtime. A rerun skips a file only after
its configuration, regenerated inputs, and implementation agree; use `--force`
only for an intentional replacement, or use a different `--output-root`.

Failed fits remain visible, with `avg_err=null` and the last accepted error
stored separately when available. There is no baseline substitution.
The summary reports completed/failed/missing counts, completed-fit means and
SEs, and partial-fit averages separately. It rejects mixed implementations
or tuning. Its outputs are `summary/comparison.csv`, `comparison.md`, and
`comparison.png`.

The package's `PracticalCalibration.enforce_source_accuracy` switch was added
in 0.1.1. The plot reads paper reference values directly, so R baseline packages
and new runs of the older SMART estimators are unnecessary.
