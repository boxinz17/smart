# SparseSMART 0.3 failure regression

This experiment retries the 60 applicable failures from the previous three-model,
four-experiment pilot, plus six seed-zero controls: exact and low-noise sources
in each model. It fits only SparseSMART. Historical results under
`result/sparse_smart_external` are preserved; new results are under
`result/sparse_smart_repairs`.

The changes are spectral projection at initialization and fixed-anchor
constrained refinement in `H=Z D^{-1}` coordinates. Both preserve the requested
rank, source frames, anchors, and original refinement objective. The initializer
projection changes singular values and records the original values and correction.
The new solver retains the singular-value weights in the L1 penalty and solves
the combined L1/anchor proximal subproblem iteratively. It reports local
constrained residuals in the H metric. Neither change certifies initialization
quality, global optimality, or the manuscript's statistical conditions.

All original training rows and the same 100 independent tuning observations
are reused. The generator seeds, coefficient truth, observed source, penalty
grid, 500-update budget, and validation selection rule are unchanged. Candidate
failures remain ineligible for selection. Completing 500 updates is distinct
from reaching the stationarity tolerance.

The new solver requires full complementary support caps, as used in this pilot.
Smaller hard caps use the original chart solver under `refinement_solver="auto"`.
The 45 structurally unsupported settings from the original grid are not retried;
the requirement `1 <= rank <= source_rank` remains in force.

## Reproduce

From `code/simulation`, with the existing `smart-boxinj` Python environment:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=../sparse-smart/src:.
export MPLCONFIGDIR=/private/tmp/sparse-smart-pilot-mpl
export XDG_CACHE_HOME=/private/tmp/sparse-smart-pilot-cache
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python run_sparse_smart_repairs.py --workers 3
python summarize_sparse_smart_repairs.py
```

The runner verifies unchanged before/after data hashes, candidate parameters,
and split metadata. It also checks the winning trajectory's objective decrease
and anchor margins. Checkpoint reuse requires identical code and configuration;
after further code changes, choose a new `--output-root`.

The external single-cell runner also accepts `--initialization-spectrum` and
`--refinement-solver`. To reproduce the earlier practical solver, use
`--initialization-spectrum reject --refinement-solver chart`. Default `auto`
selects the new practical solver with full caps and projected spectral updates;
prescribed calibration retains the original solver and initialization rejection.

## Results

The generated [repair summary](result/sparse_smart_repairs/repair_summary.md)
contains rescued failures, the six paired controls, remaining failures, and
stationarity counts. Its [audit](result/sparse_smart_repairs/repair_audit.json)
records the validation checks. This is a targeted regression experiment;
combining repaired cells with old successful cells would mix different solver
versions and should not be presented as a fresh complete pilot.
The independent [regenerated-data audit](result/sparse_smart_repairs/regenerated_data_audit.json)
also regenerates the observations and verifies coefficient errors, validation
prediction losses, and implementation fingerprints, including the separate
longer-iteration check below.

Package tests cover transformed derivatives, a constructed independent proximal
optimality certificate, tangential descent at an active anchor, constrained
stationarity, and protection against zero steps masquerading as progress.

## Separate longer-iteration check

One Model II high-noise candidate (seed index 0, source noise 0.5, initializer
penalty 0.03, left/right penalties both 0.0025) was also run for 2,000 updates.
It retained the same 300 training rows and 100 independent tuning rows.
The initializer and first 500 validation-history entries exactly match the
corresponding candidate in the main regression (501 losses including initialization).
Its validation loss decreased from 0.299927 at update 500 to 0.278595 at update
2,000; the selected coefficient error was 0.017363. It still ended at the
iteration cap, with constrained residual 0.150702, above the 1e-6 tolerance.
The [separate result](result/sparse_smart_repairs_longer/model2/exp4/SparseSMARTExternal_result_model2_exp4_sigma0=0.5_rd_seed_id=0.json)
is an iteration diagnostic, not part of the unchanged-budget 66-cell regression.
This shows that longer refinement can help this candidate after the stall is
removed; it does not justify a universal iteration-only fix for poor starts.
