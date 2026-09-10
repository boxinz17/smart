# Validation-tuned SparseSMART simulations

This runner fits only the new SparseSMART package using the existing V1 seeds
and generator. Paper comparison curves remain in `paper_reference/`; it never
runs another estimator. Use the current editable SparseSMART source for the
selection behavior described below; saved wheels retain their historical rules.

Create the [shared environment](../environment/README.md), then install from
the `code/` software repository root:

```sh
source .venv/bin/activate
python -m pip install --no-build-isolation -c python-constraints.txt \
  -e ./smart -e ./sparse-smart
```

From `code/simulation/`, run one existing seed at high source noise:

```sh
python run_sparse_smart_tuned.py 0 3 0 --setting-index 5
```

The positional arguments are model ID (0-2), experiment ID (0-3), and existing
seed ID (0-99). Omitting `--setting-index` runs the selected experiment's full
grid. `--dry-run` displays the resolved configuration without fitting.

Defaults use a deterministic 80/20 target training/validation split, initializer
penalty 0.03, and independent left/right penalty grids `(0.0025,0.01,0.04)`:
nine candidates per cell. The initializer and up to 500 accepted refinement
updates are scored on validation data. Current source compares each candidate
or iterate directly with its incumbent using
`mean((P-P_incumbent)*((P-Y)+(P_incumbent-Y)))`. A negative difference replaces
the incumbent; an exactly zero computed difference retains it. Iteration zero
is eligible. Absolute MSE and the relative `selection_score` are reporting
quantities, not tie-breakers. The latter uses one fixed initializer prediction
shared across the entire fit, avoiding subtraction of large complete losses.
Candidates train on the training subset only; the retained model is not
automatically refitted on all rows. Validation scores are selection scores.
The true coefficient is accessed after selection solely for evaluation.

Projected spectral updates enforce the configured singular-value bounds and
minimum gap while allowing movement along an active boundary. Other numerical
domain and descent checks stay active. `--spectral-step reject` selects the
original trial rule for a diagnostic comparison. `--stationarity-tol` defaults
to 1e-6. The stopping record distinguishes a small projected-gradient mapping,
maximum iterations, active-boundary stalls, and numerical/backtracking failure.
Successful fixed-iteration completion does not imply stationarity.

The default support caps are full complement **entry counts**: noisy Model I
uses 475 left and 225 right entries, while exact Model I uses 25 on each side.
Soft thresholding by the selected penalties can leave substantially fewer
entries nonzero. Configure grids explicitly when needed:

```sh
python run_sparse_smart_tuned.py 0 3 0 --setting-index 5 \
  --init-penalties .01,.03 \
  --penalties-u .00125,.0025,.005,.01,.02,.04 \
  --penalties-v .00125,.0025,.005,.01,.02,.04 \
  --support-grid 100:100,475:225 \
  --output-root result/sparse_smart_tuned_extended
```

Support counts must fit the working dimensions for every selected grid cell.
There is no heuristic uncertainty-to-support formula embedded in the defaults;
full caps avoid an arbitrary sparsity bottleneck, and an optional support grid
can be evaluated with the same target-only selection criterion.

The runner explicitly uses empirical source mode because the V1 positive-noise
grid violates the conservative source-accuracy condition. Actual source noise
and gap diagnostics remain in the selected fit. `--strict-source-check`
enforces that condition. These practical simulations have no theorem claim.

Results are atomic JSON files under `result/sparse_smart_tuned/model*/exp*/`.
They include grids, split indices and sizes, implementation/data/configuration
fingerprints, every candidate's outcome, failed partial scores, selected
penalties/supports/iterate, validation loss, training history, and termination.
Current records identify `selection_rule="pairwise-validation-loss-v1"` and
save iterate/candidate incumbent comparisons for replay. The selected ambient
factors and `validation_reference_prediction` preserve the selected prediction
and relative-score evidence. Historical records without this marker remain
readable under their original relative-score/absolute-MSE or absolute-MSE rule;
they are not relabeled as results from current selection.
Failed candidates cannot win; an all-failed cell has no coefficient error.
Resumption requires identical data, configuration, and implementation. Use a
new output root for changed experiments, or `--force` for intentional replacement.
Current source uses the portable `sparse-smart-source-content-v1` scheme and
rejects legacy/missing schemes for resume; historical summaries remain readable.
See the [shared result-reuse policy](README.md#sparsesmart-workflows-and-result-reuse)
for content hashes, saved source paths, and batch-attempt handling.

The historical completed Model I pilot uses seeds 0-4 and Experiments 1 and 4, with the
default nine-candidate grid. All 55 cells and their 495 candidate records are
successful; no cells are missing. The shared n=200/source-noise=0.01 cell is
repeated across the two experiments, so there are 50 distinct datasets. All
winning optimization runs reach the 500-update budget without meeting the
stationarity tolerance. Selection can retain an earlier iterate.

The original pilot commands below require its matching historical source and
environment. Current reruns must use fresh output roots; the later source
fingerprint and selection rule differ. The comparison also uses saved paper
curves and the earlier fixed-parameter pilot:

```sh
for seed in 0 1 2 3 4; do
    python run_sparse_smart_tuned.py 0 0 "$seed"
    python run_sparse_smart_tuned.py 0 3 "$seed"
done
python summarize_sparse_smart_tuned.py \
  --fixed-result-root result/sparse_smart_pilot
```

See [comparison report](result/sparse_smart_tuned/summary/comparison.md) and
[plot](result/sparse_smart_tuned/summary/comparison.png). The summary checks
configuration, seed, split, candidate selection, and shared-cell consistency;
its CSV and audit JSON retain detailed counts. Means use five repetitions,
while the paper curves summarize 100 repetitions under their tuning procedure.
At source noise 0.5, all five searches choose the lower penalty-grid endpoints
and iterates 499-500. This motivates a separately configured follow-up with a
wider grid and longer optimization, rather than treating this pilot as a
fully converged or definitive performance comparison.
