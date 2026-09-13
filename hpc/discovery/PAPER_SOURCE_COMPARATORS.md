# Source-aware comparators on the original paper grid

This isolated campaign pairs eight added methods with the completed
validation-selected SparseSMART v2 results. Scientific policy is documented in
`response_r1/paper_source_comparator_campaign_protocol.md` in the parent project.
Existing paper, BIC, and reviewer-stress campaigns are read-only references.

- All three original model sizes, all four supported experiment grids, seeds
  0–99: 6,600 display cases, 3,000 datasets, 33,000 tuned method tasks.
- Five source-coefficient comparators, raw Lasso–SVD initializer, target RRR,
  and target ridge RRR. No raw-source-data method is included.
- Every fit uses the reference training observations, its same 200-observation
  validation set, and the same observed source coefficient. No validation refit.
- Experiment 3 fits are aliases of independent comparator tuning. Rank-free
  centered ridge and nuclear contrast are also aliases across Experiment 2.
- Reference hashes, all display-alias array fingerprints, source-frame identity,
  and initializer-primitive source hashes must agree before production starts.
- Candidate grids are immutable. Training-derived ridge/nuclear scales and
  selected boundary hits are retained. Failed candidates cannot silently win.
- Each method retains every candidate's score, parameters, status and numerical
  certificate, plus the selected coefficient. The runtime independently checks
  the selected state/score. It does not retain/reexecute all unselected states.

## Entrypoints

`build_paper_source_comparators_snapshot.py --staging LOCAL_NEW_DIRECTORY
--remote-root REMOTE_ROOT` builds an isolated source archive and manifest.
Upload/extract it into a new `/scratch2` root. Use the project SSH session helper.

With the frozen simulation and package directories on `PYTHONPATH`, run:

```sh
python -m paper_source_comparators_20260913.plan --root REMOTE_ROOT
python REMOTE_ROOT/source/hpc/discovery/submit_paper_source_comparators.py \
  --run-dir REMOTE_ROOT --prepare-only --dry-run
python REMOTE_ROOT/source/hpc/discovery/submit_paper_source_comparators.py \
  --run-dir REMOTE_ROOT --prepare-only
```

Preparation runs focused tests, verifies every shared input, and dispatches 46
fixed seed-zero canary tasks through the same one-CPU Slurm worker used in
production. These cover every model size, fitted rank 11, and Model III source
noise zero and 0.5. Only a successful `canary.json` with the matching plan opens
the production gate. Scientific method failure remains distinct from a broken
execution or selected-state audit.

After verifying preparation, submit the full campaign:

```sh
python REMOTE_ROOT/source/hpc/discovery/submit_paper_source_comparators.py \
  --run-dir REMOTE_ROOT --workers 100 --max-tasks-per-chunk 5000 --dry-run
python REMOTE_ROOT/source/hpc/discovery/submit_paper_source_comparators.py \
  --run-dir REMOTE_ROOT --workers 100 --max-tasks-per-chunk 5000
```

Seven case-aligned GNU Parallel pools run **serially**, each requesting 100
single-CPU workers, 4 GB per CPU, and 24 hours. The campaign-wide fitting ceiling
is therefore 100 CPUs, not 700. Pools continue independent tasks after a worker
failure. A final one-CPU, 16 GB, 24-hour summary depends on all pools ending and
reports missing/failed work instead of silently dropping it. Preparation uses
one CPU, 16 GB, six hours. The existing environment is reused; no packages are
installed by jobs.

Every `sbatch` call has a durable intent/result receipt. Repeating submission
reuses active jobs. Use `--resume` only after reconciling ended allocations;
completed hash-bound tasks are skipped. An uncertain submission is never
automatically repeated. Do not alter a frozen source or plan to repair a run;
use a fresh snapshot/root and preserve the failed attempt's evidence.

## Outputs and paper use

`summary/summary.json`, `per-case.csv`, `per-setting.csv`, `task-outcomes.csv`,
and `settings.pdf/png` contain planned denominators, numerical failure counts,
selected tuning values, coefficient RMSE and Monte Carlo uncertainty, and
paired comparisons with the reference. The report explicitly labels partial
coverage. Flat-line aliases never add Monte Carlo replications.

The report combines methods under the common validation policy. It does not
merge training-only BIC outputs or claim equal optimization/search budgets.
Saved source decompositions are excluded from comparator fit-call timings;
those timings cannot establish end-to-end computational dominance.
