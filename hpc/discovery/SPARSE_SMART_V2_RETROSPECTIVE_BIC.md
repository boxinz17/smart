# Retrospective BIC on saved validation-stopped fits

This campaign reads the completed fixed-grid SparseSMART v2 results in place.
For each case it compares three selections over the same eligible candidates:

1. The original winner by validation MSE at each candidate's selected state.
2. The winner by training BIC at each candidate's validation-selected state.
3. The winner by training BIC at each candidate's terminal, validation-stopped state.

Both BIC comparisons inherit the original validation-driven trajectory and
stopping. They are retrospective hybrids, not training-only BIC fits. They do
not search unsaved iterations or the automatic rank/free-count grid.

The score and model-dimension approximation are the same as the training-only
BIC campaign. Observed training residual sum of squares supplies the likelihood
term; validation error and coefficient truth are used for reporting only after
the BIC winner is determined. Each endpoint uses its own saved chart and the
unchanged physical free rows. The direct RRR endpoint uses its RRR dimension.

The immutable plan names the old model/experiment roots, exact original cases,
and their plan, preparation and source-manifest hashes. A new source snapshot
and source manifest bind the postprocessing implementation. No raw outputs or
initializers are copied and no optimizer is called.

## Slurm sequence

Use `submit_sparse_smart_v2_retrospective_bic.py --run-dir ROOT --workers 32`.
`ROOT` must be a new directory on `/scratch2`, containing the frozen source,
manifest, plan and headerless `work-items.tsv` of case IDs.

- Preparation: one CPU, 16 GB, four hours. Run focused tests, verify provenance,
  build small case input indexes and score one case from each original subroot.
- Scoring pool: 32 tasks, one CPU and 4 GB per task, 24 hours. GNU Parallel
  dispatches one case per exclusive `srun` step; both BIC arms share the reads.
  Allocation may span nodes. Independent cases continue after a case failure.
- Summary: one CPU, 16 GB, four hours, after the pool ends. Read compact case
  records to produce coverage diagnostics, paired comparisons and figures.

The pool depends on successful preparation. The summary runs after any pool
outcome, so incomplete coverage is reported. Slurm arrays are not used.
Preflight case results are reused. A repeat submission reuses recorded job IDs;
ambiguous submissions require reconciliation and are never retried automatically.

Each case retains a compact result and hash-bound status record, candidate BIC
terms, selected identities, provenance, endpoint errors, classification counts,
and a worker log. Candidate matrices and trajectories remain only in the old
campaign. Summary artifacts can be downloaded without transferring the raw fits.
