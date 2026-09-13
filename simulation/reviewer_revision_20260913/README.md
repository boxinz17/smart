# SMART reviewer numerical revision

This package implements the additional numerical evidence requested in the SMART
reviews. The main deliverable is
`response_r1/numerical_revision_note.pdf` (with self-contained LaTeX source) at the project root. Its execution ledger
and machine-readable registry are `response_r1/numerical_revision_progress.md`
and `response_r1/numerical_evidence/execution_registry.json`.

The study does not modify the SparseSMART v2 solver or the separate paper campaign.
Scientific snapshots are immutable copies with file-level SHA-256 manifests.
Later independent auditors and report scripts have separate hashes. The current
working directory is not a substitute for those frozen snapshots when reproducing
the reported results.

## Components

| File | Purpose |
|---|---|
| `design.py` | Controlled structural violations, fitted-source regression, named independent streams, and measured generator diagnostics |
| `competitors.py` | Exact target/source-subspace RRR and ridge RRR, source-centered ridge, mixtures, and certified convex nuclear contrast |
| `published_park.py`, `.R` | Bridge to the pinned authors' all-source two-stage NR implementation; common target-holdout selection in confirmation |
| `runner.py` | Frozen fixed-rank plan, full candidate archives, validation selection, timing, and separate evaluation payload |
| `rank_study.py` | Joint operational target-rank/source-dimension selection with matched baseline tuning |
| `budget_study.py` | Fresh fits at each iteration budget, with separately evaluated selected and terminal checkpoints |
| `audit.py`, `rank_audit.py`, `budget_audit.py` | Independent reconstruction of saved losses, selection, grid coverage, and failure denominators |
| `parallel_audit.py`, `parallel_rank_audit.py` | Seed-preserving read-only audit partitions, followed by checked exact-coverage merging |
| `evidence_report.py`, `rank_report.py`, `budget_report.py`, `runtime_report.py` | Figures, paired uncertainty, timing, and selection summaries from audited evidence |
| `metadata_report.py` | Recorded realized generator diagnostics and selected penalty-boundary counts |
| `population_floor.py` | Evaluation-only lower bound for any estimate confined to the observed source subspaces |
| `complete_tables.py` | Complete condition-level risk tables, including unavailable comparisons and failures |
| `cross_study_pairing.py` | Metadata-only proof that the main and rank studies reuse identical fitting and truth payloads |
| `competitor_boundary_report.py` | Raw-source competitor penalty/stopping census with source-file hashes |
| `export_compact.py` | Audited JSON/CSV/figure packaging with per-member hashes, without numerical arrays |
| `freeze.py` | Immutable source copy and content manifest |

## Executed protocols

- **Confirmation:** 76 conditions, 30 fresh seeds (`1000`–`1029`), 2280 tasks.
  Main library: one target-RRR endpoint plus 210 chart candidates, with 500 accepted
  updates available to each chart candidate. The two sparse-stress conditions add
  matched active-cap/full-cap libraries.
- **Operational rank:** six representative conditions, the same 30 seed IDs,
  target ranks `3,5,7`, operational source dimensions `10,15`; 180 tasks.
  This is a separate analysis of joint validation selection, not 180 additional
  independent replicates of each confirmation condition.
- **Runtime:** three dimensions, five fresh seeds (`3000`–`3004`), 15 tasks;
  complete fixed-rank tuning with the confirmation library.
- **Iteration budget:** four conditions, five seeds (`2000`–`2004`), four update
  budgets, six fresh fits per budget; 80 tasks and 480 fits. It uses a reduced
  library with full caps and no direct RRR shortcut.
- **Development:** seed 0 with the earlier grid. It is excluded from confirmation.
  Its artifacts are functional/development evidence and must not be pooled with
  the fresh-seed study.

All methods in a condition use identical target training and validation arrays.
Test observations and truth are reserved for evaluation, apart from the labelled
clean-source oracle. Park additionally receives raw source observations and
estimates an intercept; generic methods use the known-zero-intercept convention.
These information differences are reported explicitly.

## Execution and reproduction

Production fitting and numerical array audits require Slurm and a `/scratch2/`
result root. Use the task-scoped SSH helper and lifecycle rules in the project's
`scripts/SSH_SESSIONS.md`. Do not borrow another task's connection or change the
other campaign's jobs. The run registry gives the actual accepted job IDs and
snapshot locations; the ledger records preparation failures and replacements.

Source `source/hpc/discovery/env.sh` from the selected frozen run before importing
numerical packages. It checks the pinned Python/library runtime, one BLAS thread,
and the common Haswell dispatch convention. Add the frozen `simulation` and
`sparse-smart-v2/src` directories to `PYTHONPATH` as the supplied batch launchers do.
The Park bridge also requires the declared R module and `R_LIBS_USER` library.

The module interfaces used by batch jobs are:

```text
python -m reviewer_revision_20260913.runner plan ROOT --seeds SEEDS --iterations 500
python -m reviewer_revision_20260913.runner run ROOT TASK_INDEX
python -m reviewer_revision_20260913.rank_study plan ROOT --seeds SEEDS --ranks 3,5,7 --source-ranks 10,15 --iterations 500
python -m reviewer_revision_20260913.rank_study run ROOT TASK_INDEX
python -m reviewer_revision_20260913.budget_study plan ROOT
python -m reviewer_revision_20260913.budget_study run ROOT TASK_INDEX
```

Use a new result root for a new plan. A plan's hash binds task identity, candidate
libraries and seeds. Do not rewrite a plan to make missing or failed work disappear.
Reconcile existing task receipts before restarting interrupted work.

Independent audits can be run from a separately hashed overlay preceding frozen
source on `PYTHONPATH`. They do not refit any estimator:

```text
python -m reviewer_revision_20260913.parallel_audit ROOT --workers 8
python -m reviewer_revision_20260913.parallel_rank_audit ROOT --workers 8
python -m reviewer_revision_20260913.budget_audit ROOT
```

Production report generators require matching, complete, passing audit receipts.
The JSON/CSV-only reports may be regenerated from compact downloaded artifacts;
the population-floor evaluation reads numerical archives and runs under Slurm.
Published figures and tables must be regenerated after any change to their input
audit or plan. An incomplete/development output is not final evidence.

## Interpretation rules

Population excess prediction risk is per response and excludes irreducible
response noise. Include any estimated intercept in that risk. Report independent
paired-seed uncertainty, rather than counting multiple settings or tuning
candidates as independent samples. Failed methods retain their planned denominator.

The main hard caps are inactive. Only the matched sparse-stress experiment can
support a statement about active-cap benefit. A target-RRR endpoint selection is
not successful transfer optimization. Retaining iteration zero is not evidence
that refinement helped. A completed iteration budget is not a stationarity or
global-optimization certificate. Timing includes failed candidates and declared
cache reuse; source preprocessing must be charged consistently.

These simulations address numerical questions. They do not establish the missing
end-to-end theory or perform donor-blocked application validation.

Final local integration check: **209 tests passed**. All four completed numerical
studies passed independent Slurm audits with zero issues; declared method failures
remain in the evidence. Final figures rendered from cached audited summaries are
separate from the immutable original compact export.
