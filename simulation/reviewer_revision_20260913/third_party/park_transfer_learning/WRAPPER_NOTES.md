# Exact authors' Park comparison

The files listed in `PROVENANCE.json` are unmodified copies from
<https://github.com/ishspsy/transfer_learning> at commit
`9c94ff38b71cded6e6c7a8b6db243e2c7a06b78d`, retrieved September 13, 2026.
The MIT license and original README are included. The wrapper verifies every
recorded SHA256 before it executes the authors' R code.

The implemented baseline calls the original `cv.twostep` in
`Function/Functions_naiveapproaches.R`. Its table label must be **Park all-source
two-stage NR**. With one available source, the source is included throughout;
this is the authors' pooled nuclear regression and target nuclear bias-correction
procedure with internally selected tuning parameters. It is not their FSD or
MSD source-selection procedure and does not replicate their original simulation
configuration. The raw source observations are available to this baseline;
the SMART and coefficient-only competitors receive only their fitted source
coefficient. This information advantage must be stated explicitly.

The wrapper does not alter the source functions, centering, standardization,
ADMM iterates, internal fold allocation or tuning rule. Its observational ADMM
wrapper records iteration counts and the authors' squared-update stopping rule
for every CV/final fit. That stopping rule is not a convex dual-gap certificate.
An iteration cap is recorded, not silently called convergence. Prediction uses
the returned target intercept; coefficient risk should be accompanied by
prediction risk that includes that intercept. The authors reset the R seed to
100 in each internal CV invocation.

The wrapper exposes user-declared, strictly increasing positive lambda grids,
ADMM eta, maximum iterations, squared-update tolerance and internal fold count.
Strict sorting is needed because the authors use `tapply` to group lambda scores
and then index the original lambda vector. The declared wrapper defaults are
lambda grids `[0.01,0.03,0.1,0.3,1,3,10]`, eta 1, 1000 iterations, squared-update
tolerance 1e-6 and five folds; these are revision tuning choices, not a claim to
use the publication's original grid. All target input rows belong to the fit;
evaluation rows must be withheld by the runner.

The author code depends only on R packages `corpcor` and base `stats` for this
path; the I/O wrapper additionally requires `jsonlite`. No `foreach`,
`doParallel` or `splitTools` dependency is needed for `cv.twostep`.

Two source-level issues prevent treating the unmodified FSD code as a reliable
single-source fallback. Its forward loops use `1:(length(auxvec)-1)`, which in
R iterates over `1,0` when the source count is one; and its no-source final branch
refers to `cvLasso` without a visible preceding definition within `FSDtrans`.
These are static findings about the pinned code, not reproduced experimental
failures of the publication. They do not invalidate `cv.twostep`. FSD and MSD
files are included only as provenance for this distinction and are not executed.

Run the small deterministic bridge check in an existing Slurm allocation after
loading R and the project Python environment:

```sh
python simulation/reviewer_revision_20260913/published_park.py --sanity --output /path/to/park_sanity.json
```

This checks bridge execution, intercept handling, original stopping diagnostics,
and near recovery in a no-noise fixture. It is not a performance simulation.
