# BI-SMART

`bi-smart` is the development package for **block-invariant SMART
(BI-SMART)**.  It follows the theorem-ready, three-fold procedure in the
appendix of the companion manuscript rather than copying the earlier ADMM
solver in `code/smart`.

The distribution name contains a hyphen, but the Python import uses an
underscore:

```python
import bi_smart
```

## Development status

This is intentionally versioned as `0.1.0.dev0`.  The appendix fully specifies
the source decomposition, tie-safe block library, exact block/cell screens,
restricted reduced-rank regression (RRR), safeguard candidates, and validation
selector.  Those stages are implemented as ordinary deterministic numerical
code.

The quotient Gauss--Newton implementation provides two backends for the
appendix's full-tangent Moore--Penrose formulation.  The dense reference
assembles the joint residual Jacobian and applies a rank-revealing SVD without
damping.  The matrix-free backend applies the same Jacobian and its exact
adjoint through zero-start LSQR, without forming the full Jacobian or a dense
vertical-coordinate matrix.  `RefinementControls` defaults to `"dense"`, which
retains the strict full-Jacobian numerical-rank policy.  Large-data runs may
explicitly select `"matrix_free"`, or select `"auto"` to use dense below the
workspace cap and matrix-free above it.  Both paths check the quotient
dimension, tangency, horizontality, the normal residual, and the descent
identity.  A singular or irregular call remains an unsuccessful branch-local
diagnostic; its exact RRR candidate at `t=0` and later safeguards are retained.

## Installation

From the surrounding `code/` directory:

```bash
conda activate smart-boxinj
python -m pip install -e './bi-smart[test]'
```

Verify the import:

```bash
python -c "import bi_smart; print(bi_smart.__version__)"
```

Unlike the current `smart` package, this scaffold deliberately exports
`__version__`, so the command above prints `0.1.0.dev0`.

## Full-sample restricted-RRR pilot

The package also exposes a deliberately minimal estimator for the initial
simulation pilot.  It uses the complete target sample once and fits rank-`r`
RRR inside the leading `r0` left and right singular subspaces of the observed
source matrix:

```python
from bi_smart import restricted_rrr

fit = restricted_rrr(
    X,
    Y,
    C0_tilde,
    target_rank=5,
    source_rank=10,
)

if not fit.successful:
    raise RuntimeError(
        f"{fit.candidate.failure_reason}: {fit.candidate.message}"
    )
C_hat = fit.coefficient
```

This entry point intentionally does **not** run the block screen, source-error
or Wedin gates, validation selector, safeguards, or Gauss--Newton refinement.
It records the observed source-boundary and RRR cutoff gaps as diagnostics but
does not use them to select among estimators.  The structural ranks must obey
`1 <= target_rank <= source_rank <= min(p, q)`.

## Quick start

The three `FoldData` objects must be statistically independent; the package
can check their dimensions, but it cannot verify how the rows were split.

```python
import numpy as np

from bi_smart import BISMART, BISMARTConfig, FoldData

# C0_tilde has shape (p, q).  These arrays are placeholders for the three
# independent target folds used by Appendix Algorithm 2.
C0_tilde = np.load("observed_source.npy")
X_in, Y_in = np.load("X_in.npy"), np.load("Y_in.npy")
X_ft, Y_ft = np.load("X_ft.npy"), np.load("Y_ft.npy")
X_val, Y_val = np.load("X_val.npy"), np.load("Y_val.npy")

config = BISMARTConfig(
    target_rank=2,                 # r
    source_rank=5,                 # r0
    source_error_bound=0.10,       # externally calibrated epsilon_hat_0
    budget_path=((2, 2), (3, 3), (5, 5)),
    source_weights=(0.1, 1.0, 10.0),
    pinv_rcond=1e-12,              # explicit target-only pseudoinverse cutoff
    atol=1e-12,                    # absolute numerical comparison tolerance
    rtol=1e-10,                    # relative numerical comparison tolerance
)

model = BISMART(C0_tilde, config)
result = model.fit_folds(
    FoldData(X_in, Y_in, name="initialization"),
    FoldData(X_ft, Y_ft, name="fitting"),
    FoldData(X_val, Y_val, name="validation"),
    enable_refinement=False,
)

C_hat = result.coefficient
print(result.selected_candidate.label)
print(model.branch_diagnostics_)  # algebraically unsuccessful branches
```

With `enable_refinement=False`, this runs every fully specified exact stage
and retains restricted RRR as iteration `t=0`.  `source_weights` belongs only
to the refinement grid and therefore does not change an exact-stage run.

To exercise the complete grid orchestration, explicitly supply the constants
that the manuscript leaves to the caller:

```python
from dataclasses import replace
from bi_smart import RefinementControls

controls = RefinementControls(
    armijo_constant=1e-4,       # c_A
    contraction=0.5,           # beta
    initial_step_size=1.0,      # bar_eta
    radius_ratio=2.0,           # q_rho
    radius_half_width=2,        # J_rho
    max_backtracking_cap=5,     # B_cal
    iteration_cap=10,           # T
    # Explicit scalable policy; use "dense" for the strict full-J SVD check.
    gauss_newton_backend="matrix_free",
    # matrix_free_max_iterations=500,  # optional explicit Krylov cap
)
model = BISMART(C0_tilde, replace(config, refinement_controls=controls))
result = model.fit_folds(
    FoldData(X_in, Y_in, name="initialization"),
    FoldData(X_ft, Y_ft, name="fitting"),
    FoldData(X_val, Y_val, name="validation"),
    enable_refinement=True,
)
```

Regular calibrated calls now produce `t=1,...,T` refinement candidates.  A
call contributes those iterates only if every requested iteration completes;
otherwise its partial path is discarded and `model.branch_diagnostics_`
records the numerical reason.

## Appendix algorithm represented by the package

The canonical API uses three independent target folds:

1. **Initialization fold:** infer source blocks and screen whole block unions.
2. **Fitting fold:** recompute exact restricted RRR and run safeguarded
   quotient Gauss--Newton refinement when explicitly enabled.
3. **Validation fold:** choose deterministically from every successful transfer
   candidate plus full-block, target-only, and zero safeguards.

In pseudocode, the complete estimator is:

```text
INPUT:
    initialization, fitting, validation folds
    observed source matrix C0_tilde
    target rank r and source truncation rank r0
    source operator-error certificate epsilon0
    ordered budget path K and source-weight grid Omega

SOURCE PROCESSING:
    compute the rank-r0 source SVD
    if the r0 versus r0+1 boundary is not certified:
        skip every source-guided branch
    else:
        identify certified spectral cuts
        build nested partitions from finest to one-block coarsest

FOR each budget pair and each partition, in deterministic order:
    form the initialization-fold source-coordinate statistic Z_in
    choose the raw or exact cell-thresholded hybrid pilot
    require a strict rank-r cutoff
    retain the pilot candidate
    perform one exact whole-block alternating screen
    retain the frozen candidate and its selected block sets

    if the observable Wedin gate fails:
        keep pilot/frozen; skip this branch's RRR and refinement
    else:
        recompute exact restricted RRR on the fitting fold
        retain RRR as iteration t=0
        for each source weight / trust radius / backtracking cap:
            run at most T quotient Gauss--Newton steps
            keep iterates only when that complete call succeeds

SAFETY CANDIDATES:
    independently run the one-block, full-budget source branch
    add minimum-norm target-only RRR
    add the zero matrix

OUTPUT:
    compute validation MSE for every successful labelled candidate
    return the first candidate attaining the minimum
```

## Package layout

```text
bi_smart/
    types.py           Validated configurations, folds, partitions, results
    linalg.py          Strict SVD/SPD/polar helpers and deterministic tie rule
    source_blocks.py   Source SVD, certified cuts, coarsenings, Wedin gate
    screening.py       Exact block/cell knapsacks and one-sweep screen
    initialization.py  Restricted and target-only reduced-rank regression
    _quotient_geometry.py  Tangent frame, gauge space, quotient dimensions
    _matrix_free_linalg.py  NumPy-only Golub--Kahan/LSQR iteration
    refinement.py      Joint state/objective, quotient-GN solve, safeguards
    candidates.py      Deterministic candidate library and validation
    estimator.py       Complete BI-SMART orchestration
```

The stage separation is deliberate.  It makes algebraic failure branches
testable and prevents the monolithic solver state from obscuring which fold is
allowed to use which responses.

## Manuscript choices kept explicit

The scaffold does not invent values that the appendix has not fixed.  In
particular, callers must make deliberate choices for ranks, fold construction,
the source-error certificate, budgets, weights, and refinement controls.
`RefinementControls` represents `c_A`, `beta`, `bar_eta`, `q_rho`, `J_rho`,
`B_cal`, and `T`; its `max_dense_work_bytes`, `gauss_newton_backend`, and
`matrix_free_max_iterations` fields are non-statistical process controls.
`"dense"` is the default and applies the strict full-Jacobian SVD policy.
`"matrix_free"` is an explicit scalable mode, while `"auto"` opts into that
mode only when the full-Jacobian workspace estimate exceeds the cap.  The
matrix-free path certifies algebraic injectivity, applies several necessary
numerical-rank screens, and checks the returned solution's normal residual,
tangency, horizontality, and descent identity.  It does not claim exact parity
with every near-rank-deficiency decision of the dense SVD, so its diagnostics
intentionally report `jacobian_rank=None` and no full-Jacobian singular values.
It still performs a compact rank SVD whose dimensions depend on `r0`, not on
the ambient sample/predictor/response dimensions.  Its worst-case workspace is
quartic and its dense SVD work is sixth-order in `r0`; a separate 256 MiB
safety guard prevents an accidental large-rank allocation.  Core/separation
thresholds use the
appendix's computable default rule.  Numerical positive-definiteness and rank
tolerances are exposed in the configuration because exact mathematical
comparisons need floating-point interpretations in code.  The configured
`atol`/`rtol` policy is threaded through source certification, screening ties,
regularity checks, local polar factors, and the quotient Jacobian; each object
uses its own local scale.  `pinv_rcond` is always passed explicitly to the
target-only pseudoinverse (default `1e-12`) so its rank decision is independent
of NumPy's implicit defaults.  Public numerical inputs must be real-valued;
complex arrays are rejected instead of having their imaginary parts discarded.

The package follows the appendix when it differs from the concise methodology
section:

- three folds rather than an unproved two-fold cross-fitting aggregation;
- the hybrid block-cell pilot rather than always truncating raw `Z_in`;
- full within-block source cores under independent tied-block basis rotations;
- the observable Wedin gate before restricted RRR and refinement;
- explicit target-only and zero safeguards;
- duplicate matrices retained under distinct deterministic labels.

## Tests

```bash
cd code/bi-smart
python -m pytest -q
```

The suite checks configuration and dimension validation, certified partition
construction, exact knapsack tie-breaking, strict rank failures, deterministic
tied target-only truncation, restricted RRR, quotient dimension/rank,
tangency, horizontality, gauge equivariance, safeguard retention, refinement
failure continuation, deterministic validation, and end-to-end runs.
