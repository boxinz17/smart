# sparse-smart

Sparse two-stage spectral transfer regression from
`notes/v1_sparse_two_stage.tex`: entrywise reduced Lasso, data-selected anchors,
and sparse refinement in coordinates that preserve orthogonality and selected
weighted-factor zeros. The distribution is `sparse-smart`; the import is
`sparse_smart`. It is independent of the existing `smart` and `bi_smart` packages.
Version 0.5 adds continuous optimization trajectories and dense validation
checkpoints for studying iteration budgets. It retains the version 0.4
checkpoint eligibility, numerical acceptance, and stopping safeguards in the
practical anchor solver. It builds on feasible spectral initialization,
anchor-aware refinement, and separate left/right penalties. These extensions
retain explicit numerical safeguards and are not theorem claims.

## Install and run

The supported runtime is **Python 3.12**, **scikit-learn 1.9.0**, NumPy 2.5.3,
and SciPy 1.18.1. Python patch releases within 3.12 are allowed. Local setup,
CI, and Discovery use the same [dependency pins](../python-constraints.txt).

From the repository root, create the shared environment as described in the
[environment guide](../environment/README.md). Then, from this directory:

```bash
python -m pip install --no-build-isolation -c ../python-constraints.txt -e '.[test]'
python ../environment/check_runtime.py
python -m pytest
python examples/exact_and_noisy.py
```

The wheel metadata also pins the scientific runtime, including scikit-learn's
joblib and threadpoolctl dependencies. The `src/` layout keeps imports independent of the checkout's
working directory. The example runs exact, noisy, and unresolved-cluster modes.

## Minimal fit

```python
import numpy as np
from sparse_smart import ExactSource, Margins, PracticalCalibration, SparseSMART

rng = np.random.default_rng(7)
X = rng.normal(size=(100, 5))
C = np.zeros((5, 4))
C[0, 0] = 2.0
Y = X @ C + 0.01 * rng.normal(size=(100, 4))

model = SparseSMART(
    rank=1,
    source_rank=3,
    sparsity=(1, 1),
    margins=Margins(d_lower=0.1, d_upper=5.0, gap=0.1),
    calibration=PracticalCalibration(
        init_penalty=0.01,
        penalty=0.001,
        step_size_inverse=2.0,
        support_limits=(0, 0),
    ),
    iterations=10,
)
model.fit(X, Y, source=ExactSource(np.eye(5)[:, :3], np.eye(4)[:, :3]))
if not model.success_:
    raise RuntimeError(f"{model.status_}: {model.message_}")
Yhat = model.predict(X)
C_hat = model.coefficient_  # shape (5, 4); Yhat = X @ C_hat
```

The numerical values in examples are practical choices, not the theorem's
calibration. Inputs are finite real arrays: `X` is `(n,p)`, `Y` is `(n,q)`.
`SparseSMART` performs no intercept fitting, centering, standardization, or
implicit sample split. The separate `SparseSMARTTuner` below creates a split.
The known dimensions satisfy `1 <= rank <= source_rank <= min(p,q)`.
`sparsity=(s_u,s_v)` gives global entrywise alignment budgets, each between
`rank` and `source_rank * rank`. They are not row counts or sparsity of the
physical coefficient. Source information, ranks, and tuning are fixed during
a fit. The same target observations enter both stages.

## Source experiments

`ExactSource(U,V)` supplies specified orthonormal frames with shapes
`(p,source_rank)` and `(q,source_rank)`. Their column ordering and orientation
define the sparse model; the package preserves them.

```python
from sparse_smart import NoisySource

source = NoisySource(
    coefficient=observed_source,  # p by q observed source coefficient
    noise_std=tau0,              # positive per-entry source noise bound
    gap_lower=g0,                # positive declared relevant-source gap bound
    cluster_size=1,              # >1: bound on relevant cluster sizes
)
model.fit(X, Y, source=source)
```

Noisy mode uses the leading `source_rank` SVD directions for initialization
and **full p- and q-dimensional frames** for refinement. This permits target
corrections outside the estimated leading source spans. The source-accuracy
check `eta0 <= g0/8` is enforced by default in both calibration modes. It
checks declared bounds; it does not estimate or certify the unknown relevant
gaps. For empirical experiments outside this condition, explicitly set
`PracticalCalibration(..., enforce_source_accuracy=False)`. This preserves
the supplied source noise and gap and reports `source_accuracy_passed`,
`source_accuracy_enforced`, and `source_accuracy_bypassed` in
`model.calibration_.diagnostics`; bypass is true only when the condition
fails. Such runs do not have the source-accuracy guarantee, and their computed
source error/tail expressions are not certified bounds. All chart and
numerical feasibility checks still apply. Prescribed calibration always
enforces source accuracy.

In cluster mode `cluster_size` inflates the declared alignment/row budgets
before calibration; no cluster partition or unknown support is an input.
Theoretical cluster coverage also requires the universal anchor regime. Its
numerical bound is reported in calibration diagnostics. Practical support
limits remain the explicitly supplied values, even in cluster mode.

## Calibration and numerical margins

`Margins` sets `d_lower`, `d_upper`, `gap`, `anchor_min`, `trial_radius`, and
`qr_threshold`. Require `d_upper > 4*d_lower > 0`, `gap > 0`,
`0 < anchor_min < 1/16`, and `qr_threshold > 1`. The adjacent target gap is
checked only for rank greater than one. Initializer anchors require minimum
singular value at least `8*anchor_min`; refinement requires `anchor_min`.

Two calibration configurations use the refinement engine:

- `PracticalCalibration(init_penalty, penalty, step_size_inverse,
  support_limits, delta=0.05, enforce_source_accuracy=True)` uses explicit
  numerical choices. `init_penalty`
  and inverse step size are positive; refinement penalty may be zero. A scalar
  `penalty` applies to both factors; `(lambda_u, lambda_v)` sets them separately.
  Each support limit is between zero and `(N_a-r)*r`, where exact mode has
  `N_u=N_v=source_rank` and noisy mode has `N_u=p,N_v=q`.
- `PrescribedCalibration(sigma, delta=0.05, support_enlargement=None,
  design_lower=0.5)` evaluates the manuscript's formulas with positive target
  noise scale. The default automatic support enlargement uses a conservative
  saturated-support upper bound for the inverse step size, resolving its
  dependence on support size. Supplying an enlargement greater than one
  evaluates that value and reports the resulting sufficient inequalities.

Use `resolve_calibration(...)` to inspect the full prescription before fitting.
Diagnostics include source errors/tails, enlarged support counts, saturation,
geometry constants, and numerical sufficient inequalities. Huge constants
may make steps unrepresentable or saturate the supports. The package does not
silently replace prescribed constants with practical ones. All fits report
`diagnostics_["theorem_certified"] == False`: finite-precision output and
observable checks do not establish the theorem's signal/design/basin assumptions.

`SparseSMART(spectral_step="auto")` uses Euclidean projection of proposed
singular values onto their bounded, gap-separated domain in practical mode.
This allows updates along an active spectral boundary. All anchor, rotation,
trust-radius, and objective-decrease checks remain active. In prescribed mode,
`auto` uses the original rejection rule. Set `spectral_step="reject"` explicitly
to reproduce that rule in a practical fit.

`initialization_spectrum="auto"` projects an infeasible reduced-Lasso spectrum
onto the declared bounds and gaps in practical mode. Singular directions are
retained, and `initialization_` still contains the original Lasso result.
Diagnostics record both spectra and the correction norm. Prescribed mode
retains rejection; `initialization_spectrum="reject"` explicitly restores it
in practical mode. Passing this feasibility check does not certify a good
statistical initializer, particularly when the source span omits signal.

`refinement_solver="auto"` uses `"anchor_projected"` for practical fits with
full complement support caps and projected spectral steps. It optimizes in
`H=Z D^{-1}` coordinates, imposing the anchor constraint through a spectral
norm ball. It preserves the original objective, fixed anchors, and L1 weights
on Z. Its proximal mapping includes the anchor, rotation, and singular-value
constraints. Gradients and step norms for this solver use the H-coordinate
metric, identified in `diagnostics_["diagnostic_coordinates"]`.

The practical anchor solver compares objective differences directly to reduce
cancellation near a stationary point. Each iteration starts its line search
at the calibrated inverse step size `initial_L`; rejected trials double it.
Feasibility, trial-radius, and sufficient-decrease checks are unchanged.
The stationarity mapping keeps a fixed reference inverse step size,
independent of backtracking. Its reported residual
separates mapping displacement from an allowance for the inner proximal solve;
when that allowance prevents a possible stationarity decision, the diagnostic
solve is tightened. Roundoff allowances remain active even when a stricter
inner tolerance is requested. Unresolved numerical stalls remain failures. Both
solvers now use the stable objective-difference calculation for acceptance;
the chart solver retains its prescribed update direction and resets its trial
inverse step size each iteration. See [the numerical policy](docs/algorithm.md)
for details.

Prescribed fits, smaller hard support caps, and `spectral_step="reject"` use
the original `"chart"` solver under `auto`. An explicit `"anchor_projected"`
request rejects incompatible caps or spectral rejection. To reproduce the
0.2 practical solver, set `initialization_spectrum="reject"` and
`refinement_solver="chart"`. The tuner accepts the same two options and stores
each candidate's diagnostics, including failures.

## Target-data tuning and best-iterate selection

```python
from sparse_smart import SparseSMARTTuner

tuned = SparseSMARTTuner(
    rank=5, source_rank=10, sparsity=(5, 5),
    margins=Margins(.05, 12., .01, anchor_min=.005),
    init_penalties=(.01, .03),
    penalties_u=(.0025, .01, .04),
    penalties_v=(.0025, .01, .04),
    support_limits=None,
    iterations=2000, iteration_budgets=(500, 2000),
    validation_fraction=.2, random_state=0,
)
tuned.fit(X, Y, source=source)
if not tuned.success_:
    raise RuntimeError(tuned.status_)
print(tuned.selected_budget_, tuned.best_params_, tuned.best_score_)
prediction = tuned.predict(X_new)
```

The tuner splits target rows reproducibly into training and validation sets,
or accepts `fit(..., validation_data=(X_validation,Y_validation))`. Both
initialization and refinement use only training rows. Each candidate is fitted
independently; its initializer and every accepted iterate are evaluated on the
validation rows. Selection targets mean squared prediction error using the
stable comparison described below. No coefficient
truth is accepted as a tuning input. `selection_history_` records every
candidate, including failures; failed partial fits cannot win. If all fail,
the tuner reports `no_successful_candidate` and exposes no winning coefficient.

`iteration_budgets=None` is the default: each grid point is fitted once at
`iterations`. An explicit schedule must contain positive, strictly increasing
integers and end at `iterations`. Each budget runs the entire parameter grid
independently, using the same training and validation rows; it does not resume
the shorter fit or warm-start from another candidate. Thus `(500, 2000)` costs
both sets of fits. There is no implicit refit on the combined training and
validation rows.

Within one tuner call, source preparation and training-data projections are shared,
and identical Lasso initializers and anchors are reused across candidates and
budgets. Support thresholding and refinement remain candidate-specific. A new
`fit` starts a fresh preparation cache; fitted public arrays remain independent.

`checkpoints_` maps each budget with successful candidates to that budget's
best fitted estimator. Selection directly compares each checkpoint's validation
predictions with the incumbent using a stable pairwise loss difference. A
negative difference replaces the incumbent; a computed zero retains the earlier
budget, then the original grid order. A later failure cannot erase a completed
earlier checkpoint, and
the failed run's partial coefficient is still ineligible. A completed
iteration budget need not be a converged fit.

`iteration_budgets_`, `selected_budget_`, and `selected_candidate_id_` identify
the resolved schedule and winner. Each `selection_history_` record includes
`iteration_budget`, a globally sequential `candidate_id`, and
`grid_candidate_id`, which identifies the same grid position across budgets.
The winning model's iteration count, selected iterate, and histories remain
its own, even if a later fit fails. Tuner diagnostics separately record
`budget_statuses`, `selected_checkpoint_status`,
`selected_checkpoint_continuations`, and
`selected_checkpoint_retained_after_failure`. Here continuations are
independent longer fits of the selected grid position, not resumed execution.
A new call to `fit` clears the previous call's results; checkpoint retention
applies within one scheduled fit.

By default `support_limits=None` permits the full weighted nonanchor blocks,
leaving the penalties to determine effective sparsity. These are **entry
counts**, `(N_u-r)*r` and `(N_v-r)*r`, rather than numbers of directions.
For noisy Model I they are 475 and 225. An optional sequence of support pairs
can also be searched. Inspect the winning supports before enlarging that grid.

`estimator_` is the winning fitted `SparseSMART`, and `best_params_`,
`best_score_`, `train_indices_`, and `validation_indices_` record selection.
`best_score_` reports absolute validation MSE; `best_selection_score_` records
a relative loss for reporting. For prediction `P`, validation response `Y`, and a
fixed reference prediction `P0`, this value is
`mean(2 * (P0 - Y) * (P - P0) + (P - P0)**2)`. It represents the MSE change
from the reference without subtracting two large absolute losses. The first
evaluated initializer supplies `P0`, shared across every candidate and budget
within one tuner `fit` and reset on the next call. A standalone estimator uses
its own initializer. Scores from different fits need not share a reference.
Selection instead evaluates
`mean((P-P_incumbent) * ((P-Y) + (P_incumbent-Y)))`, with extended-precision
products and accumulation where available. A negative difference wins; an
exactly zero computed difference retains the incumbent. Neither absolute MSE
nor the fixed-reference score breaks a pairwise tie. The reference is
available as `validation_reference_prediction_` on the fitted tuner or
standalone estimator; `diagnostics_["selection_reference"]` describes its origin.

`selection_rule_="pairwise-validation-loss-v1"` identifies this rule. Validation
history rows record `selection_rule` and `selection_comparison`, including the
incumbent iteration and loss difference. Candidate rows record the corresponding
global and per-budget comparisons, so the ordered decisions can be replayed.

The returned fit uses the training subset; there is no implicit refit using
validation outcomes as training data. Its reported validation MSE describes the
selected fit and is not an unbiased test error. Both the tuner and the estimator retain
strict source-accuracy checks by default; use the explicit empirical option
`enforce_source_accuracy=False` for experiments outside that condition.

For a single candidate, `model.fit(X_train,Y_train,source=source,
validation_data=(X_validation,Y_validation))` also selects the best iterate.
`selected_iteration_` and `best_validation_loss_` identify it;
`best_selection_score_` and each validation-history row's `selection_score`
retain the relative reporting value, while the row's `loss` remains absolute MSE.
`last_state_` and `last_coefficient_` retain the optimizer's final accepted
state separately. Validation never changes gradients or training acceptance.
History, `optimization_converged_`, and termination reasons describe the final
optimization state. `converged_` is true only when that stationary terminal
state is also selected. Gradient diagnostics describe the selected coefficient;
their `last_` counterparts describe the terminal state.

## Results and failures

`fit` returns the estimator. Inspect `success_`, `status_`, `message_`, and
`n_iter_` before using it. With `stationarity_tol=None`, `iterations=T` requests
T accepted updates; `T=0` returns the support-enforced initializer. Supplying a
positive `stationarity_tol` also permits early termination on a projected
gradient-mapping residual. No coefficient-truth stopping rule is used.
`termination_reason_` distinguishes `max_iterations`, `stationarity`, boundary
stalls, numerical stagnation, and exhausted backtracking. `converged_` requires
selection of the stationary terminal state; completing T updates is not convergence.
The mapping criterion is local and does not establish a global optimum.
In fixed-budget anchor refinement, an exact proximal fixed point may produce
repeated accepted no-op updates, including when penalties or active constraints
balance a nonzero smooth gradient. Both the fixed-reference diagnostic trial
and the line-search trial must use closed-form block proximal maps and return
the current feasible state exactly; an unresolved iterative proximal solve
does not qualify. This applies only when `stationarity_tol=None`; the
conservative mapping uncertainty remains reported and no convergence is
claimed. An explicit tolerance, however small, still requires the full
constrained residual to pass and can produce numerical stagnation.

Successful fits expose `coefficient_`, `singular_values_`, `left_factors_`,
`right_factors_`, working-coordinate `factors_`, `anchors_`, `supports_`,
`source_`, `calibration_`, `initialization_`, `chart_`, `initial_state_`,
`state_`, `result_`, `history_`, and `diagnostics_`. Supports are C-order
flattened indices into each weighted nonanchor block; use the chart's
`complement_u`/`complement_v` to map them back to rows.

For `anchor_projected`, `supports_` and history `support_u`/`support_v` report
effective numerical support: entries exceeding `1e-10 * max(1, max(abs(Z)))`
in each side's weighted block. This avoids counting tiny residual entries from
the approximate constrained proximal solve as signal. `raw_supports_`, history
`raw_support_u`/`raw_support_v`, and `raw_fitted_support_counts` in diagnostics
retain literal nonzeros. `support_tolerances_` and history
`support_tolerance_u`/`support_tolerance_v` record the thresholds. The `chart`
solver uses zero reporting tolerance. Coefficients are never thresholded for
reporting, and hard-cap checks and `support_cap_reached` use literal support.
Effective support is a numerical convention, not a statistical support guarantee.

The history contains the initial state and every accepted update's objective,
smooth loss, penalty, support counts, anchor margins, inverse step size,
step norm, and rejected-trial reasons. Reported loss includes the constant
response component outside the thin exact right source span.

For the practical anchor solver, inspect `mapping_displacement`,
`proximal_uncertainty`, `mapping_refinements`, and
`mapping_precision_limited` in `diagnostics_`; `last_` versions describe the
terminal state. The selected state also reports `objective_change` and
`relative_step_norm`. History records `line_search_start_inverse`, and
`line_search_strategy="reset_initial_inverse"` identifies the per-iteration
reset in both solvers. These are floating-point diagnostics, not
rigorous interval certificates. The terminal `precision_limited` flag marks
numerical stagnation or a precision-limited mapping; it does not imply
`optimization_converged` or make a failed partial fit eligible.

Algorithmic statuses include `source_accuracy_failed`, `initialization_failed`,
`lasso_not_converged`, `initialization_spectrum_failed`, `anchor_selection_failed`,
`handoff_failed`, `invalid_initial_state`, `line_search_failed`,
`numerical_failure`, `numerical_stagnation`, and active-constraint stalls.
Successful termination can be `completed` or `converged`. Malformed input or
unrepresentable calibration raises `ValueError`.

Refinement failures preserve the last accepted coefficient in `last_coefficient_`.
With validation, `coefficient_` instead retains the best validation iterate.
`predict` rejects
unsuccessful fits by default; `predict(X, allow_partial=True)` explicitly uses
the retained coefficient if one exists. Earlier failures have no coefficient. Set
`raise_on_failure=True` to raise `FitFailure`, whose `status` is inspectable.
Refitting clears earlier results, including when the new fit fails.

## Low-level research interface

`prepare_source`, `reduced_lasso`, `AnchorChart`, `resolve_calibration`, and
`refine` are public. `refine` accepts a fixed chart, flat initial coordinates,
working design `X @ F_L`, and working response `Y @ F_R`, plus resolved
calibration and margins. It checks feasibility and support limits, not
proximity to an unknown truth. Optional `loss_offset` accounts for the
constant loss outside the working right span.

Coordinate order is left skew parameters, right skew parameters, singular
values, left weighted complement, right weighted complement. Skew coordinates
use `(E_ij-E_ji)/sqrt(2)`, with `i<j`; complements use C order. Use
`chart.pack`, `unpack`, `initial_state`, and `reconstruct` instead of manually
laying out arrays.

## Numerical scope and development

### Continuous trajectories and iteration-budget studies

Use the opt-in continuous mode to compare maximum budgets without restarting
each parameter setting:

```python
tuner = SparseSMARTTuner(
    rank=5, source_rank=7, sparsity=(5, 5), margins=margins,
    init_penalties=(.03,), penalties_u=(.0025, .01, .04),
    penalties_v=(.0025, .01, .04),
    iterations=8000, iteration_budgets=(500, 1000, 2000, 4000, 8000),
    checkpoint_execution="continuous", checkpoint_interval=250,
    refinement_solver="anchor_projected", enforce_source_accuracy=False,
)
tuner.fit(X_train, Y_train, source=source,
          validation_data=(X_validation, Y_validation))
```

Each grid point has one initializer and one solver trajectory. Validation is
evaluated at iteration zero, every 250 updates, all comparison budgets, and
an earlier stationary endpoint if needed. The optimization state continues
between checkpoints, with the usual inverse-step reset at each iteration.
The validation sample never enters the updates. Both earlier successful
prefixes and their best validation states remain available after a later
numerical failure.
The original independent-budget mode remains the default.

To resolve minima before the first regular checkpoint, continuous mode also
accepts `validation_iterations=(10, 25, 50, 100, 150, 200)`.
These points add validation evaluations without changing the full checkpoint
schedule, gradient updates, or stationarity stopping rule. The schedule must
contain strictly increasing unique nonnegative integers; points above the
maximum budget are ignored. The default is empty, and independent mode rejects
a nonempty schedule because it already validates every accepted iterate.
An early validation winner propagates into subsequent full checkpoints. Extra
points do not establish budget coverage or rescue a trajectory that fails before
its first positive full checkpoint. `best_validation_states_` stores read-only
compact chart states only for improving extra points, including improvements
that are superseded before the next full checkpoint; scalar evaluations remain
in `validation_history_` and optimization records in `history_`.

`trajectory_models_` and `trajectory_history_` contain one entry per grid
point. Each full estimator exposes lightweight `checkpoints_` and a
`checkpoint_model(iteration)` method that reconstructs an independent fitted
view without optimization. That view's `coefficient_` is the selected state;
`last_coefficient_` is the checkpoint endpoint. At the lower level, pass
`checkpoint_iterations`, `validation_interval`, and `validation_iterations`
directly to `SparseSMART`. Estimator diagnostics record the configured extra
schedule and the actual validation iterations; tuner diagnostics declare the
planned union, while individual trajectory diagnostics show evaluations reached.
Checkpoint capture is in memory; it is not process-restart support.
The tuner reconstructs validation predictions from compact checkpoint states
and makes pairwise comparisons before constructing independent fitted models
for retained budget winners. This avoids building
dense coefficient matrices for every grid-by-budget checkpoint.

Continuous tuner `selection_history_` records remain in budget-major order.
`success` describes a retained, completed prefix; `budget_reached` separately
records whether the requested cap was reached or stationarity was established
earlier. `trajectory_checkpoint_iteration` identifies the actual prefix. For
example, after failure at update 400, a completed 250-update prefix may remain
eligible at cap 500, with `budget_reached=False`. A failed partial iterate is
never substituted for that completed checkpoint. An initializer alone is not
a fallback after failure before the first positive checkpoint.

The separate simulation budget-study runner records validation MSE, diagnostic
coefficient error, objective, movement, and stationarity at every checkpoint.
Current artifacts also preserve the pairwise decisions and compact factors.
The summary reconstructs predictions on regenerated validation data, verifies
reported scores, and replays iterate/candidate/cap comparisons. Validation gains
between caps use direct pairwise loss differences, separately from optimization
diagnostics. Legacy artifacts retain their original selection rule when audited.
Missing or failed extensions cannot establish
a plateau. See [the budget-study guide](../simulation/SPARSE_SMART_V05_BUDGET_STUDY.md).

Continuous mode stores all trajectory histories and compact checkpoint states;
it therefore uses more memory than a single-budget fit. Validation-selected
scores are tuning scores, not independent estimates of prediction error.

### Numerical implementation

See [the algorithm mapping and numerical policy](docs/algorithm.md). The
implementation differentiates the matrix square root with a Sylvester solve,
computes gradients for all potential complement coordinates, and reconstructs
orthogonal factors after thresholding. It does not require a full regression
Gram inverse or an enumeration of alignment supports.

Tests cover analytic derivatives, threshold-subproblem optima on tiny examples,
RRQR exchanges, Lasso normalization and nonconvergence, source/SVD conventions,
calibration formulas, all major solver checks, exact/noisy/cluster fits,
support changes, and initialization when `n < source_rank`.
Checkpoint tests cover eligibility after later failures, validation ties,
identical data across budgets, and reproduction of shorter validation prefixes.

Dense arrays are used. The initializer stores an `r0` by `r0`
coefficient. Noisy full frames cost `O(p^2+q^2)` storage and their deterministic
completion can be expensive. Fitting uses low-rank prediction products and
does not form a `p*q` Jacobian; selected and terminal physical coefficients are
materialized on return. Ordinary multi-output Lasso shares design processing
across responses. Thin SVD null-space completion stops at the required number
of columns, preserving the canonical basis prefix. A bounded two-state chart
cache reuses factor reconstruction and square-root decompositions during
backtracking, while public reconstructed arrays remain independent.
The implementation is numerical research software, not a
finite-precision statistical guarantee or a global optimizer.
