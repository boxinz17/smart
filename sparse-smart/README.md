# sparse-smart

Sparse two-stage spectral transfer regression from
`notes/v1_sparse_two_stage.tex`: entrywise reduced Lasso, data-selected anchors,
and sparse refinement in coordinates that preserve orthogonality and selected
weighted-factor zeros. The distribution is `sparse-smart`; the import is
`sparse_smart`. It is independent of the existing `smart` and `bi_smart` packages.
Version 0.3 adds feasible spectral initialization and anchor-aware refinement
to the practical projection, separate left/right penalties, and validation
selection introduced in 0.2. These
extensions retain explicit numerical safeguards and are not theorem claims.

## Install and run

From this directory, with Python 3.10 or newer:

```bash
python -m pip install -e '.[test]'
python -m pytest
python examples/exact_and_noisy.py
```

Runtime dependencies are NumPy, SciPy, and scikit-learn. A normal wheel install
is also supported. The `src/` layout keeps imports independent of the checkout's
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
    iterations=500, validation_fraction=.2, random_state=0,
)
tuned.fit(X, Y, source=source)
if not tuned.success_:
    raise RuntimeError(tuned.status_)
print(tuned.best_params_, tuned.best_score_)
prediction = tuned.predict(X_new)
```

The tuner splits target rows reproducibly into training and validation sets,
or accepts `fit(..., validation_data=(X_validation,Y_validation))`. Both
initialization and refinement use only training rows. Each candidate is fitted
independently; its initializer and every accepted iterate are evaluated on the
validation rows. The lowest validation mean squared prediction error selects
the candidate and its iterate, with earliest ties retained. No coefficient
truth is accepted as a tuning input. `selection_history_` records every
candidate, including failures; failed partial fits cannot win. If all fail,
the tuner reports `no_successful_candidate` and exposes no winning coefficient.

By default `support_limits=None` permits the full weighted nonanchor blocks,
leaving the penalties to determine effective sparsity. These are **entry
counts**, `(N_u-r)*r` and `(N_v-r)*r`, rather than numbers of directions.
For noisy Model I they are 475 and 225. An optional sequence of support pairs
can also be searched. Inspect the winning supports before enlarging that grid.

`estimator_` is the winning fitted `SparseSMART`, and `best_params_`,
`best_score_`, `train_indices_`, and `validation_indices_` record selection.
The returned fit uses the training subset; there is no implicit refit using
validation outcomes as training data. Its validation score is a selection
score, not an unbiased test error. Both the tuner and the estimator retain
strict source-accuracy checks by default; use the explicit empirical option
`enforce_source_accuracy=False` for experiments outside that condition.

For a single candidate, `model.fit(X_train,Y_train,source=source,
validation_data=(X_validation,Y_validation))` also selects the best iterate.
`selected_iteration_` and `best_validation_loss_` identify it;
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

Successful fits expose `coefficient_`, `singular_values_`, `left_factors_`,
`right_factors_`, working-coordinate `factors_`, `anchors_`, `supports_`,
`source_`, `calibration_`, `initialization_`, `chart_`, `initial_state_`,
`state_`, `result_`, `history_`, and `diagnostics_`. Supports are C-order
flattened indices into each weighted nonanchor block; use the chart's
`complement_u`/`complement_v` to map them back to rows.

The history contains the initial state and every accepted update's objective,
smooth loss, penalty, support counts, anchor margins, inverse step size,
step norm, and rejected-trial reasons. Reported loss includes the constant
response component outside the thin exact right source span.

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

See [the algorithm mapping and numerical policy](docs/algorithm.md). The
implementation differentiates the matrix square root with a Sylvester solve,
computes gradients for all potential complement coordinates, and reconstructs
orthogonal factors after thresholding. It does not require a full regression
Gram inverse or an enumeration of alignment supports.

Tests cover analytic derivatives, threshold-subproblem optima on tiny examples,
RRQR exchanges, Lasso normalization and nonconvergence, source/SVD conventions,
calibration formulas, all major solver checks, exact/noisy/cluster fits,
support changes, and initialization when `n < source_rank`.

Dense arrays are used in version 0.1. The initializer stores an `r0` by `r0`
coefficient. Noisy full frames cost `O(p^2+q^2)` storage and their deterministic
completion can be expensive. Fitting uses low-rank prediction products and
does not form a `p*q` Jacobian; the physical coefficient is formed once on
return. The implementation is numerical research software, not a
finite-precision statistical guarantee or a global optimizer.
