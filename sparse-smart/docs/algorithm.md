# Algorithm mapping and numerical policy

Specification: `notes/v1_sparse_two_stage.tex`, Section 3 and Appendix
"Implementation and numerical calibration". The unrestricted-estimation
and lower-bound sections describe statistical benchmarks, not this estimator.
The practical extensions below are implemented in software without changing
the manuscript or asserting that its fixed-iteration theorem covers them.

| Source label | Implementation |
| --- | --- |
| `eq:branches`, `eq:full-frames` | `source.prepare_source` |
| `eq:init-tuning`, `eq:init` | `calibration.resolve_calibration`, `initialization.reduced_lasso` |
| Anchor screening / RRQR appendix | `anchors.select_anchor` |
| `eq:chart`, `eq:outer` | `chart.AnchorChart` |
| `eq:objective`, `eq:majorizer`, `eq:update` | `chart.value_gradient`, `thresholding`, `solver.refine` |
| `eq:acceptance`, `alg:combined` | `solver.refine`, `estimator.SparseSMART.fit` |
| `eq:source-calibration`, `eq:balanced-budget`, `eq:cluster-inputs` | `calibration.resolve_calibration` |
| `eq:tuning`, `eq:basin`, `eq:primitive` | prescribed calibration diagnostics |

## Initialization and fixed coordinates

Ordinary multi-output Lasso solves the independent response-wise entrywise
penalties in one call with `1/(2n)` normalization, no intercept, and no design
rescaling. Exactly zero responses bypass the numerical solve; warning and
optimality diagnostics remain per-response where applicable. The returned coefficient
is a numerical minimizer; it is not asserted to be the minimum-Frobenius-norm
member of a nonunique minimizer set. Scikit-learn convergence warnings and an
independent KKT residual are recorded. KKT tolerance is the maximum of
`100*eps*scale` and `sqrt(lasso_tol)*scale`, where scale is the maximum of 1,
the penalty, and the initial absolute score. The solver also applies its
relative dual-gap stopping rule. Exact Lasso solutions remain an assumption
of the manuscript theorem.

SVD singular values within `tie_tol * largest_singular_value` form numerical
ties. Coordinate-order orthogonalization fixes their left bases and applies
paired right rotations/signs. Numerically null spaces are completed
independently. This can perturb reconstruction at the declared tie tolerance;
no claim of bitwise agreement across BLAS/LAPACK implementations is made.
The first left coordinate exceeding `tie_tol` fixes each pair's sign.
The same SVD tolerance applies to source preparation and initialization.

Source completion projects coordinate vectors in index order and uses two
orthogonalization passes. Residual norms below `tie_tol` are discarded.
Exact source inputs are checked against `orthogonality_tol` and kept as supplied.
Thin SVD null-space completion stops at its required column count; the result
is the identical prefix of full coordinate-order completion. Noisy working
frames still require full completion.

Row-energy ordering uses exact computed energy ties, broken by original index.
The screening eigenvalue threshold has `32*eps` absolute slack. The CPQR
and exchange numerical tie tolerance is `32*eps*max(1,largest_score)`; original
row indices order pivots and exchanges. Each exchange solves a small square
linear system. Reaching `qr_max_exchanges` reports failure. The final anchor
test and polar center use the original unthresholded factor. The screen never
truncates factor rows.

## Reconstruction and derivative

For each side H=z D^{-1}, W=sqrt(I-H'H), and the anchor factor is O W with
O=O_center cay(Omega). Positive definiteness is mandatory; an invalid square
root is not repaired by clipping or projection. The Sylvester equation
`W dW + dW W = -(H' dH + dH' H)` defines the derivative. This is well defined
for repeated positive square-root eigenvalues. The reverse derivative
propagates through both uses of D, through H and through the coefficient.

Thresholding zeros therefore remain zeros in the weighted nonanchor factors.
The gradient includes inactive coordinates, allowing supports to change.
No polar projection follows a thresholded affine update. Source frames,
anchors, reference rotations, penalties, and support limits remain fixed.
Repeated current/trial evaluations reuse a two-entry chart cache keyed by exact
state and geometry values. In-place input changes invalidate entries, public
factor outputs are independent copies, and serialization drops cached work.
Skew index arrays are shared and immutable. History anchor margins reuse the
square-root eigenvalues rather than decomposing the anchor factors again.

## Original chart solver: acceptance and finite precision

In the original `chart` solver, every iteration starts at the configured
inverse step size. Soft thresholding
followed by stable C-order hard thresholding minimizes the specified isotropic
quadratic support subproblem. The trial must pass the chart domain, displacement,
and `F(w) <= F(x) - L*||w-x||^2/4` checks. There is no permissive uphill
acceptance tolerance. An additive constant outside the exact right source span
is removed only during comparisons and restored in reported objectives.
Both solvers evaluate the objective difference through the stable residual
identity described below. This also prevents an irreducible response component
inside the working span from rounding away an increase or required decrease.

`max_backtracks` limits doublings; zero still allows the first trial.
Finite-precision overflow produces a numerical failure or rejected trial.
An overflowing penalty/L shrink threshold has the limiting all-zero output.
A trial at floating-point resolution with an unresolved independently computed
projected residual produces a numerical or active-constraint stall. This is a
numerical safeguard, not a proof of stationarity. With no stationarity stopping
requested, zero steps whose mapping is resolved can count toward T. No
convergence-to-global-optimum claim is made. The update direction, inverse-step
reset, and mathematical acceptance inequality remain the prescribed rules;
the shared stable difference calculation repairs their floating-point evaluation.

## Practical projection and validation extensions (0.2)

Practical `spectral_step="projected"` replaces rejection of an infeasible
singular-value proposal with its Euclidean projection onto the polytope
`d_lower <= d_j <= d_upper`, `d_j-d_{j+1} >= gap`. Subtracting the required
gap staircase reduces this to bounded decreasing isotonic regression. The
projection operates only on the singular-value block; it neither clips an
invalid chart square root nor relaxes anchor/rotation constraints. Thresholding
uses separate left/right penalties when supplied. Every accepted trial still
passes feasibility, trust, and sufficient-decrease tests.
`spectral_step="reject"` retains the manuscript's original trial rule and is
the automatic choice with prescribed calibration.

An independent spectral/support projected gradient mapping is used for optional stationarity
stopping. It is not inferred from the very small step produced by backtracking.
This original mapping does not project anchor or Cayley constraints, and a
nonzero residual at their boundary is not a full constrained KKT certificate.
An appreciable residual with a collapsed step produces a numerical or active
constraint stall. Iteration completion and local stationarity have separate
termination reasons. No global optimality or truth-basin membership is inferred.

The optional validation callback observes the initializer and accepted states
without altering the optimization trajectory. It keeps the earliest iterate
with minimal target validation prediction MSE. The tuning wrapper fits every
candidate using training target rows only and selects among successful fits;
failures remain in its history. Full complement entry caps are the practical
default for this wrapper, with optional support-pair candidates. The chosen
fit is retained on the training subset and is not automatically refitted on
the combined data. Candidate and iterate selection share validation data;
an independent test set or simulation evaluation is needed for performance
assessment. Neither routine accepts a true coefficient for tuning.

## Feasible initialization and anchor-aware refinement (0.3)

Practical `initialization_spectrum="auto"` projects an infeasible Lasso/SVD
spectrum onto the same bounded, gap-separated polytope before chart handoff.
The singular directions, original Lasso result, and source frames are retained.
The original/projected spectra and correction norm are recorded. Prescribed
auto mode and explicit `"reject"` retain the original initialization gate.
Projection ensures only spectral feasibility, not truth-basin membership.

For full support caps, the practical `anchor_projected` solver changes numerical
coordinates to `y=(omega_u,omega_v,d,H_u,H_v)`, with `Z_a=H_a D`. Since d is
positive, this is a bijection and preserves complement zeros. In these
coordinates the original penalty is
`sum_j d_j (lambda_u ||H_u[:,j]||_1 + lambda_v ||H_v[:,j]||_1)`.
The anchor condition is exactly `||H_a||_op <= sqrt(1-anchor_min^2)` because
the anchor factor is an orthogonal rotation times `sqrt(I-H_a.T@H_a)`.
The chart is fixed; anchors and unpenalized rows are never moved.

The smooth-gradient transformation is `grad_H_a=grad_Z_a D` and
`grad_d|H=grad_d|Z + colsum(grad_Z_u*H_u + grad_Z_v*H_v)` (separate column
sums when the sides have different row counts). H proposals solve weighted
L1 plus a spectral-ball constraint to a numerical tolerance. One shrinkage
pass followed by singular-value clipping is not used as a combined prox.
The d proposal includes the linear penalty column sums before ordered
spectral projection; Cayley proposals project onto their skew operator-norm
balls. Sufficient decrease is checked on the original objective, using the
H-coordinate step metric, and the original chart domain is checked separately.
Small inward floating-point margins avoid accepting an infeasible rounded
boundary; the chart's square-root checks remain strict.

This complete constrained mapping supplies the optional local stationarity
residual; it is computed independently of the backtracked accepted step.
Residuals have finite inner-solver accuracy and are not global optimality or
statistical certificates. `diagnostic_coordinates` distinguishes these norms
from the original Z-coordinate diagnostics. Full caps are required because
adding a hard cardinality cap makes the proximal subproblem nonconvex; auto
mode retains the chart solver for that case. Rank admissibility is unchanged.

Support reporting distinguishes effective numerical support from literal
nonzeros. In the approximate anchor solver, each Z block uses reporting
threshold `max(1e-10*scale,64*eps*scale)`, with
`scale=max(1,max(abs(Z)))`. `supports_` and history `support_u`/`support_v`
use that threshold; `raw_supports_` and `raw_support_u`/`raw_support_v`
retain literal counts. The thresholds are exposed in `support_tolerances_`
and the corresponding history fields. The original chart solver uses threshold
zero. Reporting never modifies a state, objective, or feasibility test;
`support_cap_reached` remains literal, with `effective_support_cap_reached`
available separately. These counts do not certify statistical support recovery.

## Budget checkpoints and numerical stability (0.4)

### Successful checkpoints remain eligible

`SparseSMARTTuner(iterations=T, iteration_budgets=None)` fits the grid once at
T, preserving the original single-budget behavior, including T=0. An explicit
schedule must consist of positive strictly increasing integers whose last
value is T. For example, `(500,2000)` runs each candidate once with budget 500
and independently again with budget 2000. The training and validation arrays
are split or validated once and reused unchanged. Source preparation and
training-data projections are cached within this call, and identical Lasso/anchor results
are cached per initialization configuration. Support thresholding and all
refinement state remain candidate-specific. Cached preparations are not reused
by a later `fit`, and public fitted arrays are copied independently. There is
no optimization-state resume or warm start between candidates or budgets.
Calling `fit` again clears all checkpoints from the previous call.

Budgets are traversed in increasing order, with the existing parameter-grid
order inside each budget. Every fit has a globally sequential `candidate_id`,
a `grid_candidate_id` shared across budgets, and its `iteration_budget` in
`selection_history_`. All outcomes and validation trajectories are retained.
Only the best successful fitted model per budget is kept in `checkpoints_`,
keyed by budget, to avoid storing all candidate model objects. Budgets with no
successful candidate have no model in this mapping.

The global winner uses the stable validation comparison below over all
successful checkpoints. A strictly negative pairwise loss difference is required
to replace it, so a computed zero prefers the earlier budget and then earlier
grid position. This gives the same winner as retaining every successful
candidate model. A failed longer fit
cannot invalidate the shorter fit that completed successfully. Its partial
coefficient is still excluded, even if that partial validation score is lower.
Completion at a declared budget is distinct from stationarity.

`selected_budget_` and `selected_candidate_id_` identify the winning fit;
`selected_iteration_` and `n_iter_` concern that fit, not the last fit attempted.
`budget_statuses` reports success and failure counts and the winning candidate
at each budget. `selected_checkpoint_continuations` reports the later outcomes
at the selected grid position, while
`selected_checkpoint_retained_after_failure` identifies retention despite a
later failure. These continuation records describe independent refits. There
is no implicit refit on training plus validation data, and coefficient truth
does not select a budget. Validation scores used repeatedly for this selection
are not independent test-error estimates.

### Stable validation selection

Each newly evaluated prediction `P` is compared directly with the incumbent
prediction `P_incumbent`, using

```text
loss_difference = mean((P-P_incumbent) * ((P-Y) + (P_incumbent-Y)))
```

Products and accumulation use extended precision where the platform supports
it. A negative difference replaces the incumbent; an exactly zero computed
difference retains the earlier iterate, candidate, or budget. This removes
unchanged prediction entries and avoids subtracting large complete losses.
Neither a rounded absolute MSE nor a rounded fixed-reference score breaks a
pairwise tie. The rule does not change training, the validation schedule, or
checkpoint eligibility.

One fixed validation prediction `P0` is taken from the first evaluated
initializer and shared across all candidates, iterates, and budgets in a
tuner `fit`. A standalone estimator uses its own initializer; a new `fit`
starts a new reference. For reporting, compute
`selection_score = mean(2*(P0-Y)*(P-P0) + (P-P0)**2)` with extended-precision
products and accumulation where the platform supports them. This evaluates
the MSE change from the reference without subtracting complete losses that
may share a large response-only term. It is not used to rank current fits.
Validation predictions use ambient factors in the same multiplication order
as public `predict` and the saved-factor summary checks:
`((X_validation @ left_factors) * singular_values) @ right_factors.T`.
Validation designs are not projected into source coordinates for caching.

`best_selection_score_`, history
`selection_score`, and checkpoint `best_selection_score` store the relative
value. `best_score_`/`best_validation_loss_` and history `loss` still report
absolute MSE. `validation_reference_prediction_` on the tuner or standalone
estimator and diagnostic `selection_reference` preserve the common reference
and its origin. Relative values from separate fits are not comparable unless
their references match.

The estimator and tuner expose `selection_rule_="pairwise-validation-loss-v1"`.
Saved histories include this `selection_rule` and a `selection_comparison`
containing `incumbent_iteration` and `loss_difference`; the first comparison
has null values. Candidate records use `incumbent_candidate_id` and include
both global `selection_comparison` and `budget_selection_comparison`. Replaying
these ordered decisions reconstructs the winners without ranking rounded
reporting scores. Budget artifacts additionally save factors at every evaluated
checkpoint, enabling independent validation of the differences themselves.
Historical records without the marker retain their recorded rule: relative
score with absolute-MSE tie-breaking when the relative field exists, otherwise
absolute MSE. Comparisons cannot mix these legacy rules with the current marker.

### Stable objective differences in both solvers

For a proposed fixed-chart update, let R be the current working prediction
residual and Delta be the prediction change. The smooth-loss difference is
computed as `(<R,Delta> + ||Delta||_F^2/2)/n`. Delta is assembled from telescoping
factor differences. The penalty difference is computed from the entrywise
changes in `abs(Z_u)` and `abs(Z_v)`. This avoids subtracting two complete
objective values that can round to the same number near stationarity.

The acceptance condition remains
`objective_change <= -L*||y_trial-y||^2/4` in H coordinates for the anchor solver,
or the same inequality in Z chart coordinates for the original solver, together
with the original feasibility and displacement checks. No uphill tolerance is
introduced. The recorded objective is still the original objective, including
any constant loss outside the exact-source right span; that constant cancels
from the difference. Identically rounded recorded objectives alone therefore
do not imply that the stable difference was zero.

Every iteration in both solvers starts its line search at the calibrated
inverse step size `initial_L`, then doubles L for rejected trials. The
feasibility, displacement, and sufficient-decrease checks remain unchanged.
The diagnostic reference remains `L_ref=clip(initial_L,1,1000)`, fixed throughout
the fit; backtracking cannot manufacture a small stationarity residual by
increasing L.
`line_search_start_inverse` records the initial trial inverse step at each
iteration. The estimator reports `line_search_strategy="reset_initial_inverse"`
for both solvers.

### Inner accuracy and the constrained stationarity diagnostic

The anchor solver records the mapping displacement
`m=L_ref*||y_prox-y||_2` and proximal uncertainty
`u=L_ref*sqrt(e_u^2+e_v^2)` separately. Each block allowance is
`e_a=sqrt(2*(G_a+c_a))+d_a`, where G is the numerical primal-dual gap,
c guards cancellation in its evaluation, and d is a linear arithmetic
roundoff allowance for both Dykstra and direct proximal formulas. Active
Dykstra solves also retain the cancellation allowance; the arithmetic floor
remains even when the ball-gap term is zero. Their sum
`m+u` is the reported constrained residual. The uncertainty accounts for
inexact proximal solutions; it is not statistical uncertainty.
The gaps and allowances are floating-point diagnostics, not rigorous interval
certificates accounting for every rounding error.

If the numerical residual interval straddles the stopping threshold,
`max(0,m-u) <= stationarity_tol < m+u`, the diagnostic tries tighter absolute
gap targets, with at most two refinements. A target
uncertainty eta corresponds to a per-block Dykstra target
`G_a+c_a <= eta^2/(4*L_ref^2)`. Thus tightening is based on the squared desired
mapping accuracy, not only a relative inner-iteration tolerance. The smallest
available upper residual `m+u` and its corresponding components are retained if floating-point
arithmetic prevents the stricter target from being reached. A nonzero
uncertainty allowance is never silently dropped to declare convergence.
Requesting a smaller tolerance does not remove the roundoff allowances.

Iteration and result diagnostics include `mapping_displacement`,
`proximal_uncertainty`, `mapping_refinements`, and
`mapping_precision_limited`. Estimator diagnostics expose these for the
validation-selected state and expose the corresponding `last_` fields for
the terminal state. `objective_change` and `relative_step_norm` describe the
selected step. The terminal `precision_limited` flag is true for numerical
stagnation or a precision-limited terminal mapping; optimization convergence
still requires stationarity termination. A precision-limited diagnostic does not certify
stationarity or automatically make a failed partial fit eligible. A collapsed
trial with an unresolved constrained residual produces numerical stagnation,
except for a narrow fixed-budget case in the anchor solver: with
`stationarity_tol=None`, both the fixed-reference diagnostic trial and the
line-search trial must use closed-form block proximal maps and return the
current feasible state exactly. This recognizes fixed points with a nonzero
smooth gradient balanced by an L1 penalty or active constraint. An unresolved
Dykstra iterate cannot supply this certificate. The accepted update is a no-op;
the mapping uncertainty remains in diagnostics, and these updates do not establish
stationarity. Any explicit stopping tolerance retains the full residual check;
an unresolved arithmetic floor can still cause numerical stagnation. Failed
fits remain ineligible, allowing the tuner to retain an earlier successful budget.
Neither this diagnostic nor budget-checkpoint selection establishes global
optimality or the manuscript's statistical assumptions.

## Continuous trajectories and budget assessment (v0.5)

`checkpoint_execution="continuous"` in the tuner performs one initialization
and one uninterrupted refinement call per hyperparameter grid point. The
internal H state and gradient remain in the solver across all checkpoints;
each iteration uses the ordinary reset to the calibrated inverse step size.
No H/Z round trip is used to restart an update.
The default independent-budget mode is unchanged.

The capture schedule is the union of iteration zero, multiples of
`checkpoint_interval` (250 by default in continuous mode), all requested
comparison budgets, and the maximum budget. Validation is evaluated at
those scheduled points, optional additive `validation_iterations`, and a
successful terminal iterate. The extra schedule defaults to empty, is strictly
increasing and unique, and accepts nonnegative integers; points beyond the
maximum budget are ignored. It never adds full checkpoints. A computed
pairwise loss difference of zero retains the earlier evaluated state.
Validation never modifies a gradient or an acceptance condition. The initializer participates in validation
selection once a positive prefix completes, but cannot by itself rescue a
trajectory that fails before its first positive checkpoint. A stationary
initializer is a successful terminal fit and does cover later caps.

Each estimator stores compact original-chart endpoint and selected states in
`checkpoints_`. The `checkpoint_model(t)` view reconstructs coefficients,
truncated histories, and selected/terminal diagnostics without fitting. Views
are independent of the ongoing or failed parent estimator's mutable arrays.
Nonfinite objective/gradient callback records are excluded from validation
selection and do not become successful checkpoints. Improving extra validation
states are kept in the read-only `best_validation_states_` mapping; losing
extra points keep scalar diagnostics only. Every early improvement is retained,
even if superseded before the next full checkpoint. These states do not certify
new finite prefixes: only the ordinary full checkpoints or successful stationary
termination can establish cap coverage. Prefix views truncate both retained
extra states and their validation histories. Capturing state is not a disk-resume
protocol.

Continuous tuning reconstructs canonical validation predictions from compact
checkpoint states and compares them pairwise, then constructs models only for
retained budget winners. The global winner reuses its budget's fitted view.
Checkpoint copies omit derived
dense coefficient/factor arrays before reconstructing them for the requested
prefix; mutable public outputs remain independent of the parent and other views.

For each requested cap, the continuous tuner evaluates the latest successful
checkpoint at or below that cap for each grid point. A certified stationary
stop may cover all later caps. `success` means a usable completed prefix;
`budget_reached` means the requested cap is covered. Later numerical failure
does not erase an earlier dense checkpoint, and retained fallbacks do not
establish cap coverage. Candidate records stay in budget-major order, with
a negative pairwise validation loss difference replacing the winner; a computed
zero retains earlier caps and then grid order. Per-trajectory timing is
reported once rather than charged again to every prefix.

The simulation study compares cumulative validation minima at 500, 1,000,
2,000, 4,000, and 8,000 updates. It saves compact ambient singular factors for
checkpoint endpoint and selected states. Training/validation fingerprints and
the saved factors allow independent reconstruction of prediction and coefficient
errors. Current artifacts carry `selection_rule="pairwise-validation-loss-v1"`;
the summary independently rescores the factors, checks comparison signs and
values, and replays the iterate and candidate incumbent chains. Saved
`validation_comparisons` also record direct differences between selected cap
predictions. Coefficient truth enters only this post-fit evaluation and never the
tuner or the material-gain rule.

Statistical stabilization and optimization convergence are separate outputs.
Material validation gain is configurable; the study's default is an absolute
gain exceeding `max(1e-4, 1e-3 * baseline_validation_MSE)`. This is a practical
comparison threshold, not a hypothesis test or a theorem. Current artifacts
compute gains directly between the selected predictions at two caps, using the
same stable pairwise identity. Absolute MSE supplies the baseline in the
threshold; relative reference scores remain reporting quantities. Historical
artifacts remain readable under their original relative-score/absolute-MSE or
absolute-MSE rule. Missing cells, unattained caps, and failed extensions cannot
count as evidence of a plateau.
The strict numerical residual retains its original tolerance, and a
validation-stable result can still be unconverged or precision-limited.

## Calibration

All source-error terms are set to zero before evaluating exact-source formulas.
Anchor selection costs are computed with log-gamma; no anchor pairs are
enumerated. Finite support sizes are capped before integer conversion.
The manuscript's bare K in its support formula is interpreted as K_alg.

Automatic support enlargement evaluates the upper design and stochastic
counts at fully saturated algorithmic supports. All those counts are
nondecreasing with support size, while the source approximation budget and
beta are independent of K. The resulting upper bound L_upper gives
`K = 1 + (4*L_upper/mu)^2`. This stronger numerical construction avoids a
circular dependency and does not rely on assuming the primitive perturbation
bound merely to compute K. Log K is retained if K itself is unrepresentable;
actual capped supports must remain finite. User-supplied K is also allowed.

The diagnostics distinguish clean complement sparsity, source-correction
budget, reference support, algorithmic support, and derivative-support unions.
Primitive inequalities and the universal-anchor numerical bound are diagnostics.
They do not certify Gaussian sampling, relevant gaps, stable favorable
anchors, target truth margins, or initializer basin entry. The fixed-entry
rho_I bound involving additional truth-class inputs g_T and eta_RR is not used
as a runtime check or stopping rule. Practical tuning is explicitly labeled.
