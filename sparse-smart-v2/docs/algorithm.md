# Algorithm and interpretation

The implementation follows the hard-sparse method in
[`v1_sparse_two_stage_meets_version1.tex`](../../../notes/v1_sparse_two_stage_meets_version1.tex).
Its operational choices are supplied through `Margins`, `PracticalCalibration`,
`free_directions`, and the iteration/stopping settings. It provides no automatic
prescribed statistical calibration.

## Fixed source coordinates and initialization

For an observed coefficient, compute one deterministic ordered SVD and retain
all of its directions. Complete its rectangular thin factors to orthogonal
matrices `F_L` and `F_R` in ambient dimensions `p` and `q`. Numerical ties and
null directions use the shared deterministic conventions. Exact source-frame
inputs preserve their supplied columns and append deterministic complements.
The full frames are used even with exact source information.

Let `Ubar = F_L[:, :r0]` and `Vbar = F_R[:, :r0]`, with `r0=source_rank`.
Initialization solves

```text
min_B ||Y Vbar - X Ubar B||_F² / (2 n) + lambda_I ||B||_1
```

and takes the rank-`r` coefficient SVD of its result. Positive `lambda_I` uses
the reduced Lasso; zero uses reduced minimum-norm least squares before the
same coefficient-SVD truncation. The initializer is calculated from training
observations. Validation observations contribute no terms to this objective.
For positive-penalty Lasso ties on a rank-deficient reduced design, a convex
quadratic problem selects the numerical minimum-Frobenius-norm coefficient on
the tied face. It preserves the fitted predictions and L1 norm, using the
equicorrelation signs and nonnegative magnitudes. The equality constraints and
coefficient scale are normalized before the numerical solve. The result must
pass prediction, L1, norm, and numerical norm-optimality checks; a failed solve
or check is an explicit initialization failure.
The implementation checks numerical solver success and the supplied spectrum
bounds. It does not replace an infeasible initializer spectrum by a repaired
one. Practical fitting does not certify the statistical assumptions required
by the note's initializer theorem.

## Anchors and additional free rows

The original sparse package's normalized energy-screening and strong
rank-revealing QR rule chooses `r` anchor rows from each initialized reduced
factor. The selected blocks must pass the declared anchor-margin check.
The anchors need not be the first `r` source directions, including when
`ru=rv=r`. Ties follow deterministic index conventions.

For each side, form its unpenalized row set `J` from the selected anchors and
the first `r_a-r` nonanchor indices in fixed source order. Increasing `r_a`
keeps the anchors and gives nested free sets for that anchor choice. Extra
free rows can extend beyond the initializer's first `r0` directions. They
remain available for refinement even if their initialized entries are zero.

The free-coordinate dimension is

```text
d0 = r * (ru + rv - r).
```

This includes the singular values, two anchor rotations, and nonanchor
coordinates lying in the expanded free sets. The maximum numbers of penalized
entries are `M_u=(p-ru)*r` and `M_v=(q-rv)*r`. Hard caps apply to entries,
not rows. The algorithm neither chooses global best anchors nor searches
arbitrary free subsets; it uses the specified constructive rules.

## Chart objective and update

Write `C = F_L P D Q.T F_R.T` with orthonormal columns in `P` and `Q` and
positive ordered diagonal `D`. The anchor chart uses free skew rotation
coordinates, the diagonal of `D`, and weighted nonanchor coordinates
`z_u = P_nonanchor D`, `z_v = Q_nonanchor D`. Outside the free rows, the
weighted factors in the penalty are exactly the selected entries of `z`.

```text
J(x) = ||Y - X C(x)||_F² / (2 n)
       + lambda_u ||(z_u)_outside Ju||_1
       + lambda_v ||(z_v)_outside Jv||_1

number of nonzero (z_u)_outside Ju <= ku
number of nonzero (z_v)_outside Jv <= kv
```

At initialization, hard truncation is applied only to the outside coordinates,
then chart feasibility is checked. On a refinement trial with inverse step
size `L`, first set `v = x - grad(f(x))/L`. Leave the free entries of `v`
unchanged and, on each outside block, apply

```text
w_new = hard_k(soft_{lambda/L}(v_outside)).
```

`hard_k` keeps the largest `k` magnitudes with a stable index tie rule. This
is the exact separable quadratic-model update with the cardinality constraints.
Reconstruct orthogonal factors through the chart; there is no post-thresholding
factor rotation that fills in the selected outside zeros.

The proposed point must pass the chart/spectral margins and trust-step bound,
and satisfy

```text
J(x_new) <= J(x) - L ||x_new-x||² / 4.
```

Every iteration starts at `L0 = calibration.step_size_inverse`. An unsuccessful
trial doubles `L` within the finite backtracking budget; the next iteration
resets to `L0` rather than inheriting the preceding accepted `L`.
The estimator reports failure if initialization, handoff, or all allowed
refinement trials fail. It does not modify the chosen margins or invoke an
alternative estimator to rescue the fit. Zero penalties leave hard caps
active; only explicitly choosing a full cap makes hard truncation an identity.

## Validation, stopping, and memory

The training objective and its gradients use only training observations.
The same supplied validation set may select both an iterate and a tuning
candidate. The estimator evaluates the initial point, scheduled validation
points, requested checkpoints, and a successful terminal point. It keeps
the lowest observed validation loss, using stable pairwise differences and
earlier iterations for exact ties. Absolute MSE is retained for reporting and
the validation-patience rule; equal rounded MSE values can hide a pairwise
detectable improvement.
With no validation set, the selected coefficient is the terminal coefficient.

`validation_patience=None` disables validation-based compute stopping. With
positive patience, `validation_min_iterations` prevents an early stop and
`validation_min_relative_improvement` determines whether an improvement resets
patience. Patience counts accepted iterations, not the number of validation
evaluations. Changing validation frequency can change the selected iterate
and stopping time. Completing an iteration budget or a validation stop does
not establish optimization convergence.

The numerical stationarity diagnostic uses a fixed reference inverse step
`L_ref = clip(L0, 1, 1000)`, independent of the backtracked trial value. It
evaluates the same masked soft/hard proposal and checks its chart and trust
feasibility. An exactly zero evaluated mapping stops refinement even when
`stationarity_tol=None`; a positive supplied tolerance also permits a small
mapping norm. Free-coordinate gradients and retained-coordinate algebraic
expressions avoid mistaking rounded state subtraction for a zero mapping.
A tiny or rounded-to-zero trial step by itself is numerical stagnation, not
stationarity. This diagnostic is a feasible hard-prox fixed-point condition;
it is neither a global optimality test nor a KKT certificate for the chart's
boundary constraints. A terminal stationarity flag also does not make an
earlier validation-selected state stationary.

The tuner deep-copies every candidate and prepares source frames once per
search. Candidates share `source_rank` and the training/validation split.
Within a search, validated source frames and reduced initializers are cached;
the initializer key includes target rank and initialization penalty, and all
entries use the same training observations. Candidate margins, anchors,
refinement, and validation choices are evaluated separately. This cache is
discarded after the search and is not retained on fitted models.
Successful candidates compete by finite validation MSE; a failed entire fit
is ineligible even if it has an earlier retained state. The tuner preserves
the winning estimator plus compact configuration/status/validation audits.
It does not retain all losing trajectories or refit the winner on validation
observations. The estimator stores full states for requested checkpoints and
its initial/terminal checkpoints, rather than for every validation evaluation.

For efficient iteration selection, use one maximum `iterations` budget per
configuration and schedule losses through `validation_iterations` or
`validation_interval`. Add `checkpoint_iterations` only when the full prefix
states are needed as well. Separate candidates with different iteration
budgets provide an explicit comparison of separately fitted trajectories;
they repeat refinement from initialization and are not required merely to
select an earlier evaluated iterate.

## Scope of the guarantees

This package's success checks are numerical and algorithmic. The companion
note's coefficient and fixed-design prediction guarantees additionally need
target gaps, anchors, source approximation, adequate initialization, restricted
design/curvature, and localization. Practical tuning does not verify these
unknown-truth conditions: `theorem_certified` is always false.

Expanded free blocks change both approximation and statistical costs. The
note's free dimension is `r*(ru+rv-r)` rather than `r²`, and uniform coefficient
identifiability on the free predictor block needs `n >= ru`. Sparse outside
supports permit restricted-design analysis when `n < p`; they do not remove
initializer or noisy-source basin requirements. These are mathematical
qualifications, not runtime sample-size certificates.

The implementation is a separate local estimator package. It adds neither
an automatic RRR branch nor an RRR validation candidate. Existing simulation
campaigns, cluster scripts, and production runs remain outside this package's
integration scope.
