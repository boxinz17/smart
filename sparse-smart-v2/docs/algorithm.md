# Algorithm and interpretation

The implementation follows the hard-sparse method in
[`v1_sparse_two_stage_meets_version1.tex`](../../../notes/v1_sparse_two_stage_meets_version1.tex).
Its operational choices are supplied through `Margins`, `PracticalCalibration`,
`free_directions`, and the iteration/stopping settings. It provides no automatic
prescribed statistical calibration. The explicit RRR endpoint below is a
practical extension of that note; the remaining sections describe the chart path.

## Direct RRR for unpenalized configurations

Before source preparation, initialization, or chart entry, the default
`rrr_shortcut=True` checks each side's outside capacity `M_a=(dimension-r_a)*r`.
It dispatches to target-only RRR exactly when `k_a=M_a` on both sides and
`lambda_a=0` or `M_a=0` on each side. Thus all rows free on both sides triggers
RRR even for positive nominal penalties; a remaining positive effective penalty
or a restrictive cap prevents the shortcut. Invalid counts/caps still raise.

For the numerical thin SVD `X=U_x diag(s_x) V_x.T`, form `B=U_x.T @ Y` and
retain a best rank-at-most-r approximation `B_r`. The output is

```text
C_RRR = (V_x / s_x) @ B_r
optimal_loss = (||Y - U_x B||_F^2 + sum_{j>r} sigma_j(B)^2) / (2n).
```

This gives the minimum-norm lift of the chosen optimal fitted response.
No Gram inverse, source information, spectral floor, positive fitted gap,
anchor bound, or chart projection enters this calculation. Numerical design
rank uses `eps*max(n,p)*sigma_max(X)`. The response numerical rank and tied
cutoff convention are separately recorded. Tied response spaces use a
deterministic coordinate convention; minimum coefficient norm is asserted for
that selected fitted response, not across different tied optimal responses.
The attained loss must pass the recorded numerical optimum certificate before
the fit is labeled successful. Effective rank can be below r, including zero.

The sole output is this training-only coefficient. Validation scores it for
candidate selection but performs no iterate selection or early stopping.
Metadata labels `fit_method="target_rrr"` and
`termination_reason="target_rrr_closed_form"`; `n_iter=selected_iteration=0`
means zero chart updates, not an initializer. Physical coefficient/factor
arrays and a dedicated checkpoint replace chart states. Repeated tuning
candidates share one direct solve per training split and requested rank.
Source preparation is lazy, so all-direct searches need no source decomposition.

`rrr_shortcut=False` retains legacy chart behavior and is used for initializer
preflight. New Discovery plans freeze the enabled flag, allow RRR candidates
past failed source-initializer preflight, and save physical artifacts under
schema 3. Audits independently check rank, optimal loss, minimum-norm row-space
membership and validation metrics. Old plans missing the flag retain legacy
behavior. Failed penalized fits never trigger RRR implicitly.

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

Optional adaptive anchors keep these exact physical free row sets fixed.
If refinement stops at an anchor or Cayley boundary, two deterministic searches
start from the current anchor and a restricted pivoted QR anchor. Each uses at
most 16 improving one-row exchanges within the same free rows. The best result
is considered. A near-boundary block may be replaced when its minimum singular
value strictly exceeds the current margin plus `256 * eps * max(1, margin)`,
while satisfying the declared anchor floor. If neither primary start gives a
numerically meaningful gain, a fallback tries at most `r*(|J|-r)` one-row
neighbors of the current anchor as starting points, with at most 16 improving
swaps per start, and stops at the first acceptable result. Successful primary
paths are preserved. There is no additional absolute replacement floor. Absence of
such a block leaves that anchor boundary unresolved.
This is a numerical continuation rule, distinct from the initializer's
stronger anchor certificate. It does not search all subsets or promise to
find every possible feasible anchor.

A side whose Cayley coordinate has operator norm at least 0.45 may also be
recentered at its current polar rotation. This does not enlarge the Cayley
domain: the same factors receive zero rotation coordinates in the new chart,
and all subsequent trials still obey the operator-norm limit 0.5. If its
anchor rows do not need replacement, they remain identical, as do its weighted
nonanchor coordinates. This can operate even when the free set has exactly
`r` rows. Continuation is attempted only after an eligible failed refinement;
these thresholds do not trigger extra evaluations or switches on every update.

On a switch, the current factors and singular values are represented in a
new chart with new polar centers on the switched sides. Physical coordinates
outside the fixed free sets retain their weighted entries, including exact
zeros. Rebuilding the masks from the same free rows preserves the objective,
L1 penalties, and hard support counts. Only subsequent coordinate updates
change. A switch consumes no accepted iteration and resets neither validation
patience nor the incumbent selected prediction. At most
`max_anchor_switches=16` switches are allowed when `adaptive_anchors=True`;
fixed anchors remain the package default. Each switch records its global
iteration, action on each side, old/new anchors and margins, rotation norms,
and hashes of the old/new centers.

Checkpoint states and their validation-selected earlier states each retain
their own chart. The final `chart_` belongs to the terminal state and
`selected_chart_` belongs to the selected state. Reinterpreting an old state
under the newest chart would change its fitted coefficient and is prohibited.
Adaptive anchors extend the note's fixed-chart algorithm; its fixed-chart
local guarantees are not automatically a theorem for this continuation.

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
size `L`, first set `v = x - grad(f(x))/L`. Leave the rotation and unpenalized
nonanchor entries of `v` unchanged and, on each outside block, apply

```text
w_new = hard_k(soft_{lambda/L}(v_outside)).
```

`hard_k` keeps the largest `k` magnitudes with a stable index tie rule.
Project the singular-value block `v_d` onto the existing spectral constraints:

```text
d_new = argmin_d ||d - v_d||²
        subject to d_lower <= d[r-1], d[0] <= d_upper,
                   d[j] - d[j+1] >= gap.
```

This small convex projection is solved by shifting out the required gaps,
applying bounded nonincreasing isotonic regression, and restoring the gaps.
It preserves column identities; it does not independently sort singular values
while leaving their factors in place. Float boundary corrections are limited
to rounding scale so the unchanged strict spectral checks accept the result.

Together these are the exact separable quadratic-model update with spectral
and cardinality constraints, up to floating-point arithmetic.
Reconstruct orthogonal factors through the chart; there is no post-thresholding
factor rotation that fills in the selected outside zeros.

The spectral proximal block is an implementation extension of the note's
free singular-value gradient step followed by a domain rejection test. It
retains the same objective and declared feasible spectra, but changes the
optimization trajectory when a spectral constraint is active. This avoids
the observed repeated gap rejections and vanishing steps. It is not an
initializer repair or a projection onto the full coupled chart domain;
anchor and Cayley inequalities still use the checks below. No extended
end-to-end theoretical guarantee is claimed for this change.

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
evaluates the same masked soft/hard and constrained spectral proposal and checks its chart and trust
feasibility. An exactly zero evaluated mapping stops refinement even when
`stationarity_tol=None`; a positive supplied tolerance also permits a small
mapping norm. Free-coordinate gradients and retained-coordinate algebraic
expressions avoid mistaking rounded state subtraction for a zero mapping.
For the singular-value block the mapping is the constrained proximal residual,
computed with stable pooled-block expressions rather than the unconstrained
gradient. An outward gradient at a spectral boundary is therefore not by
itself evidence that a feasible improving step exists.
A tiny or rounded-to-zero trial step by itself is numerical stagnation, not
stationarity. This diagnostic is a feasible hard-prox fixed-point condition;
it is neither a global optimality test nor a KKT certificate for the chart's
boundary constraints. A terminal stationarity flag also does not make an
earlier validation-selected state stationary.

The tuner deep-copies every candidate and prepares source frames at most once
per search, when a chart candidate needs them. Candidates share `source_rank`
and the training/validation split.
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

## Optional full-capacity constrained update

`refinement_solver="masked_anchor_projected"` is a practical alternative when
the hard support caps equal the full capacities outside the free rows. It uses
`H_s=Z_s D^{-1}` while preserving the original masked L1 objective
`sum_s lambda_s sum_ij M_sij d_j |H_sij|`. At the current diagonal, the H block
uses weighted L1 proximal weights `lambda_s M_sij d_j / L` together with
`||H_s||op <= sqrt(1-anchor_min^2)`. Its diagonal block includes the masked
column penalty `sum_s lambda_s sum_i M_sij |H_sij|` and the same bounded,
gap-separated spectral projection. Cayley blocks project onto their existing
operator-norm balls. Backtracking checks the actual masked objective, including
the simultaneous H/diagonal bilinear penalty remainder.

The smooth chain rule is `g_Hs=g_Zs D` and
`g_d(H)=g_d(Z)+sum_rows(g_Zu*H_u)+sum_rows(g_Zv*H_v)`.
Unpenalized complementary rows have zero L1 weights but remain inside the
joint anchor-feasibility ball. Certified inner-proximal error bounds enter the
fixed-step residual; unresolved proximal solves cannot certify stationarity.
Only an active numerical boundary receives the inherited inward roundoff guard,
which never lowers the declared floor. Shared v1 failure caching and its
failure-aware step policy limit repeated exhausted proximal solves.

All returned states and checkpoints use the original Z encoding. Step norms,
gradient norms, and sufficient decrease use the H metric; trial radius remains
in the Z metric. Audits recompute the declared metric independently. Proximal
work is accumulated across chart-continuation segments. Earlier checkpoint
views label unretained prefix work as unavailable rather than reporting zero.

Operator-norm projection can fill zeros, so this option explicitly rejects
restrictive hard caps. It does not replace the hard-sparse default method,
alter source/initializer settings, adapt the anchor floor, or establish a new
theorem guarantee. Candidate configuration and campaign plans record the solver.

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

The explicit unpenalized RRR branch has a separate numerical optimum audit;
the manuscript's sparse chart theorem is not claimed for this branch. Both
paths retain `theorem_certified=False` in practical result metadata.
