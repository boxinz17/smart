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

Independent response-wise Lasso solves the entrywise penalty with `1/(2n)`
normalization, no intercept, and no design rescaling. The returned coefficient
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

## Acceptance and finite precision

Every iteration starts at the configured inverse step size. Soft thresholding
followed by stable C-order hard thresholding minimizes the specified isotropic
quadratic support subproblem. The trial must pass the chart domain, displacement,
and `F(w) <= F(x) - L*||w-x||^2/4` checks. There is no permissive uphill
acceptance tolerance. An additive constant outside the exact right source span
is removed only during comparisons and restored in reported objectives.

`max_backtracks` limits doublings; zero still allows the first trial.
Finite-precision overflow produces a numerical failure or rejected trial.
An overflowing penalty/L shrink threshold has the limiting all-zero output.
When an entire trial rounds to its current state but the core/active-coordinate
first-order residual exceeds a local scale times `64*eps`, the solver reports
`numerical_stagnation`. The core scale uses its own gradient; active complement
scales also include the penalty. At available support capacity it checks unused
coordinates whose scores exceed the penalty. This is a numerical safeguard,
not a proof of stationarity. Exact zero steps with no resolvable local direction
are accepted and counted toward T. No convergence-to-global-optimum claim is made.

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
