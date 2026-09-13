# SparseSMART v2

`sparse-smart-v2` implements the hard-sparse two-stage procedure in
[`notes/v1_sparse_two_stage_meets_version1.tex`](../../notes/v1_sparse_two_stage_meets_version1.tex).
It is a separate package and imports reusable chart, anchor, and numerical
components from `sparse-smart`. It does not modify the original estimator.

The penalized path uses fixed full source frames. A reduced Lasso–SVD initializer
selects well-conditioned anchor rows. Refinement leaves those rows and any
additional requested source directions unpenalized, while applying entrywise
soft thresholding and hard support limits outside them. Unpenalized configurations
with full support capacities now return target-only RRR directly. This explicit
shortcut is a practical extension of the note's chart-based procedure.

## Unpenalized RRR endpoint

With the default `rrr_shortcut=True`, fitting returns ordinary target-only
rank-at-most-`rank` reduced-rank least squares when both effective refinement
penalties vanish and both support caps equal their full outside capacities:

- `penalty=(0, 0)` with `support_limits=((p-ru)*rank, (q-rv)*rank)`.
- `free_directions=(p, q)` with `support_limits=(0, 0)`, regardless of penalties.
- The corresponding mixed case: one side has every row free and the other
  side has zero penalty and full support capacity.

This dispatch occurs before source initialization or chart checks. It ignores
the initializer penalty, anchor floor, spectral bounds, spectral gaps, rotation
limits, and iteration budget when computing the coefficient. It uses the target
training rows only; there is no implicit refit on validation data. Validation
scores this single coefficient and can select it against other tuning candidates,
but cannot replace it with an initializer or an earlier iterate.

The calculation uses an SVD of the design and rank truncation of its projected
response, not truncation of the ordinary least-squares coefficient. A singular
design uses the minimum-norm lift of the selected fitted response, with numerical
rank tolerances and deterministic boundary-tie conventions recorded in
`rrr_certificate_`. The certificate checks attainment of the optimal training
loss on the numerical design range. No floor or jitter is added to the design.

Direct fits have `method_ == "target_rrr"`, termination reason
`"target_rrr_closed_form"`, and zero iterative updates. Their single checkpoint
is numbered zero and holds the physical coefficient, not an initializer or a
chart state. `left_factors_`, `right_factors_`, and `factors_` are in physical
coordinates, with as many columns as the effective fitted rank. Source and
chart attributes are absent. `checkpoint_model(0)` returns an independent copy.

Restrictive hard caps or a nonzero effective penalty keep the chart path.
`rrr_shortcut=False` explicitly restores legacy behavior for reproducibility
or initializer preflight. RRR is never substituted after a failed penalized fit.
The tuner caches one direct solve per training split and requested rank. New
Discovery plans record the shortcut; old frozen plans without the setting keep
their original behavior. No running campaign is changed by editing local code.

## Install and run locally

Use the repository's Python 3.12 environment at `code/.venv`. From the **SMART
project root**, install both editable packages with the shared dependency pins:

```bash
python3.12 -m venv code/.venv  # only if this environment does not already exist
code/.venv/bin/python -m pip install -c code/python-constraints.txt setuptools wheel
code/.venv/bin/python -m pip install --no-build-isolation -c code/python-constraints.txt \
  -e './code/sparse-smart[test]' -e './code/sparse-smart-v2[test]'
code/.venv/bin/python -m pip check
code/.venv/bin/python code/environment/check_runtime.py
code/.venv/bin/python -m pytest code/sparse-smart-v2/tests
code/.venv/bin/python code/sparse-smart-v2/examples/quickstart.py
```

The original editable install provides shared numerical components; the
import for this estimator is `sparse_smart_v2`. These commands run local unit
checks and a small example. This package is not registered automatically with
the existing simulation campaigns, and installation submits no cluster jobs.

## Fit one model

```python
from sparse_smart_v2 import (
    Margins, ObservedSource, PracticalCalibration, SparseSMARTv2,
)

model = SparseSMARTv2(
    rank=1,
    source_rank=3,
    free_directions=(2, 2),
    margins=Margins(d_lower=0.01, d_upper=10.0, gap=0.005, anchor_min=0.005),
    calibration=PracticalCalibration(
        init_penalty=0.02,
        penalty=(0.01, 0.01),
        step_size_inverse=20.0,
        support_limits=(2, 2),
    ),
    iterations=200,
)
model.fit(X_train, Y_train, source=ObservedSource(C_source),
          validation_data=(X_valid, Y_valid))
if model.success_:
    prediction = model.predict(X_test)
else:
    print(model.status_, model.message_)
```

The example assumes compatible matrix shapes and an initializer satisfying the
declared margins; these illustrative tuning values do not ensure a successful
fit for arbitrary data. The estimator does not center or standardize data and
does not fit an intercept. Any preprocessing must be fitted on training data
and applied consistently to validation/test observations and source coordinates.

`rank` is the fitted target rank. `source_rank` is the number of leading source
directions used by initialization, with `rank <= source_rank <= min(p, q)` on
the chart path. Direct RRR does not require `rank <= source_rank`.
The full working frames still have sizes `p × p` and `q × q`.

`ObservedSource(C_source)` accepts a source coefficient without asserting a
noise level or spectral-gap bound. A matrix supplied directly is equivalent.
`NoisySource(coefficient, noise_std, gap_lower, cluster_size=1)` retains the
supplied uncertainty metadata; practical fitting does not turn those numbers
into a source-accuracy certificate. `ExactSource(U, V)` accepts the original
package's orthonormal source-frame interface: its supplied columns are
preserved and completed to full ambient frames. Even this exact-frame adapter
uses full frames in refinement. With a source coefficient, all observed
ordered SVD directions are retained, including directions beyond `source_rank`;
deterministic numerical tie and null-space conventions complete the frames.

## Choose tuning values

Pass a list of estimator configurations to the tuner. This makes the axes of
the search explicit, including initialization penalties and anchor margins.
To tune iteration count efficiently, fit each configuration once to a maximum
budget and select among its validation evaluations:

```python
from itertools import product
from sparse_smart_v2 import SparseSMARTv2Tuner

candidates = [
    SparseSMARTv2(
        rank=1, source_rank=3, free_directions=free,
        margins=Margins(0.01, 10.0, 0.005, anchor_min=b),
        calibration=PracticalCalibration(
            init_penalty=init, penalty=(lu, lv),
            step_size_inverse=20.0, support_limits=caps,
        ),
        iterations=200, validation_interval=200,
        validation_iterations=(50, 100), checkpoint_iterations=(100,),
    )
    for init, free, lu, lv, caps, b in product(
        (0.01, 0.03), ((1, 1), (2, 2)),
        (0.0, 0.02), (0.0, 0.02), ((1, 1), (2, 2)),
        (0.005,),
    )
]
tuner = SparseSMARTv2Tuner(candidates, random_state=17)
tuner.fit(X_train, Y_train, source=ObservedSource(C_source),
          validation_data=(X_valid, Y_valid))
if tuner.success_:
    print(tuner.best_index_, tuner.best_score_, tuner.selected_iteration_)
    prediction = tuner.predict(X_test)
```

The supplied cap pairs must satisfy `ku <= (p-ru)*rank` and
`kv <= (q-rv)*rank` for each candidate. Use a filtered list when different
free-set sizes make some cap combinations invalid. The list above illustrates
the interface; it is not a recommended exhaustive grid.

`validation_iterations` adds scoring points without retaining a full state
at each point. Use `checkpoint_iterations` where an independently inspectable
prefix is also needed. The example evaluates iterations 0, 50, 100, and 200,
subject to early termination. Separate candidates with different `iterations`
are available for an explicit comparison of separately fitted budgets, but
they repeat the refinement work; shared initialization does not avoid that cost.

Without `validation_data`, the tuner reserves `ceil(n/3)` observations by
default using its seeded random split. All candidates use the same split and
share `source_rank`. Source frames are prepared only if a chart candidate needs
them. Each template is
deep-copied before fitting. Within that search, candidates with the same
initialization penalty and rank reuse the reduced initializer calculated on
the shared training rows. Validation choices and refinement runs remain
separate. Validation selects each candidate's evaluated
iterate and then the candidate with minimum mean squared error; exact ties
prefer the earlier candidate. Ranking uses stable pairwise loss differences,
so rounded absolute-MSE ties do not hide a detectable improvement. Failed fits are ineligible even when they retain
a partial coefficient estimate. The winner is kept on the training split,
without an implicit refit on all observations.

Inspect `results_` (also `candidate_results_`) for configuration, eligibility,
status, validation history, and selected iteration. `best_estimator_` is the
winning fitted model. The tuner retains compact candidate audits, not every
candidate trajectory. `SparseSMARTTuner` is an alias for `SparseSMARTv2Tuner`,
with this candidate-list constructor rather than the original package's grid
constructor. Validation reused for these choices is not an independent test
estimate; use separate test data for performance reporting.

## Interpret fits and stopping

- `free_directions=(ru, rv)` specifies row counts, not fixed anchor indices.
  The default is `(rank, rank)`. Each free set contains its selected anchors,
  then the earliest remaining source-SVD indices until its count is reached.
- `support_limits=(ku, kv)` counts **entries outside the free rows**. Free
  entries have no penalty or hard cap. Enlarging free sets, enlarging caps,
  and reducing penalties have different effects.
- `adaptive_anchors=True` permits a bounded continuation when refinement
  stalls at an anchor or Cayley rotation boundary. It selects better anchor rows only within
  the original, unchanged free sets. The source frames, fitted coefficient,
  spectrum, penalty, and hard outside support are preserved at the switch.
  `max_anchor_switches=16` limits these attempts; the default
  `adaptive_anchors=False` keeps a fixed chart. Each switch is logged in
  `anchor_switches_`. The global iteration budget and validation patience
  continue across switches. This coordinate extension does not carry an
  additional theoretical guarantee.
  An anchor replacement must strictly improve its margin beyond numerical
  slack while meeting the supplied floor; no percentage buffer is required. Rotation recentering
  can retain the same anchor indices with new centers. `anchor_switches_`
  records both kinds of chart transition, with explicit per-side actions.
  Numerical work reports total chart transitions, anchor-row changes, and
  rotation recenterings separately. If both primary anchor searches fail to
  find a positive gain, a bounded fallback searches from one-row neighbors
  of the current anchor, stopping at the first useful result. Every search
  stays inside the exact original free rows.
- Setting either refinement penalty to zero preserves a restrictive hard cap. Setting
  `init_penalty=0` uses reduced minimum-norm least squares followed by
  coefficient-SVD truncation in the same initialization path. It does not
  by itself trigger the RRR shortcut; the refinement penalties and full caps
  determine that dispatch. Full caps must be supplied explicitly.
- For a nonunique positive-penalty Lasso solution, initialization uses a
  checked minimum-norm quadratic solve on its tied solution face. The solve
  normalizes coefficient scale and verifies the fit, L1 norm, and numerical
  norm optimality; an unsuccessful check produces an initialization failure.
- On the chart path, initialization applies strict spectrum checks and rejects an infeasible
  spectrum without repairing it. Refinement includes the declared singular-value
  bounds and adjacent gaps in its quadratic proximal model, using bounded
  isotonic projection of the proposed singular-value block. This lets an update
  move along an active gap boundary instead of repeatedly rejecting an outward
  gradient step. Anchor and Cayley chart checks remain separate.
  Backtracking halves the step until a valid decreasing update is found or
  the trial limit is reached. Each iteration starts again at the supplied
  `step_size_inverse`; it does not inherit the preceding backtracked value.
  Failures have inspectable statuses; set
  `raise_on_failure=True` to raise `FitFailure`. A retained partial estimate
  can be inspected explicitly but cannot win a tuning search.
- On the chart path, with validation data, `coefficient_` is the best evaluated iterate and
  `last_coefficient_` is the last accepted iterate. `selected_iteration_`,
  `validation_history_`, `history_`, and `metadata_` record that distinction.
  Requested `checkpoint_iterations` retain additional full states; ordinary
  validation evaluations need only loss records and the current best state.
  With adaptive anchors, `chart_` describes the terminal state and
  `selected_chart_` describes the validation-selected state. Every saved
  checkpoint retains its endpoint and selected charts separately; an earlier
  state must never be reconstructed using a later chart.
- Optional `validation_patience` stops after a specified number of accepted
  iterations without meaningful validation improvement, evaluated according
  to `validation_interval` and `validation_iterations`. This is a compute
  stopping rule. It does not establish stationarity. Refinement recognizes an
  exactly zero feasible hard-prox mapping even with `stationarity_tol=None`;
  supplying a positive tolerance additionally permits an approximate mapping
  criterion. This uses a fixed reference step, independently of backtracking.
  A tiny or rounded-to-zero step alone is stagnation, not stationarity. The
  mapping criterion is a numerical fixed-point condition, not a global
  optimality or chart-boundary KKT guarantee.

All practical runs record `metadata_["theorem_certified"] == False`.
`success_` reports a successfully completed computational run, including a
finite iteration budget or validation stop; it does not establish the
manuscript's source, restricted-curvature, initialization, or basin conditions.
See [the algorithm description](docs/algorithm.md) for the exact update and
the boundary between the implementation and the theory.

### Optional constraint-aware solver

Set `refinement_solver="masked_anchor_projected"` to optimize in `H = Z / d`
coordinates and include the anchor and Cayley constraints in each proximal
update. The masked L1 objective, source frames, free rows, spectral margins,
and declared anchor floor remain the same. This practical option uses the
certified constrained-proximal routines from SparseSMART v1; it does not
replace the initializer or repair an invalid initial spectrum.

This option requires `support_limits` equal to the full outside capacities
`((p-ru)*rank, (q-rv)*rank)`. Restrictive hard caps raise an error because an
operator-norm projection can change entry support. The default remains
`masked_chart_spectral_soft_hard`, which supports restrictive caps.

Saved states remain Z-encoded. Step and gradient diagnostics use H coordinates;
trial-radius checks remain in Z coordinates. Proximal work and uncertainty are
logged, and small unresolved steps remain numerical failures. This alternative
does not carry a new statistical guarantee. Discovery plans select it explicitly
with `plan --refinement-solver masked_anchor_projected`.
