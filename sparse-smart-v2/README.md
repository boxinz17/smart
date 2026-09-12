# SparseSMART v2

`sparse-smart-v2` implements the hard-sparse two-stage procedure in
[`notes/v1_sparse_two_stage_meets_version1.tex`](../../notes/v1_sparse_two_stage_meets_version1.tex).
It is a separate package and imports reusable chart, anchor, and numerical
components from `sparse-smart`. It does not modify the original estimator.

Both stages use fixed full source frames. A reduced Lasso–SVD initializer
selects well-conditioned anchor rows. Refinement leaves those rows and any
additional requested source directions unpenalized, while applying entrywise
soft thresholding and hard support limits outside them. There is one procedure
at every penalty value, including zero.

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
directions used by initialization, with `rank <= source_rank <= min(p, q)`.
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
the same full source frames; they must share `source_rank`. Each template is
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
- Setting either refinement penalty to zero preserves its hard cap. Setting
  `init_penalty=0` uses reduced minimum-norm least squares followed by
  coefficient-SVD truncation in the same initialization path. It does not
  dispatch to a target-only RRR solver. Full caps must be supplied explicitly.
- For a nonunique positive-penalty Lasso solution, initialization uses a
  checked minimum-norm quadratic solve on its tied solution face. The solve
  normalizes coefficient scale and verifies the fit, L1 norm, and numerical
  norm optimality; an unsuccessful check produces an initialization failure.
- Initialization and refinement apply strict spectrum and chart checks.
  Infeasible singular values are rejected rather than clipped or repaired.
  Backtracking halves the step until a valid decreasing update is found or
  the trial limit is reached. Each iteration starts again at the supplied
  `step_size_inverse`; it does not inherit the preceding backtracked value.
  Failures have inspectable statuses; set
  `raise_on_failure=True` to raise `FitFailure`. A retained partial estimate
  can be inspected explicitly but cannot win a tuning search.
- With validation data, `coefficient_` is the best evaluated iterate and
  `last_coefficient_` is the last accepted iterate. `selected_iteration_`,
  `validation_history_`, `history_`, and `metadata_` record that distinction.
  Requested `checkpoint_iterations` retain additional full states; ordinary
  validation evaluations need only loss records and the current best state.
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
