# Codex handoff: deploy verified SparseSMART and test on Discovery

The production line-search correction and local gate below have passed. Carry
out the remaining cluster experiment using the corrected source: verify the
cluster runtime, then run three targeted full-tuning-grid cases on Discovery.
Proceed to the five-seed budget study only if the later gate below passes.
Do not start a 100-seed campaign or fit competing methods. No cluster jobs have
been submitted as part of the local verification.

Local verification completed on 2026-09-09: 346 SparseSMART tests and 353
simulation tests passed. The matched 2,000-update production fit recovered
validation MSE `0.2916988417713056`; all 2,001 states matched the isolated
original solver exactly. The terminal residual was `0.08238381680428128`, so
this is a successful finite-budget check, not numerical convergence. Detailed
local evidence is in
`simulation/result/sparse_smart_local_reset_check_20260910T010839Z/LOCAL_CHECK.md`.
That ignored result directory is not transferred by a Git push. The correction
and its regression tests are included in the source; do not reapply them or
repeat the completed local fits when the checked source is unchanged.

## Workspace and current state

The local software repository is
`/Users/mladen.kolar/Library/CloudStorage/Dropbox/Projects/SMART_BoxinJinchi/code`.
It is a separate Git repository inside the manuscript repository. Its branch
is `WIP_bi_smart`; `d4b6e7b` is the earlier commit before the verified correction.
Deploy the subsequent commit containing this handoff and the reset fix, not
that earlier commit. Inspect current status and history before editing;
preserve any later changes. Read applicable `AGENTS.md` files.
This is software work; no manuscript edits are needed.

Discovery is reached with `ssh discovery`. The configured deployment is
`$HOME/projects/smart`, environment `$HOME/envs/smart`, Slurm account
`mkolar_1314`, and partition `main`. Verify these against the actual cluster
before submission. Its checkout may differ from the local branch. Inspect and
preserve remote changes; prepare an isolated checkout or snapshot of the fixed
source rather than replacing a dirty deployment. Do not force-reset or force-push.

Useful repository paths, relative to the software root:

- `sparse-smart/src/sparse_smart/anchor_solver.py`
- `sparse-smart/tests/test_anchor_stability.py`
- `simulation/run_sparse_smart_budget_study.py`
- `simulation/summarize_sparse_smart_budget_study.py`
- `simulation/SPARSE_SMART_V05_BUDGET_STUDY.md`
- `environment/README.md`, `environment/check_runtime.py`
- `hpc/discovery/README.md`, `hpc/discovery/env.sh`

## 1. Verify the included correction

The earlier solver at `d4b6e7b` started each search with:

```python
start_L = max(initial_L, last_accepted_L / 2.)
```

The corrected production solver restores the per-iteration starting inverse step:

```python
start_L = initial_L
```

Obsolete reuse-specific state/recovery branches have been removed while
preserving the general numerical-resolution safeguard. Keep the stable
objective-difference calculation, all feasibility and decrease checks, fixed
mapping metric, inner-proximal uncertainty/refinement, stationarity tolerance,
and successful-checkpoint retention. Do not loosen tolerances, change
penalties, or increase the budget to disguise the regression.

The reuse-specific tests have been replaced. The new deterministic regression
demonstrates recovery of a larger valid step after transient anchor curvature;
it fails with the old reuse rule and passes with the correction by checking
actual objective progress. Retain it and the tests of objective descent,
feasibility, honest stationarity, and preservation of earlier checkpoints after
failure. Package documentation and diagnostic metadata describe the reset.

### Evidence already established locally

The controlled case is Model I, source noise 0.5, seed ID 1, initializer penalty
0.03, left/right penalties `(0.0025, 0.01)`, and 2,000 updates. It uses 200
training rows and 100 independent validation rows.

| Variant | Endpoint validation MSE | Training objective |
|---|---:|---:|
| Committed reuse rule | 0.36863169717173067 | 8.070864436595183 |
| Current solver with only inverse-step reset | 0.2916988417713056 | 7.122437195042997 |
| Original v0.3 solver | 0.2916988417713056 | 7.122437195042997 |

The corrected and original solvers matched all 2,001 objective/step-history
records, all nine recorded validation points, and all 59 saved state arrays.
The first divergence from the faulty solver occurred at update 8. Both needed
inverse step 1,310,720 at update 7, near an active anchor constraint. Reuse then
accepted 655,360; resetting found 10,240, followed by 160 and 20 at updates 9–10.

Optional local evidence is under
`simulation/result/sparse_smart_v05_regression_investigation/`. These ignored
artifacts are not guaranteed to exist on Discovery. Their old diagnostic script
loads the original solver from `HEAD`; that reference is now stale. If using
or adapting it, obtain the original solver from the immutable commit
`650d1b8:sparse-smart/src/sparse_smart/anchor_solver.py`, never current `HEAD`.
Keep the current estimator and dependencies when isolating solver variants.

The matched pair has been reproduced locally with the production correction
and isolated original-solver reference, using identical generated arrays. An
uncorrected control is allowed on this same pair if needed. Limit this control
to 2,000 updates and at most three variants. Preserve it in separately labeled
diagnostic artifacts, not ordinary study records with misleading fingerprints.
After the local gate passes, repeat this bounded check on one cluster node to
verify portability before starting the three full-grid cases.

Expect agreement within numerical precision between fixed and original
trajectories on the same platform. Use 1e-8 absolute validation-MSE agreement
and 1e-8 relative/absolute objective agreement as an initial check; investigate
larger discrepancies rather than relaxing the criterion silently. Historical
macOS floats and array hashes are evidence, not a requirement for bitwise
identity on a different BLAS/platform. Record any platform-dependent difference.

## 2. Verify the standardized runtime and source

Use Python 3.12.x and the exact shared pins in `python-constraints.txt`:
NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.0, joblib 1.6.0, threadpoolctl 3.6.0.
Discovery's configured module is `gcc/13.3.0 python/3.12.8`. Local runs use
the software repository's `.venv/bin/python`, not the older Miniforge environment.

Run `python environment/check_runtime.py` and `python -m pip check` in the
environment actually used for tests/fits. If cluster dependencies differ,
repair an isolated Python environment in a Slurm setup job using the shared
constraints. Do not update an environment used by running jobs. No R package
installation is needed for this SparseSMART task.

### Local gate: passed; retain these checks for later changes

Use the repository's `.venv/bin/python` to run the runtime checks, package tests,
and simulation regression tests with the production correction. From the
software root:

```bash
.venv/bin/python environment/check_runtime.py
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q sparse-smart/tests --import-mode=importlib
.venv/bin/python -m pytest -q simulation/tests
```

Also run the single matched Model I case above through 2,000 updates. Require
the new regression test to fail under the previous reuse rule and pass under
the production correction, and require the matched fit to recover agreement
with the isolated original solver. Do not rerun all pilot estimates locally.
If any check fails, investigate locally and stop before cluster deployment.
An earlier isolated monkey-patch experiment does not satisfy this gate for the
production code.

Once this local gate passes, freeze the corrected source. Repeat runtime and
relevant regression checks in the cluster environment. Use Slurm for cluster
tests and fitting; do not run these on the login node. For example, from the
frozen software root in the allocated environment:

```bash
python -m pytest -q sparse-smart/tests --import-mode=importlib
python -m pytest -q simulation/tests
```

The simulation suite uses its existing default pytest import mode. The verified
local source passed 346 SparseSMART and 353 simulation tests. Record actual
cluster results rather than copying those counts. Preserve the frozen source's
commit or complete diff and a file-hash
manifest.

## 3. Run three targeted cases with all nine penalty pairs

Use a dedicated budget-study Slurm wrapper. The existing `submit.sh` rejects
`run_sparse_smart_budget_study.py`, whose CLI is different from the per-seed
drivers. Do not bypass the allowlist or silently switch to
`run_sparse_smart_external.py`: that would change the continuous-trajectory
experiment. Do not use `smoke.sbatch`, which also fits restricted RRR.

For these three sequential cases, start with one node, one task, one CPU,
8 GB memory, and a two-hour time limit. These are allocation bounds, not
measured requirements. Confirm availability/account limits. Keep BLAS/OpenMP
threads at one and use `--workers 1`.

Create an immutable source snapshot that includes uncommitted fixes when
necessary, shared environment files, and the saved seed CSV. Exclude virtual
environments, old results, caches, and build products. Reuse the snapshot and
archive conventions in the Discovery scripts, with a dedicated budget wrapper.
Set `SMART_SOURCE_ROOT` to the snapshot's absolute path **before** sourcing
its `hpc/discovery/env.sh`; then change into its `simulation` directory.

Choose a unique run parent and assign its absolute path to `BUDGET_RUN_ROOT`.
Keep logs, source snapshots, and Slurm metadata outside the fresh runner output
directories. Each runner root must be empty or absent before its first run;
the runner creates its own manifest.

From the snapshot's `simulation` directory, dry-run these exact commands by
adding `--dry-run`. Verify one cell and nine trajectories for each before
executing without that flag inside the Slurm allocation:

```bash
python run_sparse_smart_budget_study.py \
  --models 0 --experiments 3 --setting-index 5 --seed-ids 1 \
  --profile difficult --workers 1 \
  --iteration-budgets 500 1000 2000 4000 8000 --checkpoint-interval 250 \
  --init-penalties 0.03 --penalties-u 0.0025,0.01,0.04 --penalties-v 0.0025,0.01,0.04 \
  --output-root "$BUDGET_RUN_ROOT/model1_noise05_seed1"

python run_sparse_smart_budget_study.py \
  --models 2 --experiments 2 --setting-index 3 --seed-ids 1 \
  --profile difficult --workers 1 \
  --iteration-budgets 500 1000 2000 4000 8000 --checkpoint-interval 250 \
  --init-penalties 0.03 --penalties-u 0.0025,0.01,0.04 --penalties-v 0.0025,0.01,0.04 \
  --output-root "$BUDGET_RUN_ROOT/model3_rs7_seed1"

python run_sparse_smart_budget_study.py \
  --models 2 --experiments 2 --setting-index 3 --seed-ids 3 \
  --profile difficult --workers 1 \
  --iteration-budgets 500 1000 2000 4000 8000 --checkpoint-interval 250 \
  --init-penalties 0.03 --penalties-u 0.0025,0.01,0.04 --penalties-v 0.0025,0.01,0.04 \
  --output-root "$BUDGET_RUN_ROOT/model3_rs7_seed3"
```

Here "full tuning grid" means the nine penalty pairs, not `--profile full`.
Model I retains 200 training rows; Model III retains 500. Each has 100 additional
independent validation observations. Do not split away training rows or refit
on validation data. Use `simulation/data/random_seeds/experiment_seeds.csv`;
never invent replacement seeds. Coefficient truth is diagnostic only.

Summarize each case immediately in the same frozen source/environment, ideally
within the same job/node. Both paths below must be explicit:

```bash
python summarize_sparse_smart_budget_study.py \
  --result-root "$BUDGET_RUN_ROOT/model1_noise05_seed1" \
  --output-root "$BUDGET_RUN_ROOT/model1_noise05_seed1/summary" \
  --manifest-scope
```

Repeat for the other two roots. Verify recorded source fingerprints explicitly
against the frozen source, and verify regenerated data and saved factor scores.
The summary checks recorded implementation-hash consistency but does not itself
recompute the current implementation hash. Runner implementation hashes include
absolute source paths; relocating identical code changes them. Keep the original
snapshot path for fitting and runner resume/verification. Archive source, seeds,
hashes, runtime records, raw results, summaries, logs, and exit statuses without
rewriting recorded hashes. Preserve partial outputs on failure or timeout.

### Gate before expanding

Inspect actual records; neither runner nor summary exit code zero establishes
successful optimization or complete study coverage. Require all of the following:

1. Relevant tests and runtime checks pass, and the matched-pair correction
   reproduces the original solver on the same cluster data/environment.
2. Exactly the three requested cells exist, with nine declared candidates each,
   no manifest exceptions or missing cells, and correct training/validation use.
3. Independent factor/data audits and explicit source-fingerprint checks pass.
   Accepted steps obey the existing feasibility and sufficient-decrease rules.
4. Every candidate reaches the 8,000 cap or has verified earlier stationarity.
   Earlier successful checkpoints survive later failure, but such a failure
   still makes the longer-budget comparison unresolved. If any remain unresolved,
   report the diagnostics and pause expansion instead of changing eligibility.
5. No unexplained performance regression remains in the targeted checks. For
   the known Model I issue, inspect the full-grid trajectory with
   `grid_candidate_id=1`, penalties `(init=.03, U=.0025, V=.01)`, and its
   **terminal 2,000-update** objective/MSE; the overall nine-grid winner is not
   the matched comparison. Confirm the parameters, rather than relying only
   on the numeric grid ID.

Strict stationarity is not required for every finite-budget fit if the cap is
reached and reporting is honest. Continuing validation gains are a reason to
study budgets, not to label a fit converged. Model III seed 3 was chosen because
it previously had late precision-limited stops; do not hide or reclassify them.
If a gate fails, complete the diagnostic report and stop before the larger study.

## 4. Conditional five-seed study; never 100 seeds in this task

Only after the gate passes, run the default difficult-setting design: three
model sizes, source ranks 5 and 7 and source noise 0.5, seeds 0–4. This is
**45 cells and 405 trajectories**, not the complete paper grid. It uses all
original training rows (200/300/500 by model) plus 100 independent validation rows.

Use a new output root; do not broaden a three-case manifest in place. The runner's
ProcessPoolExecutor is node-local. Request one node, one task, three CPUs,
24 GB memory, and initially an eight-hour limit, with `--workers 3` and one BLAS
thread per worker. These are conservative request bounds; verify Slurm limits
and use observed smoke timings to report a runtime estimate first. Do not use
multi-node allocations or a 128-worker pool for this runner. If the bounds are
insufficient, report that rather than automatically launching larger jobs.

First verify this command with `--dry-run`, then remove that flag inside the
allocated job only after all gates above pass:

```bash
python run_sparse_smart_budget_study.py \
  --models 0 1 2 --experiments 2 3 --profile difficult --seed-count 5 --workers 3 \
  --iteration-budgets 500 1000 2000 4000 8000 --checkpoint-interval 250 \
  --init-penalties 0.03 --penalties-u 0.0025,0.01,0.04 --penalties-v 0.0025,0.01,0.04 \
  --output-root "$BUDGET_RUN_ROOT/five_seed_budget_study" --dry-run

python summarize_sparse_smart_budget_study.py \
  --result-root "$BUDGET_RUN_ROOT/five_seed_budget_study" \
  --output-root "$BUDGET_RUN_ROOT/five_seed_budget_study/summary" \
  --manifest-scope --absolute-threshold 0.0001 --relative-threshold 0.001
```

Run the summary after fitting, not after the dry run. Compare 2k–4k, 4k–8k,
and 2k–8k validation gains using the declared practical threshold
`max(1e-4, 0.001 * earlier_validation_MSE)`. Exclude incomplete candidate
extensions from plateau evidence. Do not tune stopping with coefficient truth
or claim a universal sufficient budget from five seeds. Do not automatically
increase the 8,000-update cap in this task.

## Deliverables

Return the reviewed solver/test change, actual test results, Slurm job IDs,
requested resources and observed elapsed times, source/runtime/data provenance,
and durable artifact locations. Report the three-case gate outcome explicitly.
If the five-seed study ran, include paired validation gains and uncertainty,
selected iterations, coverage/failure counts, objective/movement/inner-solve
diagnostics, and a recommendation on the next iteration budget. Keep statistical
stabilization separate from numerical convergence. No competing-method fits,
manuscript changes, automatic 100-seed run, or automatic Git push are part of
this handoff.
