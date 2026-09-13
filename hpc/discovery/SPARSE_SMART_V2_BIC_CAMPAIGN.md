# SparseSMART v2: paired fixed and automatic BIC campaign

This campaign compares training-only BIC with the completed 100-seed,
validation-selected SparseSMART v2 campaign. It does not compare unpublished
earlier SMART algorithms. All scientific computation runs under Slurm on
Discovery; all new output belongs on `/scratch2`.

## Frozen scientific policy

- Models I–III, seeds 0–99, all supported settings in Experiments 1–4.
- The fixed arm preserves the previous supplied ranks/free counts, including
  rank 11 with initializer dimension and free counts expanded to 11.
- Automatic BIC searches fitted ranks 1, 3, 5, 7, 9, 11; for each rank it uses
  tied free counts from 5, 7, 10, 15, 20 that are at least the rank, plus the
  rank itself and the all-free RRR endpoint.
- Initializer dimension is `max(10, fitted_rank)`. Initializer penalties remain
  0.003, 0.03, 0.1, 0.3, 1, 3. Refinement grids remain
  U = 0, 0.001, 0.0025, 0.01, 0.04 and V = 0, 0.001, 0.0025, 0.01.
- Full observed source SVD, nonbinding hard caps, anchor floor 0.04, adaptive
  chart continuation, and `masked_anchor_projected` refinement remain fixed.
- Preserve lower singular-value/gap bounds 0.001/0.0001 in Experiments 1–2 and
  0.05/0.01 in Experiments 3–4. Identical observations under different bounds
  are different estimator protocols and cannot share iterative fits.
- Stop on the existing training stationarity criterion or at 2,000 accepted
  updates. Return the terminal state. No validation input, validation stopping,
  or validation checkpoint selection is permitted in fitting.
- BIC uses observed RSS and dimension
  `r*(free_u+free_v-r) + support_u + support_v`. Supports count exactly nonzero
  masked weighted coordinates (frozen absolute tolerance zero). This is a
  model-dimension approximation, not a proved adaptive degrees-of-freedom
  formula. RRR is charged `k*(rank(X)+q-k)` for
  `k=min(fitted_rank,rank(X),q)`.
- Validation and true coefficients are used only for downstream evaluation.

## Sharing and storage

The fixed candidates are contained in the automatic library. A single shared
fitting campaign therefore produces both selection arms. There are 3,000
unique datasets and 3,300 dataset/protocol groups, 118,800 dispatched grouped
tasks, and 10,553,400 candidate outcomes. The final display retains 6,600
fixed-setting cases. Automatic results are horizontal across the supplied
rank/free-count sensitivity axes in Experiments 2–3.

Each dispatched task covers one dataset/protocol, fitted rank, and initializer
penalty. Its initializer is reused across free counts and their 19 nonzero
U/V penalty pairs. The first initializer task also computes one RRR endpoint
for that rank; zero-penalty and all-free aliases require no duplicate fits.

Preparation binds SHA256-verified data and source-frame NPZ files from the
previous campaign in place. The reference is read-only. No raw data copies or
per-fit archival transfers are performed. Each grouped task writes candidate
score/status records and retains states and histories only for candidates
that could win fixed or automatic selection. Completed free-count blocks are
restartable. Numerical exclusions are recorded separately from execution
failures; partial or failed trajectories cannot silently win.

## Entrypoints

- `simulation/sparse_smart_v2_bic_plan.py`: freezes the complete plan and aliases.
- `simulation/run_sparse_smart_v2_bic.py prepare`: verifies/binds reference data.
- `simulation/run_sparse_smart_v2_bic.py fit --task ID`: grouped training fits.
- `hpc/discovery/submit_sparse_smart_v2_bic.py`: resumable Slurm submission.
- `simulation/summarize_sparse_smart_v2_bic.py`: integrity and winner audits,
  paired results, tables, and comparison plots.

Freeze an isolated source snapshot and manifest in a fresh run root first.
Planning must use that frozen source. Run the launcher's dry run before actual
submission. Use `--task-ids ... --no-summary` for a small canary; after it ends,
`--resume` submits only missing tasks. Full fits use GNU Parallel within
100-CPU allocations, with an exclusive one-CPU `srun` per dispatched worker.
There are no Slurm arrays. Each pool depends on successful preparation, not
on earlier fitting pools. A separate one-CPU summary follows all fitting pools.

Every submission has an immutable attempt record. Reconcile any uncertain
submission against that record and Slurm before retrying. Read SSH session
instructions in the parent manuscript repository before remote operations.

The comparison reports paired coefficient RMSE, Monte Carlo uncertainty,
selected ranks/free counts/penalties, RRR frequency, numerical outcomes, and
computation time. Since the old trajectories used validation stopping, the
fixed BIC comparison changes both stopping and selection. It does not isolate
the scoring criterion alone. Budget completion does not certify convergence.
