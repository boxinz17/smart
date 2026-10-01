# SparseSMART v2: bounded RSC/BIC pilot

This campaign investigates automatic selection at **200 accepted updates**, with
earlier training-stationarity stopping. It does not run 500/2,000-update controls,
use validation stopping, change a prior campaign, or launch 100 seeds.

The split and pair criterion are frozen before ranking or fitting. Development
uses seeds 0–4; assessment uses 5–9, across Models I–III and all 33 canonical
dataset/protocol groups per seed. Display aliases do not receive extra weight.
These assessment seeds are held out from shortcut design, not claimed to be
independent of all previous analyses.

## Selection and estimator

Rank is estimated once per observed training dataset with original SMART's RSC
implementation, `theta=1`. The residual noise denominator is
`q * (n - rank(X))`. Undefined/nonfinite or nonpositive noise estimates stop
preparation. Original SMART caps the result at the number of singular values of
observed `C0` exceeding `1e-10`, and promotes a raw zero estimate to one when that
cap permits it. A zero source cap is explicitly unsupported. Even or previously
unsearched ranks are retained; no rounding to the old odd-rank grid occurs.

For selected rank `r`, use initializer dimension `max(10,r)` and tied free counts
`{r} union {5,7,10,15,20: s >= r}`, bounded by ambient dimensions. Both phases fit
the six-initializer control bank (`0.003,0.03,0.1,0.3,1,3`) with tied positive
penalties `0.001,0.0025,0.01`. One zero-penalty/all-free RRR endpoint is shared
across initializers. At rank five, the bank contains 91 candidates per group;
the final two-initializer subset contains 31. Each grouped task reuses its
initializer across free counts and penalties.

After complete development coverage, compare all 15 initializer pairs against
the six-initializer bank using mean normalized training-BIC regret
`(pair BIC - bank BIC)/(n*q)`, equally weighting dataset/protocol groups. Ties use
ascending initializer pairs. Missing computations block pair freezing; recorded
initializer exclusions remain outcomes and are reported separately. Assessment
fitting is released only after the pair artifact is frozen and verified.

The full observed source SVD, nonbinding hard sparsity caps, anchor floor `0.04`,
adaptive chart continuation, `masked_anchor_projected` solver, protocol-specific
singular-value/gap bounds and terminal BIC dimension approximation are preserved.
Budget completion is distinguished from convergence.

## Snapshot and submission

Run the builder locally from the code repository, using fresh absolute paths:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  hpc/discovery/build_sparse_smart_v2_cheap_bic_snapshot.py \
  --local-root /absolute/local/result/sparse-smart-v2-cheap-bic-TIMESTAMP \
  --remote-root /scratch2/mkolar/smart/runs/sparse-smart-v2-cheap-bic-TIMESTAMP
```

The baseline manifest supplies only the source-file inventory. The builder
copies current files, records Git commit/status/diff hash and per-file hashes,
freezes the split, and writes a sibling `.tar.gz` with paths relative to the new
root. It includes no observed arrays or raw fit directories. Transfer and extract
into the matching fresh remote root using the project's task-private SSH helper
and `scripts/SSH_SESSIONS.md` from the parent project. Close that connection when
orchestration is finished.

Use the existing persistent Python environment to preview and then submit:

```bash
/home1/mkolar/envs/smart/bin/python \
  "$PILOT_ROOT/source/hpc/discovery/submit_sparse_smart_v2_cheap_bic.py" \
  --root "$PILOT_ROOT" --dry-run
/home1/mkolar/envs/smart/bin/python \
  "$PILOT_ROOT/source/hpc/discovery/submit_sparse_smart_v2_cheap_bic.py" \
  --root "$PILOT_ROOT"
```

`PILOT_ROOT` must be the builder's exact `/scratch2` remote root. Submission uses
durable intent ledgers; reconcile uncertain submissions before retrying.

The sequence is preparation/RSC (one CPU), development fitting (100 CPUs),
development audit and pair freeze (one CPU), assessment fitting (100 CPUs), and
assessment summary (one CPU). Each fit pool uses GNU Parallel inside one Slurm
allocation, with exclusive one-CPU `srun` workers; no arrays are used. Pools have
a six-hour limit and use 4 GB per CPU; preparation/audit jobs have 16 GB and a
two-hour limit. Completed fits release their CPUs before separate aggregation.

## Reuse, audit and outputs

Existing data/source arrays are verified and read in place on `/scratch2`.
Matching certified RRR endpoints can be reused with bound source/data/rank
identities. A longer run's terminal record does not reconstruct its 200-update
state; unavailable iterative fits remain explicitly unavailable retrospectively.

Retain `pilot.json`, the source manifest/snapshot, split and rank records,
rank-selection timing, phase plans, task identities/arguments, outcomes,
selected states, process/Slurm logs, the frozen pair and compact summary. There
is no per-fit archival copy. Reports compare the six- and two-initializer banks
on paired groups and aggregate uncertainty by seed blocks, retaining repeated
settings within each seed. Report selection distributions, numerical exclusions,
execution coverage, RRR frequency and measured time including rank/initializer
overhead. Historical runtime reconstructions are estimates; this pilot alone
does not establish a final-paper superiority claim.
