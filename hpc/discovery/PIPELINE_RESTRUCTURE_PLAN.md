# Discovery rollout: fit-only jobs first, aggregation later

Status: the fit-only shell overlay was deployed and verified on Discovery on September 10, 2026. All 39 migrated jobs use 64 workers/CPUs per batch. Canary 11859552 started at 23:16:24 UTC. The user then explicitly authorized releasing the remaining jobs while the canary was progressing successfully, superseding the earlier requirement to wait for its completion. Jobs 11859553–11859590 were all released and verified pending without holds at 23:43:25 UTC. Deployment and release are complete. Canary 11859552 passed terminal operational validation at 2026-09-11 01:24:45 UTC: all 10,000 task identities, compact manifests, exit markers and expected result paths matched; eight result JSONs passed sampled scalar metadata checks. Full numerical/convergence/coverage auditing is separate. Phase-two case aggregation and compact reduction are now implemented locally; Discovery deployment remains pending.

## 1. Agreed direction

First simplify the jobs currently on hold so they only fit the already-planned tuning tasks, write their full results and essential execution records, and exit. Remove aggregation, campaign summaries and archival from their execution paths. A new dependent-job orchestrator is not a prerequisite for this change.

The change was prepared and tested locally, then deployed while all migration targets remained held and legacy jobs continued under their existing code in verified separate directories. Develop aggregation and its Slurm orchestration afterward, using the preserved outputs.

This supersedes the earlier proposal to build and deploy the entire fitting/aggregation/summary/archive DAG before releasing the held fitting jobs.

At the September 10, 09:28 Pacific snapshot, jobs 11859552–11859590 were held (39 jobs) and nine legacy campaign jobs were running. These are historical counts; the deployment and release record below supersedes them. The user subsequently authorized execution of phase one and then explicitly authorized changing the helper to permit deployment alongside isolated running legacy jobs.

## 2. Phase one: the smallest fit-only change

The fitting job should do four things:

1. Load and verify the existing numerical environment and frozen source/configuration.
2. Dispatch its existing work table with GNU Parallel and single-CPU srun steps.
3. Write the existing complete per-task scientific JSON and essential execution/provenance records.
4. Finish the job after the last task terminates and the job log/status records are flushed.

Keep fitting worker count flexible; jobs 11859552–11859590 now request 64 workers per fitting batch. Each worker retains one CPU and 8 GiB of memory, giving 512 GiB per batch; the time limit remains 24 hours. Slurm may distribute workers across available nodes. Legacy jobs retain their original requests. Do not chain fitting batches to one another.

### Remove from the fitting path

- Per-task archival in both budget_worker.sh exit traps, including all separate archive-marker cp calls.
- The recursive whole-run archive and final archive-marker copy in budget_pool.sbatch's exit trap.
- The post-fitting environment inventory, aggregation command and standalone scientific-summary command in budget_pool.sbatch.
- Any aggregation or bulk/per-task archival embedded in the campaign's external wrapper, including its exit traps. Compact execution/provenance copies may remain.
- Mandatory archive-directory creation/copying in the fit-only submission/preparation path. Preparing a fit-only batch must not depend on scratch2 being writable or available.

Do not add new aggregation arrays, audit receipts, summary jobs, archive jobs, dependency orchestration or new result formats in phase one. Existing scientific aggregation programs can remain as reference/compatibility tools; fitting jobs must not invoke them. A generic legacy-versus-new pipeline mode is not required for this first rollout because legacy jobs keep their original, isolated snapshots while held jobs receive the fit-only shell overlay.

### Preserve the inputs needed for later aggregation

- Full result JSONs with every currently retained trajectory, coefficient-factor checkpoint, validation history, selected-iteration record and numerical diagnostic. Save unsuccessful and inapplicable outcomes as before.
- Case identity, seed file/hash, grid, budgets, validation schedule and study/work-item manifests.
- Frozen scientific source provenance and environment configuration. Retain source once per batch or immutable source identity, not once per fit.
- Process, worker and launcher exit codes; the GNU Parallel job log; pool exit/stage status; commands and worker stdout/stderr needed to diagnose failures.
- Task locks and atomic result writing so duplicate execution and incomplete writes remain detectable.
- Explicit archival and postprocessing statuses such as deferred, without claiming that either stage succeeded. Status tools should report fits finished; aggregation pending rather than campaign complete when a fit-only job exits successfully.

Retain genuine fitting/launch failures. Do not rewrite old archive-copy failures as successful historical executions. Execution completion, budget coverage and optimizer convergence remain distinct.

The existing 250-iteration checkpoints are in-memory scientific snapshots serialized into the final result JSON after fitting. This change does not add restart checkpoints or make interrupted fits resumable.

### Local files to change in phase one

| File | Change |
|---|---|
| hpc/discovery/budget_worker.sh | Remove task archive scans/transfers; preserve local outcomes/logs and deferred-archive status |
| hpc/discovery/budget_pool.sbatch | End after fitting/status writes; remove postprocessing and bulk archival from normal and failure exits |
| hpc/discovery/submit_budget_study.sh | Prepare fit-only submissions and primary provenance without requiring/copying to an archive destination; update help/dry-run output |
| hpc/discovery/README.md | Describe fit-only behavior, retained outputs, flexible workers and separately deferred aggregation |
| simulation/tests/test_discovery_budget_launcher.py and related launcher fixtures | Verify the new execution boundary and preservation of input/output contracts |

Inspection on Discovery found that all 39 held jobs have the same saved outer wrapper. It delegates execution to private per-run pool/worker shell scripts, so deployment preserved the existing job IDs. The outer wrapper copies a few compact status/provenance files to home; its conditional summary-copy branch remains inactive because fit-only jobs produce no summary directory. Migration verified that the held runs had no prior results or postprocessing artifacts. The saved outer wrapper itself and scientific Python sources remain unchanged.

### Focused validation before deployment

Use the existing fake Slurm/runner fixtures; no additional scientific simulations are needed to validate this operational change.

- Successful tasks and real fit/launch failures retain their correct primary exit markers and logs.
- Normal completion, task failure, launcher failure and handled termination invoke no archive rsync or archival cp operation.
- A fit-only batch never invokes aggregation, summary, archive or post-fitting package inventory commands.
- Fit-only preparation works without archive storage; paths with spaces and shell characters remain literal.
- Source/seed/configuration hashes, work tables, exact runner arguments, worker counts and result paths remain unchanged by the fit-only migration. The separately authorized resource amendment below changes only the worker-count configuration line and corresponding scheduler requests, with before/after hashes recorded separately.
- GNU Parallel continues independent tasks after an individual failure and the batch reports the overall fitting outcome correctly.
- The resulting raw layout still passes the existing collector's input/identity checks. Do not delete status or manifest files just to reduce file count.
- Dry-run prepares/displays the fit-only plan and invokes no scheduler or estimator.

Validation completed locally: 403 SparseSMART tests and 649 simulation tests passed, including 60 focused budget-launcher tests. Shell syntax and git diff whitespace checks passed. The fixtures cover failure/termination paths, forbidden archive/postprocessing commands, unusable archive destinations, literal paths and the existing collector's input contract. Eleven additional local tests of the campaign-specific migration helper passed, covering isolation from running legacy jobs, scheduler gates, changed inputs, unavailable archive storage, successful/repeated migration, rollback, an external release during rollback, and unexpected files, symlinks and nonregular source files. Seven canary-checker tests also passed. Scientific runner/solver modifications are outside phase one.

## 3. Deployment to held jobs alongside isolated legacy jobs

### Verified deployment record

The reviewed v3 migration helper has SHA-256 `9ce07b3183f99482799785efee85902d0b04085d37c88ce9cfa3c9c9f16780a7`. It completed all 78 shell replacements: budget_worker.sh and budget_pool.sbatch in each of the 39 held jobs' private source directories. The campaign's frozen scientific source remains at commit `2818bed30ea4c4155d832345528bc9cd35672e7c`; full raw outputs and scientific settings are preserved, and aggregation/archival remain deferred for migrated jobs.

| Time on September 10, 2026 (UTC) | Verified event |
|---|---|
| 18:43:22 | Migration completed; all 78 shell operations verified |
| 18:44:42 | Postverification confirmed all 39 targets still held, original resources/submission unchanged, and script/env/config hashes identical for all 26 legacy jobs |
| 18:44:57 | Campaign status reader installed to distinguish finished fits from deferred postprocessing |
| 18:45:24 | Existing canary 11859552 released with its original 150-CPU request; pending for Priority at this snapshot; jobs 11859553–11859590 remain held |
| 20:16:04 | Separate resource amendment completed: all 39 jobs held and verified at 64 CPUs/tasks/workers, 8 GiB per CPU, 512 GiB total and a 24-hour limit |
| 20:17:54 | Resource-aware campaign status reader installed, preserving original submission and campaign-plan records |
| 20:17:59 | Canary 11859552 released with 64 workers; pending at this snapshot; all other 38 jobs verified still held |
| 23:16:24 | Canary 11859552 started with 64 CPUs distributed across 44 nodes; start-monitor automation subsequently paused |
| 23:43:21 | Canary progressing in fit-only mode: 4,068 of 10,000 tasks finished, zero nonzero worker outcomes; aggregation and archival deferred |
| 23:43:25 | Under the user's explicit early-release instruction, all remaining 38 jobs released and verified pending without holds or dependencies, retaining 64 CPUs/tasks, 8 GiB per CPU and a 24-hour limit |

The durable overlay journal and original/new script blobs are under `/home1/mkolar/smart-results/campaigns/20260910_100seeds_2818bed_200_49XCM7/provenance/fit-only-phase1-overlay/`; `journal.json` records completion and per-operation hashes. The original plan required terminal canary validation before releasing the remaining jobs. The user's later early-release instruction superseded that scheduling gate. Healthy progress supports the release decision but does not establish the terminal output checks or fit-only allocation boundary.

The later resource amendment is recorded separately under `provenance/workers-64-20260910/`. Its completed `journal.json` preserves original configurations, before/after hashes, scheduler records and unchanged scientific-input evidence for all 39 jobs. Thirty-seven jobs changed from 150 workers and two from 100 workers. No job was canceled or resubmitted. Slurm initially retained the canary's old total-memory request after reducing its CPU/task counts; verification stopped the update with every job held. Explicitly reasserting `MinMemoryCPU=8192` recalculated the total to 512 GiB. The failed attempt and correction are retained alongside the completed amendment.

For subsequent canary checks, run `provenance/workers-64-20260910/check_canary_workers64.py --checker provenance/fit-only-phase1-staging-v3/check_canary.py` from the campaign directory. This wrapper requires the completed resource ledger, permits exactly the recorded worker-count line change, and retains the original scientific and terminal-output checks. The original baseline/checker remain unchanged. A `waiting` verdict is not a terminal validation pass; the early release rests on the user's explicit instruction. Focused local fixtures verified the resource recovery, amended checker and status installation; no scientific simulations were added for this resource change.

The completed release receipt is `provenance/workers-64-20260910/remaining38-release.json`. It records the early-release authorization, canary progress, per-job release intents/results, and verification of all 38 original job IDs and resource requests. The reviewed release helper has SHA-256 `0315cba6442ef04f63bb517714e7cbfd9a7fed79ec73890ca7212d1a6875ef72`. Release changed no scientific inputs, scripts, resources or historical submission records. Do not rerun the release helper; the jobs are already released.

### Migration procedure and original release gate

1. Keep the held jobs held while making and testing the local change. Do not patch helpers, source snapshots or wrappers used by currently running jobs.
2. Refresh all campaign states. Known legacy jobs may be running or completing only in their own verified directories. Check that no active run, source, output or operational file overlaps a held target. Record legacy states and hashes before/after migration, preserving their outputs and original behavior. Reject unknown active attempts and any migration target that is no longer user-held or has started.
3. Import and verify the existing campaign/submission manifests and restrict migration to the still-held intended jobs. Preserve their scientific scopes and original result roots.
4. Inspect each distinct saved Slurm script using scontrol write batch_script, plus every delegated script/configuration and exit trap. Slurm spools submitted scripts: editing the original script file does not replace the saved body.
5. Keep existing held job IDs if their saved wrappers delegate the expensive work through safely replaceable per-run operational files. Patch only those files, atomically, retaining before/after copies and hashes. Verify that no hidden wrapper still runs aggregation or archival.
6. If a saved wrapper embeds unavoidable postprocessing, prepare exact replacements for the held jobs and an old-to-new ID map. Do not promise an in-place update before inspecting the spool, and do not cancel/resubmit anything merely because this plan mentions the fallback.
7. Keep the current campaign's scientific source at commit 2818bed30ea4c4155d832345528bc9cd35672e7c. Change the operational shell layer independently. In particular, the runner's scientific fingerprint includes merge_sparse_smart_budget_shards.py, so replacing scientific snapshot files would change provenance even if the fits appeared otherwise identical.
8. Before removing per-task secondary copies, check primary-storage capacity/file quota, expected retention and the intended time until separate archival. Preserve one verified source/seed/configuration record; no raw-result deletion is part of migration. The archive directory's name alone does not establish backup or retention guarantees.
9. Record the migration and validate all fit-only entrypoints before releasing work. Initially release one existing short applicable held batch as the canary; this runs already-authorized work rather than adding fits. The original sequence called for confirming its allocation exits immediately after task completion before releasing the rest. The user subsequently authorized release based on successful ongoing canary progress; terminal validation remains a separate outstanding check.
10. No downstream aggregation jobs need to exist before the held fitting work is released. Failed fits and incomplete tuning coverage remain visible for the later aggregation stage.

Official script/update behavior: [sbatch](https://slurm.schedmd.com/sbatch.html) and [scontrol](https://slurm.schedmd.com/scontrol.html).

## 4. Phase two: add aggregation independently

Begin this work after the fit-only change is deployed. It can be developed while the newly released jobs fit; it does not require rerunning any completed fit.

The local implementation is described in [CASE_AGGREGATION.md](CASE_AGGREGATION.md). It provides independent case aggregation, immutable restart generations, explicitly scoped model indexes, compact selected-factor publications and a lean campaign reducer. The separate Slurm array launcher is now implemented locally; Discovery deployment remains pending. The design is:

- Aggregate one model/experiment/setting/seed case at a time across all of that case's tuning shards. Keep deterministic selection and exact grid/identity checks.
- Use modest, configurable case-level parallelism. Each array element requests one CPU and processes a short sequential chunk (default 25 cases); the array concurrency cap defaults to four and can be chosen as 16, 32, etc. Size memory and time using existing-result benchmarks before adopting production defaults. One slow case retains only one CPU allocation.
- Preserve the full merged numerical audit in memory, then publish compact audit/provenance and distinct selected factors by default. Full all-factor merged JSON remains an explicit output mode. Preserve scientific checks and avoid invoking a third full factor audit solely to produce a report. Parsing and initial hashing share one raw read; the final full content rehash remains.
- Make case publication atomic and resumable using exact input/output, plan, scientific-source and analysis-code hashes. Changes to a processing generation must not relabel old fit provenance.
- Add a final reducer over compact audited case records. It should not traverse raw task directories, regenerate datasets, load factors or fit models.
- Use audited within-case winners to calculate statistics across seeds. Preserve the general 100-combination/8,000-iteration configuration, expanded source-rank-5 144-combination/16,000 configuration, and source-rank-7 100-combination/16,000 configuration. Do not pretend the campaign uses one common grid/budget.
- Recompute means/SEs from individual seed outcomes, preserve requested versus usable counts, and keep incomplete-grid available-winner summaries separate from complete-grid statistics. Do not average batch means or SEs.
- Track execution, scientific audit, tuning coverage, convergence and archive status independently.
- Preserve the paper results in a smaller validated package; do not restore per-task archive copies. Selected-factor export and compact diagnostics are implemented. Long-term storage and cleanup remain separate decisions. Keep raw results until a verified replacement and an explicit cleanup decision exist.

The first rollout uses completed Model II producers and the immutable campaign roster. Already-finished producers may have left Slurm's controller; verify terminal state and durable outputs rather than depend blindly on old job IDs. The launcher submits the final summary with an afterany dependency on its aggregation array, keeping failure and missing-case diagnostics visible. A single frozen analysis snapshot is shared by all array elements, separately from fitting snapshots. Durable submission intents prevent accidental duplicate submissions after interrupted or uncertain scheduler responses. Keep final summary and archival outside fitting allocations.

## 5. Completion criteria

Phase-one deployment and release are complete: the held work was migrated without changing running legacy jobs, and all 39 migrated jobs have been released. The completed canary passed the operational output gate: 10,000 unique completed tasks, matching manifests and result-file presence, no operational issues, and allocation release within about one second of the last worker. This was not a full scientific result audit. The new fitting jobs contain no aggregation/summary/archival operations, and preserve their scientific inputs and full raw-result format. Aggregation implementation is deliberately not a prerequisite.

The local worker, pool, submission helper, launcher tests and README implement the fit-only behavior. The initial deployment replaced only the two private runtime shell scripts per held run, preserved all job IDs and original resources/submission, and verified the legacy script/env/config hashes unchanged. The subsequent resource amendment reduced all 39 migrated jobs to 64 workers while preserving job IDs, scientific inputs and historical submission records. The revised campaign status reader reports these current requests. Canary 11859552 is completed; all remaining 38 jobs were released. The canary operational report is preserved under campaign provenance/workers-64-20260910/canary-output-verification-parallel-20260911T012253Z.json. Phase two is implemented locally and validated with synthetic saved outputs; no aggregation jobs have been submitted on Discovery.
