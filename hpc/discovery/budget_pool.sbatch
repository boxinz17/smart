#!/usr/bin/env bash
# One distributed allocation; one independent single-CPU step per planned cell.
set -euo pipefail
[[ $# == 1 && -f "$1/budget-job-config.sh" ]] || { printf 'Use submit_budget_study.sh.\n' >&2; exit 2; }
source "$1/budget-job-config.sh"
[[ "$pool_workers" =~ ^[1-9][0-9]*$ && ${SLURM_NTASKS:-} == "$pool_workers" && -n ${SLURM_JOB_ID:-} ]] || { printf 'Unexpected Slurm allocation\n' >&2; exit 2; }
[[ -s "$run_dir/study-plan.json" && -s "$run_dir/work-items.tsv" ]] || { printf 'Missing study plan\n' >&2; exit 2; }
mkdir -p "$run_dir/logs/steps"

finish() {
    status=$?
    trap - EXIT
    printf 'Pool exit code %s at %s\n' "$status" "$(date -u +%FT%TZ)"
    printf '%s\n' "$status" > "$run_dir/pool-exit-code.txt"
    if ! rsync -a --exclude='source/' --exclude='tmp/' --exclude='matplotlib/' "$run_dir/" "$archive_dir/"; then
        printf 'Archive failed; recover data from %s\n' "$run_dir" >&2
        [[ "$status" -ne 0 ]] || status=74
    fi
    printf '%s\n' "$status" > "$run_dir/pool-exit-code.txt"
    if ! cp "$run_dir/pool-exit-code.txt" "$archive_dir/pool-exit-code.txt"; then
        [[ "$status" -ne 0 ]] || status=74
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM

if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge
module load gcc/13.3.0 "${SMART_PARALLEL_MODULE:-parallel/20240522}"
command -v parallel >/dev/null || { printf 'GNU Parallel is unavailable\n' >&2; exit 2; }
parallel --plain --version > "$run_dir/logs/parallel-version.txt"
cd "$run_dir"
printf 'stage\texit_code\n' > "$run_dir/stage-status.tsv"
overall_status=0
# --quote handles shell characters; fixed paths also avoid interpreting braces
# in a custom root as GNU Parallel replacement strings. No resume/retry occurs.
set +e
parallel --plain --quote --jobs "$pool_workers" --delay 0.2 --halt never \
    --joblog "$run_dir/logs/parallel-joblog.tsv" --colsep '\t' \
    bash ./source/hpc/discovery/budget_worker.sh dispatch . '{1}' '{2}' '{3}' '{4}' '{5}' \
    :::: "$run_dir/work-items.tsv"
worker_status=$?
set -e
printf 'workers\t%s\n' "$worker_status" >> "$run_dir/stage-status.tsv"
if (( worker_status != 0 )); then overall_status=$worker_status; fi

# Aggregation/audit run even when individual workers fail. They use one process,
# and the scientific summary never fits estimators or invents missing records.
export SMART_SOURCE_ROOT="$run_dir/source"
export MPLCONFIGDIR="$run_dir/matplotlib"
set +e
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh" > "$run_dir/logs/environment.txt" 2>&1
environment_status=$?
if (( environment_status == 0 )); then
    python -m pip list --format=freeze >> "$run_dir/logs/environment.txt" 2>&1
fi
# Collection is metadata-only and can still describe failures if runtime
# validation failed. Never certify a scientific summary under a failed runtime.
"$VENV/bin/python" "$SMART_SOURCE_ROOT/simulation/discovery_budget_study.py" aggregate \
    --run-root "$run_dir" --output-root "$run_dir/results" \
    > "$run_dir/logs/aggregate.log" 2>&1
aggregate_status=$?
if (( environment_status == 0 )); then
    python "$SMART_SOURCE_ROOT/simulation/summarize_sparse_smart_budget_study.py" \
        --result-root "$run_dir/results" --output-root "$run_dir/results/summary" --manifest-scope \
        > "$run_dir/logs/summary.log" 2>&1
    summary_status=$?
else
    summary_status=$environment_status
    printf 'Scientific summary skipped because runtime validation failed; see environment.txt.\n' > "$run_dir/logs/summary.log"
fi
set -e
printf 'environment\t%s\naggregate\t%s\nsummary\t%s\n' "$environment_status" "$aggregate_status" "$summary_status" >> "$run_dir/stage-status.tsv"
if (( overall_status == 0 && environment_status != 0 )); then overall_status=$environment_status; fi
if (( overall_status == 0 && aggregate_status != 0 )); then overall_status=$aggregate_status; fi
if (( overall_status == 0 && summary_status != 0 )); then overall_status=$summary_status; fi
exit "$overall_status"
