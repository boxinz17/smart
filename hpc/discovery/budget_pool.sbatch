#!/usr/bin/env bash
# Fit-only allocation; one single-CPU step per planned tuning subset.
set -euo pipefail
[[ $# == 1 && -f "$1/budget-job-config.sh" ]] || { printf 'Use submit_budget_study.sh.\n' >&2; exit 2; }
source "$1/budget-job-config.sh"
[[ "$pool_workers" =~ ^[1-9][0-9]*$ && ${SLURM_NTASKS:-} == "$pool_workers" && -n ${SLURM_JOB_ID:-} ]] || { printf 'Unexpected Slurm allocation\n' >&2; exit 2; }
[[ -s "$run_dir/study-plan.json" && -s "$run_dir/work-items.tsv" ]] || { printf 'Missing study plan\n' >&2; exit 2; }
mkdir -p "$run_dir/logs/steps"

finish() {
    status=$?
    trap - EXIT
    printf 'Fit-only pool exit code %s at %s; postprocessing and archival deferred.\n' "$status" "$(date -u +%FT%TZ)"
    printf '%s\n' "$status" > "$run_dir/pool-exit-code.txt"
    if (( status == 0 )); then
        printf 'finished\n' > "$run_dir/fit-status.txt"
    else
        printf 'failed\n' > "$run_dir/fit-status.txt"
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM

printf 'fit-only\n' > "$run_dir/pipeline-mode.txt"
printf 'running\n' > "$run_dir/fit-status.txt"
printf 'deferred\n' > "$run_dir/postprocessing-status.txt"
printf 'deferred\n' > "$run_dir/archive-status.txt"
printf 'stage\texit_code\n' > "$run_dir/stage-status.tsv"

if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge
module load gcc/13.3.0 "${SMART_PARALLEL_MODULE:-parallel/20240522}"
command -v parallel >/dev/null || { printf 'GNU Parallel is unavailable\n' >&2; exit 2; }
parallel --plain --version > "$run_dir/logs/parallel-version.txt"
cd "$run_dir"
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
exit "$worker_status"
