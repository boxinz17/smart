#!/usr/bin/env bash
# Dispatch on the batch node; execute inside one exclusive single-CPU step.
set -euo pipefail
[[ $# == 3 && ( "$1" == dispatch || "$1" == execute ) && "$3" =~ ^(0|[1-9][0-9]*)$ && -n ${SLURM_JOB_ID:-} ]] || { printf 'Expected dispatch|execute RUN_DIR INTEGER_TASK in Slurm.\n' >&2; exit 2; }
mode=$1
run_dir=$(cd -- "$2" && pwd -P)
task_id=$3
printf -v task_name '%06d' "$task_id"
task_dir="$run_dir/tasks/$task_name"
[[ -s "$run_dir/preparation.json" ]] || { printf 'Missing preparation marker.\n' >&2; exit 2; }
mkdir -p "$task_dir" "$run_dir/logs/steps"
if [[ "$mode" == dispatch ]]; then
    exec >> "$run_dir/logs/steps/$task_name.out" 2>> "$run_dir/logs/steps/$task_name.err"
    finish_dispatch() {
        status=$?
        trap - EXIT
        printf '%s\n' "$status" > "$task_dir/launcher-exit-code.txt"
        printf 'job_id\tparallel_sequence\texit_code\tfinished_utc\n%s\t%s\t%s\t%s\n' \
            "$SLURM_JOB_ID" "${PARALLEL_SEQ:-unknown}" "$status" "$(date -u +%FT%TZ)" > "$task_dir/launcher-status.tsv"
        exit "$status"
    }
    trap finish_dispatch EXIT
    trap 'exit 143' TERM
    printf 'Dispatch task %s at %s\n' "$task_id" "$(date -u +%FT%TZ)"
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --kill-on-bad-exit=0 \
        bash "$run_dir/source/hpc/discovery/sparse_smart_v2_bic_worker.sh" execute "$run_dir" "$task_id" </dev/null
    exit 0
fi

[[ ${SLURM_CPUS_PER_TASK:-1} == 1 && -n ${SLURM_STEP_ID:-} ]] || { printf 'Expected a single-CPU Slurm step.\n' >&2; exit 2; }
exec >> "$task_dir/slurm.out" 2>> "$task_dir/slurm.err"
finish_process() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$task_dir/process-exit-code.txt"
    printf 'job_id\tstep_id\tpid\texit_code\tfinished_utc\n%s\t%s\t%s\t%s\t%s\n' \
        "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$$" "$status" "$(date -u +%FT%TZ)" > "$task_dir/process-status.tsv"
    exit "$status"
}
trap finish_process EXIT
trap 'exit 143' TERM
mkdir -p "$task_dir/tmp" "$task_dir/matplotlib"
export TMPDIR="$task_dir/tmp" MPLCONFIGDIR="$task_dir/matplotlib"
export VENV=${VENV:-/home1/mkolar/envs/smart} SMART_SOURCE_ROOT="$run_dir/source"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
export PYTHONPATH="$SMART_SOURCE_ROOT/sparse-smart-v2/src:$PYTHONPATH"
cd "$task_dir"
command_args=("$VENV/bin/python" "$SMART_SOURCE_ROOT/simulation/run_sparse_smart_v2_bic.py" fit --root "$run_dir" --task "$task_id")
{
    printf 'Job: %s\nStep: %s\nHost: %s\nTask: %s\nParallel sequence: %s\n' \
        "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$(hostname)" "$task_id" "${PARALLEL_SEQ:-unknown}"
    printf 'Source: %s\nTemporary directory: %s\nPython path: %s\nCommand: ' "$SMART_SOURCE_ROOT" "$TMPDIR" "$PYTHONPATH"
    printf '%q ' "${command_args[@]}"
    printf '\n'
} > "$task_dir/environment.txt"
"${command_args[@]}"
