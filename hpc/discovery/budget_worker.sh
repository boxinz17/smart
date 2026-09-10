#!/usr/bin/env bash
# dispatch runs on the batch node; execute runs once on a Slurm-assigned CPU.
set -euo pipefail
[[ $# == 7 && ( "$1" == dispatch || "$1" == execute ) && -f "$2/budget-job-config.sh" ]] || { printf 'Expected dispatch|execute RUN_DIR TASK MODEL EXPERIMENT SEED SETTING.\n' >&2; exit 2; }
worker_mode=$1
source "$2/budget-job-config.sh"
task_id=$3 model=$4 experiment=$5 seed=$6 setting=$7
[[ -n ${SLURM_JOB_ID:-} && "$model" =~ ^[0-2]$ && "$experiment" =~ ^[0-3]$ && "$seed" =~ ^(0|[1-9][0-9]?)$ && "$setting" =~ ^[0-6]$ ]] || { printf 'Invalid allocation or work-item identity\n' >&2; exit 2; }
[[ "$task_id" == "m${model}_e${experiment}_s${seed}_k${setting}" ]] || { printf 'Task identity mismatch\n' >&2; exit 2; }
task_dir="$run_dir/tasks/$task_id"
mkdir -p "$task_dir" "$run_dir/logs/steps"

archive_task() {
    mkdir -p "$archive_dir/tasks/$task_id" && \
        rsync -a --exclude='tmp/' --exclude='matplotlib/' "$task_dir/" "$archive_dir/tasks/$task_id/"
}
if [[ "$worker_mode" == dispatch ]]; then
    exec >> "$run_dir/logs/steps/$task_id.out" 2>> "$run_dir/logs/steps/$task_id.err"
    dispatch_finish() {
        status=$?
        trap - EXIT
        printf '%s\n' "$status" > "$task_dir/launcher-exit-code.txt"
        if ! archive_task; then
            printf 'Could not archive launch outcome for %s\n' "$task_id" >&2
            printf 'failed\n' > "$task_dir/archive-status.txt"
            [[ "$status" -ne 0 ]] || status=74
        fi
        printf '%s\n' "$status" > "$task_dir/launcher-exit-code.txt"
        if ! cp "$task_dir/launcher-exit-code.txt" "$archive_dir/tasks/$task_id/"; then
            printf 'failed\n' > "$task_dir/archive-status.txt"
            [[ "$status" -ne 0 ]] || status=74
            printf '%s\n' "$status" > "$task_dir/launcher-exit-code.txt"
        fi
        exit "$status"
    }
    trap dispatch_finish EXIT
    trap 'exit 143' TERM
    printf 'Dispatch %s; Parallel sequence %s at %s\n' "$task_id" "${PARALLEL_SEQ:-unknown}" "$(date -u +%FT%TZ)"
    # The allocation spans arbitrary nodes; each step uses one CPU on one node.
    # Inherit SLURM_MEM_PER_CPU; do not start local multiprocessing per step.
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --kill-on-bad-exit=0 \
        bash "$run_dir/source/hpc/discovery/budget_worker.sh" execute "$run_dir" \
        "$task_id" "$model" "$experiment" "$seed" "$setting" </dev/null
    exit 0
fi

exec >> "$task_dir/slurm.out" 2>> "$task_dir/slurm.err"
if ! mkdir "$task_dir/started.lock"; then
    printf 'Task already started; use a fresh submission rather than an implicit retry.\n' >&2
    exit 75
fi
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$task_dir/process-exit-code.txt"
    printf '%s\n' "$status" > "$task_dir/exit-code.txt"
    printf 'Finished %s with process exit %s at %s\n' "$task_id" "$status" "$(date -u +%FT%TZ)"
    if archive_task; then
        printf 'complete\n' > "$task_dir/archive-status.txt"
    else
        printf 'failed\n' > "$task_dir/archive-status.txt"
        printf 'Archive failed; recover task from %s\n' "$task_dir" >&2
        [[ "$status" -ne 0 ]] || status=74
    fi
    printf '%s\n' "$status" > "$task_dir/exit-code.txt"
    if ! cp "$task_dir/exit-code.txt" "$task_dir/archive-status.txt" "$archive_dir/tasks/$task_id/"; then
        printf 'failed\n' > "$task_dir/archive-status.txt"
        [[ "$status" -ne 0 ]] || status=74
        printf '%s\n' "$status" > "$task_dir/exit-code.txt"
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
mkdir -p "$task_dir/tmp"
export TMPDIR="$task_dir/tmp" MPLCONFIGDIR="$task_dir/matplotlib"
export SMART_SOURCE_ROOT="$run_dir/source"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
cd "$task_dir"
command_args=(python "$SMART_SOURCE_ROOT/simulation/run_sparse_smart_budget_study.py"
    --models "$model" --experiments "$experiment" --seed-ids "$seed" --setting-index "$setting"
    --profile full --workers 1 --output-root "$task_dir/results"
    --seed-file "$SMART_SOURCE_ROOT/simulation/data/random_seeds/experiment_seeds.csv"
    "${budget_args[@]}")
{
    printf 'Job: %s\nStep: %s\nHost: %s\nParallel sequence: %s\nTask: %s\n' "$SLURM_JOB_ID" "${SLURM_STEP_ID:-unknown}" "$(hostname)" "${PARALLEL_SEQ:-unknown}" "$task_id"
    printf 'Source: %s\nPython: ' "$SMART_SOURCE_ROOT"; python --version
    printf 'Command: '; printf '%q ' "${command_args[@]}"; printf '\n'
} > "$task_dir/environment.txt"
"${command_args[@]}"
