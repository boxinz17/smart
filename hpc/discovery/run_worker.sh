#!/usr/bin/env bash
# Execute one isolated seed or seed/setting item inside an allocated Slurm task.
# This script must not create a nested srun step.
set -euo pipefail

[[ $# == 4 && -f "$1/job-config.sh" ]] || { printf 'Expected RUN_DIR TASK_ID SEED SETTING.\n' >&2; exit 2; }
source "$1/job-config.sh"
task_id=$2 seed=$3 setting=$4
[[ "$seed" =~ ^[0-9]+$ && "$seed" -le 99 ]] || { printf 'Invalid seed ID\n' >&2; exit 2; }
if [[ "$setting" == - ]]; then
    [[ "$task_id" == "$seed" ]] || { printf 'Invalid seed task identity\n' >&2; exit 2; }
else
    [[ "$setting" =~ ^[0-9]+$ && "$task_id" == "s${seed}_k${setting}" ]] || { printf 'Invalid setting task identity\n' >&2; exit 2; }
fi
task_dir="$run_dir/tasks/$task_id"
mkdir -p "$task_dir"
exec >> "$task_dir/slurm.out" 2>> "$task_dir/slurm.err"

finish() {
    status=$?
    archive_status=0
    trap - EXIT
    printf 'Job %s item %s finished with process exit code %s at %s\n' "${SLURM_JOB_ID:-unknown}" "$task_id" "$status" "$(date -u +%FT%TZ)"
    printf '%s\n' "$status" > "$task_dir/exit-code.txt"
    if ! mkdir -p "$archive_dir/tasks/$task_id"; then
        printf 'failed\n' > "$task_dir/archive-status.txt"
        printf 'Cannot create archive directory; recover this item from %s\n' "$task_dir" >&2
        [[ "$status" -ne 0 ]] || status=74
        exit "$status"
    fi
    # Keep result paths relative to the private source tree. Include partial
    # outputs and logs on failure without duplicating source in the archive.
    if ! rsync -a --exclude='source/' --exclude='matplotlib/' "$task_dir/" "$archive_dir/tasks/$task_id/"; then
        archive_status=74
        printf 'Archive copy failed; recover this item from %s\n' "$task_dir" >&2
        [[ "$status" -ne 0 ]] || status=74
    fi
    if [[ -d "$task_dir/source/simulation/result" ]]; then
        if ! rsync -a "$task_dir/source/simulation/result/" "$archive_dir/tasks/$task_id/result/"; then
            archive_status=74
            printf 'Result archive copy failed; recover results from %s\n' "$task_dir/source/simulation/result" >&2
            [[ "$status" -ne 0 ]] || status=74
        fi
    fi
    if [[ "$archive_status" == 0 ]]; then
        printf 'complete\n' > "$task_dir/archive-status.txt"
    else
        printf 'failed\n' > "$task_dir/archive-status.txt"
    fi
    if ! cp "$task_dir/archive-status.txt" "$archive_dir/tasks/$task_id/archive-status.txt"; then
        printf 'failed\n' > "$task_dir/archive-status.txt"
        printf 'Could not persist archive status; recover this item from %s\n' "$task_dir" >&2
        [[ "$status" -ne 0 ]] || status=74
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM

printf 'Start UTC: %s\nJob: %s\nStep: %s\nHost: %s\nTask: %s\nSeed ID: %s\nSetting: %s\nWorker rank: %s\nParallel sequence: %s\n' \
    "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-unknown}" "${SLURM_STEP_ID:-unknown}" "$(hostname)" "$task_id" "$seed" "$setting" "${SLURM_PROCID:-unknown}" "${PARALLEL_SEQ:-none}"
rsync -a "$run_dir/source/" "$task_dir/source/"
export SMART_SOURCE_ROOT="$task_dir/source"
export MPLCONFIGDIR="$task_dir/matplotlib"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"

args=()
while IFS= read -r -d '' argument; do args+=("$argument"); done < "$run_dir/invocation.args"
[[ ${#args[@]} -ge 3 ]] || { printf 'Missing invocation metadata\n' >&2; exit 2; }
runner=${args[0]} model=${args[1]} experiment=${args[2]}
cd "$SMART_SOURCE_ROOT/simulation"
for model_dir in model1 model2 model3; do
    for experiment_dir in exp1 exp2 exp3 exp4; do mkdir -p "result/$model_dir/$experiment_dir"; done
done
command_args=(python "$runner" "$model" "$experiment" "$seed")
if [[ "$setting" != - ]]; then command_args+=(--setting-index "$setting"); fi
command_args+=("${args[@]:3}")
{
    printf 'Working directory: %s\n' "$PWD"
    printf 'Job: %s\nStep: %s\nParallel sequence: %s\nWorker rank: %s\nNode index: %s\nHost: %s\n' \
        "${SLURM_JOB_ID:-unknown}" "${SLURM_STEP_ID:-unknown}" "${PARALLEL_SEQ:-none}" "${SLURM_PROCID:-unknown}" "${SLURM_NODEID:-unknown}" "$(hostname)"
    printf 'Command: '; printf '%q ' "${command_args[@]}"; printf '\n'
    printf 'Python: '; python --version
    printf 'Modules:\n'; module list 2>&1
    # This reads the existing environment; it does not install packages.
    printf 'Installed Python distributions:\n'; python -m pip list --format=freeze
} > "$task_dir/environment.txt"
printf 'Running: '; printf '%q ' "${command_args[@]}"; printf '\n'
"${command_args[@]}"
