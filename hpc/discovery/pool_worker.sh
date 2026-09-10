#!/usr/bin/env bash
# GNU Parallel calls this wrapper once per manifest row on the batch node.
# srun places that single item on a free CPU anywhere in the job allocation.
set -euo pipefail

[[ $# == 4 && -f "$1/job-config.sh" ]] || { printf 'Expected RUN_DIR TASK_ID SEED SETTING.\n' >&2; exit 2; }
source "$1/job-config.sh"
[[ -n ${SLURM_JOB_ID:-} ]] || { printf 'Expected an existing Slurm allocation\n' >&2; exit 2; }
task_id=$2 seed=$3 setting=$4
[[ "$seed" =~ ^(0|[1-9][0-9]?)$ ]] || { printf 'Invalid seed ID\n' >&2; exit 2; }
if [[ "$setting" == - ]]; then
    [[ "$task_id" == "$seed" ]] || { printf 'Invalid seed task identity\n' >&2; exit 2; }
else
    [[ "$setting" =~ ^[0-6]$ && "$task_id" == "s${seed}_k${setting}" ]] || { printf 'Invalid setting task identity\n' >&2; exit 2; }
fi
mkdir -p "$run_dir/logs/steps"
# Capture srun errors even if the simulation worker never starts. The pool's
# exit trap archives these logs and the GNU Parallel job log.
exec >> "$run_dir/logs/steps/$task_id.out" 2>> "$run_dir/logs/steps/$task_id.err"
printf 'Dispatch item %s (seed %s, setting %s), Parallel sequence %s at %s\n' \
    "$task_id" "$seed" "$setting" "${PARALLEL_SEQ:-unknown}" "$(date -u +%FT%TZ)"
# Limit each step to one CPU on one node; the allocation may span many nodes.
# SLURM_MEM_PER_CPU is inherited from the pool's sbatch memory request.
# exec propagates srun's status to GNU Parallel without hiding failures.
exec srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --kill-on-bad-exit=0 \
    bash "$run_dir/source/hpc/discovery/run_worker.sh" "$run_dir" "$task_id" "$seed" "$setting" </dev/null
