#!/usr/bin/env bash
# Array resources and controller output paths are supplied by submit.sh.
set -euo pipefail

[[ $# == 1 && -f "$1/job-config.sh" ]] || { printf 'Use hpc/discovery/submit.sh to submit this job.\n' >&2; exit 2; }
source "$1/job-config.sh"
seed=${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}
[[ "$seed" =~ ^[0-9]+$ && "$seed" -le 99 ]] || { printf 'Invalid seed task ID\n' >&2; exit 2; }

srun --ntasks=1 --cpus-per-task=1 \
    bash "$run_dir/source/hpc/discovery/run_worker.sh" "$run_dir" "$seed" "$seed" -
