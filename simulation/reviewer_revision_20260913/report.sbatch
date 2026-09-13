#!/usr/bin/env bash
# Report-only overlay: ROOT STUDY_DIR RUNTIME_DIR OUTPUT_PREFIX REPORT_SCRIPT.
set -euo pipefail
if [[ $# -ne 5 ]]; then
  printf 'Usage: report.sbatch ROOT STUDY_DIR RUNTIME_DIR OUTPUT_PREFIX REPORT_SCRIPT\n' >&2
  exit 2
fi
run_root=$1
study_dir=$2
runtime_dir=$3
output_prefix=$4
report_script=$5
export SMART_SOURCE_ROOT="$run_root/source"
export VENV=/home1/mkolar/envs/smart
export TMPDIR="$run_root/tmp/${SLURM_JOB_ID:?Report launcher requires Slurm}"
mkdir -p "$TMPDIR"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
exec "$VENV/bin/python" "$report_script" "$study_dir" \
  --runtime-root "$runtime_dir" --output-prefix "$output_prefix"
