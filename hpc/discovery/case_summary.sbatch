#!/usr/bin/env bash
# Submitted afterany:ARRAY so failures and missing cases remain in the report.
set -euo pipefail
[[ $# -eq 3 ]] || { printf 'Expected source root, launch manifest, and SHA256\n' >&2; exit 2; }
export SMART_SOURCE_ROOT=$1
LAUNCH=$2
LAUNCH_SHA=$3
[[ ${SLURM_NTASKS:-} == 1 && ${SLURM_CPUS_PER_TASK:-} == 1 ]] || { printf 'Expected one task and one CPU\n' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH
python3 -B -S "$SMART_SOURCE_ROOT/simulation/discovery_case_aggregation.py" verify --launch "$LAUNCH" --expected-sha "$LAUNCH_SHA"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
exec python -B "$SMART_SOURCE_ROOT/simulation/discovery_case_aggregation.py" summary \
    --launch "$LAUNCH" --expected-sha "$LAUNCH_SHA"
