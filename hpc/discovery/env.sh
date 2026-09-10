#!/usr/bin/env bash
# Source this file from a Discovery batch job after setting SMART_SOURCE_ROOT.

SMART_RUNTIME_ROOT=${SMART_SOURCE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)} || return 1
for SMART_RUNTIME_FILE in .python-version python-constraints.txt environment/check_runtime.py; do
    if [[ ! -r "$SMART_RUNTIME_ROOT/$SMART_RUNTIME_FILE" ]]; then
        printf 'Missing shared runtime configuration: %s/%s\n' "$SMART_RUNTIME_ROOT" "$SMART_RUNTIME_FILE" >&2
        return 1
    fi
done

if ! type module >/dev/null 2>&1; then
    if [[ -r /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    else
        printf 'The Discovery module command is unavailable.\n' >&2
        return 1
    fi
fi

module purge || return 1
module load gcc/13.3.0 python/3.12.8 || return 1

export VENV="${VENV:-$HOME/envs/smart}"
if [[ ! -x "$VENV/bin/python" ]]; then
    printf 'Missing Python environment: %s. Run bootstrap.sbatch first.\n' "$VENV" >&2
    return 1
fi
source "$VENV/bin/activate" || return 1

# Import each package from this job's source snapshot, including the src layout.
# Putting the repository root itself on PYTHONPATH would shadow smart's package.
if [[ -n "${SMART_SOURCE_ROOT:-}" ]]; then
    export PYTHONPATH="$SMART_SOURCE_ROOT/smart:$SMART_SOURCE_ROOT/bi-smart:$SMART_SOURCE_ROOT/sparse-smart/src"
else
    unset PYTHONPATH
fi
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# Use one kernel across heterogeneous Discovery nodes. The runtime check below
# verifies AVX2/FMA before importing NumPy and rejects silent BLAS fallback.
export OPENBLAS_CORETYPE=Haswell
# NumPy 2.5.3's audited Linux wheel uses X86_V2 as its compiled baseline.
# Disable every optional dispatch group so node-specific SIMD stays disabled.
export NPY_DISABLE_CPU_FEATURES=X86_V3,X86_V4,AVX512_ICL,AVX512_SPR
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/smart-matplotlib-${USER}-${SLURM_JOB_ID:-interactive}}"
mkdir -p "$MPLCONFIGDIR" || return 1

# Reject an incompatible environment before any worker imports project code.
# Pins and the allowed Python minor come from this job's source snapshot.
"$VENV/bin/python" "$SMART_RUNTIME_ROOT/environment/check_runtime.py" \
    --root "$SMART_RUNTIME_ROOT" --blas-core Haswell || return 1
unset SMART_RUNTIME_ROOT SMART_RUNTIME_FILE
