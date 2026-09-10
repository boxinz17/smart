#!/bin/bash
#SBATCH --job-name=smart-smoke
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
SMART_REPO=${SMART_REPO:-${SLURM_SUBMIT_DIR:?}}
source "$SMART_REPO/hpc/discovery/env.sh"
SMOKE_ROOT=/scratch1/$USER/smart/smoke-$SLURM_JOB_ID
SMOKE_ARCHIVE=$HOME/smart-results/smoke-$SLURM_JOB_ID
mkdir -p "$SMOKE_ROOT" "$SMOKE_ARCHIVE"
export SMOKE_ROOT
trap 'rc=$?; cp -a "$SMOKE_ROOT/." "$SMOKE_ARCHIVE/" || rc=1; exit "$rc"' EXIT

# Test the new methods and their supporting Python implementation.
# The legacy SMART package supplies data generation; paper methods are not rerun.
cd "$SMART_REPO/bi-smart"
python -m pytest -q tests/test_public_api.py tests/test_estimator.py
cd "$SMART_REPO/sparse-smart"
python -m pytest -q tests/test_estimator.py tests/test_tuning_checkpoints.py

cd "$SMART_REPO/simulation"
python run_restricted_rrr.py 0 0 0 --setting-index 0 --output-root "$SMOKE_ROOT/restricted"
python run_sparse_smart.py 0 0 0 --setting-index 0 --iterations 2 --output-root "$SMOKE_ROOT/sparse"
python - <<'PY'
import json, os, pickle
from pathlib import Path
import numpy as np
root = Path(os.environ['SMOKE_ROOT'])
for method in ('restricted', 'sparse'):
    files = list((root / method).rglob('*.pkl' if method == 'restricted' else '*.json'))
    assert files, f'No result written by {method}'
    for path in files:
        if path.suffix == '.pkl':
            with path.open('rb') as stream:
                result = pickle.load(stream)
        else:
            result = json.loads(path.read_text())
        assert result['success'], (path, result.get('status'))
        assert np.isfinite(result['avg_err']), (path, result.get('avg_err'))
        print('Validated', path, result['status'], result['avg_err'])
PY

printf 'Slurm smoke checks passed on %s at %s\n' "$(hostname)" "$(date -u +%FT%TZ)"
