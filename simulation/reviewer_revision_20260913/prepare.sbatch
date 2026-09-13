#!/usr/bin/env bash
set -euo pipefail
run_dir=$1
seeds=$2
iterations=$3
pilot=${4:-false}
export SMART_SOURCE_ROOT="$run_dir/source"
export VENV=/home1/mkolar/envs/smart
export TMPDIR="$run_dir/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$run_dir/logs"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
export PYTHONPATH="$SMART_SOURCE_ROOT/simulation:$SMART_SOURCE_ROOT/sparse-smart-v2/src:$PYTHONPATH"
module load r/4.4.3
export R_LIBS_USER="$VENV/R/library"
Rscript -e 'if (!requireNamespace("corpcor", quietly=TRUE)) install.packages("corpcor", repos="https://cloud.r-project.org"); stopifnot(requireNamespace("corpcor",quietly=TRUE),requireNamespace("jsonlite",quietly=TRUE))'
cd "$SMART_SOURCE_ROOT"
# Guard fixtures deliberately test execution outside Slurm. The test process
# remains on this allocated compute node but must not inherit the job marker.
env -u SLURM_JOB_ID "$VENV/bin/python" -m pytest simulation/reviewer_revision_20260913/tests -q
"$VENV/bin/python" -m reviewer_revision_20260913.published_park --sanity --output "$run_dir/park-sanity.json"
"$VENV/bin/python" -m reviewer_revision_20260913.published_park --sanity-validation --output "$run_dir/park-validation-sanity.json"
args=()
if [[ "$pilot" == true ]]; then args+=(--pilot); fi
"$VENV/bin/python" -m reviewer_revision_20260913.runner plan "$run_dir" --seeds "$seeds" --iterations "$iterations" "${args[@]}"
"$VENV/bin/python" - "$run_dir" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
manifest=json.loads((root/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==digest,name
(root/'preparation.json').write_text(json.dumps({'complete':True,'verified_files':len(manifest['files'])}))
PY
