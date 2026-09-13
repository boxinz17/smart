#!/usr/bin/env bash
# No fitting, arrays, raw-state copies or per-worker package installation.
set -euo pipefail
[[ $# == 3 && "$3" =~ ^[1-9][0-9]*$ && -n ${SLURM_JOB_ID:-} ]] || exit 2
run_dir=$(cd -- "$1" && pwd -P)
mode=$2
workers=$3
[[ "$mode" == prepare || "$mode" == pool || "$mode" == summary ]] || exit 2
attempt_dir="$run_dir/campaign/attempts/$mode-attempt-0001"
mkdir -p "$attempt_dir/tmp" "$attempt_dir/matplotlib" "$run_dir/logs/cases"
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$attempt_dir/exit-code.txt"
    printf 'job_id\texit_code\tfinished_utc\n%s\t%s\t%s\n' \
        "$SLURM_JOB_ID" "$status" "$(date -u +%FT%TZ)" > "$attempt_dir/process-status.tsv"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
export TMPDIR="$attempt_dir/tmp" MPLCONFIGDIR="$attempt_dir/matplotlib"
export VENV=${VENV:-/home1/mkolar/envs/smart} SMART_SOURCE_ROOT="$run_dir/source" SLURM_EXPORT_ENV=ALL
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
export PYTHONPATH="$SMART_SOURCE_ROOT/sparse-smart-v2/src:$PYTHONPATH"
cd "$run_dir"
"$VENV/bin/python" - "$run_dir" "$mode" "$workers" <<'PY'
import importlib.util,os,pathlib,sys
root=pathlib.Path(sys.argv[1]);mode=sys.argv[2];workers=int(sys.argv[3])
spec=importlib.util.spec_from_file_location('retro_submit', root/'source/hpc/discovery/submit_sparse_smart_v2_retrospective_bic.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.validate(root)
record=module.core.read(root/f'campaign/attempts/{mode}-attempt-0001/submission.json')
assert record['kind']==mode and record['workers']==workers
assert record['plan_sha256']==module.core.sha(root/'plan.json')
assert record['work_items_sha256']==module.core.sha(root/'work-items.tsv')
assert int(os.environ['SLURM_NTASKS'])==(workers if mode=='pool' else 1)
assert int(os.environ.get('SLURM_CPUS_PER_TASK','1'))==1
PY
runner="$SMART_SOURCE_ROOT/simulation/run_sparse_smart_v2_retrospective_bic.py"
if [[ "$mode" == prepare ]]; then
    "$VENV/bin/python" -m pytest -q "$SMART_SOURCE_ROOT/sparse-smart-v2/tests/test_bic_selection.py" \
        "$SMART_SOURCE_ROOT/simulation/tests/test_sparse_smart_v2_retrospective_bic.py" \
        "$SMART_SOURCE_ROOT/simulation/tests/test_summarize_sparse_smart_v2_retrospective_bic.py" \
        "$SMART_SOURCE_ROOT/simulation/tests/test_sparse_smart_v2_retrospective_bic_launcher.py" \
        --import-mode=importlib -p no:cacheprovider
    "$VENV/bin/python" "$runner" --plan "$run_dir/plan.json" --prepare
    # The first seed/case from each original model/experiment is reused by pool.
    "$VENV/bin/python" - "$run_dir" <<'PY'
import json,pathlib,subprocess,sys,time
root=pathlib.Path(sys.argv[1]);plan=json.loads((root/'plan.json').read_text())
seen=set();rows=[]
for entry in plan['cases']:
    if entry['source_root'] in seen: continue
    seen.add(entry['source_root']);start=time.monotonic()
    subprocess.run([sys.executable,str(root/'source/simulation/run_sparse_smart_v2_retrospective_bic.py'),
                    '--plan',str(root/'plan.json'),'--case-id',entry['case_id']],check=True)
    rows.append({'case_id':entry['case_id'],'elapsed_seconds':time.monotonic()-start})
(root/'preflight.json').write_text(json.dumps({'success':True,'cases':rows},indent=2)+'\n')
PY
elif [[ "$mode" == pool ]]; then
    module load "${SMART_PARALLEL_MODULE:-parallel/20240522}"
    parallel --plain --version > "$attempt_dir/parallel-version.txt"
    parallel --plain --quote --jobs "$workers" --delay 0.1 --halt never \
        --joblog "$attempt_dir/parallel-joblog.tsv" \
        bash ./source/hpc/discovery/sparse_smart_v2_retrospective_bic_worker.sh . '{}' \
        :::: "$run_dir/work-items.tsv"
else
    "$VENV/bin/python" "$SMART_SOURCE_ROOT/simulation/summarize_sparse_smart_v2_retrospective_bic.py" --root "$run_dir"
fi
