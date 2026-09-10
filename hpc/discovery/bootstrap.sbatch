#!/bin/bash
#SBATCH --job-name=smart-setup
#SBATCH --account=mkolar_1314
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Submit from the repository root, with --output pointing to an existing directory.
# Installation only: never invokes simulations, test suites, or comparisons.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
SMART_REPO=${SMART_REPO:-${SLURM_SUBMIT_DIR:?}}
VENV=${VENV:-$HOME/envs/smart}
SMART_SETUP_LOGS=${SMART_SETUP_LOGS:-$HOME/smart-results/setup}
export SMART_SETUP_LOGS
mkdir -p "$SMART_SETUP_LOGS" "${VENV%/*}"
module purge
module load gcc/13.3.0 python/3.12.8 "${SMART_R_MODULE:-r/4.4.3}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
python3 "$SMART_REPO/environment/check_runtime.py" --root "$SMART_REPO" --python-only
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export TZ=UTC
export TMPDIR=/scratch1/$USER/smart/setup/tmp-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --constraint "$SMART_REPO/python-constraints.txt" setuptools wheel
python -m pip install --no-build-isolation --constraint "$SMART_REPO/python-constraints.txt" \
    -e "$SMART_REPO/smart[test]" -e "$SMART_REPO/bi-smart[test]" -e "$SMART_REPO/sparse-smart[test]"
python -m pip install --constraint "$SMART_REPO/python-constraints.txt" pdfplumber
python -m pip check
python "$SMART_REPO/environment/check_runtime.py" --root "$SMART_REPO"
python -m pip list --format=freeze > "$SMART_SETUP_LOGS/python-packages-$SLURM_JOB_ID.txt"
python -m pip freeze --exclude-editable > "$SMART_SETUP_LOGS/python-requirements-$SLURM_JOB_ID.txt"

# Record installed distributions without importing or running project code.
cd "$TMPDIR"
python - <<'PY'
from importlib.metadata import version
for name in ('smart-transfer', 'bi-smart', 'sparse-smart', 'pdfplumber'):
    print(name, version(name))
PY

export R_LIBS_USER=${R_LIBS_USER:-$VENV/R/library}
export RETICULATE_PYTHON=$VENV/bin/python
mkdir -p "$R_LIBS_USER"
Rscript --vanilla - <<'RS'
packages <- c("rrpack", "jsonlite", "MASS", "reticulate")
available <- rownames(installed.packages())
missing <- setdiff(packages, available)
if (length(missing)) install.packages(missing, repos="https://cloud.r-project.org", Ncpus=2)
installed <- installed.packages()
stopifnot(all(packages %in% rownames(installed)))
write.csv(installed[,c("Package","Version")],
          file=file.path(Sys.getenv("SMART_SETUP_LOGS"),
                         paste0("r-packages-", Sys.getenv("SLURM_JOB_ID"), ".csv")), row.names=FALSE)
print(installed[packages,c("Package","Version"),drop=FALSE])
RS
printf 'Package installation completed on %s at %s\n' "$(hostname)" "$(date -u +%FT%TZ)"
