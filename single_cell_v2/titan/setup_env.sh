#!/bin/bash
# Create the Titan environment for code/single_cell_v2 (run once, on the login node).
#
#   bash code/single_cell_v2/titan/setup_env.sh
#
# A conda environment from the user's miniforge supplies Python 3.12 and R
# (with corpcor and jsonlite for the Park et al. bridge). Python packages are
# then installed with pip under the repository's version pins, matching the
# local environment, plus editable installs of the three project packages.
set -euo pipefail
cd /shared/home/mladen.kolar/SMART_BoxinJinchi
ENV=/shared/home/mladen.kolar/miniforge3/envs/smart-singlecell
if [ ! -x "$ENV/bin/python" ]; then
    ~/miniforge3/bin/mamba create -y -p "$ENV" -c conda-forge python=3.12 r-base r-corpcor r-jsonlite pip
fi
PY="$ENV/bin/python"
"$PY" -m pip install -q -c code/python-constraints.txt setuptools wheel
"$PY" -m pip install -q -c code/python-constraints.txt numpy scipy scikit-learn pandas scanpy anndata h5py
"$PY" -m pip install -q --no-build-isolation -c code/python-constraints.txt \
    -e ./code/smart -e ./code/sparse-smart -e ./code/sparse-smart-v2
"$PY" -m pip check
"$PY" -c "import numpy, scipy, sklearn, scanpy, anndata, smart, sparse_smart, sparse_smart_v2; \
print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'sklearn', sklearn.__version__, \
'scanpy', scanpy.__version__, 'anndata', anndata.__version__)"
"$ENV/bin/Rscript" -e 'cat("R", R.version$major, R.version$minor, "corpcor", as.character(packageVersion("corpcor")), "jsonlite", as.character(packageVersion("jsonlite")), "\n")'
"$PY" -m pip freeze > code/single_cell_v2/titan/environment-freeze.txt
