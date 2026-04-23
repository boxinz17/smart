# SMART: A Spectral Transfer Approach to Multi-Task Learning

This repository contains the reference implementation and experiment code for

> Boxin Zhao, Mladen Kolar, Jinchi Lv. *SMART: A Spectral Transfer Approach to
> Multi-Task Learning.* arXiv preprint [arXiv:2604.20161](https://arxiv.org/abs/2604.20161), 2026.

SMART estimates a low-rank coefficient matrix for a target multi-task regression
problem by transferring a spectral subspace learned from a related source task.
The estimator is solved with an ADMM / Riemannian-optimization hybrid and is
paired with data-driven rank selection (Bunea–She–Wegkamp RSC) and
cross-validated structural-rank tuning.

The repository is organized into a standalone Python package and two experiment
folders that reproduce the paper end to end:

```
smart/          Standalone, pip-installable Python package (with its own tests)
simulation/     Section 5 — synthetic multi-task regression experiments
single_cell/    Section 6 — CITE-seq transfer-learning case study
```

## Installation

Python 3.9 or newer is required; the experiments in the paper were run with
Python 3.12. We recommend a fresh virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ./smart
pip install -r requirements.txt
```

The first command installs the `smart` package in editable mode; the second
adds the extras used by the experiment scripts (for example `scanpy` and
`anndata` for the single-cell pipeline, and `pytest` for the test suite).

The R baselines need an R installation with the packages listed in
[`R_packages.txt`](R_packages.txt):

```r
install.packages(c("rrpack", "jsonlite", "reticulate", "MASS"))
```

## Tests

The pytest suite lives inside the package so that `smart/` is fully
self-contained:

```bash
cd smart
pytest -q
```

It exercises the data generator, rank selector, ADMM solver, and high-level
`SMART` wrapper with small synthetic inputs, runs in a few minutes on a
laptop, and is wired into GitHub Actions in
[`.github/workflows/tests.yml`](.github/workflows/tests.yml).

## Reproducing the paper

Each experiment folder contains a self-contained README with step-by-step
commands and expected outputs.

- [`simulation/README.md`](simulation/README.md) — Section 5. Covers seed
  generation, the three problem sizes and four sub-experiments, the R
  baselines (RRR, SRRR, RSSVD, SOFAR), the SMART runs
  (`run_SMART.py`, `run_SMARTCV.py`), and the figure scripts.
- [`single_cell/README.md`](single_cell/README.md) — Section 6. Covers the
  raw CITE-seq download from
  [GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122),
  preprocessing, estimation of the seven source matrices, target-only and
  source-only baselines, the SMART sweep, and the final table and boxplot.

Large artifacts — raw data, processed matrices, per-seed result pickles,
generated figures, and HPC logs — are excluded from the repository (see
[`.gitignore`](.gitignore)) and are regenerated deterministically from the
checked-in random seeds.

## Package API in one glance

```python
from smart import SMART, generate_data, evaluate_model_avg_err

data = generate_data(n=400, p=100, q=50, sigma0=0.01, random_seed=0)
X, Y, C_true, C0 = data["X"], data["Y"], data["C_star"], data["C0"]

model = SMART(X, Y, C0=C0)
model.run_full_selection(fit_kwargs={"use_optuna": True, "n_trials": 10})
C_hat = model.get_estimates()["C_hat"]
print(evaluate_model_avg_err(C_hat, C_true))
```

See [`smart/README.md`](smart/README.md) for a full tour of the public API
(`SMART`, `SMARTSolver`, `RankSelectorRSC`, `MultitaskMarginalRegression`,
`generate_data`, `evaluate_model_avg_err`, `extract_svd_subspaces`,
`fit_baseline`).

## HPC submission scripts

The `simulation/submit_*.sh` and `single_cell/submit_*.sh` files are Slurm
array examples we used on a cluster. They are kept as convenient templates;
the main replication path documented in the experiment READMEs is plain
shell + Python from a virtual environment, and the Slurm files can be
adapted or ignored.

Before running them, replace the placeholder values (`--account=YOUR_ACCOUNT`,
the `module load` names, and the `${VENV:-$HOME/SMART_experiments}` path) with
whatever matches your cluster. The experiment READMEs spell this out in
detail.

## Citation

```bibtex
@article{zhao2026smart,
  title         = {SMART: A Spectral Transfer Approach to Multi-Task Learning},
  author        = {Zhao, Boxin and Kolar, Mladen and Lv, Jinchi},
  year          = {2026},
  eprint        = {2604.20161},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2604.20161},
  url           = {https://arxiv.org/abs/2604.20161},
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for the full text.
