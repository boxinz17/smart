# smart

Reference implementation of **SMART** (a Spectral transfer approach to Multi-tAsk RegTession) from
*Zhao, Kolar, Lv, "SMART: A Spectral Transfer Approach to Multi-Task Learning"
(arXiv preprint [arXiv:2604.20161](https://arxiv.org/abs/2604.20161), 2026)*.
SMART estimates a low-rank target coefficient matrix by transferring a spectral subspace
obtained from a related source task, solved with an ADMM / Riemannian-optimization hybrid.

The top-level companion repository
([github.com/boxinz17/smart](https://github.com/boxinz17/smart)) contains scripts that
reproduce every simulation and real-data result from the paper. This subfolder is the
standalone Python package and can be installed and used on its own.

## Installation

From a clone of the repository:

```bash
cd smart
pip install -e .
```

Or, from PyPI-style wheel build:

```bash
cd smart
pip install .
```

Python 3.9 or newer is required. Runtime dependencies (NumPy, SciPy, scikit-learn,
Matplotlib, joblib, Optuna, autograd, Pymanopt) are resolved automatically.

## Quick start

```python
import numpy as np
from smart import SMART, generate_data, evaluate_model_avg_err

data = generate_data(n=400, p=100, q=50, sigma0=0.01, random_seed=0)
X, Y, C_true, C0 = data["X"], data["Y"], data["C_star"], data["C0"]

model = SMART(X, Y, C0=C0)
model.run_full_selection(
    verbose=False,
    use_optuna_rank=False,
    fit_kwargs={"use_optuna": True, "n_trials": 10},
)
C_hat = model.get_estimates()["C_hat"]
print("Avg Frobenius error:", evaluate_model_avg_err(C_hat, C_true))
```

The `SMART` class orchestrates intrinsic rank estimation (via `RankSelectorRSC`),
cross-validated structural rank selection, and BIC-based tuning of the ADMM
sparsity weights. For low-level control, instantiate `SMARTSolver` directly.

## Public API

| Symbol | Purpose |
|---|---|
| `SMART` | High-level model-selection wrapper (rank + structural ranks + lambdas) |
| `SMARTSolver` | ADMM / Riemannian solver for a fixed configuration |
| `RankSelectorRSC` | Rank estimator based on Bunea–She–Wegkamp (2011) |
| `MultitaskMarginalRegression` | Marginal feature screening used in the single-cell pipeline |
| `generate_data` | Synthetic data generator used in the simulation study |
| `evaluate_model_avg_err`, `extract_svd_subspaces`, `fit_baseline` | Helpers |

## A note on the ADMM inflation factor

`SMARTSolver` has a `gamma` argument controlling the ADMM penalty inflation factor.
The default, `gamma=2.0`, matches the value used for the main simulation and
real-data experiments in the paper. Other values (e.g. `gamma=1.5`) can be passed
explicitly for sensitivity analysis.

## Tests

After installing the package, run the test suite from this folder:

```bash
pytest -q
```

The tests are lightweight synthetic checks and run in a few minutes on a laptop.

## Citation

If you use this code, please cite:

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
