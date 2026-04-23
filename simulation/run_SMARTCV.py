#!/usr/bin/env python
"""run_SMARTCV.py

Batch‑run simulation experiments with the SMART algorithm.

* Mirrors the behaviour of **run_SMART.py** (which benchmarks SMARTSolver),
  but focuses on SMART with cross validation so that:
    • Only **exp1** (varying sample size *n*) and **exp4** (varying noise level
      *sigma0*) are supported.
    • True rank (r) and structural rank (r_s) are **fixed** at 5 and 10 when
      *generating* the synthetic data — they are **not** passed to SMART.
* Results are saved in the same folder structure / pickle format as
  run_SMART.py so downstream analysis scripts keep working.

Usage
-----
$ python run_SMARTCV.py <model_id: 0‑2> <exp_id: 0|3> <rd_seed_id: 0‑99>

  model_id : 0 → model1, 1 → model2, 2 → model3
  exp_id   : 0 → exp1   (vary n)
             3 → exp4   (vary sigma0)
  rd_seed_id: index into data/random_seeds/experiment_seeds.csv (0‑99)

Example
-------
Run the *sigma‑sweep* experiment for model 2 with random‑seed index 5:

$ python run_SMARTCV.py 2 3 5
"""

import sys
import os
import time
import pickle
import numpy as np
# ――― smart package imports ―――
from smart import (
    generate_data,
    evaluate_model_avg_err,
    SMART,
)

# --------------------------------------------------------------------------------------
#  Global experiment configuration (mirrors run_SMART.py)
# --------------------------------------------------------------------------------------

# (p, q) dimensions for each model type
pq_dict = {
    "model1": (100, 50),
    "model2": (150, 100),
    "model3": (300, 200),
}

exp_list = ["exp1", "exp2", "exp3", "exp4"]  # SMART only supports 0 and 3
model_list = list(pq_dict)

# Pre‑generated seeds for reproducibility
experiment_seeds = np.loadtxt(
    "data/random_seeds/experiment_seeds.csv",
    dtype=int, delimiter=",", skiprows=1, usecols=1,
)

# Sample‑size grids (per model) used in exp1 (identical to run_SMART.py)
model_to_n = {
    0: [200, 400, 600, 800, 1000],
    1: [300, 500, 700, 1000, 1200],
    2: [500, 700, 1000, 1200, 1500],
}

# --------------------------------------------------------------------------------------
#  Helper utilities
# --------------------------------------------------------------------------------------

def _get_int(arg: str, min_val: int, max_val: int, label: str) -> int:
    """Parse and range‑check an integer command‑line argument."""
    try:
        val = int(arg)
        if not (min_val <= val <= max_val):
            raise ValueError
        return val
    except ValueError:
        print(f"{label} must be an integer in [{min_val}, {max_val}].")
        sys.exit(1)


def _run_simulation(*, n: int, p: int, q: int, sigma0: float, suffix: str):
    """Generate data, fit SMART, evaluate error, and persist result."""
    print(
        f"Running SMART simulation with n={n}, p={p}, q={q}, "
        f"sigma0={sigma0}, rd_seed_id={rd_seed_id}"
    )

    tic = time.time()

    # ‑‑ Synthetic data ‑‑
    data = generate_data(
        n=n,
        p=p,
        q=q,
        sigma0=sigma0,
        r_star=5,   # true rank (for data generation only)
        r0_star=10, # structural rank for constructing C0 (data generation only)
        random_seed=random_seed,
    )
    X, Y, C_true, C0 = data["X"], data["Y"], data["C_star"], data["C0"]

    # ‑‑ Fit SMART ‑‑
    smartcv = SMART(X, Y, C0=C0)
    smartcv.run_full_selection(
        verbose=True,            # flip to True for detailed logs
        use_optuna_rank=False,    # choose rank by grid search; set True if you prefer Optuna
        fit_kwargs={"use_optuna": True},  # use Optuna when tuning (λ_u, λ_v)
    )

    C_hat = smartcv.get_estimates()["C_hat"]

    avg_err = evaluate_model_avg_err(C_hat, C_true)
    elapsed = time.time() - tic

    print(f"Average Frobenius error: {avg_err:.4f}")
    print(f"Elapsed time: {elapsed:.2f} s")

    # ‑‑ Persist result ‑‑
    result = {
        "C_true": C_true,
        "C_hat": C_hat,
        "avg_err": avg_err,
        "elapsed_time_sec": elapsed,
    }
    out_dir = f"result/{model}/{exp}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = (
        f"{out_dir}/SMARTCV_result_{model}_{exp}_{suffix}_rd_seed_id={rd_seed_id}.pkl"
    )
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved result to {out_path}\n")


# --------------------------------------------------------------------------------------
#  Entry‑point
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python run_SMARTCV.py <model_id: 0-2> <exp_id: 0|3> "
            "<rd_seed_id: 0-99>"
        )
        sys.exit(1)

    model_id   = _get_int(sys.argv[1], 0, 2, "model_id")
    exp_id     = _get_int(sys.argv[2], 0, 3, "exp_id")
    rd_seed_id = _get_int(sys.argv[3], 0, 99, "rd_seed_id")

    if exp_id not in (0, 3):
        print("Only exp_id 0 (exp1) and 3 (exp4) are supported for SMART.")
        sys.exit(1)

    model = model_list[model_id]
    exp   = exp_list[exp_id]
    random_seed = experiment_seeds[rd_seed_id]
    p, q = pq_dict[model]

    print(
        f"Running SMART with model={model}, exp={exp}, rd_seed_id={rd_seed_id}"
    )

    n_grid = model_to_n[model_id]

    if exp == "exp1":
        for n in n_grid:
            _run_simulation(n=n, p=p, q=q, sigma0=0.01, suffix=f"n={n}")

    elif exp == "exp4":
        n_fixed = n_grid[0]  # same convention as run_SMART.py
        for sigma0 in [0.0, 0.01, 0.02, 0.05, 0.1, 0.5]:
            _run_simulation(
                n=n_fixed, p=p, q=q, sigma0=sigma0, suffix=f"sigma0={sigma0}"
            )
