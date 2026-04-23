#!/usr/bin/env python

"""
run_real_SMARTCV.py

Run SMARTCV on real-world target data using source matrices obtained by
RRR, SRRR, RSSVD, SOFAR, Ridge, OLS, or Lasso.

Usage
-----
$ python run_real_SMARTCV.py <method_id: 0-6> <rd_id: 0-99>

method_id:
  0 -> RRR
  1 -> SRRR
  2 -> RSSVD
  3 -> SOFAR
  4 -> Ridge
  5 -> OLS
  6 -> Lasso

rd_id:
  Random seed ID (0-99), used for splitting train/test set.

Example
-------
Run SMARTCV using Lasso source matrix, repetition 7:
$ python run_real_SMARTCV.py 6 7
"""

import sys
import os
import time
import pickle
import numpy as np
from smart import SMART

# --------------------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------------------

source_matrix_dict = {
    0: "source_matrix/source_matrix_RRR.npy",
    1: "source_matrix/source_matrix_SRRR.npy",
    2: "source_matrix/source_matrix_RSSVD.npy",
    3: "source_matrix/source_matrix_SOFAR.npy",
    4: "source_matrix/source_matrix_ridge.npy",
    5: "source_matrix/source_matrix_ols.npy",
    6: "source_matrix/source_matrix_lasso.npy",
}

method_name_dict = {
    0: "RRR",
    1: "SRRR",
    2: "RSSVD",
    3: "SOFAR",
    4: "Ridge",
    5: "OLS",
    6: "Lasso",
}

seed_list_path = "./random_seeds/realdata_seeds.csv"

# --------------------------------------------------------------------------------------
# Helper function
# --------------------------------------------------------------------------------------

def _get_int(arg: str, min_val: int, max_val: int, label: str) -> int:
    try:
        val = int(arg)
        if not (min_val <= val <= max_val):
            raise ValueError
        return val
    except ValueError:
        print(f"{label} must be an integer in [{min_val}, {max_val}].")
        sys.exit(1)

# --------------------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_real_SMARTCV.py <method_id: 0-6> <rd_id: 0-99>")
        sys.exit(1)

    method_id = _get_int(sys.argv[1], 0, 6, "method_id")
    rd_id = _get_int(sys.argv[2], 0, 99, "rd_id")

    method_name = method_name_dict[method_id]
    source_matrix_path = source_matrix_dict[method_id]

    # Load data
    X_target = np.load("processed_data/X_target.npy")
    Y_target = np.load("processed_data/Y_target.npy")
    C0 = np.load(source_matrix_path)

    # Load seed
    if not os.path.exists(seed_list_path):
        raise FileNotFoundError(f"Random seed file not found: {seed_list_path}")

    seed_list = np.loadtxt(seed_list_path, dtype=int, delimiter=",", skiprows=1, usecols=1)
    random_seed = seed_list[rd_id]

    np.random.seed(random_seed)
    n_samples = X_target.shape[0]
    idx = np.random.permutation(n_samples)

    n_train = int(0.7 * n_samples)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_train, Y_train = X_target[train_idx], Y_target[train_idx]
    X_test, Y_test = X_target[test_idx], Y_target[test_idx]

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")

    # Load estimated rank for target data
    r_hat_path = "./processed_data/r_hat_target.csv"
    if not os.path.exists(r_hat_path):
        raise FileNotFoundError(f"Target rank file not found: {r_hat_path}")

    r_hat_target = int(np.loadtxt(r_hat_path, delimiter=","))
    print(f"Loaded fixed target rank (r_hat): {r_hat_target}")

    # Fit SMARTCV
    print(f"Fitting SMARTCV using source method {method_name}, repetition number: {rd_id}")
    tic = time.time()

    smartcv = SMART(X_train, Y_train, C0=C0)
    smartcv.run_full_selection(
        verbose=True,
        use_optuna_rank=False,
        fit_kwargs={"use_optuna": True},
        fixed_rank=r_hat_target
    )

    elapsed = time.time() - tic
    print("SMARTCV fitting finished!")

    C_hat = smartcv.get_estimates()["C_hat"]
    Y_pred = X_test @ C_hat
    frob_err = np.linalg.norm(Y_pred - Y_test, "fro") / np.sqrt(Y_test.shape[0] * Y_test.shape[1])

    print(f"Frobenius error: {frob_err:.4f}")
    print(f"Elapsed time: {elapsed:.2f} s")

    # Save result
    os.makedirs("result", exist_ok=True)
    out_path = f"result/SMARTCV_realdata_{method_name}_rd_id={rd_id}.pkl"

    result = {
        "frob_error": frob_err,
        "elapsed_time_sec": elapsed,
        "method_name": method_name,
        "rd_id": rd_id,
    }

    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Saved result to {out_path}")