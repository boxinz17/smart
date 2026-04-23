#!/usr/bin/env python

"""
run_baseline_target_only_linear.py

Run multi-task Ridge, OLS, and Lasso regression on real-world target data.

Usage:
$ python run_baseline_target_only_linear.py <rd_id: 0-99>
"""

import sys
import os
import time
import pickle
import numpy as np
from smart import fit_baseline

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

model_list = ["ridge", "ols", "lasso"]
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
    if len(sys.argv) != 2:
        print("Usage: python run_real_baseline_models.py <rd_id: 0-99>")
        sys.exit(1)

    rd_id = _get_int(sys.argv[1], 0, 99, "rd_id")

    # Load data
    X_target = np.load("processed_data/X_target.npy")
    Y_target = np.load("processed_data/Y_target.npy")

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

    os.makedirs("result", exist_ok=True)

    for model_type in model_list:
        print(f"\nFitting {model_type} baseline, repetition number: {rd_id}")
        tic = time.time()

        if model_type == "ridge" or model_type == "lasso":
            C_hat, _ = fit_baseline(X_train, Y_train, model_type=model_type, alphas=np.logspace(-3, 2, 30))
        else:
            C_hat, _ = fit_baseline(X_train, Y_train, model_type=model_type)

        Y_pred = X_test @ C_hat
        elapsed = time.time() - tic
        frob_err = np.linalg.norm(Y_pred - Y_test, "fro") / np.sqrt(Y_test.shape[0] * Y_test.shape[1])

        print(f"{model_type} Frobenius error: {frob_err:.4f}")
        print(f"{model_type} Elapsed time: {elapsed:.2f} s")

        out_path = f"result/baseline_target_only_{model_type}_rd_id={rd_id}.pkl"
        result = {
            "frob_error": frob_err,
            "elapsed_time_sec": elapsed,
            "method_name": f"TARGET_ONLY_{model_type}",
            "rd_id": rd_id
        }

        with open(out_path, "wb") as f:
            pickle.dump(result, f)

        print(f"Saved result to {out_path}")

