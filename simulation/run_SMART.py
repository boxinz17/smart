"""
run_SMART.py

This script runs simulation experiments for the SMaRT algorithm with structured low-rank regression.

Usage:
    python run_SMART.py <model_id: 0-2> <exp_id: 0-3> <rd_seed_id: 0-99>

Arguments:
    model_id   : Index specifying the problem dimension.
                 - 0 → model1 (p=100, q=50)
                 - 1 → model2 (p=150, q=100)
                 - 2 → model3 (p=300, q=200)

    exp_id     : Index specifying the experiment type.
                 - 0 → exp1: vary sample size n
                 - 1 → exp2: vary intrinsic rank r
                 - 2 → exp3: vary source subspace rank r_s
                 - 3 → exp4: vary source noise level σ₀

    rd_seed_id : Random seed index from preloaded seed file
                 (data/random_seeds/experiment_seeds.csv)

Outputs:
    - Stores result dictionaries containing true C, estimated C_hat, average Frobenius error,
      and elapsed time.
    - Results are saved as pickled files in `result/` folder, with descriptive filenames.

Example:
    python run_SMART.py 0 1 42
    → Runs exp2 on model1 with random seed ID 42, varying target rank r.
"""

import sys
import time
import pickle
import numpy as np
from smart import generate_data, evaluate_model_avg_err, fit_baseline
from smart import SMARTSolver

# Setup
pq_dict = {'model1': (100, 50), 'model2': (150, 100), 'model3': (300, 200)}
exp_list = ['exp1', 'exp2', 'exp3', 'exp4']
model_list = list(pq_dict)
experiment_seeds = np.loadtxt(
    "data/random_seeds/experiment_seeds.csv",
    dtype=int, delimiter=",", skiprows=1, usecols=1,
)

def get_input(arg, min_val, max_val, label):
    try:
        val = int(arg)
        if not (min_val <= val <= max_val):
            raise ValueError
        return val
    except ValueError:
        print(f"{label} must be between {min_val} and {max_val}")
        sys.exit(1)

def run_simulation(n, p, q, sigma0, r, r_s, suffix):
    print(f"Running simulation with n={n}, p={p}, q={q}, sigma0={sigma0}, r={r}, r_s={r_s}, rd_seed_id={rd_seed_id}")

    start = time.time()
    data = generate_data(n=n, p=p, q=q, sigma0=sigma0, random_seed=random_seed)
    X, Y, C_true, C0 = data["X"], data["Y"], data["C_star"], data["C0"]

    C_ridge, _ = fit_baseline(X, Y, model_type="ridge", alphas=np.logspace(-3, 2, 10))
    print("Ridge fitted")

    solver = SMARTSolver(C0=C0, r_u=r_s, r_v=r_s, t_max=500)
    solver.initialize(r=r, X=X, Y=Y, C=C_ridge)
    solution = solver.select_hyperparameters_via_bic(C_init=C_ridge, use_optuna=True, n_trials=10)
    C_smart = solution['C_hat']
    err = evaluate_model_avg_err(C_smart, C_true)
    elapsed = time.time() - start

    print(f"\n=== Average Frobenius Norm Errors ===")
    print(f"SMART (BIC): {err:.4f}")
    print(f"Elapsed time: {elapsed:.2f}s")

    result = {
        'C_true': C_true, 'C_hat': C_smart,
        'avg_err': err, 'elapsed_time_sec': elapsed
    }
    path = f"result/{model}/{exp}/SMART_result_{model}_{exp}_{suffix}_rd_seed_id={rd_seed_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved result to {path}\n")

# === Parse command line arguments ===
if len(sys.argv) != 4:
    print("Usage: python run_smart.py <model_id: 0-2> <exp_id: 0-3> <rd_seed_id: 0-99>")
    sys.exit(1)

model_id = get_input(sys.argv[1], 0, 2, "model_id")
exp_id = get_input(sys.argv[2], 0, 3, "exp_id")
rd_seed_id = get_input(sys.argv[3], 0, 99, "rd_seed_id")

model = model_list[model_id]
exp = exp_list[exp_id]
random_seed = experiment_seeds[rd_seed_id]
p, q = pq_dict[model]
print(f"Running with: model={model}, exp={exp}, rd_seed_id={rd_seed_id}")

# === Run based on experiment ===
model_to_n = {
    0: [200, 400, 600, 800, 1000],
    1: [300, 500, 700, 1000, 1200],
    2: [500, 700, 1000, 1200, 1500]
}
n_list = model_to_n[model_id]

if exp == "exp1":
    for n in n_list:
        run_simulation(n=n, p=p, q=q, sigma0=0.01, r=5, r_s=10, suffix=f"n={n}")

elif exp == "exp2":
    for r in [1, 3, 5, 7, 9, 11]:
        run_simulation(n=n_list[0], p=p, q=q, sigma0=0.01, r=r, r_s=10, suffix=f"r={r}")

elif exp == "exp3":
    for r_s in [0, 3, 5, 7, 10, 15, 20]:
        run_simulation(n=n_list[0], p=p, q=q, sigma0=0.01, r=5, r_s=r_s, suffix=f"rs={r_s}")

elif exp == "exp4":
    for sigma0 in [0.0, 0.01, 0.02, 0.05, 0.1, 0.5]:
        run_simulation(n=n_list[0], p=p, q=q, sigma0=sigma0, r=5, r_s=10, suffix=f"sigma0={sigma0}")
