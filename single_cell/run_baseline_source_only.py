import sys
import os
import numpy as np
import pickle

# === Source matrix options ===
matrix_options = [
    "lasso", "ols", "ridge",
    "RRR", "RSSVD", "SOFAR", "SRRR"
]

# === Argument parsing ===
if len(sys.argv) != 3:
    print("Usage: python run_baseline_source_only.py <matrix_id: 0-6> <rd_id: 0-99>")
    sys.exit(1)

try:
    matrix_id = int(sys.argv[1])
    rd_id = int(sys.argv[2])
    assert 0 <= matrix_id < len(matrix_options)
    assert 0 <= rd_id <= 99
except:
    print("matrix_id must be in [0, 6] and rd_id must be in [0, 99]")
    sys.exit(1)

matrix_name = matrix_options[matrix_id]
print(f"Using source matrix: {matrix_name}, repetition ID: {rd_id}")

# === Load data ===
X_target = np.load("processed_data/X_target.npy")
Y_target = np.load("processed_data/Y_target.npy")

# === Load seed and split ===
seed_list = np.loadtxt("random_seeds/realdata_seeds.csv", dtype=int, delimiter=",", skiprows=1, usecols=1)
np.random.seed(seed_list[rd_id])
n = X_target.shape[0]
idx = np.random.permutation(n)
n_train = int(0.7 * n)
train_idx = idx[:n_train]
test_idx = idx[n_train:]

X_test = X_target[test_idx]
Y_test = Y_target[test_idx]

# === Load source matrix ===
matrix_path = f"source_matrix/source_matrix_{matrix_name}.npy"
if not os.path.exists(matrix_path):
    raise FileNotFoundError(f"Source matrix not found: {matrix_path}")

C0 = np.load(matrix_path)

# === Compute prediction and error ===
Y_pred = X_test @ C0
frob_error = np.linalg.norm(Y_pred - Y_test, ord="fro") / np.sqrt(Y_test.shape[0] * Y_test.shape[1])

print(f"Frobenius error (source only): {frob_error:.4f}")

# === Save result ===
os.makedirs("result", exist_ok=True)
out_path = f"result/baseline_source_only_{matrix_name}_rd_id={rd_id}.pkl"

result = {
    "frob_error": frob_error,
    "method_name": "source_only",
    "source_matrix_type": matrix_name,
    "rd_id": rd_id
}

with open(out_path, "wb") as f:
    pickle.dump(result, f)

print(f"Result saved to: {out_path}")
