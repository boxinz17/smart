import os
import numpy as np
from smart import fit_baseline

# === Create output directory ===
output_dir = "source_matrix"
os.makedirs(output_dir, exist_ok=True)

# === Load X_source and Y_source ===
print("Loading data...")
X_source = np.load("./processed_data/X_source.npy")
Y_source = np.load("./processed_data/Y_source.npy")
print(f"Loaded X_source: {X_source.shape}")
print(f"Loaded Y_source: {Y_source.shape}")

# === Define alpha grid ===
alphas = np.logspace(-4, 4, 30)

# === Fit models with different regularization types ===
model_types = ["ridge", "ols", "lasso"]

for model_type in model_types:
    print(f"Running {model_type.upper()} regression...")
    if model_type == "ols":
        # alphas not used for OLS
        C_hat, _ = fit_baseline(X_source, Y_source, model_type=model_type)
    else:
        C_hat, _ = fit_baseline(X_source, Y_source, model_type=model_type, alphas=alphas)

    output_path = os.path.join(output_dir, f"source_matrix_{model_type}.npy")
    np.save(output_path, C_hat)
    print(f"{model_type.upper()} estimation saved to: {output_path}")