import scanpy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import MultiTaskLassoCV
from scipy.linalg import svd
from numpy.linalg import norm

from smart import RankSelectorRSC
from smart import MultitaskMarginalRegression

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

# File path to the processed dataset
DATA_PATH = '../raw_data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad'

# Define selected source and target cell types
SOURCE_CELL_TYPE = 'NK'          # Large-sample source (5434 cells)
TARGET_CELL_TYPE = 'ILC1'        # Data‑sparse target (552 cells)

HVG_K = 3000

# Output folder
OUTPUT_DIR = '../processed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load the dataset
# ------------------------------------------------------------------

print("Loading dataset …")
adata = sc.read_h5ad(DATA_PATH)
adata.var_names_make_unique()
print("Dataset loaded. Shape:", adata.shape)

# ------------------------------------------------------------------
# Split GEX and ADT features
# ------------------------------------------------------------------

is_gex = adata.var['feature_types'] == 'GEX'
is_adt = adata.var['feature_types'] == 'ADT'

adata_gex = adata[:, is_gex].copy()
adata_adt = adata[:, is_adt].copy()

print("GEX shape:", adata_gex.shape)
print("ADT shape:", adata_adt.shape)

# ------------------------------------------------------------------
# Preprocess GEX matrix
# ------------------------------------------------------------------

print("Preprocessing GEX matrix (normalize → log1p → HVG_K → scale)…")
sc.pp.normalize_total(adata_gex, target_sum=1e4)
sc.pp.log1p(adata_gex)
sc.pp.highly_variable_genes(adata_gex, n_top_genes=HVG_K)
adata_gex = adata_gex[:, adata_gex.var.highly_variable].copy()
sc.pp.scale(adata_gex)
print("GEX shape after preprocessing:", adata_gex.shape)

# ------------------------------------------------------------------
# Preprocess ADT matrix (CLR normalisation)
# ------------------------------------------------------------------

print("Applying CLR normalisation to ADT matrix …")
adt_matrix = adata_adt.X.toarray() if not isinstance(adata_adt.X, np.ndarray) else adata_adt.X
adt_clr = np.log1p(adt_matrix) - np.log1p(adt_matrix).mean(axis=1, keepdims=True)
adata_adt.X = adt_clr
print("ADT shape after CLR normalisation:", adata_adt.shape)

# ------------------------------------------------------------------
# Filter cells for source and target tasks
# ------------------------------------------------------------------

print(f"Filtering source cells: {SOURCE_CELL_TYPE}")
idx_source = adata.obs['cell_type'] == SOURCE_CELL_TYPE
adata_gex_source = adata_gex[idx_source, :].copy()
adata_adt_source = adata_adt[idx_source, :].copy()

print(f"Filtering target cells: {TARGET_CELL_TYPE}")
idx_target = adata.obs['cell_type'] == TARGET_CELL_TYPE
adata_gex_target = adata_gex[idx_target, :].copy()
adata_adt_target = adata_adt[idx_target, :].copy()

print("Source GEX shape:", adata_gex_source.shape)
print("Source ADT shape:", adata_adt_source.shape)
print("Target GEX shape:", adata_gex_target.shape)
print("Target ADT shape:", adata_adt_target.shape)

# ------------------------------------------------------------------
# Prepare (X, Y) matrices
# ------------------------------------------------------------------

print("Preparing matrices for SMART …")

# Convert to dense if necessary
X_source = adata_gex_source.X.toarray() if not isinstance(adata_gex_source.X, np.ndarray) else adata_gex_source.X
Y_source = adata_adt_source.X.toarray() if not isinstance(adata_adt_source.X, np.ndarray) else adata_adt_source.X

X_target = adata_gex_target.X.toarray() if not isinstance(adata_gex_target.X, np.ndarray) else adata_gex_target.X
Y_target = adata_adt_target.X.toarray() if not isinstance(adata_adt_target.X, np.ndarray) else adata_adt_target.X

print("Matrices prepared.")

# ------------------------------------------------------------------
# Remove columns with zero variance in both source and target
# ------------------------------------------------------------------

print("Removing features and outputs with zero variance …")

# Feature-wise (GEX, i.e. X columns)
x_var_source = np.var(X_source, axis=0)
x_var_target = np.var(X_target, axis=0)
x_keep_mask = (x_var_source > 0) & (x_var_target > 0)

# Output-wise (ADT, i.e. Y columns)
y_var_source = np.var(Y_source, axis=0)
y_var_target = np.var(Y_target, axis=0)
y_keep_mask = (y_var_source > 0) & (y_var_target > 0)

# Apply filters
X_source = X_source[:, x_keep_mask]
X_target = X_target[:, x_keep_mask]
Y_source = Y_source[:, y_keep_mask]
Y_target = Y_target[:, y_keep_mask]

print(f"Retained {X_source.shape[1]} GEX features and {Y_source.shape[1]} ADT outputs after filtering.")

# ------------------------------------------------------------------
# Feature selection utility function
# ------------------------------------------------------------------

def select_features(X_source, Y_source, X_target, Y_target, method="marginal", k=150):
    """
    Select features using either multi-task Lasso or marginal regression.

    Parameters:
        X_source (np.ndarray): Source design matrix (n₁ × p)
        Y_source (np.ndarray): Source response matrix (n₁ × q)
        X_target (np.ndarray): Target design matrix (n₂ × p)
        Y_target (np.ndarray): Target response matrix (n₂ × q)
        method (str): Feature selection strategy. Choose from:
                      - "lasso"   → MultiTaskLassoCV (row-sparse regression)
                      - "marginal"→ MultitaskMarginalRegression using Pearson correlation
        k (int): Number of top features to select from each task (source and target)

    Returns:
        selected_features (np.ndarray): Sorted union of selected feature indices
        X_source_new (np.ndarray): Source matrix restricted to selected features
        X_target_new (np.ndarray): Target matrix restricted to selected features
    """
    
    if method == "lasso":
        print("Running multi‑task Lasso feature selection (row‑sparse) …")

        # Fit multi-task Lasso on source data
        print("Fitting MultiTaskLassoCV on source data …")
        model_source = MultiTaskLassoCV(cv=5, random_state=666)
        model_source.fit(X_source, Y_source)
        W_source = model_source.coef_.T  # shape: [p, q]
        row_norms_source = norm(W_source, axis=1)
        top_source = np.argsort(-row_norms_source)[:k]

        # Fit multi-task Lasso on target data
        print("Fitting MultiTaskLassoCV on target data …")
        model_target = MultiTaskLassoCV(cv=5, random_state=666)
        model_target.fit(X_target, Y_target)
        W_target = model_target.coef_.T
        row_norms_target = norm(W_target, axis=1)
        top_target = np.argsort(-row_norms_target)[:k]

    elif method == "marginal":
        print("Running marginal regression feature selection …")

        # Fit marginal regression on source with Pearson correlation
        print("Fitting marginal model on source data …")
        model_source = MultitaskMarginalRegression(X_source, Y_source, marginal_method="pearson")
        top_source = model_source.fit(manual_k=k)

        # Fit marginal regression on target with Pearson correlation
        print("Fitting marginal model on target data …")
        model_target = MultitaskMarginalRegression(X_target, Y_target, marginal_method="pearson")
        top_target = model_target.fit(manual_k=k)

    else:
        raise ValueError("Invalid method. Use 'lasso' or 'marginal'.")

    # Take the union of top-k features from both tasks
    selected_features = np.union1d(top_source, top_target)
    print(f"Selected {len(selected_features)} features after union.")

    # Restrict source/target matrices to selected features
    X_source_new = X_source[:, selected_features]
    X_target_new = X_target[:, selected_features]

    return selected_features, X_source_new, X_target_new

# ------------------------------------------------------------------
# Feature selection (choose method: 'lasso' or 'marginal')
# ------------------------------------------------------------------

selected_features, X_source, X_target = select_features(
    X_source, Y_source, X_target, Y_target,
    method="marginal",  # or "lasso"
    k=150
)

# ------------------------------------------------------------------
# Sanity check: Estimate ranks of regression matrices
# ------------------------------------------------------------------

print("Estimating rank of regression matrix on source data …")
selector = RankSelectorRSC()
r_hat_source = selector.select_rank(X_source, Y_source)
print(f"Estimated intrinsic rank on source data: r̂_source = {r_hat_source}")

print("Estimating rank of regression matrix on target data …")
selector = RankSelectorRSC()
r_hat_target = selector.select_rank(X_target, Y_target)
print(f"Estimated intrinsic rank on target data: r̂_target = {r_hat_target}")

# Save ranks as CSV files
np.savetxt(os.path.join(OUTPUT_DIR, 'r_hat_source.csv'), np.array([r_hat_source]), fmt='%d')
np.savetxt(os.path.join(OUTPUT_DIR, 'r_hat_target.csv'), np.array([r_hat_target]), fmt='%d')
print("Saved rank estimates → r_hat_source.csv, r_hat_target.csv")

# ------------------------------------------------------------------
# Save matrices to file
# ------------------------------------------------------------------

print("Saving matrices …")
np.save(os.path.join(OUTPUT_DIR, 'X_source.npy'), X_source)
np.save(os.path.join(OUTPUT_DIR, 'Y_source.npy'), Y_source)
np.save(os.path.join(OUTPUT_DIR, 'X_target.npy'), X_target)
np.save(os.path.join(OUTPUT_DIR, 'Y_target.npy'), Y_target)
print("Matrices saved →", OUTPUT_DIR)

# ------------------------------------------------------------------
# (Optional) Singular‑value spectra sanity check
# ------------------------------------------------------------------

print("\n=== Optional: Singular‑value spectra check ===")
print("Fitting Ridge regressions …")

reg_source = Ridge(alpha=1.0).fit(X_source, Y_source)
C_source_est = reg_source.coef_.T  # p × q

reg_target = Ridge(alpha=1.0).fit(X_target, Y_target)
C_target_est = reg_target.coef_.T

# Singular values
_, s_source, _ = svd(C_source_est, full_matrices=False)
_, s_target, _ = svd(C_target_est, full_matrices=False)

plt.figure(figsize=(8, 6))
plt.plot(s_source, label=f'Source ({SOURCE_CELL_TYPE})')
plt.plot(s_target, label=f'Target ({TARGET_CELL_TYPE})')
plt.xlabel('Singular‑value index')
plt.ylabel('Singular‑value magnitude')
plt.title('Singular‑value spectra of estimated coefficient matrices')
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = 'singular_value_spectra.png'
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Figure saved → {save_path}")
print("Script finished.")

