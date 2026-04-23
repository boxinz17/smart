# === Load required packages ===
library(rrpack)
library(reticulate)

# === Create output directory ===
output_dir <- "source_matrix"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# === Set up Python interface ===
np <- import("numpy")

# === Load X_source and Y_source ===
cat("Loading data...\n")
X_source <- np$load("./processed_data/X_source.npy")
Y_source <- np$load("./processed_data/Y_source.npy")
cat(sprintf("Loaded X_source: (%d, %d)\n", dim(X_source)[1], dim(X_source)[2]))
cat(sprintf("Loaded Y_source: (%d, %d)\n", dim(Y_source)[1], dim(Y_source)[2]))

# === Load r_hat_source from CSV ===
cat("Loading estimated rank from file...\n")
r_hat_path <- "./processed_data/r_hat_source.csv"
r_hat <- as.integer(read.csv(r_hat_path, header = FALSE)[1, 1])
cat(sprintf("Loaded rank (r_hat): %d\n", r_hat))

# === Run Reduced-Rank SVD (RSSVD) ===
cat("Running RSSVD...\n")
rssvd_fit <- rssvd(Y_source, X_source, nrank = r_hat, ic.type = "BIC")

# Form coefficient matrix
if (length(rssvd_fit$D) == 1) {
  C_hat_rssvd <- rssvd_fit$U %*% t(rssvd_fit$V) * rssvd_fit$D
} else {
  C_hat_rssvd <- rssvd_fit$U %*% diag(rssvd_fit$D) %*% t(rssvd_fit$V)
}

# Save result
np$save(file.path(output_dir, "source_matrix_RSSVD.npy"), C_hat_rssvd)
cat("RSSVD estimation saved successfully.\n")
