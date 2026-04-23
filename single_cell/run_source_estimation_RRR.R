# === Load required packages ===
library(rrpack)
library(reticulate)

# Create output directory
output_dir <- "source_matrix"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# === Load X_source and Y_source ===
np <- import("numpy")

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

# === Run Reduced-Rank Regression (RRR) ===
cat("Running RRR...\n")
rrr_fit <- rrr.fit(Y_source, X_source, nrank = r_hat)
C_hat_rrr <- rrr_fit$coef
np$save(file.path(output_dir, "source_matrix_RRR.npy"), C_hat_rrr)

cat("RRR estimation saved successfully.\n")