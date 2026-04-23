# === Load libraries ===
library(rrpack)
library(reticulate)

# === Python interface ===
np <- import("numpy")

# === Read arguments ===
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: Rscript run_baseline_methods.R <rd_id: 0-99>")
}
rd_id <- as.integer(args[1])
if (is.na(rd_id) || rd_id < 0 || rd_id > 99) {
  stop("rd_id must be an integer between 0 and 99.")
}

# === Load data ===
X_target <- np$load("processed_data/X_target.npy")
Y_target <- np$load("processed_data/Y_target.npy")

# === Load seed list and split data ===
seed_list <- as.integer(np$loadtxt("random_seeds/realdata_seeds.csv", delimiter = ",", skiprows = 1L, usecols = 1L))
set.seed(seed_list[rd_id + 1])  # R is 1-based

n <- dim(X_target)[1]
idx <- sample(1:n)
n_train <- floor(0.7 * n)
train_idx <- idx[1:n_train]
test_idx <- idx[(n_train + 1):n]

X_train <- X_target[train_idx, ]
Y_train <- Y_target[train_idx, ]
X_test <- X_target[test_idx, ]
Y_test <- Y_target[test_idx, ]

# === Create result directory ===
dir.create("result", showWarnings = FALSE)

# === Load r_hat_target from CSV ===
cat("Loading estimated rank from file...\n")
r_hat_path <- "./processed_data/r_hat_target.csv"
r_hat <- as.integer(read.csv(r_hat_path, header = FALSE)[1, 1])
cat(sprintf("Loaded rank (r_hat): %d\n", r_hat))

# === Target-Only RRR ===
cat("Running target-only RRR...\n")
rrr_fit <- rrr.fit(Y_train, X_train, nrank = r_hat)
C_hat <- rrr_fit$coef
Y_pred <- X_test %*% C_hat
frob_error <- norm(Y_pred - Y_test, type = "F") / sqrt(nrow(Y_test) * ncol(Y_test))
reticulate::py_save_object(list(frob_error = frob_error, method_name = "TARGET_ONLY", rd_id = rd_id),
                           sprintf("result/baseline_target_only_RRR_rd_id=%d.pkl", rd_id))

# === Target-Only SRRR ===
cat("Running target-only SRRR...\n")
srrr_fit <- srrr(Y_train, X_train, nrank = r_hat, method = "adglasso", ic.type = "BIC")
C_hat <- srrr_fit$coef
Y_pred <- X_test %*% C_hat
frob_error <- norm(Y_pred - Y_test, type = "F") / sqrt(nrow(Y_test) * ncol(Y_test))
reticulate::py_save_object(list(frob_error = frob_error, method_name = "TARGET_ONLY", rd_id = rd_id),
                           sprintf("result/baseline_target_only_SRRR_rd_id=%d.pkl", rd_id))

# === Target-Only RSSVD ===
cat("Running target-only RSSVD...\n")
rssvd_fit <- rssvd(Y_train, X_train, nrank = r_hat, ic.type = "BIC")
if (length(rssvd_fit$D) == 1) {
  C_hat <- rssvd_fit$U %*% t(rssvd_fit$V) * rssvd_fit$D
} else {
  C_hat <- rssvd_fit$U %*% diag(rssvd_fit$D) %*% t(rssvd_fit$V)
}
Y_pred <- X_test %*% C_hat
frob_error <- norm(Y_pred - Y_test, type = "F") / sqrt(nrow(Y_test) * ncol(Y_test))
reticulate::py_save_object(list(frob_error = frob_error, method_name = "TARGET_ONLY", rd_id = rd_id),
                           sprintf("result/baseline_target_only_RSSVD_rd_id=%d.pkl", rd_id))

# === Target-Only SOFAR ===
cat("Running target-only SOFAR...\n")
sofar_fit <- sofar(Y_train, X_train, nrank = r_hat, ic.type = "BIC")
if (length(sofar_fit$D) == 1) {
  C_hat <- sofar_fit$U %*% sofar_fit$D %*% t(sofar_fit$V)
} else {
  C_hat <- sofar_fit$U %*% diag(sofar_fit$D) %*% t(sofar_fit$V)
}
Y_pred <- X_test %*% C_hat
frob_error <- norm(Y_pred - Y_test, type = "F") / sqrt(nrow(Y_test) * ncol(Y_test))
reticulate::py_save_object(list(frob_error = frob_error, method_name = "TARGET_ONLY", rd_id = rd_id),
                           sprintf("result/baseline_target_only_SOFAR_rd_id=%d.pkl", rd_id))
