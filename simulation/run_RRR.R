# === Load required packages and utilities ===
library(rrpack)
library(jsonlite)
source("utils_R.R")

# Settings
pq_dict <- list(model1 = c(100, 50), model2 = c(150, 100), model3 = c(300, 200))
exp_list <- c("exp1", "exp2", "exp3", "exp4")
model_list <- names(pq_dict)

# Parse arguments
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
model_id <- as.integer(args[1])
exp_id <- as.integer(args[2])
rd_seed_id <- as.integer(args[3])

stopifnot(model_id %in% 0:2, exp_id %in% 0:3, rd_seed_id %in% 0:99)

model <- model_list[[model_id + 1]]
exp <- exp_list[[exp_id + 1]]
random_seed <- as.integer(read.csv("data/random_seeds/experiment_seeds.csv")$seed[rd_seed_id + 1])
p <- pq_dict[[model]][1]
q <- pq_dict[[model]][2]

cat(sprintf("Running with: model=%s, exp=%s, rd_seed_id=%d\n", model, exp, rd_seed_id))

# === Helper function for simulation and saving ===
run_simulation <- function(n, r, suffix) {
  cat(sprintf("%s: %s\n", toupper(gsub("=.*", "", suffix)), gsub(".*=", "", suffix)))
  
  data <- generate_data(n = n, p = p, q = q, sigma0 = 0.01, random_seed = random_seed)
  X <- data$X; Y <- data$Y; C_true <- data$C_star; C0 <- data$C0
  
  extract_svd_subspaces(C0, r_u = 10, r_v = 10)  # not used in RRR, but kept for consistency
  
  res_rrr <- rrr.fit(Y, X, nrank = r)
  C_rrr <- res_rrr$coef
  err_rrr <- evaluate_model_avg_err(C_rrr, C_true)
  
  cat(sprintf("\n=== Average Frobenius Norm Error ===\nRRR: %.4f\n", err_rrr))
  
  # Save result as JSON
  result <- list(
    C_true = unname(split(as.matrix(C_true), row(C_true))),
    C_hat = unname(split(as.matrix(C_rrr), row(C_rrr))),
    avg_err = err_rrr
  )
  
  out_dir <- sprintf("result/%s/%s", model, exp)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_file <- sprintf("%s/RRR_result_%s_%s_%s_rd_seed_id=%d.json", out_dir, model, exp, suffix, rd_seed_id)
  
  write_json(result, out_file, pretty = TRUE, auto_unbox = TRUE)
  cat(sprintf("Saved result to %s\n\n", out_file))
}

# === Run experiments ===
if (exp == "exp1") {
  model_to_n <- list(
    "model1" = c(200, 400, 600, 800, 1000),
    "model2" = c(300, 500, 700, 1000, 1200),
    "model3" = c(500, 700, 1000, 1200, 1500)
  )
  n_list <- model_to_n[[model]]
  for (n in n_list) {
    run_simulation(n = n, r = 5, suffix = sprintf("n=%d", n))
  }
}

if (exp == "exp2") {
  for (r in seq(1, 11, by = 2)) {
    run_simulation(n = 200, r = r, suffix = sprintf("r=%d", r))
  }
}
