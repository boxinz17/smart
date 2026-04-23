generate_data <- function(n, p, q, sigma0 = 0.01, sigma = 0.5, r0_star = 10, r_star = 5, random_seed = NULL) {
  if (!is.null(random_seed)) {
    set.seed(random_seed)
  }
  
  # Step 1: Source Regression Coefficient Matrix
  U_prime <- matrix(rnorm(p * r0_star), nrow = p)
  V_prime <- matrix(rnorm(q * r0_star), nrow = q)
  U0_star <- qr.Q(qr(U_prime))
  V0_star <- qr.Q(qr(V_prime))
  D0_star <- diag(seq(1.0, 10.0, length.out = r0_star))
  C0_star <- U0_star %*% D0_star %*% t(V0_star)
  E0 <- sigma0 * matrix(rnorm(p * q), nrow = p)
  C0 <- C0_star + E0
  
  # Step 2: Feature Matrix X
  Sigma_x <- outer(1:p, 1:p, function(i, j) 0.5 ^ abs(i - j))
  X <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma_x)
  
  # Step 3: Target Regression Coefficient Matrix
  col_indices_U <- sample(r0_star, r_star)
  col_indices_V <- sample(r0_star, r_star)
  U_star <- U0_star[, col_indices_U, drop = FALSE]
  V_star <- V0_star[, col_indices_V, drop = FALSE]
  D_star <- diag(seq(3.0, 5.0, length.out = r_star))
  C_star <- U_star %*% D_star %*% t(V_star)
  
  # Step 4: Response Matrix Y
  E <- sigma * matrix(rnorm(n * q), nrow = n)
  Y <- X %*% C_star + E
  
  return(list(
    X = X,
    Y = Y,
    C_star = C_star,
    C0 = C0,
    C0_star = C0_star,
    U0_star = U0_star,
    V0_star = V0_star,
    U_star = U_star,
    V_star = V_star,
    D_star = D_star
  ))
}

evaluate_model_relav_err <- function(C_hat, C_true) {
  norm(C_hat - C_true, type = "F") / norm(C_true, type = "F")
}

evaluate_model_avg_err <- function(C_hat, C_true) {
  norm(C_hat - C_true, type = "F") / sqrt(nrow(C_hat) * ncol(C_hat))
}

extract_svd_subspaces <- function(C0, r_u, r_v) {
  svd_result <- svd(C0)
  U0 <- svd_result$u
  D0 <- svd_result$d
  V0 <- svd_result$v
  
  r0_star <- sum(D0 > 1e-10)
  
  if (!(r_u > 0 && r_u <= r0_star && r_v > 0 && r_v <= r0_star)) {
    stop(sprintf("r_u and r_v must be in the range (0, r0_star=%d]", r0_star))
  }
  
  U_s <- U0[, 1:r_u, drop = FALSE]
  V_s <- V0[, 1:r_v, drop = FALSE]
  
  return(list(U_s = U_s, V_s = V_s))
}
