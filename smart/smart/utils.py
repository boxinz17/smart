import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV

def generate_data(n, p, q, sigma0=0.01, sigma=0.5, r0_star=10, r_star=5, random_seed=None):
    """
    Generate synthetic data for structured low-rank regression experiments.

    The function creates a source matrix `C0`, a true coefficient matrix `C_star`,
    and corresponding feature/response data (X, Y) under a controlled low-rank
    structure with additive Gaussian noise.

    Parameters
    ----------
    n : int
        Number of samples.
    p : int
        Number of predictors (features).
    q : int
        Number of response variables.
    sigma0 : float, optional
        Standard deviation of noise added to the source matrix C0 (default is 0.01).
    sigma : float, optional
        Standard deviation of noise added to the response matrix Y (default is 0.5).
    r0_star : int, optional
        Rank of the source coefficient matrix C0_star (default is 10).
    r_star : int, optional
        Rank of the true coefficient matrix C_star (default is 5).
    random_seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
            - "X": Feature matrix of shape (n, p)
            - "Y": Response matrix of shape (n, q)
            - "C_star": True coefficient matrix of shape (p, q)
            - "C0": Noisy source matrix used in SMART
            - "C0_star": Clean source regression matrix
            - "U0_star", "V0_star": Left and right singular bases of C0_star
            - "U_star", "V_star", "D_star": Components of C_star
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # ------------------------------------------------------------
    # Step 1: Source Regression Coefficient Matrix (C0_star + noise)
    # ------------------------------------------------------------
    U_prime = np.random.randn(p, r0_star)          # Random matrix for left basis
    V_prime = np.random.randn(q, r0_star)          # Random matrix for right basis
    U0_star, _ = np.linalg.qr(U_prime)             # Orthonormal left basis
    V0_star, _ = np.linalg.qr(V_prime)             # Orthonormal right basis
    D0_star = np.diag(np.linspace(1.0, 10.0, r0_star))  # Diagonal values spread across a range
    C0_star = U0_star @ D0_star @ V0_star.T        # Clean source matrix (rank r0_star)
    E0 = sigma0 * np.random.randn(p, q)            # Additive Gaussian noise
    C0 = C0_star + E0                              # Noisy version of the source matrix

    # ------------------------------------------------------------
    # Step 2: Feature Matrix X ~ N(0, Σ_x)
    # ------------------------------------------------------------
    Sigma_x = np.fromfunction(lambda i, j: 0.5 ** np.abs(i - j), (p, p))  # AR(1) covariance structure
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma_x, size=n)

    # ------------------------------------------------------------
    # Step 3: True Target Coefficient Matrix (C_star)
    # ------------------------------------------------------------
    col_indices_U = np.random.choice(r0_star, size=r_star, replace=False)  # Select r_star basis vectors
    col_indices_V = np.random.choice(r0_star, size=r_star, replace=False)
    U_star = U0_star[:, col_indices_U]            # Subspace of U0_star
    V_star = V0_star[:, col_indices_V]            # Subspace of V0_star
    D_star = np.diag(np.linspace(3.0, 5.0, r_star))  # Singular values for C_star
    C_star = U_star @ D_star @ V_star.T           # Final low-rank coefficient matrix

    # ------------------------------------------------------------
    # Step 4: Generate Noisy Response Y = X C_star + ε
    # ------------------------------------------------------------
    E = sigma * np.random.randn(n, q)             # Output noise
    Y = X @ C_star + E                            # Linear model with noise

    # ------------------------------------------------------------
    # Return all generated components
    # ------------------------------------------------------------
    return {
        "X": X,
        "Y": Y,
        "C_star": C_star,
        "C0": C0,
        "C0_star": C0_star,
        "U0_star": U0_star,
        "V0_star": V0_star,
        "U_star": U_star,
        "V_star": V_star,
        "D_star": D_star
    }

def evaluate_model_relav_err(C_hat, C_true):
    """
    Compute the relative Frobenius error between the estimated and true coefficient matrices.

    This metric evaluates the estimation accuracy of C_hat by comparing it against C_true
    and normalizing by the Frobenius norm of the ground truth.

    Parameters
    ----------
    C_hat : np.ndarray
        Estimated coefficient matrix.
    C_true : np.ndarray
        Ground truth coefficient matrix.

    Returns
    -------
    float
        Relative Frobenius error: ||C_hat - C_true||_F / ||C_true||_F
    """
    return np.linalg.norm(C_hat - C_true, 'fro') / np.linalg.norm(C_true, 'fro')

def evaluate_model_avg_err(C_hat, C_true):
    """
    Compute the average per-entry Frobenius error between C_hat and C_true.

    This metric gives the root mean square error per entry, useful for assessing
    absolute accuracy independent of the scale of C_true.

    Parameters
    ----------
    C_hat : np.ndarray
        Estimated coefficient matrix.
    C_true : np.ndarray
        Ground truth coefficient matrix.

    Returns
    -------
    float
        Average Frobenius error per entry: ||C_hat - C_true||_F / sqrt(p * q),
        where (p, q) = shape of C_hat.
    """
    return np.linalg.norm(C_hat - C_true, 'fro') / np.sqrt(C_hat.shape[0] * C_hat.shape[1])

def extract_svd_subspaces(C0: np.ndarray, r_u: int, r_v: int):
    """
    Compute the subspace matrices Us and Vs from the SVD of C0.

    Parameters
    ----------
    C0 : np.ndarray
        The source regression coefficient matrix of shape (m, n).
    r_u : int
        Number of columns to select from U^0 (top left singular vectors).
    r_v : int
        Number of columns to select from V^0 (top right singular vectors).

    Returns
    -------
    Us : np.ndarray
        Left singular vector matrix of shape (m, r_u).
    Vs : np.ndarray
        Right singular vector matrix of shape (n, r_v).
    """
    # Validate input
    if not isinstance(C0, np.ndarray):
        raise TypeError("C0 must be a NumPy array.")
    
    m, n = C0.shape

    if not (0 < r_u <= m and 0 < r_v <= n):
        raise ValueError(f"r_u must be in (0, {m}] and r_v must be in (0, {n}]. "
                         f"Got r_u={r_u}, r_v={r_v}.")

    # Compute SVD
    U0, _, V0t = np.linalg.svd(C0, full_matrices=True)
    V0 = V0t.T

    # Extract top-r_u and top-r_v subspaces
    U_s = U0[:, :r_u]
    V_s = V0[:, :r_v]

    return U_s, V_s

def fit_baseline(X, Y, model_type="ols", alphas=None):
    """
    Fit a baseline model (OLS, Ridge, or Lasso) column-wise.

    Args:
        X: Feature matrix (n x p)
        Y: Response matrix (n x q)
        model_type: "ols", "ridge", or "lasso"
        alphas: List or array of alpha values to try (only used for ridge/lasso)

    Returns:
        C_hat: Estimated coefficient matrix (p x q)
        best_alphas: List of selected alpha values (None for OLS)
    """
    n, p = X.shape
    _, q = Y.shape
    C_hat = np.zeros((p, q))
    best_alphas = []

    for j in range(q):
        y_j = Y[:, j]
        if model_type == "ols":
            model = LinearRegression()
        elif model_type == "ridge":
            model = RidgeCV(alphas=alphas)
        elif model_type == "lasso":
            model = LassoCV(alphas=alphas, cv=5, max_iter=10000, precompute=False)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X, y_j)
        C_hat[:, j] = model.coef_
        if model_type in ["ridge", "lasso"]:
            best_alphas.append(model.alpha_)
        else:
            best_alphas.append(None)

    return C_hat, best_alphas