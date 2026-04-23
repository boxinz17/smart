import numpy as np
from numpy.linalg import svd, pinv, eigvalsh, matrix_rank

class RankSelectorRSC:
    """
    Implements the Rank Selection Criterion (RSC) for determining the rank of the coefficient matrix 
    in multivariate response regression. Based on Bunea, She, and Wegkamp (2011).

    Attributes:
        mu (float or None): User-specified threshold for eigenvalue truncation. If None, it is computed.
        sigma (float or None): Noise standard deviation. If None, it is estimated from residuals.
        theta (float): Inflation factor used in the threshold formula when computing mu from sigma.

    Example
    -------
    >>> import numpy as np
    >>> from smart import RankSelectorRSC
    >>> 
    >>> # Simulated data
    >>> n, p, q, r = 100, 30, 20, 5
    >>> U = np.linalg.qr(np.random.randn(p, r))[0]
    >>> V = np.linalg.qr(np.random.randn(q, r))[0]
    >>> D = np.diag(np.linspace(1.5, 0.5, r))
    >>> A = U @ D @ V.T
    >>> X = np.random.randn(n, p)
    >>> Y = X @ A + 0.5 * np.random.randn(n, q)
    >>>
    >>> # Use RSC to estimate rank
    >>> selector = RankSelectorRSC()
    >>> estimated_rank = selector.select_rank(X, Y)
    >>> print(f"Estimated rank: {estimated_rank}")
    """
    def __init__(self, mu=None, sigma=None, theta=1.0):
        """
        Initialize the RankSelectorRSC with optional tuning parameters.

        Parameters:
            mu (float or None): Threshold for eigenvalue truncation. If None, computed based on sigma.
            sigma (float or None): Noise standard deviation. If None, estimated automatically.
            theta (float): Tuning parameter for computing mu when sigma is available or estimated.
        """
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def compute_projection_matrix(self, X):
        """
        Compute the projection matrix onto the column space of X.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
            P (ndarray): Projection matrix (n_samples x n_samples)
        """
        return X @ pinv(X.T @ X) @ X.T

    def estimate_sigma(self, Y, Y_proj, n, q):
        """
        Estimate the noise standard deviation sigma using residual Frobenius norm.

        Parameters:
            Y (ndarray): Response matrix (n x q)
            Y_proj (ndarray): Projected response P Y
            n (int): Number of responses (columns of Y)
            q (int): Rank of the projection matrix (i.e., rank(X))

        Returns:
            float: Estimated sigma
        """
        residual = Y - Y_proj
        df = n * (Y.shape[0] - q)  # degrees of freedom
        sigma_squared = np.sum(residual ** 2) / df
        return np.sqrt(sigma_squared)

    def compute_mu(self, sigma, n, q):
        """
        Compute the eigenvalue threshold mu based on estimated or given sigma.

        Parameters:
            sigma (float): Noise standard deviation
            n (int): Number of responses
            q (int): Rank of X

        Returns:
            float: Regularization threshold mu
        """
        return (1 + self.theta) ** 2 * sigma ** 2 * (np.sqrt(n) + np.sqrt(q)) ** 2

    def select_rank(self, X, Y):
        """
        Select the rank of the coefficient matrix in Y = X A + E using RSC.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features)
            Y (ndarray): Response matrix of shape (n_samples, n_responses)

        Returns:
            r_hat (int): Estimated rank of the regression coefficient matrix
        """
        n, q = Y.shape

        # Compute projection matrix P onto Col(X), and project Y to get PY
        P = self.compute_projection_matrix(X)
        Y_proj = P @ Y

        # Compute eigenvalues of Y^T P Y (equivalently, singular values of PY squared)
        Y_proj_cross = Y_proj.T @ Y_proj
        eigenvalues = eigvalsh(Y_proj_cross)[::-1]  # Sort in descending order

        # Determine threshold mu
        mu = self.mu
        if mu is None:
            sigma = self.sigma
            q_proj = matrix_rank(X)  # Effective rank of the design matrix
            if sigma is None:
                sigma = self.estimate_sigma(Y, Y_proj, q, q_proj)
            mu = self.compute_mu(sigma, q, q_proj)

        # Rank is the number of eigenvalues greater than or equal to mu
        r_hat = np.sum(eigenvalues >= mu)
        return r_hat