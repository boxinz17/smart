# ----------------------------------------
# Multitask Marginal Regression
# ----------------------------------------

import numpy as np
from sklearn.linear_model import LinearRegression

class MultitaskMarginalRegression:
    def __init__(self, X, Y, scoring="l2", delta=0.05, marginal_method="regression"):
        """
        Initialize the multitask marginal regression object.

        Parameters:
            X (np.ndarray): Feature matrix (n x p)
            Y (np.ndarray): Response matrix (n x T)
            scoring (str): Scoring method ('l1', 'l2', 'linf')
            delta (float): Confidence parameter for thresholding
            marginal_method (str): Method to compute marginal coefficients ('regression' or 'pearson')
        """
        if marginal_method not in ["regression", "pearson"]:
            raise ValueError("marginal_method must be 'regression' or 'pearson'")

        self.X = X
        self.Y = Y
        self.n, self.p = X.shape
        _, self.T = Y.shape
        self.scoring = scoring
        self.delta = delta
        self.marginal_method = marginal_method
        self.ranked_features = None
        self.scores = None
        self.sigma_squared_est = None
        self.selected_features = None

    def compute_marginal_scores(self):
        if self.marginal_method == "pearson":
            # Compute Pearson correlation matrix between columns of X and Y
            # Standardize X and Y
            X_centered = self.X - self.X.mean(axis=0, keepdims=True)
            X_std = X_centered / (np.std(X_centered, axis=0, keepdims=True) + 1e-10)
            Y_centered = self.Y - self.Y.mean(axis=0, keepdims=True)
            Y_std = Y_centered / (np.std(Y_centered, axis=0, keepdims=True) + 1e-10)
            marginal_coeffs = X_std.T @ Y_std / self.n
        else:
            # Use regression coefficients
            dot_products = self.X.T @ self.Y  # shape (p, T)
            feature_norms_squared = np.sum(self.X ** 2, axis=0)  # shape (p,)
            feature_norms_squared_safe = np.maximum(feature_norms_squared, 1e-8)
            marginal_coeffs = dot_products / feature_norms_squared_safe[:, np.newaxis]

        # Score aggregation
        if self.scoring == "l2":
            self.scores = np.linalg.norm(marginal_coeffs, axis=1)
        elif self.scoring == "l1":
            self.scores = np.sum(np.abs(marginal_coeffs), axis=1)
        elif self.scoring == "linf":
            self.scores = np.max(np.abs(marginal_coeffs), axis=1)
        else:
            raise ValueError("Invalid scoring method. Choose from 'l1', 'l2', 'linf'.")

        self.ranked_features = np.argsort(-self.scores)

    def estimate_sigma_squared(self, selected_features):
        k = len(selected_features)
        X_selected = self.X[:, selected_features]
        sigma_squared_tasks = []

        for t in range(self.T):
            y_t = self.Y[:, t]
            model = LinearRegression(fit_intercept=False)
            model.fit(X_selected, y_t)
            residuals = y_t - model.predict(X_selected)
            sigma_squared_t = np.sum(residuals ** 2) / (self.n - k)
            sigma_squared_tasks.append(sigma_squared_t)

        self.sigma_squared_est = np.mean(sigma_squared_tasks)

    def select_number_of_features(self):
        X_ranked = self.X[:, self.ranked_features]
        threshold = (self.T + 2 * np.sqrt(self.T * np.log(2/self.delta)) + 2 * np.log(2/self.delta)) * self.sigma_squared_est

        previous_proj_Y = np.zeros((self.n, self.T))

        for k in range(1, min(self.n-1, len(self.ranked_features))):
            X_k = X_ranked[:, :k+1]

            # QR decomposition (economy size)
            Q, R = np.linalg.qr(X_k, mode='reduced')

            # Projection onto span of Q
            proj_Y = Q @ (Q.T @ self.Y)

            # Difference from previous projection
            diff = proj_Y - previous_proj_Y
            residual_variance = np.sum(diff ** 2)

            if residual_variance > threshold:
                return k

            previous_proj_Y = proj_Y.copy()

        return len(self.ranked_features)

    def fit(self, manual_k=None):
        """
        Fit the model.

        Parameters:
            manual_k (int or None): Manually specify number of features to keep. If None, estimate it automatically.
        """
        self.compute_marginal_scores()

        # Preselect a moderately large set to estimate sigma^2
        prelim_k = min(2 * self.n // int(np.log(self.n)), len(self.ranked_features))
        prelim_features = self.ranked_features[:prelim_k]

        self.estimate_sigma_squared(prelim_features)

        if manual_k is not None:
            selected_k = manual_k
        else:
            selected_k = self.select_number_of_features()

        self.selected_features = self.ranked_features[:selected_k]

        return self.selected_features

# ----------------------------------------
# Example Usage (Synthetic Data)
# ----------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    n, p, T = 100, 500, 20
    X = np.random.randn(n, p)

    true_features = np.random.choice(p, size=10, replace=False)
    beta_true = np.zeros((p, T))
    for j in true_features:
        beta_true[j, :] = np.random.randn(T)

    noise = 0.5 * np.random.randn(n, T)
    Y = X @ beta_true + noise

    model = MultitaskMarginalRegression(X, Y, marginal_method="pearson")
    selected_features = model.fit(manual_k=15)  # Example with manually specified number of features

    print(f"Selected features (indices): {selected_features}")
    print(f"True features: {true_features}")