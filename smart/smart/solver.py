import numpy as np
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient
from pymanopt import function
import matplotlib.pyplot as plt
from itertools import product
from joblib import Parallel, delayed
import optuna
from optuna.samplers import TPESampler

class SMARTSolver:
    """
    Solver class for the SMaRT algorithm, which estimates structured low-rank factorizations
    using ADMM with structural alignment to source domain singular vectors.

    Attributes:
        lambda_u (float): Regularization weight for the structural penalty on U.
        lambda_v (float): Regularization weight for the structural penalty on V.
        U_s (ndarray): Source left singular vectors.
        V_s (ndarray): Source right singular vectors.
        U_s_orth (ndarray): Orthonormal basis for the orthogonal complement of U_s.
        V_s_orth (ndarray): Orthonormal basis for the orthogonal complement of V_s.
        gamma (float): Multiplicative inflation factor for the ADMM penalty parameter rho.
        t_max (int): Maximum number of iterations for the optimization loop.
    """

    def __init__(self, C0, r_u, r_v, lambda_u=1.0, lambda_v=1.0, gamma=2.0, t_max=500):
        """
        Initialize the SMaRT solver using a source regression matrix C0 and structural ranks.

        Parameters:
            C0 (ndarray): Source regression matrix (p x q).
            r_u (int): Structural rank for U.
            r_v (int): Structural rank for V.
            lambda_u (float): Regularization weight for U structure.
            lambda_v (float): Regularization weight for V structure.
            gamma (float): ADMM penalty inflation factor. Default 2.0 matches the
                value used for the main simulation and real-data experiments reported
                in the SMART paper (Zhao, Kolar, Lv, 2026; arXiv:2604.20161).
            t_max (int): Max number of iterations for ADMM.
        """
        self.C0 = C0
        self.r_u = r_u
        self.r_v = r_v
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.gamma = gamma
        self.t_max = t_max

        # Perform full SVD of C0
        # U0: (p x p), V0: (q x q)
        U0, D0, V0t = np.linalg.svd(C0, full_matrices=True)
        V0 = V0t.T

        # Store full orthonormal bases
        self.U0_full = U0
        self.V0_full = V0

        p, q = C0.shape

        # Define structured subspaces and their orthogonal complements
        self.U_s = U0[:, :r_u] if r_u > 0 else np.zeros((p, 0))
        self.U_s_orth = U0[:, r_u:] if r_u < p else np.zeros((p, 0))

        self.V_s = V0[:, :r_v] if r_v > 0 else np.zeros((q, 0))
        self.V_s_orth = V0[:, r_v:] if r_v < q else np.zeros((q, 0))

    def _get_dynamic_tolerance(self, t, tol_0=1e-2, tol_min=1e-6, decay=0.9):
        """
        Compute adaptive tolerance for Riemannian gradient norm in subproblem solver.

        Parameters:
            t (int): Current ADMM iteration.
            tol_0 (float): Initial loose tolerance.
            tol_min (float): Minimum allowed tolerance.
            decay (float): Exponential decay factor.

        Returns:
            float: Tolerance value for current iteration.
        """
        return max(tol_min, tol_0 * (decay ** t))

    def initialize(self, r, X, Y, C=None):
        """
        Initialize optimization variables at iteration t = 0.

        Parameters:
            r (int): Target rank of the low-rank factorization.
            X (ndarray): Input feature matrix.
            Y (ndarray): Output/response matrix.
            C (ndarray or None): Optional matrix of shape (X.shape[1], Y.shape[1]) to initialize U, D, V via SVD.
        """
        # -------------------------------
        # Sanity check: required attributes must exist
        # -------------------------------
        if not hasattr(self, "r_u") or not hasattr(self, "r_v"):
            raise AttributeError("SMARTSolver must be initialized with 'r_u' and 'r_v' before calling initialize().")

        if not hasattr(self, "C0"):
            raise AttributeError("SMARTSolver must be initialized with source matrix 'C0' before calling initialize().")
        
        n, p = X.shape
        q = Y.shape[1]
        self.X = X
        self.Y = Y
        self.n = n
        self.rank = r

        if C is None:
            # Initialize U and V as orthonormal random matrices
            self.U = np.random.randn(p, r)
            self.U, _ = np.linalg.qr(self.U)
            self.V = np.random.randn(q, r)
            self.V, _ = np.linalg.qr(self.V)
            self.D = np.eye(r)
        else:
            # Check shape of C
            if C.shape != (p, q):
                raise ValueError(f"Expected C to have shape ({p}, {q}), but got {C.shape}")
            # Initialize U, D, V from truncated SVD of C
            U_C, s_C, Vt_C = np.linalg.svd(C, full_matrices=False)
            self.U = U_C[:, :r]
            self.D = np.diag(s_C[:r])
            self.V = Vt_C[:r, :].T  # Transpose to get (q x r)

        # Initialize ADMM auxiliary and dual variables
        self.Omega_u = np.zeros((self.U_s_orth.shape[1], r))
        self.Omega_v = np.zeros((self.V_s_orth.shape[1], r))
        self.Gamma_u = np.zeros_like(self.Omega_u)
        self.Gamma_v = np.zeros_like(self.Omega_v)

        # Track objective values and residuals across iterations
        self.objective_history = []
        self.primal_residual_history = []
        self.dual_residual_history = []

        # Initialize penalty parameter and iteration counter
        self.rho = 1.0
        self.t = 0

        # Precompute frequently used Gram matrices scaled by (1/n) for efficiency
        self.XtX_over_n = (1 / self.n) * self.X.T @ self.X
        self.XtY_over_n = (1 / self.n) * self.X.T @ self.Y
        self.YtX_over_n = (1 / self.n) * self.Y.T @ self.X

        # define manifods for sub-problems
        self.manifold_U = Stiefel(self.X.shape[1], self.rank)
        self.manifold_V = Stiefel(self.Y.shape[1], self.rank)

    def update_U(self, tolerance=1e-6):
        """
        Update U by solving the constrained minimization problem on the Stiefel manifold:
           min_U (1/2n) ||X U D||_F^2 + (rho/2) ||U_s_orth^T U||_F^2 + tr(U^T B_u)
            subject to U^T U = I_r
        This is solved using the Pymanopt package with the Riemannian conjugate gradient method.
        """
        # Compute B_u matrix (see paper for definition)
        # This matrix consolidates terms for the trace penalty in the objective
        Bu = (
            self.U_s_orth @ self.Gamma_u                      # Dual term
            - self.XtY_over_n @ self.V @ self.D             # Data-fitting interaction term
            - self.rho * self.U_s_orth @ self.Omega_u              # ADMM penalty term
        )

        # Define the cost function on the Stiefel manifold
        @function.numpy(self.manifold_U)
        def cost(U):
            # Term 1: reconstruction penalty ||XUD||_F^2 scaled by 1/(2n)
            term1 = (1 / (2 * self.n)) * np.linalg.norm(self.X @ U @ self.D, "fro") ** 2
        
            # Term 2: structural penalty on deviation from U_s_orth's subspace
            term2 = (self.rho / 2) * np.linalg.norm(self.U_s_orth.T @ U, "fro") ** 2
        
            # Term 3: linear trace term tr(U^T B_u)
            term3 = np.trace(U.T @ Bu)

            # Total cost
            return term1 + term2 + term3
        
        # Define the Euclidean gradient function
        @function.numpy(self.manifold_U)
        def euclidean_gradient(U):
            """
            Compute the Euclidean gradient of the cost function with respect to U.

            The cost function is:
                L(U) = (1/2n) * ||X U D||_F^2 + (rho/2) * ||U_s_orth^T U||_F^2 + tr(U^T B_u)

            Gradient terms:
                grad1: Gradient of the reconstruction loss (1/2n) * ||X U D||_F^2
                grad2: Gradient of the structural penalty (rho/2) * ||U_s_orth^T U||_F^2
                grad3: Gradient of the linear trace term tr(U^T B_u)

            Parameters:
                U (ndarray): Current value of U (p x r)

            Returns:
                ndarray: Euclidean gradient (same shape as U)
            """
            # Gradient of (1/2n) * ||X U D||_F^2 = (1/n) * Xᵀ X U D²
            grad1 = self.XtX_over_n @ (U @ self.D @ self.D)

            # Gradient of (rho/2) * ||U_s_orth^T U||_F^2 = rho * U_s_orth (U_s_orthᵀ U)
            grad2 = self.rho * self.U_s_orth @ (self.U_s_orth.T @ U)

            # Gradient of tr(Uᵀ B_u) = B_u
            grad3 = Bu

            # Sum all components
            return grad1 + grad2 + grad3

        # Set up the constrained optimization problem
        problem = Problem(manifold=self.manifold_U, cost=cost, euclidean_gradient=euclidean_gradient)

        # Solve the problem using Riemannian conjugate gradient optimizer
        # ------------------------------------------------------------------
        # Choose CG β-rule.
        #
        # • Hestenes–Stiefel (Pymanopt default) occasionally divides by
        #   ⟨diff, d_k⟩, which is 0 when the search space is rank-1.
        # • When rank == 1 we therefore fall back to Fletcher–Reeves, whose
        #   denominator is ⟨grad, grad⟩ ≥ 0 and never vanishes, eliminating
        #   the “invalid value encountered in divide” warning.
        # ------------------------------------------------------------------
        if self.rank == 1:
            optimizer = ConjugateGradient(
                beta_rule="FletcherReeves",      # safe for rank-1
                verbosity=0,
                min_gradient_norm=tolerance      # Use adaptive tolerance
            )
        else:
            optimizer = ConjugateGradient(       # default HS β-rule
                verbosity=0,
                min_gradient_norm=tolerance      # Use adaptive tolerance
            )

        # Defensive checks before optimization
        if not np.all(np.isfinite(self.U)):
            raise ValueError("Invalid values in U before update_U.")
        if not np.all(np.isfinite(self.D)):
            raise ValueError("Invalid values in D before update_U.")
        if not np.all(np.isfinite(self.X)):
            raise ValueError("Invalid values in X before update_U.")

        # run optimization
        result = optimizer.run(problem, initial_point=self.U)

        # Update U with the optimized value
        self.U = result.point

    def update_V(self, tolerance=1e-6):
        """
        Update V by solving the constrained minimization problem on the Stiefel manifold:
            min_V (rho/2) ||V_s_orth^T V||_F^2 + tr(V^T B_v)
            subject to V^T V = I_r
        This is solved using the Pymanopt package with the Riemannian conjugate gradient method.
        """
        # Compute B_v matrix (see paper for definition)
        # This matrix consolidates terms for the trace penalty in the objective
        Bv = (
            self.V_s_orth @ self.Gamma_v                         # Dual term
            - self.YtX_over_n @ self.U @ self.D                # Data-fitting interaction term
            - self.rho * self.V_s_orth @ self.Omega_v                 # ADMM penalty term
        )

        # Define the cost function on the Stiefel manifold
        @function.numpy(self.manifold_V)
        def cost(V):
            # Term 1: structural penalty on deviation from V_s_orth's subspace
            term1 = (self.rho / 2) * np.linalg.norm(self.V_s_orth.T @ V, "fro") ** 2

            # Term 2: linear trace term tr(V^T B_v)
            term2 = np.trace(V.T @ Bv)

            # Total cost
            return term1 + term2
        
        # Define the Euclidean gradient function
        @function.numpy(self.manifold_V)
        def euclidean_gradient(V):
            """
            Compute the Euclidean gradient of the cost function with respect to V.

            The cost function is:
                L(V) = (rho/2) * ||V_s_orthᵀ V||_F^2 + tr(Vᵀ B_v)

            Gradient terms:
                grad1: Gradient of the structural penalty ||V_s_orthᵀ V||_F^2
                grad2: Gradient of the linear trace term tr(Vᵀ B_v)

            Parameters:
                V (ndarray): Current value of V (q x r)

            Returns:
                ndarray: Euclidean gradient (same shape as V)
            """
            # === Term 1: Gradient of (rho/2) * ||V_s_orthᵀ V||_F^2 ===
            # This is a quadratic form: tr(Vᵀ P V), where P = V_s_orth V_s_orthᵀ
            # Gradient: ∇_V = rho * V_s_orth (V_s_orthᵀ V)
            grad1 = self.rho * self.V_s_orth @ (self.V_s_orth.T @ V)

            # === Term 2: Gradient of tr(Vᵀ B_v) ===
            # Gradient is simply the matrix B_v
            grad2 = Bv

            # === Total Gradient ===
            return grad1 + grad2

        # Set up the constrained optimization problem
        problem = Problem(manifold=self.manifold_V, cost=cost, euclidean_gradient=euclidean_gradient)

        # Solve the problem using Riemannian conjugate gradient optimizer
        # ------------------------------------------------------------------
        # Choose CG β-rule.
        #
        # • Hestenes–Stiefel (Pymanopt default) occasionally divides by
        #   ⟨diff, d_k⟩, which is 0 when the search space is rank-1.
        # • When rank == 1 we therefore fall back to Fletcher–Reeves, whose
        #   denominator is ⟨grad, grad⟩ ≥ 0 and never vanishes, eliminating
        #   the “invalid value encountered in divide” warning.
        # ------------------------------------------------------------------
        if self.rank == 1:
            optimizer = ConjugateGradient(
                beta_rule="FletcherReeves",      # safe for rank-1
                verbosity=0,
                min_gradient_norm=tolerance      # Use adaptive tolerance
            )
        else:
            optimizer = ConjugateGradient(       # default HS β-rule
                verbosity=0,
                min_gradient_norm=tolerance      # Use adaptive tolerance
            )

        # Defensive checks before optimization
        if not np.all(np.isfinite(self.V)):
            raise ValueError("Invalid values in V before update_V.")
        if not np.all(np.isfinite(self.D)):
            raise ValueError("Invalid values in D before update_V.")
        if not np.all(np.isfinite(self.Y)):
            raise ValueError("Invalid values in Y before update_V.")

        result = optimizer.run(problem, initial_point=self.V)

        # Update V with the optimized value
        self.V = result.point

    def update_D(self):
        """
        Update D using a closed-form solution based on least squares:
            Let A_d = (X U)ᵀ X U
                B_d = (1/n * Y V)ᵀ X U
            Then D is a diagonal matrix with:
                D_{jj} = B_d[j, j] / A_d[j, j]
            for j = 1, ..., r
        """
        # Retrieve current variables
        X, Y = self.X, self.Y                     # Input and output matrices
        U, V = self.U, self.V                     # Current estimates of U and V
        n = self.n                                # Sample size

        # Compute intermediate matrices
        XU = X @ U                                 # Shape: (n x r)
        YV = Y @ V                                 # Shape: (n x r)

        # Compute A_d = (XU)^T XU and B_d = (YV)^T XU
        A_d = (1 / n) * XU.T @ XU                            # Shape: (r x r), symmetric
        B_d = (1 / n) * YV.T @ XU                            # Shape: (r x r), generally not symmetric

        # Initialize diagonal matrix D
        r = self.rank
        D_new = np.zeros((r, r))
        eps = 1e-8  # Small number to avoid division by zero

        # Update each diagonal entry D_{jj} = B_d[j, j] / A_d[j, j]
        for j in range(r):
            if A_d[j, j] < eps or not np.isfinite(A_d[j, j]):
                print(f"Warning: A_d[{j},{j}] is too small ({A_d[j,j]:.2e}) or invalid. Setting D[{j},{j}] = 0")
                D_new[j, j] = 0.0
            else:
                D_new[j, j] = B_d[j, j] / A_d[j, j]

        if not np.all(np.isfinite(D_new)):
            raise ValueError("NaN or Inf detected in updated D matrix.")

        # Update internal variable
        self.D = D_new

    def update_Omega_u(self):
        """
        Update Omega_u using a proximal operator (soft-thresholding) to solve:
            min_Omega (rho/2) * ||Omega - (U_s_orth^T U + Gamma_u / rho)||_F^2 + lambda_u * ||Omega||_1

        The solution is given in closed-form as:
            Omega_u[i, j] = S_{lambda_u / rho}((U_s_orth^T U + Gamma_u / rho)[i, j])
        """
        # Retrieve current variables
        U, U_s_orth = self.U, self.U_s_orth
        Gamma_u = self.Gamma_u
        rho = self.rho
        lam = self.lambda_u

        # Compute the proximal input matrix
        Z = U_s_orth.T @ U + (1 / rho) * Gamma_u

        # Apply elementwise soft-thresholding
        kappa = lam / rho
        self.Omega_u = self._soft_thresholding(Z, kappa)

    def update_Omega_v(self):
        """
        Update Omega_v using a proximal operator (soft-thresholding) to solve:
            min_Omega (rho/2) * ||Omega - (V_s_orth^T V + Gamma_v / rho)||_F^2 + lambda_v * ||Omega||_1

        The solution is given in closed-form as:
            Omega_v[i, j] = S_{lambda_v / rho}((V_s_orth^T V + Gamma_v / rho)[i, j])
        """
        # Retrieve current variables
        V, V_s_orth = self.V, self.V_s_orth
        Gamma_v = self.Gamma_v
        rho = self.rho
        lam = self.lambda_v

        # Compute the proximal input matrix
        Z = V_s_orth.T @ V + (1 / rho) * Gamma_v

        # Apply elementwise soft-thresholding
        kappa = lam / rho
        self.Omega_v = self._soft_thresholding(Z, kappa)

    @staticmethod
    def _soft_thresholding(X, kappa):
        """
        Apply the elementwise soft-thresholding operator:
            S_kappa(a) = sign(a) * max(|a| - kappa, 0)

        Parameters:
            X (ndarray): Input matrix
            kappa (float): Threshold value

        Returns:
            ndarray: Thresholded matrix
        """
        return np.sign(X) * np.maximum(np.abs(X) - kappa, 0.0)

    def update_Gamma_u(self):
        """
        Update the dual variable Gamma_u using ADMM rule.
        """
        self.Gamma_u += self.rho * (self.U_s_orth.T @ self.U - self.Omega_u)

    def update_Gamma_v(self):
        """
        Update the dual variable Gamma_v using ADMM rule.
        """
        self.Gamma_v += self.rho * (self.V_s_orth.T @ self.V - self.Omega_v)

    def update_rho(self, primal_residual, dual_residual, mu=10, tau_inc=2.0, tau_dec=2.0):
        if primal_residual > mu * dual_residual:
            self.rho *= tau_inc
        elif dual_residual > mu * primal_residual:
            self.rho /= tau_dec

    def compute_objective(self):
        """
        Compute the value of the ADMM objective function at the current parameters.

        Returns:
            float: The current value of the ADMM objective function.
        """
        # Reconstruction loss
        residual = self.Y - self.X @ self.U @ self.D @ self.V.T
        loss_reconstruction = (1 / (2 * self.n)) * np.linalg.norm(residual, 'fro')**2

        # L1 penalties on Omega_u and Omega_v
        penalty_u = self.lambda_u * np.sum(np.abs(self.Omega_u))
        penalty_v = self.lambda_v * np.sum(np.abs(self.Omega_v))

        # Dual inner products
        inner_u = np.sum(self.Gamma_u * (self.U_s_orth.T @ self.U - self.Omega_u))
        inner_v = np.sum(self.Gamma_v * (self.V_s_orth.T @ self.V - self.Omega_v))

        # Quadratic penalty terms
        penalty_rho_u = (self.rho / 2) * np.linalg.norm(self.U_s_orth.T @ self.U - self.Omega_u, 'fro')**2
        penalty_rho_v = (self.rho / 2) * np.linalg.norm(self.V_s_orth.T @ self.V - self.Omega_v, 'fro')**2

        # Total objective
        objective = (
            loss_reconstruction +
            penalty_u +
            penalty_v +
            inner_u +
            inner_v +
            penalty_rho_u +
            penalty_rho_v
        )

        return objective
    
    def compute_residuals(self):
        """
        Compute the primal and dual residuals based on Boyd et al.'s ADMM criteria.

        Returns:
            (float, float): primal_residual, dual_residual
        """
        # === PRIMAL RESIDUAL ===
        R_p_u = self.U_s_orth.T @ self.U - self.Omega_u
        R_p_v = self.V_s_orth.T @ self.V - self.Omega_v

        p, r = self.U.shape
        q = self.V.shape[0]
        r_u = self.U_s.shape[1]
        r_v = self.V_s.shape[1]

        eps = 1e-10  # small constant to prevent division by zero
        denom_u = np.sqrt(max((p - r_u) * r, eps))
        denom_v = np.sqrt(max((q - r_v) * r, eps))
        primal_residual = 0.5 * (
            np.linalg.norm(R_p_u, 'fro') / denom_u
            + np.linalg.norm(R_p_v, 'fro') / denom_v
        )

        # === DUAL RESIDUAL (via Riemannian gradients) ===

        # --- Riemannian gradient of U ---
        X, Y, D, V = self.X, self.Y, self.D, self.V
        rho, n = self.rho, self.n
        U_s_orth, Gamma_u, Omega_u = self.U_s_orth, self.Gamma_u, self.Omega_u

        Bu = (
            U_s_orth @ Gamma_u
            - self.XtY_over_n @ V @ D
            - rho * U_s_orth @ Omega_u
        )

        manifold_U = Stiefel(p, r)
        
        # Define the Euclidean gradient function
        @function.numpy(manifold_U)
        def euclidean_gradient_U(U):
            grad1 = self.XtX_over_n @ U @ D @ D
            grad2 = rho * U_s_orth @ (U_s_orth.T @ U)
            grad3 = Bu
            return grad1 + grad2 + grad3
        
        egrad_U = euclidean_gradient_U(self.U)
        rgrad_U = manifold_U.euclidean_to_riemannian_gradient(self.U, egrad_U)

        # --- Riemannian gradient of V ---
        U = self.U
        V_s_orth, Gamma_v, Omega_v = self.V_s_orth, self.Gamma_v, self.Omega_v

        Bv = (
            V_s_orth @ Gamma_v
            - self.YtX_over_n @ U @ D
            - rho * V_s_orth @ Omega_v
        )

        manifold_V = Stiefel(q, r)
        
        # Define the Euclidean gradient function
        @function.numpy(manifold_V)
        def euclidean_gradient_V(V):
            grad1 = rho * V_s_orth @ (V_s_orth.T @ V)
            grad2 = Bv
            return grad1 + grad2
        
        egrad_V = euclidean_gradient_V(self.V)
        rgrad_V = manifold_V.euclidean_to_riemannian_gradient(self.V, egrad_V)

        dual_residual = 0.5 * (
            np.linalg.norm(rgrad_U, 'fro') / np.sqrt(p * r)
            + np.linalg.norm(rgrad_V, 'fro') / np.sqrt(q * r)
        )

        return primal_residual, dual_residual


    def fit(self, tol=1e-3, verbose=True, converg_warning=False):
        """
        Run the main optimization loop of the SMaRT algorithm.

        Parameters:
            tol (float): Tolerance for early stopping.
            verbose (bool): Whether to print residuals and objective values each iteration.
            converg_warning (bool): Whether to print warning when the algorithm did not converge.

        Returns:
            dict: Dictionary containing the final estimates and convergence diagnostics.
        """
        while self.t < self.t_max:
            # Compute adaptive tolerance for this iteration
            curr_tol = self._get_dynamic_tolerance(self.t)

            # === Main ADMM updates ===
            self.update_U(tolerance=curr_tol)
            self.update_V(tolerance=curr_tol)
            self.update_D()
            self.update_Omega_u()
            self.update_Omega_v()
            self.update_Gamma_u()
            self.update_Gamma_v()

            # === Track objective and residuals ===
            obj_val = self.compute_objective()
            self.objective_history.append(obj_val)

            # Compute residuals and log them
            primal_residual, dual_residual = self.compute_residuals()
            self.primal_residual_history.append(primal_residual)
            self.dual_residual_history.append(dual_residual)

            # adjust rho
            self.update_rho(primal_residual, dual_residual)

            if verbose:
                print(
                    f"[Iter {self.t:3d}] "
                    f"Obj: {obj_val:.6e} | "
                    f"Primal Res: {primal_residual:.2e} | "
                    f"Dual Res: {dual_residual:.2e} | "
                    f"tolerance = {tol:.2e}"
                )

            if primal_residual <= tol and dual_residual <= tol:
                if verbose:
                    print("Converged: both residuals below tolerance.")
                break

            self.t += 1

        # Warn if we exited due to hitting t_max without convergence
        if self.t == self.t_max and converg_warning:
            print(
                f"WARNING: Algorithm did not converge after {self.t_max} iterations.\n"
                f"Final primal residual = {primal_residual:.2e}, "
                f"dual residual = {dual_residual:.2e}, "
                f"tolerance = {tol:.2e}"
            )

        return {
            "U": self.U,
            "V": self.V,
            "D": self.D,
            "Omega_u": self.Omega_u,
            "Omega_v": self.Omega_v,
            "Gamma_u": self.Gamma_u,
            "Gamma_v": self.Gamma_v,
            "objective_history": self.objective_history,
            "primal_residual_history": self.primal_residual_history,
            "dual_residual_history": self.dual_residual_history
        }
    
    def plot_convergence(self, filename="SMaRT_convergence_plot.pdf"):
        """
        Plot and save the convergence curves for the objective value, 
        primal residual, and dual residual.

        Parameters:
            filename (str): Path to save the PDF figure.
        """
        iterations = range(len(self.objective_history))

        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        # Plot objective values
        axs[0].plot(iterations, self.objective_history, marker='o')
        axs[0].set_ylabel("Objective Value")
        axs[0].set_title("SMaRT Convergence")

        # Plot primal residual
        axs[1].plot(iterations, self.primal_residual_history, marker='o', label="Primal Residual")
        axs[1].set_ylabel("Primal Residual")
        axs[1].set_yscale("log")
        axs[1].legend()

        # Plot dual residual
        axs[2].plot(iterations, self.dual_residual_history, marker='o', label="Dual Residual", color="orange")
        axs[2].set_ylabel("Dual Residual")
        axs[2].set_xlabel("Iteration")
        axs[2].set_yscale("log")
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(filename, format="pdf")
        plt.show()

        print(f"Convergence plot saved to '{filename}'.")

    def select_hyperparameters_via_bic(self, lambda_grid_u=None, lambda_grid_v=None, tie_lambdas=True,
                                       C_init=None, tol=1e-3, verbose=False, 
                                       parallel=False, use_optuna=False, 
                                       n_trials=10, optuna_verbose=False, n_jobs=1, converg_warning=False):
        """
        Select optimal (lambda_u, lambda_v) by minimizing the Bayesian Information Criterion (BIC).

        Parameters:
            lambda_grid_u (list of float or None): Candidate values for lambda_u. Default is logspace(-3, 2, 10).
            lambda_grid_v (list of float or None): Candidate values for lambda_v. Default is logspace(-3, 2, 10).
            tie_lambdas (bool): If True, tune lambda_u == lambda_v jointly to reduce search space.
            C_init (ndarray or None): Optional initialization for C.
            tol (float): Tolerance for solver convergence.
            verbose (bool): If True, print BIC values during search.
            parallel (bool): If True, use parallel computation over grid search.
            use_optuna (bool): If True, use Optuna for hyperparameter optimization.
            n_trials (int): Number of trials to run with Optuna.
            optuna_verbose (bool): If True, show progress during Optuna optimization.
            n_jobs (int): Number of parallel jobs to run.
            converg_warning (bool): Whether to print warning when the algorithm did not converge.

        Returns:
            dict: Dictionary with best (lambda_u, lambda_v), BIC, and model parameters.
        """
        X, Y, r = self.X, self.Y, self.rank
        n, p = X.shape
        q = Y.shape[1]

        if lambda_grid_u is None:
            lambda_grid_u = np.logspace(-3, 2, 10)
        if lambda_grid_v is None:
            lambda_grid_v = np.logspace(-3, 2, 10)

        if tie_lambdas and not np.allclose(lambda_grid_u, lambda_grid_v):
            raise ValueError("When tie_lambdas=True, lambda_grid_u must be equal to lambda_grid_v.")

        def compute_bic(lam_u, lam_v):
            solver = SMARTSolver(C0=self.C0, r_u=self.r_u, r_v=self.r_v, 
                                 lambda_u=lam_u, lambda_v=lam_v, 
                                 gamma=self.gamma, t_max=self.t_max)
            solver.initialize(r=r, X=X, Y=Y, C=C_init)
            solver.fit(tol=tol, verbose=False, converg_warning=converg_warning)

            C_hat = solver.U @ solver.D @ solver.V.T
            residual = Y - X @ C_hat
            rss = np.linalg.norm(residual, 'fro') ** 2

            bic = np.log(rss / (n * q)) + (np.count_nonzero(solver.Omega_u) + np.count_nonzero(solver.Omega_v)) * np.log(n * q) / (n * q)

            if verbose:
                print(f"[lambda_u={lam_u:.2e}, lambda_v={lam_v:.2e}] BIC = {bic:.4f}")

            return {
                "lambda_u": lam_u,
                "lambda_v": lam_v,
                "bic": bic,
                "solver": solver,
                "C_hat": C_hat,
                "U_hat": solver.U,
                "D_hat": solver.D,
                "V_hat": solver.V,
                "Omega_u_hat": solver.Omega_u,
                "Omega_v_hat": solver.Omega_v,
                "Gamma_u_hat": solver.Gamma_u,
                "Gamma_v_hat": solver.Gamma_v,
                "primal_residual": solver.primal_residual_history[-1],
                "dual_residual": solver.dual_residual_history[-1]
            }

        if use_optuna:
            def objective(trial):
                log_lam = trial.suggest_float("log_lambda", np.log10(min(lambda_grid_u)), np.log10(max(lambda_grid_u)))
                lam = 10 ** log_lam
                if tie_lambdas:
                    result = compute_bic(lam, lam)
                else:
                    log_lam_v = trial.suggest_float("log_lambda_v", np.log10(min(lambda_grid_v)), np.log10(max(lambda_grid_v)))
                    result = compute_bic(lam, 10 ** log_lam_v)
                trial.set_user_attr("result", result)
                return result["bic"]

            if not optuna_verbose:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            sampler = TPESampler(seed=42)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs if parallel else 1)

            best_trial = study.best_trial
            return best_trial.user_attrs["result"]
        else:
            if tie_lambdas:
                grid = [(lam, lam) for lam in lambda_grid_u]
            else:
                grid = list(product(lambda_grid_u, lambda_grid_v))

            if parallel:
                results = Parallel(n_jobs=n_jobs)(delayed(compute_bic)(lam_u, lam_v) for lam_u, lam_v in grid)
            else:
                results = [compute_bic(lam_u, lam_v) for lam_u, lam_v in grid]

            best_params = min(results, key=lambda d: d["bic"])
            return best_params