import numpy as np
from sklearn.model_selection import KFold
import optuna
from optuna.samplers import TPESampler

from .solver import SMARTSolver
from .rank_selector import RankSelectorRSC
from .utils import fit_baseline

class SMART:
    """
    SMART: Model‑selection wrapper for the SMART algorithm.

    ----------------------------
    Overview
    ----------------------------
    `SMART` augments the SMART solver with an *automatic* model‑selection
    pipeline.  It orchestrates three key tasks in succession:

    1. **Intrinsic rank estimation** (`select_rank`) using `RankSelectorRSC`.
    2. **Structural subspace selection** (`select_structural_ranks`) to choose
       the left/right ranks `(r_u, r_v)` that define the subspaces supplied to
       SMART.
    3. **Hyper‑parameter tuning & fitting** (`fit`) by minimising the Bayesian
       Information Criterion (BIC), optionally powered by Optuna.

    ----------------------------
    High‑level Workflow
    ----------------------------
    >>> model = SMART(X, Y, C0)
    >>> model.run_full_selection(verbose=True)
    >>> est = model.get_estimates()

    The helper `run_full_selection` chains **all** stages so that a single call
    returns a fitted model with minimal boilerplate.

    ----------------------------
    Attributes (set in `__init__`)
    ----------------------------
    X, Y, C0 : ndarray
        Core data matrices supplied by the user.
    rank_selector_kwargs : dict
        Optional keyword arguments forwarded to `RankSelectorRSC`.

    ----------------------------
    Attributes (populated later)
    ----------------------------
    r_hat : int
        Intrinsic rank estimated in Stage 1.
    best_r_u, best_r_v : int
        Structural ranks selected in Stage 2.
    fitted_solver : dict or None
        Dictionary of outputs returned by SMART after Stage 3.

    See individual method docstrings for fine‑grained details.
    """
    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 C0: np.ndarray, 
                 rank_selector_kwargs: dict = None):
        """
        Initialize the SMART model selector.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n, p), where n is the number of samples
                            and p is the number of predictors.
            Y (np.ndarray): Response matrix of shape (n, q), where q is the number of response variables.
            C0 (np.ndarray): Source matrix of shape (p, q), used for extracting structured subspaces.
                             It is typically a noisy version of the ground truth regression matrix.
            rank_selector_kwargs (dict, optional): Optional dictionary of keyword arguments passed
                                                   to the rank selector class (RankSelectorRSC).
        """

        # ----------------------------
        # Validate input types
        # ----------------------------
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array.")
        if not isinstance(Y, np.ndarray):
            raise TypeError("Y must be a NumPy array.")
        if not isinstance(C0, np.ndarray):
            raise TypeError("C0 must be a NumPy array.")

        # ----------------------------
        # Validate dimensions
        # Ensure that the source matrix C0 matches the dimensions implied by X and Y
        # ----------------------------
        if X.shape[1] != C0.shape[0]:
            raise ValueError(f"X.shape[1] = {X.shape[1]} does not match C0.shape[0] = {C0.shape[0]}")
        if Y.shape[1] != C0.shape[1]:
            raise ValueError(f"Y.shape[1] = {Y.shape[1]} does not match C0.shape[1] = {C0.shape[1]}")

        # ----------------------------
        # Assign inputs to object attributes
        # ----------------------------
        self.X: np.ndarray = X                      # (n, p) design matrix
        self.Y: np.ndarray = Y                      # (n, q) response matrix
        self.C0: np.ndarray = C0                    # (p, q) source regression matrix
        self.rank_selector_kwargs: dict = rank_selector_kwargs or {}

        # ----------------------------
        # Initialize internal state variables
        # These will be populated during model selection and fitting
        # ----------------------------
        self.r_hat: int = None                      # Estimated rank of coefficient matrix
        self.best_r_u: int = None                   # Optimal structural rank for U
        self.best_r_v: int = None                   # Optimal structural rank for V
        self.fitted_solver = None                   # Final fitted solver object after SMART optimization
        self.rank_selection_scores: list = []       # List to store CV scores for each (r_u, r_v) candidate
    
    def compute_C0_rank(self, tol: float = 1e-10):
        """
        Compute the numerical rank of the C0 matrix using SVD.

        Parameters:
            tol (float): Threshold for determining non-zero singular values. 
                         Default is 1e-10.

        Returns:
            int: Numerical rank of self.C0.
        """
        if self.C0 is None:
            raise ValueError("C0 is not defined. Please provide a valid C0 matrix.")

        _, singular_values, _ = np.linalg.svd(self.C0, full_matrices=False)
        numerical_rank = np.sum(singular_values > tol)
        return numerical_rank

    # ------------------------------------------------------------------
    # Stage 1 – Intrinsic Rank Selection
    # ------------------------------------------------------------------
    def select_rank(self, min_rank=1):
        """
        Estimate the intrinsic rank `r` of the regression coefficient matrix
        using the RankSelectorRSC method. If the estimated rank exceeds the
        rank of the source matrix, it is capped accordingly.

        The result is stored in `self.r_hat`.

        Parameters
        ----------
        min_rank : int, default=1
            Smallest rank returned.  Use min_rank=1 to bypass rank-0 pathology.

        Returns:
            int: The estimated (and possibly capped) rank of the regression matrix.
        """
        selector = RankSelectorRSC(**self.rank_selector_kwargs)
        self.r_hat = selector.select_rank(self.X, self.Y)

        if self.r_hat == 0:
            print("Warning: Estimated rank is zero. Please verify your data or adjust the rank selection parameters.")

        C0_rank = self.compute_C0_rank()
        if self.r_hat > C0_rank:
            print("Warning: Estimated target rank exceeds source rank. Capping target rank to match source rank.")
            self.r_hat = C0_rank

        # --- promote if below min_rank and feasible ---
        if self.r_hat < min_rank:
            print(
                f"Note: Estimated rank r̂ = {self.r_hat} is below the minimum threshold "
                f"({min_rank}). Promoting to r̂ = {min_rank} for stability."
            )
            self.r_hat = min(min_rank, C0_rank)

        return self.r_hat
    
    def _construct_structural_rank_grid(self, r: int, C0_rank: int, max_dim: int) -> np.ndarray:
        """
        Construct a default grid of candidate structural ranks including:
          (0) Range from 0 to r (for boundary cases),
          (1) Range from r to C0_rank (source rank),
          (2) Range from C0_rank to max ambient dimension.

        Parameters:
            r (int): Intrinsic rank (estimated or fixed).
            C0_rank (int): Numerical rank of the source matrix C0.
            max_dim (int): max(p, q) ambient dimension.

        Returns:
            np.ndarray: Sorted unique grid of candidate ranks.
        """
        # Stage 0: Include values from 0 to r
        default_grid_0 = np.arange(0, r + 1)

        # Stage 1: From r to C0_rank
        if 5 * r >= C0_rank:
            default_grid_1 = np.round(np.linspace(r, C0_rank, 5)).astype(int)
        elif 10 * r >= C0_rank:
            default_grid_1 = np.round(np.linspace(r, C0_rank, 10)).astype(int)
        else:
            default_grid1a = (np.arange(1, 6) * r).astype(int)
            default_grid1b = np.round(np.linspace(6 * r, C0_rank, 5)).astype(int)
            default_grid_1 = np.concatenate((default_grid1a, default_grid1b))

        # Stage 2: From C0_rank to max_dim
        if 5 * C0_rank >= max_dim:
            default_grid_2 = np.round(np.linspace(C0_rank, max_dim, 5)).astype(int)
        elif 10 * C0_rank >= max_dim:
            default_grid_2 = np.round(np.linspace(C0_rank, max_dim, 10)).astype(int)
        else:
            default_grid2a = (np.arange(1, 6) * C0_rank).astype(int)
            default_grid2b = np.round(np.linspace(6 * C0_rank, max_dim, 5)).astype(int)
            default_grid_2 = np.concatenate((default_grid2a, default_grid2b))

        # Combine and deduplicate
        return np.unique(np.concatenate((default_grid_0, default_grid_1, default_grid_2)))

    # ------------------------------------------------------------------
    # Stage 2 – Structural rank selection (grid / Optuna)
    # ------------------------------------------------------------------
    def select_structural_ranks(self, candidate_ru=None, candidate_rv=None,
                                tie_ranks=True, verbose=False,
                                n_folds=5, fit_kwargs=None,
                                use_optuna_rank=False, n_trials=15):
        """
        Select structural ranks (r_u, r_v) using either grid search with cross-validation
        or Optuna-based optimization. The goal is to find the (r_u, r_v) pair that minimizes
        the average validation error over folds.

        Parameters:
            candidate_ru (list[int], optional): Candidate values for r_u.
            candidate_rv (list[int], optional): Candidate values for r_v.
            tie_ranks (bool): If True, only consider r_u == r_v.
            verbose (bool): If True, print intermediate results during evaluation.
            n_folds (int): Number of folds for cross-validation.
            fit_kwargs (dict, optional): Additional keyword arguments passed to `fit()`.
            use_optuna_rank (bool): If True, use Optuna to select structural ranks.
            n_trials (int): Number of Optuna trials to run (if used).
        """
        if self.C0 is None:
            raise ValueError("select_structural_ranks requires a non-null C0 matrix.")
        if self.r_hat is None:
            raise ValueError("Please run select_rank() before selecting structural ranks.")
        
        fit_kwargs = fit_kwargs or {}

        # Compute the upper bound of rank from C0
        C0_rank = self.compute_C0_rank()
        r = self.r_hat
        p, q = self.C0.shape
        max_dim = max(p, q)

        # ------------------------------------------------------------------------
        # Construct a default grid of candidate structural ranks (r_u and r_v)
        # ------------------------------------------------------------------------

        default_grid = self._construct_structural_rank_grid(r, C0_rank, max_dim)

        # Use provided or default candidate rank grids
        if candidate_ru is None:
            candidate_ru = default_grid
        if candidate_rv is None:
            candidate_rv = default_grid

        # If using Optuna for rank selection
        if use_optuna_rank:
            candidate_ru = np.unique(candidate_ru)
            candidate_rv = np.unique(candidate_rv)

            def objective(trial):
                # Sample r_u and r_v from range
                r_u = trial.suggest_int("r_u", int(min(candidate_ru)), int(max(candidate_ru)))
                r_v = r_u if tie_ranks else trial.suggest_int("r_v", int(min(candidate_rv)), int(max(candidate_rv)))
                
                r_u = min(r_u, p)
                r_v = min(r_v, q)

                # Perform K-fold CV to evaluate error for this trial
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
                cv_errors = []
                for train_idx, val_idx in kf.split(self.X):
                    X_train, X_val = self.X[train_idx], self.X[val_idx]
                    Y_train, Y_val = self.Y[train_idx], self.Y[val_idx]
                    model = SMART(X_train, Y_train, C0=self.C0, rank_selector_kwargs=self.rank_selector_kwargs)
                    model.r_hat = self.r_hat
                    model.fit(r_u=r_u, r_v=r_v, **fit_kwargs)
                    C_hat = model.get_estimates()["C_hat"]
                    val_error = np.linalg.norm(Y_val - X_val @ C_hat, 'fro')**2 / Y_val.size
                    cv_errors.append(val_error)

                score = np.mean(cv_errors)
                if verbose:
                    print(f"[Optuna] (r_u={r_u}, r_v={r_v}) → CV = {score:.4f}")
                return score

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = TPESampler(seed=42)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(objective, n_trials=n_trials)

            # Store best structural ranks
            self.best_r_u = study.best_params["r_u"]
            self.best_r_v = self.best_r_u if tie_ranks else study.best_params["r_v"]
            if verbose:
                print(f"Optuna best (r_u, r_v) = ({self.best_r_u}, {self.best_r_v})")
            return

        # Otherwise, do exhaustive grid search
        candidate_pairs = [(r, r) for r in candidate_ru] if tie_ranks else [
            (ru, rv) for ru in candidate_ru for rv in candidate_rv]
        self.rank_selection_scores = []
        best_score = np.inf
        best_result = None

        for r_u_raw, r_v_raw in candidate_pairs:
            r_u = min(r_u_raw, p)
            r_v = min(r_v_raw, q)

            # Perform K-fold CV
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            cv_errors = []
            for train_idx, val_idx in kf.split(self.X):
                X_train, X_val = self.X[train_idx], self.X[val_idx]
                Y_train, Y_val = self.Y[train_idx], self.Y[val_idx]
                model = SMART(X_train, Y_train, C0=self.C0, rank_selector_kwargs=self.rank_selector_kwargs)
                model.r_hat = self.r_hat
                model.fit(r_u=r_u, r_v=r_v, **fit_kwargs)
                C_hat = model.get_estimates()["C_hat"]
                val_error = np.linalg.norm(Y_val - X_val @ C_hat, 'fro')**2 / Y_val.size
                cv_errors.append(val_error)

            score = np.mean(cv_errors)
            self.rank_selection_scores.append({"r_u": r_u, "r_v": r_v, "score": score})
            if verbose:
                print(f"Evaluated (r_u={r_u}, r_v={r_v}) → CV = {score:.4f}")
            if score < best_score:
                best_score = score
                best_result = (r_u, r_v)

        if best_result is None:
            raise RuntimeError("No valid (r_u, r_v) pair found in the candidate grid.")

        # Save best found rank pair
        self.best_r_u, self.best_r_v = best_result
        if verbose:
            print(f"Best (r_u, r_v) = ({self.best_r_u}, {self.best_r_v}) with CV = {best_score:.4f}")

    # ------------------------------------------------------------------
    # Stage 3 – Model fitting
    # ------------------------------------------------------------------
    def fit(self, r_u, r_v, lambda_grid_u=None, lambda_grid_v=None, tie_lambdas=True,
            use_optuna=False, parallel=False, n_trials=10, n_jobs=1, tol=1e-3, t_max=500,
            converg_warning=False):
        """
        Fit the SMART solver using selected structural ranks and BIC-based hyperparameter tuning.

        Parameters:
            r_u (int): Structural rank for left singular vectors.
            r_v (int): Structural rank for right singular vectors.
            lambda_grid_u, lambda_grid_v (list[float] or None): Grid for regularization parameters.
            tie_lambdas (bool): Whether to restrict lambda_u == lambda_v.
            use_optuna (bool): Use Optuna for hyperparameter tuning.
            parallel (bool): Enable parallel evaluation.
            n_trials (int): Number of Optuna trials.
            n_jobs (int): Number of parallel jobs (if applicable).
            tol (float): Tolerance for convergence.
            t_max (int): Maximum iterations for solver.
            converg_warning (bool): Whether to show convergence warnings.
        """
        p = self.X.shape[1]
        q = self.Y.shape[1]

        if not (0 <= r_u <= p):
            raise ValueError(f"Invalid r_u = {r_u}. It must lie in [0, {p}]")

        if not (0 <= r_v <= q):
            raise ValueError(f"Invalid r_v = {r_v}. It must lie in [0, {q}]")

        if self.r_hat is None:
            self.select_rank()
        
        solver = SMARTSolver(C0=self.C0, r_u=r_u, r_v=r_v, t_max=t_max)
        solver.initialize(r=self.r_hat, X=self.X, Y=self.Y)
        C_init, _ = fit_baseline(self.X, self.Y, model_type="ridge", alphas=np.logspace(-3, 2, 10))
        solution = solver.select_hyperparameters_via_bic(
            C_init=C_init, use_optuna=use_optuna, n_trials=n_trials, parallel=parallel,
            lambda_grid_u=lambda_grid_u, lambda_grid_v=lambda_grid_v, tie_lambdas=tie_lambdas,
            n_jobs=n_jobs, tol=tol, converg_warning=converg_warning
        )
        self.fitted_solver = solution

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_estimates(self):
        """
        Return the estimated coefficient matrix and its low-rank decomposition.

        Returns:
            dict: Dictionary with keys 'C_hat', 'U', 'D', and 'V' representing the final estimate.
        """
        if self.fitted_solver is not None:
            return {
                "C_hat": self.fitted_solver["C_hat"],
                "U": self.fitted_solver["U_hat"],
                "D": self.fitted_solver["D_hat"],
                "V": self.fitted_solver["V_hat"]
            }
        else:
            raise RuntimeError("Model has not been fitted. Please call fit() before requesting estimates.")

    def run_full_selection(self, candidate_ru=None, candidate_rv=None,
                           tie_ranks=True, verbose=False,
                           n_folds=5, fit_kwargs=None,
                           use_optuna_rank=False, n_trials=10,
                           fixed_rank=None):
        """
        Perform full model selection and estimation:
        1. Select or assign intrinsic rank r_hat.
        2. Select structural ranks (r_u, r_v).
        3. Fit the SMART model using selected ranks.

        Parameters:
            candidate_ru (list[int], optional): Candidate values for r_u.
            candidate_rv (list[int], optional): Candidate values for r_v.
            tie_ranks (bool): Whether to enforce r_u == r_v.
            verbose (bool): Whether to print progress.
            n_folds (int): Number of CV folds.
            fit_kwargs (dict, optional): Additional arguments for `fit()`.
            use_optuna_rank (bool): If True, use Optuna to choose structural ranks.
            n_trials (int): Number of trials for Optuna.
            fixed_rank (int or None): If provided, bypass intrinsic rank estimation and use this fixed rank.
        """
        fit_kwargs = fit_kwargs or {}

        # ------------------------------------------------------------------------
        # STEP 1: Intrinsic rank estimation (RSC) or use fixed value
        # ------------------------------------------------------------------------
        if fixed_rank is not None:
            if not isinstance(fixed_rank, int):
                raise TypeError(f"fixed_rank must be an integer or None. Got {type(fixed_rank)} instead.")
            self.r_hat = fixed_rank
            if verbose:
                print(f"Using pre-specified fixed rank: {self.r_hat}")
        else:
            self.select_rank(min_rank=1)
            if verbose:
                print(f"Estimated rank is {self.r_hat}")

        # ------------------------------------------------------------------------
        # STEP 2: Structural subspace selection
        # ------------------------------------------------------------------------
        self.select_structural_ranks(
            candidate_ru=candidate_ru,
            candidate_rv=candidate_rv,
            tie_ranks=tie_ranks,
            verbose=verbose,
            n_folds=n_folds,
            use_optuna_rank=use_optuna_rank,
            n_trials=n_trials,
            fit_kwargs={'tol': 1e-2, 't_max': 100, **fit_kwargs}
        )

        # ------------------------------------------------------------------------
        # STEP 3: Final SMART model fitting
        # ------------------------------------------------------------------------
        self.fit(
            r_u=self.best_r_u,
            r_v=self.best_r_v,
            **fit_kwargs
        )