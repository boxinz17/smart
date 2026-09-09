"""Public two-stage estimator with explicit source and tuning inputs."""
from __future__ import annotations

import numbers
import numpy as np

from .anchors import AnchorFailure, select_anchor
from .calibration import (CalibrationError, Margins, PracticalCalibration,
                          PrescribedCalibration, SourceAccuracyError, resolve_calibration)
from .chart import AnchorChart
from .initialization import InitializationFailure, reduced_lasso
from .solver import refine
from .source import ExactSource, NoisySource, _positive_real, prepare_source
from .spectral import project_singular_values
from .thresholding import hard_threshold


class FitFailure(RuntimeError):
    """An algorithmic or numerical failure, carrying an inspectable status."""
    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(f"{status}: {message}")


def _data(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    x = np.asarray(value, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a nonempty finite two-dimensional array")
    return x


class SparseSMART:
    """Sparse spectral transfer regression from a supplied source experiment.

    Parameters correspond to the declared inputs of Algorithm Two-stage SMART.
    There is no implicit centering, scaling, rank selection, or sample split.
    ``coefficient_`` has shape (n_features, n_responses), so Yhat=X@coefficient_.
    The package uses finite precision; successful fits are not theorem certificates.
    """

    def __init__(
        self, *, rank: int, source_rank: int, sparsity: tuple[int, int],
        margins: Margins, calibration: PrescribedCalibration | PracticalCalibration,
        iterations: int = 100, max_backtracks: int = 60,
        lasso_tol: float = 1e-9, lasso_max_iter: int = 20000,
        qr_max_exchanges: int = 10000, orthogonality_tol: float = 1e-9,
        tie_tol: float = 1e-12, raise_on_failure: bool = False,
        spectral_step: str = "auto", stationarity_tol: float | None = None,
        initialization_spectrum: str = "auto", refinement_solver: str = "auto",
    ):
        if not isinstance(initialization_spectrum, str) or initialization_spectrum not in ("auto", "projected", "reject"):
            raise ValueError("initialization_spectrum must be 'auto', 'projected', or 'reject'")
        if not isinstance(refinement_solver, str) or refinement_solver not in ("auto", "chart", "anchor_projected"):
            raise ValueError("refinement_solver must be 'auto', 'chart', or 'anchor_projected'")
        self.rank, self.source_rank, self.sparsity = rank, source_rank, sparsity
        self.margins, self.calibration = margins, calibration
        self.iterations, self.max_backtracks = iterations, max_backtracks
        self.lasso_tol, self.lasso_max_iter = lasso_tol, lasso_max_iter
        self.qr_max_exchanges = qr_max_exchanges
        self.orthogonality_tol, self.tie_tol = orthogonality_tol, tie_tol
        self.raise_on_failure = raise_on_failure
        self.spectral_step, self.stationarity_tol = spectral_step, stationarity_tol
        self.initialization_spectrum = initialization_spectrum
        self.refinement_solver = refinement_solver

    def _failure(self, status, message):
        self.status_, self.message_, self.success_ = status, str(message), False
        if getattr(self, "termination_reason_", None) is None:
            self.termination_reason_ = status
        self.converged_ = False
        self.diagnostics_["status"] = status
        self.diagnostics_["termination_reason"] = self.termination_reason_
        if self.raise_on_failure:
            raise FitFailure(status, str(message))
        return self

    def fit(self, X, Y, *, source: ExactSource | NoisySource, validation_data=None):
        """Fit both stages to the same observations; return self.

        Invalid input raises ValueError. Algorithmic failures are returned in
        status_ (or raise FitFailure if requested). Following a refinement
        failure, last_coefficient_ represents the last accepted state; prediction
        requires explicit allow_partial=True in that case. Optional validation
        data select the best accepted iterate, including the initializer; they
        never enter initialization, gradients, or line-search acceptance.
        """
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        X, Y = _data(X, "X"), _data(Y, "Y")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of observations")
        if self.spectral_step not in ("auto", "projected", "reject"):
            raise ValueError("spectral_step must be 'auto', 'projected', or 'reject'")
        if not isinstance(self.initialization_spectrum, str) or self.initialization_spectrum not in ("auto", "projected", "reject"):
            raise ValueError("initialization_spectrum must be 'auto', 'projected', or 'reject'")
        if not isinstance(self.refinement_solver, str) or self.refinement_solver not in ("auto", "chart", "anchor_projected"):
            raise ValueError("refinement_solver must be 'auto', 'chart', or 'anchor_projected'")
        if self.stationarity_tol is not None:
            _positive_real(self.stationarity_tol, "stationarity_tol")
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
                raise ValueError("validation_data must contain (X_validation, Y_validation)")
            XV, YV = (_data(value, name) for value, name in zip(
                validation_data, ("X_validation", "Y_validation")))
            if XV.shape[0] != YV.shape[0] or XV.shape[1] != X.shape[1] or YV.shape[1] != Y.shape[1]:
                raise ValueError("validation_data must have matching rows, predictors, and responses")
        for name, value, minimum in (
            ("iterations", self.iterations, 0), ("max_backtracks", self.max_backtracks, 0),
            ("lasso_max_iter", self.lasso_max_iter, 1), ("qr_max_exchanges", self.qr_max_exchanges, 1),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (("lasso_tol", self.lasso_tol),
                            ("orthogonality_tol", self.orthogonality_tol), ("tie_tol", self.tie_tol)):
            _positive_real(value, name)
        if not isinstance(source, (ExactSource, NoisySource)):
            raise ValueError("source must be ExactSource or NoisySource")
        if not isinstance(self.margins, Margins):
            raise ValueError("margins must be Margins")
        self.n_features_in_, self.n_responses_ = X.shape[1], Y.shape[1]
        self.n_iter_, self.history_, self.success_ = 0, [], False
        self.termination_reason_, self.converged_, self.optimization_converged_ = None, False, False
        self.diagnostics_ = {"theorem_certified": False, "arithmetic": "float64",
                             "same_target_data": True, "input_rescaling": False}
        source_noise = source.noise_std if isinstance(source, NoisySource) else 0.
        source_gap = source.gap_lower if isinstance(source, NoisySource) else None
        cluster_size = source.cluster_size if isinstance(source, NoisySource) else 1
        try:
            self.calibration_ = resolve_calibration(
                self.calibration, n=X.shape[0], p=X.shape[1], q=Y.shape[1],
                rank=self.rank, source_rank=self.source_rank, sparsity=self.sparsity,
                margins=self.margins, source_noise=source_noise,
                source_gap=source_gap, cluster_size=cluster_size,
            )
        except SourceAccuracyError as error:
            return self._failure("source_accuracy_failed", error)
        except CalibrationError as error:
            # Calibration errors are malformed inputs or unrepresentable constants.
            raise ValueError(str(error)) from error
        self.diagnostics_["calibration"] = self.calibration_.diagnostics
        self.diagnostics_["calibration_mode"] = self.calibration_.mode
        self.spectral_step_ = ("reject" if self.calibration_.mode == "prescribed" else "projected"
                               ) if self.spectral_step == "auto" else self.spectral_step
        self.initialization_spectrum_ = ("reject" if self.calibration_.mode == "prescribed" else "projected"
                                         ) if self.initialization_spectrum == "auto" else self.initialization_spectrum
        self.diagnostics_.update(spectral_step=self.spectral_step_,
            initialization_spectrum_requested=self.initialization_spectrum,
            initialization_spectrum=self.initialization_spectrum_,
            stationarity_tol=self.stationarity_tol,
            validation_selection=validation_data is not None,
            practical_extension=(self.spectral_step_ == "projected" or validation_data is not None
                                 or self.stationarity_tol is not None
                                 or self.initialization_spectrum_ == "projected"))
        try:
            self.source_ = prepare_source(source, self.source_rank, X.shape[1], Y.shape[1],
                                          orthogonality_tol=self.orthogonality_tol, tie_tol=self.tie_tol)
        except np.linalg.LinAlgError as error:
            return self._failure("numerical_failure", error)
        maximum_support = tuple((dimension - self.rank) * self.rank for dimension in
                                (self.source_.left.shape[1], self.source_.right.shape[1]))
        full_support = tuple(self.calibration_.support_limits) == maximum_support
        self.refinement_solver_ = self.refinement_solver
        if self.refinement_solver_ == "auto":
            self.refinement_solver_ = ("anchor_projected" if self.calibration_.mode == "practical"
                and full_support and self.spectral_step_ == "projected" else "chart")
        if self.refinement_solver_ == "anchor_projected":
            if not full_support:
                raise ValueError("anchor_projected requires full complement support limits; use chart for hard caps")
            if self.spectral_step_ != "projected":
                raise ValueError("anchor_projected requires spectral_step='projected'")
            self.diagnostics_["practical_extension"] = True
        self.diagnostics_.update(refinement_solver_requested=self.refinement_solver,
            refinement_solver=self.refinement_solver_,
            diagnostic_coordinates="omega,d,H" if self.refinement_solver_ == "anchor_projected" else "omega,d,Z",
            stationarity_scope="full_chart_constraints" if self.refinement_solver_ == "anchor_projected"
                else "spectral_and_support_mapping_with_domain_check")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                ZI, WI = X @ self.source_.leading_left, Y @ self.source_.leading_right
                initial = reduced_lasso(ZI, WI, self.rank, self.calibration_.init_penalty,
                                        tol=self.lasso_tol, max_iter=self.lasso_max_iter, tie_tol=self.tie_tol)
        except InitializationFailure as error:
            return self._failure("initialization_failed", error)
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            return self._failure("numerical_failure", error)
        self.initialization_ = initial
        self.diagnostics_["lasso"] = {"dual_gaps": initial.dual_gaps,
                                      "kkt_residual": initial.kkt_residual,
                                      "converged": initial.converged,
                                      "n_iter": initial.n_iter}
        if not initial.converged:
            return self._failure("lasso_not_converged", "Lasso did not meet its numerical optimality tolerances.")
        # Preserve the original reduced-Lasso result for inspection. Practical
        # initialization may project its singular values into the declared
        # feasible set while retaining the estimated singular directions.
        d = initial.d.copy()
        self.diagnostics_.update(
            initialization_singular_values_original=initial.d.copy(),
            initialization_singular_values_projected=d.copy(),
            initialization_spectrum_correction_norm=0.,
            initialization_spectrum_repaired=False,
        )
        if (np.any(d < self.margins.d_lower) or np.any(d > self.margins.d_upper)
                or (self.rank > 1 and np.any(d[:-1] - d[1:] < self.margins.gap))):
            if self.initialization_spectrum_ == "reject":
                return self._failure("initialization_spectrum_failed", "Initializer violates declared target spectral margins.")
            try:
                d = project_singular_values(d, d_lower=self.margins.d_lower,
                                             d_upper=self.margins.d_upper, gap=self.margins.gap)
            except (ValueError, FloatingPointError) as error:
                return self._failure("initialization_spectrum_failed", error)
            self.diagnostics_.update(
                initialization_singular_values_projected=d.copy(),
                initialization_spectrum_correction_norm=float(np.linalg.norm(d - initial.d)),
                initialization_spectrum_repaired=not np.array_equal(d, initial.d),
            )
        try:
            au = select_anchor(initial.P, anchor_min=self.margins.anchor_min,
                               qr_threshold=self.margins.qr_threshold, max_exchanges=self.qr_max_exchanges)
            av = select_anchor(initial.Q, anchor_min=self.margins.anchor_min,
                               qr_threshold=self.margins.qr_threshold, max_exchanges=self.qr_max_exchanges)
        except AnchorFailure as error:
            return self._failure("anchor_selection_failed", error)
        self.anchors_ = {"u": au, "v": av}
        self.chart_ = AnchorChart(self.source_.left.shape[1], self.source_.right.shape[1],
                                  au.indices, av.indices, au.center, av.center)
        P = np.zeros((self.chart_.n_u, self.rank))
        Q = np.zeros((self.chart_.n_v, self.rank))
        P[:self.source_rank], Q[:self.source_rank] = initial.P, initial.Q
        try:
            x0 = self.chart_.initial_state(P, d, Q)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return self._failure("handoff_failed", error)
        for sl, k in zip((self.chart_.z_u_slice, self.chart_.z_v_slice), self.calibration_.support_limits):
            x0[sl] = hard_threshold(x0[sl], k)
        self.initial_state_ = x0.copy()
        domain = self.chart_.domain_reason(x0, d_lower=self.margins.d_lower,
                  d_upper=self.margins.d_upper, gap=self.margins.gap, anchor_min=self.margins.anchor_min)
        if domain:
            return self._failure("handoff_failed", domain)
        Z, W = X @ self.source_.left, Y @ self.source_.right
        # Stable constant response loss omitted by the thin exact-source basis.
        if self.source_.right.shape[1] == Y.shape[1]:
            loss_offset = 0.
        else:
            outside = Y - W @ self.source_.right.T
            loss_offset = float(np.sum(outside * outside) / (2 * X.shape[0]))
        self.validation_history_ = []
        self.best_validation_loss_ = None
        selected_state = None
        selected_iteration = None
        if validation_data is not None:
            validation_design = XV @ self.source_.left

        def observe(iteration, state, record):
            nonlocal selected_state, selected_iteration
            if validation_data is None:
                return
            P, d, Q = self.chart_.reconstruct(state)
            prediction = ((validation_design @ P) * d) @ (self.source_.right @ Q).T
            with np.errstate(over="ignore", invalid="ignore"):
                score = float(np.mean((YV - prediction) ** 2))
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation prediction loss")
            self.validation_history_.append({"iteration": iteration, "loss": score})
            if self.best_validation_loss_ is None or score < self.best_validation_loss_:
                self.best_validation_loss_ = score
                selected_state, selected_iteration = state.copy(), iteration

        solver_options = dict(calibration=self.calibration_, margins=self.margins,
            iterations=self.iterations, max_backtracks=self.max_backtracks, loss_offset=loss_offset,
            stationarity_tol=self.stationarity_tol,
            iterate_callback=observe if validation_data is not None else None)
        if self.refinement_solver_ == "anchor_projected":
            from .anchor_solver import refine_anchor_projected
            result = refine_anchor_projected(self.chart_, x0, Z, W, **solver_options)
        else:
            result = refine(self.chart_, x0, Z, W, spectral_step=self.spectral_step_, **solver_options)
        self.result_, self.last_state_ = result, result.state.copy()
        self.state_ = self.last_state_.copy() if selected_state is None else selected_state
        self.selected_iteration_ = result.n_iter if selected_iteration is None else selected_iteration
        self.history_, self.n_iter_ = result.history, result.n_iter
        self.termination_reason_ = result.termination_reason
        self.optimization_converged_ = self.termination_reason_ == "stationarity"
        self.converged_ = self.optimization_converged_ and self.selected_iteration_ == self.n_iter_
        selected_record = next((record for record in self.history_
                                if record.iteration == self.selected_iteration_), None)
        P_last, d_last, Q_last = self.chart_.reconstruct(self.last_state_)
        self.last_coefficient_ = ((self.source_.left @ P_last) * d_last) @ (self.source_.right @ Q_last).T
        P, d, Q = self.chart_.reconstruct(self.state_)
        self.factors_ = {"P": P, "d": d.copy(), "Q": Q}
        self.left_factors_, self.right_factors_ = self.source_.left @ P, self.source_.right @ Q
        self.singular_values_ = d.copy()
        self.coefficient_ = (self.left_factors_ * d) @ self.right_factors_.T
        self.supports_ = {"u": np.flatnonzero(self.state_[self.chart_.z_u_slice]),
                          "v": np.flatnonzero(self.state_[self.chart_.z_v_slice])}
        self.diagnostics_["last_rejection"] = result.last_rejection
        self.diagnostics_.update(termination_reason=self.termination_reason_,
            projected_gradient_norm=(selected_record.projected_gradient_norm if selected_record else None),
            raw_gradient_norm=(selected_record.raw_gradient_norm if selected_record else None),
            last_projected_gradient_norm=result.projected_gradient_norm,
            last_raw_gradient_norm=result.raw_gradient_norm,
            optimization_converged=self.optimization_converged_, selected_converged=self.converged_,
            selected_iteration=self.selected_iteration_,
            best_validation_loss=self.best_validation_loss_,
            fitted_support_counts=(len(self.supports_["u"]), len(self.supports_["v"])),
            support_cap_reached=tuple(len(self.supports_[side]) == limit
                for side, limit in zip(("u", "v"), self.calibration_.support_limits)))
        if not result.success:
            return self._failure(result.status, result.message)
        self.status_, self.message_, self.success_ = result.status, result.message, True
        self.diagnostics_["status"] = result.status
        return self

    def predict(self, X, *, allow_partial: bool = False):
        """Predict with low-rank products; failed partial fits require opt-in."""
        if not hasattr(self, "coefficient_"):
            raise FitFailure(getattr(self, "status_", "not_fitted"), "No coefficient estimate is available.")
        if not self.success_ and not allow_partial:
            raise FitFailure(self.status_, "Fit did not complete; allow_partial=True uses its retained accepted iterate.")
        X = _data(X, "X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of predictors")
        return ((X @ self.left_factors_) * self.singular_values_) @ self.right_factors_.T
