"""Public two-stage estimator with explicit source and tuning inputs."""
from __future__ import annotations

import numbers
from copy import deepcopy
from dataclasses import dataclass
import numpy as np

from .anchors import AnchorFailure, select_anchor
from .calibration import (CalibrationError, Margins, PracticalCalibration,
                          PrescribedCalibration, SourceAccuracyError, resolve_calibration)
from .chart import AnchorChart
from .initialization import InitializationFailure, reduced_lasso
from .solver import refine, _result
from .source import ExactSource, NoisySource, _positive_real, prepare_source
from .spectral import project_singular_values
from .support import coordinate_support
from .thresholding import hard_threshold
from .validation import SELECTION_RULE, ValidationReference, validation_loss_difference


class FitFailure(RuntimeError):
    """An algorithmic or numerical failure, carrying an inspectable status."""
    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(f"{status}: {message}")


class _FitPreparationCache:
    """Private cache owned by one tuner fit, never retained by its models."""

    def __init__(self):
        self.context = None
        self.entries = {}
        self.validation_reference = ValidationReference()
        self.validation_context = {}

    def bind(self, context):
        if self.context is None:
            self.context = context
        elif self.context != context:
            raise ValueError("fit preparation cache cannot be reused with different observations or source")

    def get(self, key, compute):
        if key not in self.entries:
            self.entries[key] = compute()
        return self.entries[key]


def _data(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    x = np.asarray(value, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a nonempty finite two-dimensional array")
    return x


def _factor_prediction(design, left, singular_values, right):
    """Canonical ambient factor association for validation and public prediction."""
    return ((design @ left) * singular_values) @ right.T


@dataclass(frozen=True)
class TrajectoryCheckpoint:
    """Lightweight successful finite prefix; states use original Z coordinates."""
    iteration: int
    state: np.ndarray
    selected_state: np.ndarray
    selected_iteration: int
    best_validation_loss: float | None
    history_length: int
    validation_history_length: int
    status: str
    termination_reason: str
    message: str
    best_selection_score: float | None = None


def _checkpoint_schedule(values, iterations):
    if values is None:
        return None
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError("checkpoint_iterations must be a sequence of distinct iteration numbers") from error
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
           or value < 0 or value > iterations for value in values):
        raise ValueError("checkpoint_iterations must contain integers between zero and iterations")
    if len(set(values)) != len(values):
        raise ValueError("checkpoint_iterations must contain distinct iteration numbers")
    return tuple(sorted(int(value) for value in values))


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
        checkpoint_iterations=None, validation_interval: int = 1,
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
        self.checkpoint_iterations, self.validation_interval = checkpoint_iterations, validation_interval

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
        cache = self.__dict__.pop("_fit_cache", None)
        validation_reference = cache.validation_reference if cache is not None else ValidationReference()
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
        if cache is not None:
            cache.bind((id(X), id(Y), id(source),
                        id(XV) if validation_data is not None else None,
                        id(YV) if validation_data is not None else None))

        def prepared(key, compute, *, public=False):
            value = compute() if cache is None else cache.get(key, compute)
            # Public fitted attributes stay independently mutable. Solvers only
            # read the cached design/response projections.
            return deepcopy(value) if cache is not None and public else value
        for name, value, minimum in (
            ("iterations", self.iterations, 0), ("max_backtracks", self.max_backtracks, 0),
            ("lasso_max_iter", self.lasso_max_iter, 1), ("qr_max_exchanges", self.qr_max_exchanges, 1),
            ("validation_interval", self.validation_interval, 1),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        requested_checkpoints = _checkpoint_schedule(self.checkpoint_iterations, self.iterations)
        self.checkpoints_, self.checkpoint_iterations_ = {}, ()
        for name, value in (("lasso_tol", self.lasso_tol),
                            ("orthogonality_tol", self.orthogonality_tol), ("tie_tol", self.tie_tol)):
            _positive_real(value, name)
        if not isinstance(source, (ExactSource, NoisySource)):
            raise ValueError("source must be ExactSource or NoisySource")
        if not isinstance(self.margins, Margins):
            raise ValueError("margins must be Margins")
        self.n_features_in_, self.n_responses_ = X.shape[1], Y.shape[1]
        self.selection_rule_ = SELECTION_RULE
        self.n_iter_, self.history_, self.success_ = 0, [], False
        self.termination_reason_, self.converged_, self.optimization_converged_ = None, False, False
        self.diagnostics_ = {"theorem_certified": False, "arithmetic": "float64",
                             "selection_rule": SELECTION_RULE,
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
            source_key = ("source", self.source_rank, self.orthogonality_tol, self.tie_tol)
            self.source_ = prepared(source_key, lambda: prepare_source(
                source, self.source_rank, X.shape[1], Y.shape[1],
                orthogonality_tol=self.orthogonality_tol, tie_tol=self.tie_tol), public=True)
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
                ZI, WI = prepared(("leading_data", source_key), lambda:
                    (X @ self.source_.leading_left, Y @ self.source_.leading_right))
                initial_key = ("initial", source_key, self.rank, self.calibration_.init_penalty,
                               self.lasso_tol, self.lasso_max_iter)
                initial = prepared(initial_key, lambda: reduced_lasso(
                    ZI, WI, self.rank, self.calibration_.init_penalty,
                    tol=self.lasso_tol, max_iter=self.lasso_max_iter, tie_tol=self.tie_tol), public=True)
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
            anchor_key = ("anchors", initial_key, self.margins.anchor_min,
                          self.margins.qr_threshold, self.qr_max_exchanges)
            au, av = prepared(anchor_key, lambda: tuple(select_anchor(
                factor, anchor_min=self.margins.anchor_min, qr_threshold=self.margins.qr_threshold,
                max_exchanges=self.qr_max_exchanges) for factor in (initial.P, initial.Q)), public=True)
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
        def training_projection():
            Z, W = ((ZI, WI) if isinstance(source, ExactSource) else
                    (X @ self.source_.left, Y @ self.source_.right))
            # Stable constant response loss omitted by the thin exact-source basis.
            if self.source_.right.shape[1] == Y.shape[1]:
                offset = 0.
            else:
                outside = Y - W @ self.source_.right.T
                offset = float(np.sum(outside * outside) / (2 * X.shape[0]))
            return Z, W, offset

        Z, W, loss_offset = prepared(("training_data", source_key), training_projection)
        self.validation_history_ = []
        self.best_validation_loss_ = None
        self.best_selection_score_ = None
        selected_state = None
        selected_iteration = None
        selected_prediction = None
        def observe(iteration, state, record):
            nonlocal selected_state, selected_iteration
            self.history_.append(record)
            self.n_iter_ = iteration
            eligible = (iteration == 0 or iteration % self.validation_interval == 0
                        or requested_checkpoints is not None and iteration in requested_checkpoints)
            if validation_data is not None and eligible:
                evaluate_validation(iteration, state)
            if requested_checkpoints is not None and iteration in requested_checkpoints:
                capture(iteration, state, record)

        def evaluate_validation(iteration, state):
            nonlocal selected_state, selected_iteration, selected_prediction
            if self.validation_history_ and self.validation_history_[-1]["iteration"] == iteration:
                return
            P, d, Q = self.chart_.reconstruct(state)
            prediction = _factor_prediction(XV, self.source_.left @ P, d, self.source_.right @ Q)
            with np.errstate(over="ignore", invalid="ignore"):
                score = float(np.mean((YV - prediction) ** 2))
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation prediction loss")
            selection_score = validation_reference.score(prediction, YV,
                metadata={"reference_validation_mse": score,
                          **(cache.validation_context if cache is not None else {})})
            self.diagnostics_["selection_reference"] = validation_reference.metadata.copy()
            difference = (None if selected_prediction is None else
                          validation_loss_difference(prediction, selected_prediction, YV))
            self.validation_history_.append({"iteration": iteration, "loss": score,
                "selection_score": selection_score, "selection_rule": SELECTION_RULE,
                "selection_comparison": {"incumbent_iteration": selected_iteration,
                                         "loss_difference": difference}})
            if selected_prediction is None or difference < 0:
                self.best_validation_loss_ = score
                self.best_selection_score_ = selection_score
                selected_state, selected_iteration = state.copy(), iteration
                selected_prediction = prediction.copy()

        def capture(iteration, state, record, *, terminal=None):
            # Solvers also observe an accepted state whose subsequent gradient
            # evaluation failed. That record is not a successful finite prefix.
            if (not np.all(np.isfinite(state))
                    or any(value is None or not np.isfinite(value) for value in
                           (record.objective, record.smooth_loss, record.penalty_value,
                            record.raw_gradient_norm))):
                return
            status, reason = "completed", "max_iterations"
            message = f"Completed {iteration} refinement updates in a retained trajectory prefix."
            if terminal is not None:
                status, reason, message = terminal.status, terminal.termination_reason, terminal.message
            elif (iteration > 0 and self.stationarity_tol is not None
                  and record.mapping_domain_reason is None
                  and record.projected_gradient_norm is not None
                  and record.projected_gradient_norm <= self.stationarity_tol):
                status, reason = "converged", "stationarity"
                message = "The retained trajectory prefix meets the stationarity tolerance."
            endpoint = state.copy()
            chosen = state.copy() if selected_state is None else selected_state.copy()
            endpoint.setflags(write=False)
            chosen.setflags(write=False)
            self.checkpoints_[iteration] = TrajectoryCheckpoint(
                iteration, endpoint, chosen, iteration if selected_iteration is None else selected_iteration,
                self.best_validation_loss_, len(self.history_), len(self.validation_history_),
                status, reason, message, self.best_selection_score_)
            self.checkpoint_iterations_ = tuple(sorted(self.checkpoints_))

        solver_options = dict(calibration=self.calibration_, margins=self.margins,
            iterations=self.iterations, max_backtracks=self.max_backtracks, loss_offset=loss_offset,
            stationarity_tol=self.stationarity_tol,
            iterate_callback=observe if validation_data is not None or requested_checkpoints is not None else None)
        if self.refinement_solver_ == "anchor_projected":
            from .anchor_solver import refine_anchor_projected
            result = refine_anchor_projected(self.chart_, x0, Z, W, **solver_options)
        else:
            result = refine(self.chart_, x0, Z, W, spectral_step=self.spectral_step_, **solver_options)
        if result.success and result.termination_reason == "stationarity":
            if validation_data is not None:
                evaluate_validation(result.n_iter, result.state)
            if requested_checkpoints is not None and result.history:
                capture(result.n_iter, result.state, result.history[-1], terminal=result)
        if cache is None and validation_reference.prediction is not None:
            self.validation_reference_prediction_ = validation_reference.prediction.copy()
        return self._finalize_refinement(result, selected_state, selected_iteration)

    def _finalize_refinement(self, result, selected_state, selected_iteration):
        """Finalize either the solver result or a captured successful prefix."""
        self.result_, self.last_state_ = result, result.state.copy()
        self.state_ = self.last_state_.copy() if selected_state is None else selected_state
        self.selected_iteration_ = result.n_iter if selected_iteration is None else selected_iteration
        self.history_, self.n_iter_ = result.history, result.n_iter
        self.termination_reason_ = result.termination_reason
        self.optimization_converged_ = self.termination_reason_ == "stationarity"
        self.converged_ = self.optimization_converged_ and self.selected_iteration_ == self.n_iter_
        P_last, d_last, Q_last = self.chart_.reconstruct(self.last_state_)
        self.last_coefficient_ = ((self.source_.left @ P_last) * d_last) @ (self.source_.right @ Q_last).T
        P, d, Q = self.chart_.reconstruct(self.state_)
        self.factors_ = {"P": P, "d": d.copy(), "Q": Q}
        self.left_factors_, self.right_factors_ = self.source_.left @ P, self.source_.right @ Q
        self.singular_values_ = d.copy()
        self.coefficient_ = (self.left_factors_ * d) @ self.right_factors_.T
        self.supports_, self.raw_supports_, self.support_tolerances_, metadata = self._refinement_metadata(
            result, self.state_, self.selected_iteration_, self.best_validation_loss_, self.best_selection_score_)
        self.diagnostics_.update(metadata)
        if not result.success:
            return self._failure(result.status, result.message)
        self.status_, self.message_, self.success_ = result.status, result.message, True
        return self

    def _refinement_metadata(self, result, state, selected_iteration, best_validation_loss,
                             best_selection_score=None):
        """Derive selected/terminal diagnostics without constructing coefficients."""
        selected_record = next((record for record in result.history
                                if record.iteration == selected_iteration), None)
        effective = self.refinement_solver_ == "anchor_projected"
        supports, raw_supports, tolerances = {}, {}, {}
        for side, sl in (("u", self.chart_.z_u_slice), ("v", self.chart_.z_v_slice)):
            supports[side], tolerances[side] = coordinate_support(state[sl], effective=effective)
            raw_supports[side], _ = coordinate_support(state[sl])
        optimization_converged = result.termination_reason == "stationarity"
        metadata = dict(status=result.status, last_rejection=result.last_rejection,
            termination_reason=result.termination_reason,
            projected_gradient_norm=(selected_record.projected_gradient_norm if selected_record else None),
            raw_gradient_norm=(selected_record.raw_gradient_norm if selected_record else None),
            last_projected_gradient_norm=result.projected_gradient_norm,
            last_raw_gradient_norm=result.raw_gradient_norm,
            mapping_displacement=(selected_record.mapping_displacement if selected_record else None),
            proximal_uncertainty=(selected_record.proximal_uncertainty if selected_record else None),
            mapping_refinements=(selected_record.mapping_refinements if selected_record else 0),
            mapping_precision_limited=(selected_record.mapping_precision_limited if selected_record else False),
            last_mapping_displacement=result.mapping_displacement,
            last_proximal_uncertainty=result.proximal_uncertainty,
            last_mapping_refinements=result.mapping_refinements,
            last_mapping_precision_limited=result.mapping_precision_limited,
            precision_limited=(result.status == "numerical_stagnation" or result.mapping_precision_limited),
            objective_change=(selected_record.objective_change if selected_record else None),
            relative_step_norm=(selected_record.relative_step_norm if selected_record else None),
            line_search_strategy="reset_initial_inverse",
            optimization_converged=optimization_converged,
            selected_converged=optimization_converged and selected_iteration == result.n_iter,
            selected_iteration=selected_iteration, best_validation_loss=best_validation_loss,
            best_selection_score=best_selection_score,
            fitted_support_counts=tuple(len(supports[side]) for side in ("u", "v")),
            raw_fitted_support_counts=tuple(len(raw_supports[side]) for side in ("u", "v")),
            support_tolerances=tolerances.copy(),
            support_reporting="effective_numerical" if effective else "literal_nonzero",
            support_cap_reached=tuple(len(raw_supports[side]) == limit
                for side, limit in zip(("u", "v"), self.calibration_.support_limits)),
            effective_support_cap_reached=tuple(len(supports[side]) == limit
                for side, limit in zip(("u", "v"), self.calibration_.support_limits)))
        return supports, raw_supports, tolerances, metadata

    def _checkpoint_snapshot(self, iteration):
        if (isinstance(iteration, (bool, np.bool_)) or not isinstance(iteration, numbers.Integral)
                or iteration < 0):
            raise ValueError("checkpoint iteration must be a nonnegative integer")
        snapshot = getattr(self, "checkpoints_", {}).get(int(iteration))
        if snapshot is None:
            raise FitFailure("checkpoint_unavailable", f"No successful finite prefix is available at iteration {iteration}.")
        return snapshot

    def _checkpoint_summary(self, iteration):
        """Lightweight selection record; public fitted views are created only for winners."""
        snapshot = self._checkpoint_snapshot(iteration)
        result = _result(snapshot.state, snapshot.status, snapshot.message, snapshot.iteration,
                         self.history_[:snapshot.history_length], termination_reason=snapshot.termination_reason)
        _, _, _, metadata = self._refinement_metadata(
            result, snapshot.selected_state, snapshot.selected_iteration, snapshot.best_validation_loss,
            snapshot.best_selection_score)
        diagnostics = deepcopy(self.diagnostics_)
        diagnostics.update(metadata)
        return dict(success=result.success, status=snapshot.status, message=snapshot.message,
                    validation_mse=snapshot.best_validation_loss, selected_iteration=snapshot.selected_iteration,
                    selection_score=snapshot.best_selection_score, selection_rule=SELECTION_RULE,
                    n_iter=snapshot.iteration, termination_reason=snapshot.termination_reason,
                    validation_history=deepcopy(self.validation_history_[:snapshot.validation_history_length]),
                    diagnostics=diagnostics)

    def _checkpoint_prediction(self, iteration, X):
        """Predict a selected prefix directly, without constructing a fitted view."""
        snapshot = self._checkpoint_snapshot(iteration)
        P, d, Q = self.chart_.reconstruct(snapshot.selected_state)
        return _factor_prediction(X, self.source_.left @ P, d, self.source_.right @ Q)

    def checkpoint_model(self, iteration):
        """Reconstruct an independent fitted view of an available finite prefix.

        This does not rerun initialization or refinement. The selected state
        and validation minimum are limited to observations eligible by this
        checkpoint; the endpoint is retained separately as ``last_state_``.
        """
        snapshot = self._checkpoint_snapshot(iteration)
        view = object.__new__(type(self))
        excluded = {"checkpoints_", "history_", "validation_history_", "result_",
                    "coefficient_", "last_coefficient_", "factors_", "left_factors_", "right_factors_",
                    "singular_values_", "state_", "last_state_", "supports_", "raw_supports_",
                    "validation_reference_prediction_"}
        view.__dict__ = deepcopy({key: value for key, value in vars(self).items() if key not in excluded})
        view.iterations = snapshot.iteration
        view.checkpoints_ = deepcopy({key: value for key, value in self.checkpoints_.items()
                                     if key <= snapshot.iteration})
        view.checkpoint_iterations_ = tuple(sorted(view.checkpoints_))
        view.checkpoint_iterations = view.checkpoint_iterations_
        view.validation_history_ = deepcopy(self.validation_history_[:snapshot.validation_history_length])
        view.best_validation_loss_ = snapshot.best_validation_loss
        view.best_selection_score_ = snapshot.best_selection_score
        result = _result(snapshot.state.copy(), snapshot.status, snapshot.message, snapshot.iteration,
                         deepcopy(self.history_[:snapshot.history_length]),
                         termination_reason=snapshot.termination_reason)
        return view._finalize_refinement(result, snapshot.selected_state.copy(), snapshot.selected_iteration)

    def predict(self, X, *, allow_partial: bool = False):
        """Predict with low-rank products; failed partial fits require opt-in."""
        if not hasattr(self, "coefficient_"):
            raise FitFailure(getattr(self, "status_", "not_fitted"), "No coefficient estimate is available.")
        if not self.success_ and not allow_partial:
            raise FitFailure(self.status_, "Fit did not complete; allow_partial=True uses its retained accepted iterate.")
        X = _data(X, "X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of predictors")
        return _factor_prediction(X, self.left_factors_, self.singular_values_, self.right_factors_)
