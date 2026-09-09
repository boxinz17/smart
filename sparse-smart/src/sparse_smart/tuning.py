"""Held-out target prediction tuning for practical SparseSMART experiments.

Candidates are fit independently on training observations only. Validation
selects both each candidate's iterate and the winning candidate. The selected
model is retained as fitted; there is no implicit refit on all observations.
"""

from __future__ import annotations

from itertools import product
from numbers import Integral, Real
import time

import numpy as np

from .calibration import Margins, PracticalCalibration
from .estimator import FitFailure, SparseSMART, _data
from .source import ExactSource, NoisySource


_DEFAULT_PENALTIES = (.00125, .0025, .005, .01, .02, .04)


def _grid(values, name, *, positive=False):
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty sequence of numbers") from error
    if not result:
        raise ValueError(f"{name} must be nonempty")
    for value in result:
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0 or (positive and value == 0)):
            raise ValueError(f"{name} must contain finite {'positive' if positive else 'nonnegative'} numbers")
    return tuple(float(value) for value in result)


def _supports(values, maximum):
    if values is None:
        return (maximum,)
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError("support_limits must contain one or more support pairs") from error
    # Accept either one pair or a sequence of candidate pairs.
    if len(values) == 2 and all(isinstance(k, Integral) for k in values):
        values = (values,)
    if not values:
        raise ValueError("support_limits must contain at least one support pair")
    result = []
    for pair in values:
        try:
            pair = tuple(pair)
        except TypeError as error:
            raise ValueError("each support limit must be a pair of integers") from error
        if len(pair) != 2 or any(
            isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 0 or k > m
            for k, m in zip(pair, maximum)
        ):
            raise ValueError(f"support pairs must be nonnegative integers bounded by {maximum}")
        result.append(tuple(int(k) for k in pair))
    return tuple(result)


class SparseSMARTTuner:
    """Select practical penalties and an iterate using held-out target data.

    ``support_limits=None`` allows every actual complement coordinate: exact
    source frames have dimension ``source_rank``; noisy frames have dimensions
    ``p`` and ``q``. Supply a pair or sequence of pairs to tune smaller budgets.
    Penalties on the two complement blocks are selected independently.

    ``fit(X,Y,validation_data=None)`` makes a deterministic holdout split.
    With explicit validation data, all supplied X/Y rows are training rows.
    No coefficient truth is accepted or used. Validation is used repeatedly for
    selection, so its score is not an independent test-set performance estimate.
    Only successful candidate fits can win; failed partial fits remain records.
    """

    def __init__(
        self, *, rank, source_rank, sparsity, margins: Margins,
        init_penalties=(.03,), penalties_u=_DEFAULT_PENALTIES,
        penalties_v=_DEFAULT_PENALTIES, support_limits=None,
        iterations=500, step_size_inverse=20., validation_fraction=.2,
        random_state=0, enforce_source_accuracy=True, spectral_step="projected",
        stationarity_tol=1e-6, delta=.05, max_backtracks=60,
        lasso_tol=1e-9, lasso_max_iter=20000,
        initialization_spectrum="auto", refinement_solver="auto",
    ):
        self.rank, self.source_rank, self.sparsity = rank, source_rank, sparsity
        self.margins = margins
        self.init_penalties = init_penalties
        self.penalties_u, self.penalties_v = penalties_u, penalties_v
        self.support_limits = support_limits
        self.iterations, self.step_size_inverse = iterations, step_size_inverse
        self.validation_fraction, self.random_state = validation_fraction, random_state
        self.enforce_source_accuracy = enforce_source_accuracy
        self.spectral_step, self.stationarity_tol = spectral_step, stationarity_tol
        self.delta, self.max_backtracks = delta, max_backtracks
        self.lasso_tol, self.lasso_max_iter = lasso_tol, lasso_max_iter
        self.initialization_spectrum = initialization_spectrum
        self.refinement_solver = refinement_solver

    def _split(self, X, Y, validation_data):
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
                raise ValueError("validation_data must be (X_validation, Y_validation)")
            Xv, Yv = _data(validation_data[0], "X_validation"), _data(validation_data[1], "Y_validation")
            if Xv.shape[0] != Yv.shape[0] or Xv.shape[1] != X.shape[1] or Yv.shape[1] != Y.shape[1]:
                raise ValueError("validation data must have matching rows, predictor dimensions, and response dimensions")
            self.train_indices_, self.validation_indices_ = None, None
            self.split_mode_ = "explicit_validation"
            return X, Y, Xv, Yv
        fraction = self.validation_fraction
        if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real)
                or not np.isfinite(fraction) or not 0 < fraction < 1):
            raise ValueError("validation_fraction must lie strictly between zero and one")
        if (isinstance(self.random_state, (bool, np.bool_))
                or not isinstance(self.random_state, Integral) or self.random_state < 0):
            raise ValueError("random_state must be a nonnegative integer")
        n_validation = max(1, int(np.ceil(X.shape[0] * fraction)))
        if n_validation >= X.shape[0]:
            raise ValueError("holdout splitting requires at least one training and one validation row")
        order = np.random.default_rng(int(self.random_state)).permutation(X.shape[0])
        self.validation_indices_ = np.sort(order[:n_validation])
        self.train_indices_ = np.sort(order[n_validation:])
        self.split_mode_ = "deterministic_holdout"
        return X[self.train_indices_], Y[self.train_indices_], X[self.validation_indices_], Y[self.validation_indices_]

    @staticmethod
    def _validation_mse(model, Xv, Yv, *, allow_partial=False):
        prediction = np.asarray(model.predict(Xv, allow_partial=allow_partial))
        if prediction.shape != Yv.shape or not np.isfinite(prediction).all():
            raise ValueError("candidate validation predictions must be finite and match validation responses")
        with np.errstate(over="raise", invalid="raise"):
            score = float(np.mean((prediction - Yv) ** 2))
        if not np.isfinite(score):
            raise ValueError("candidate validation loss is nonfinite")
        return score

    def fit(self, X, Y, *, source: ExactSource | NoisySource, validation_data=None):
        """Fit training-only candidates and retain the successful validation winner.

        Failed candidates are not eligible, even if they have usable partial
        coefficients. If every candidate fails, return with ``success_=False``
        and status ``no_successful_candidate``; prediction then raises FitFailure.
        Equal validation scores retain the earliest candidate in grid order.
        """
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        X, Y = _data(X, "X"), _data(Y, "Y")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of observations")
        if not isinstance(source, (ExactSource, NoisySource)):
            raise ValueError("source must be ExactSource or NoisySource")
        if not isinstance(self.margins, Margins):
            raise ValueError("margins must be Margins")
        if not isinstance(self.initialization_spectrum, str) or self.initialization_spectrum not in ("auto", "projected", "reject"):
            raise ValueError("initialization_spectrum must be 'auto', 'projected', or 'reject'")
        if not isinstance(self.refinement_solver, str) or self.refinement_solver not in ("auto", "chart", "anchor_projected"):
            raise ValueError("refinement_solver must be 'auto', 'chart', or 'anchor_projected'")
        for name, value in (("rank", self.rank), ("source_rank", self.source_rank)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.rank <= self.source_rank <= min(X.shape[1], Y.shape[1]):
            raise ValueError("require rank <= source_rank <= min(p, q)")
        init_grid = _grid(self.init_penalties, "init_penalties", positive=True)
        grid_u = _grid(self.penalties_u, "penalties_u")
        grid_v = _grid(self.penalties_v, "penalties_v")
        dimensions = ((self.source_rank, self.source_rank) if isinstance(source, ExactSource)
                      else (X.shape[1], Y.shape[1]))
        maximum = tuple((dimension - self.rank) * self.rank for dimension in dimensions)
        support_grid = _supports(self.support_limits, maximum)
        Xtr, Ytr, Xv, Yv = self._split(X, Y, validation_data)
        self.n_features_in_, self.n_responses_ = X.shape[1], Y.shape[1]
        self.training_sample_count_, self.validation_sample_count_ = Xtr.shape[0], Xv.shape[0]
        self.selection_history_ = []
        self.success_, self.status_ = False, "no_successful_candidate"
        self.best_params_, self.best_score_, self.selected_iteration_ = None, None, None
        self.model_, self.estimator_ = None, None
        self.diagnostics_ = {
            "theorem_certified": False, "selection_metric": "mean_validation_squared_prediction_error",
            "refit_on_all_data": False, "split_mode": self.split_mode_,
            "training_rows": Xtr.shape[0], "validation_rows": Xv.shape[0],
            "validation_score_is_independent_test_estimate": False,
            "source_accuracy_enforced": self.enforce_source_accuracy,
            "maximum_complement_support": maximum,
            "initialization_spectrum": self.initialization_spectrum,
            "refinement_solver": self.refinement_solver,
        }
        for candidate_id, (init, pu, pv, limits) in enumerate(product(init_grid, grid_u, grid_v, support_grid)):
            params = dict(init_penalty=init, penalty_u=pu, penalty_v=pv,
                          support_limits=limits, step_size_inverse=self.step_size_inverse)
            record = dict(candidate_id=candidate_id, params=params, success=False,
                          status=None, message=None, validation_mse=None,
                          partial_validation_mse=None, has_partial_coefficient=False,
                          selected_iteration=None, n_iter=0, termination_reason=None,
                          validation_history=[], elapsed_time_sec=None)
            model = None
            started = time.perf_counter()
            try:
                calibration = PracticalCalibration(
                    init_penalty=init, penalty=(pu, pv), step_size_inverse=self.step_size_inverse,
                    support_limits=limits, delta=self.delta,
                    enforce_source_accuracy=self.enforce_source_accuracy,
                )
                model = SparseSMART(
                    rank=self.rank, source_rank=self.source_rank, sparsity=self.sparsity,
                    margins=self.margins, calibration=calibration, iterations=self.iterations,
                    max_backtracks=self.max_backtracks, lasso_tol=self.lasso_tol,
                    lasso_max_iter=self.lasso_max_iter, spectral_step=self.spectral_step,
                    stationarity_tol=self.stationarity_tol,
                    initialization_spectrum=self.initialization_spectrum,
                    refinement_solver=self.refinement_solver,
                )
                model.fit(Xtr, Ytr, source=source, validation_data=(Xv, Yv))
                record.update(success=bool(model.success_), status=model.status_,
                              message=model.message_, n_iter=int(model.n_iter_),
                              selected_iteration=getattr(model, "selected_iteration_", None),
                              termination_reason=getattr(model, "termination_reason_", model.status_),
                              validation_history=getattr(model, "validation_history_", []))
                record["diagnostics"] = getattr(model, "diagnostics_", {}).copy()
                if model.success_:
                    score = self._validation_mse(model, Xv, Yv)
                    record["validation_mse"] = score
                    if self.best_score_ is None or score < self.best_score_:
                        self.best_score_, self.best_params_ = score, params.copy()
                        self.model_ = self.estimator_ = model
                elif hasattr(model, "coefficient_"):
                    record["has_partial_coefficient"] = True
                    record["partial_validation_mse"] = self._validation_mse(model, Xv, Yv, allow_partial=True)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, FitFailure) as error:
                record.update(success=False, status=type(error).__name__, message=str(error))
                if model is not None:
                    record["has_partial_coefficient"] = hasattr(model, "coefficient_")
            record["elapsed_time_sec"] = time.perf_counter() - started
            self.selection_history_.append(record)
        self.n_candidates_ = len(self.selection_history_)
        if self.model_ is None:
            self.message_ = "No candidate completed successfully; failed partial fits were not selected."
            return self
        self.success_, self.status_ = True, "selected"
        self.message_ = "Selected a successful candidate by held-out prediction loss; no refit on all data."
        self.coefficient_ = self.model_.coefficient_.copy()
        self.selected_iteration_ = getattr(self.model_, "selected_iteration_", self.model_.n_iter_)
        self.n_iter_ = self.model_.n_iter_
        self.best_validation_loss_ = self.best_score_
        self.validation_history_ = getattr(self.model_, "validation_history_", [])
        return self

    def predict(self, X):
        """Predict with the retained training-only winner; never use failed fits."""
        if not getattr(self, "success_", False) or self.model_ is None:
            raise FitFailure(getattr(self, "status_", "not_fitted"), "No successful validation-selected model is available.")
        return self.model_.predict(X)
