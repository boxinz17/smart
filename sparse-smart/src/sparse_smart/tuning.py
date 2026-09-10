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
from .estimator import FitFailure, SparseSMART, _FitPreparationCache, _data, _validation_schedule
from .source import ExactSource, NoisySource
from .validation import SELECTION_RULE, validation_loss_difference


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


def _iteration_budgets(values, iterations):
    if (isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, Integral)
            or iterations < 0):
        raise ValueError("iterations must be a nonnegative integer")
    if values is None:
        return (int(iterations),)
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError("iteration_budgets must be a nonempty sequence of positive integers") from error
    if not result or any(isinstance(value, (bool, np.bool_))
                         or not isinstance(value, Integral) or value <= 0 for value in result):
        raise ValueError("iteration_budgets must be a nonempty sequence of positive integers")
    if any(a >= b for a, b in zip(result, result[1:])):
        raise ValueError("iteration_budgets must be strictly increasing and unique")
    if result[-1] != iterations:
        raise ValueError("the last iteration budget must equal iterations")
    return tuple(int(value) for value in result)


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

    ``iteration_budgets=None`` fits each grid point once, at ``iterations``.
    An explicit increasing schedule ends at ``iterations`` and fits each budget
    independently on the same observations. Each budget's best successful fit
    is retained in ``checkpoints_``, keyed by budget; every candidate has a
    globally sequential ID in ``selection_history_``.
    Later failures cannot disqualify earlier successful checkpoints. Equal
    validation scores prefer the earlier budget, then the existing grid order.

    With ``checkpoint_execution='continuous'``, each grid point is fit once
    to the maximum budget. ``checkpoint_interval`` defaults to 250 in this
    mode and controls dense checkpoint capture and validation evaluation.
    Completed prefixes survive later failures; ``budget_reached`` separately
    records cap coverage. The initializer alone cannot rescue a trajectory
    that fails before its first positive checkpoint. Continuous histories
    are available in ``trajectory_models_`` and ``trajectory_history_``.
    ``validation_iterations=()`` adds sorted, unique nonnegative validation
    points in continuous mode without adding full checkpoints or budget
    coverage. Points above the maximum iteration budget are ignored. Independent
    mode already validates every accepted iterate and rejects a nonempty extra
    schedule. Generic defaults retain the existing validation schedules.
    """

    def __init__(
        self, *, rank, source_rank, sparsity, margins: Margins,
        init_penalties=(.03,), penalties_u=_DEFAULT_PENALTIES,
        penalties_v=_DEFAULT_PENALTIES, support_limits=None,
        iterations=500, iteration_budgets=None, step_size_inverse=20., validation_fraction=.2,
        random_state=0, enforce_source_accuracy=True, spectral_step="projected",
        stationarity_tol=1e-6, delta=.05, max_backtracks=60,
        lasso_tol=1e-9, lasso_max_iter=20000,
        initialization_spectrum="auto", refinement_solver="auto",
        checkpoint_execution="independent", checkpoint_interval=None, validation_iterations=(),
    ):
        self.rank, self.source_rank, self.sparsity = rank, source_rank, sparsity
        self.margins = margins
        self.init_penalties = init_penalties
        self.penalties_u, self.penalties_v = penalties_u, penalties_v
        self.support_limits = support_limits
        self.iterations, self.step_size_inverse = iterations, step_size_inverse
        self.iteration_budgets = iteration_budgets
        self.validation_fraction, self.random_state = validation_fraction, random_state
        self.enforce_source_accuracy = enforce_source_accuracy
        self.spectral_step, self.stationarity_tol = spectral_step, stationarity_tol
        self.delta, self.max_backtracks = delta, max_backtracks
        self.lasso_tol, self.lasso_max_iter = lasso_tol, lasso_max_iter
        self.initialization_spectrum = initialization_spectrum
        self.refinement_solver = refinement_solver
        self.checkpoint_execution = checkpoint_execution
        self.checkpoint_interval = checkpoint_interval
        self.validation_iterations = validation_iterations

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
    def _validation_mse(model, Xv, Yv, *, allow_partial=False, selection_reference=None,
                        return_prediction=False):
        prediction = np.asarray(model.predict(Xv, allow_partial=allow_partial))
        result = SparseSMARTTuner._prediction_score(prediction, Yv, selection_reference)
        if return_prediction:
            return (*result, prediction)
        return result if selection_reference is not None else result[0]

    @staticmethod
    def _prediction_score(prediction, Yv, selection_reference=None):
        if prediction.shape != Yv.shape or not np.isfinite(prediction).all():
            raise ValueError("candidate validation predictions must be finite and match validation responses")
        with np.errstate(over="raise", invalid="raise"):
            score = float(np.mean((prediction - Yv) ** 2))
        if not np.isfinite(score):
            raise ValueError("candidate validation loss is nonfinite")
        if selection_reference is not None:
            return score, selection_reference.score(prediction, Yv)
        return score, None

    @staticmethod
    def _comparison(prediction, incumbent_prediction, Yv, incumbent_id):
        return dict(incumbent_candidate_id=incumbent_id,
            loss_difference=None if incumbent_prediction is None else
            validation_loss_difference(prediction, incumbent_prediction, Yv))

    def _continuous_search(self, grid, Xtr, Ytr, Xv, Yv, source, preparation):
        """Fit each grid point once; compare certified prefixes at each cap.

        A prefix completed at a scheduled checkpoint is a successful finite
        fit even when a subsequent update fails. ``budget_reached`` separately
        states whether the trajectory actually covered a requested cap. A
        retained earlier checkpoint cannot establish stabilization at that cap.
        """
        schedule = sorted({0, self.iterations, *self.iteration_budgets_,
                           *range(self.checkpoint_interval_, self.iterations + 1,
                                  self.checkpoint_interval_)})
        self.trajectory_models_, self.trajectory_history_ = [], []
        self.diagnostics_.update(checkpoint_execution="continuous_trajectory",
            checkpoint_interval=self.checkpoint_interval_, checkpoint_iterations=schedule,
            validation_iterations=self.validation_iterations_,
            validation_schedule=sorted({*schedule, *(t for t in self.validation_iterations_
                                                    if t <= self.iterations)}),
            candidate_records_are_checkpoint_prefixes=True,
            elapsed_time_scope="one_measurement_per_trajectory")
        for grid_id, (init, pu, pv, limits) in enumerate(grid):
            params = dict(init_penalty=init, penalty_u=pu, penalty_v=pv,
                          support_limits=limits, step_size_inverse=self.step_size_inverse)
            model, exception = None, None
            started = time.perf_counter()
            try:
                calibration = PracticalCalibration(init_penalty=init, penalty=(pu, pv),
                    step_size_inverse=self.step_size_inverse, support_limits=limits,
                    delta=self.delta, enforce_source_accuracy=self.enforce_source_accuracy)
                model = SparseSMART(rank=self.rank, source_rank=self.source_rank,
                    sparsity=self.sparsity, margins=self.margins, calibration=calibration,
                    iterations=self.iterations, max_backtracks=self.max_backtracks,
                    lasso_tol=self.lasso_tol, lasso_max_iter=self.lasso_max_iter,
                    spectral_step=self.spectral_step, stationarity_tol=self.stationarity_tol,
                    initialization_spectrum=self.initialization_spectrum,
                    refinement_solver=self.refinement_solver,
                    checkpoint_iterations=schedule, validation_interval=self.checkpoint_interval_,
                    validation_iterations=self.validation_iterations_)
                model._fit_cache = preparation
                preparation.validation_context = dict(grid_candidate_id=grid_id,
                                                       iteration_budget=self.iterations)
                model.fit(Xtr, Ytr, source=source, validation_data=(Xv, Yv))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, ArithmeticError, FitFailure) as error:
                exception = error
            elapsed = time.perf_counter() - started
            success = exception is None and bool(getattr(model, "success_", False))
            status = type(exception).__name__ if exception else getattr(model, "status_", "fit_failed")
            self.trajectory_models_.append(model)
            self.trajectory_history_.append(dict(grid_candidate_id=grid_id, params=params,
                success=success, status=status,
                message=str(exception) if exception else getattr(model, "message_", status),
                n_iter=int(getattr(model, "n_iter_", 0)),
                termination_reason=(status if exception else getattr(model, "termination_reason_", status)),
                checkpoint_iterations=list(getattr(model, "checkpoint_iterations_", ())),
                diagnostics=getattr(model, "diagnostics_", {}).copy(), elapsed_time_sec=elapsed))

        budget_scores, winner_ids = {}, {}
        global_prediction = None
        for budget in self.iteration_budgets_:
            budget_records = []
            for grid_id, (full_model, trajectory) in enumerate(zip(self.trajectory_models_, self.trajectory_history_)):
                candidate_id = len(self.selection_history_)
                record = dict(candidate_id=candidate_id, grid_candidate_id=grid_id,
                    iteration_budget=budget, params=trajectory["params"].copy(),
                    success=False, status=trajectory["status"], message=trajectory["message"],
                    validation_mse=None, selection_score=None, selection_rule=SELECTION_RULE,
                    partial_validation_mse=None,
                    has_partial_coefficient=hasattr(full_model, "coefficient_"),
                    selected_iteration=None, n_iter=min(budget, trajectory["n_iter"]),
                    termination_reason=trajectory["termination_reason"], validation_history=[],
                    elapsed_time_sec=None, trajectory_elapsed_time_sec=trajectory["elapsed_time_sec"],
                    trajectory_status=trajectory["status"], trajectory_success=trajectory["success"],
                    trajectory_termination_reason=trajectory["termination_reason"],
                    trajectory_n_iter=trajectory["n_iter"], trajectory_checkpoint_iteration=None,
                    budget_reached=False, diagnostics={})
                terminal_stationary = (trajectory["success"]
                    and trajectory["termination_reason"] == "stationarity"
                    and trajectory["n_iter"] <= budget)
                available = [t for t in trajectory["checkpoint_iterations"] if t <= budget
                             and (t > 0 or terminal_stationary or self.iterations == 0)]
                if available:
                    endpoint = max(available)
                    try:
                        summary = full_model._checkpoint_summary(endpoint)
                        score = summary["validation_mse"]
                        selection_score = summary["selection_score"]
                        if not summary["success"]:
                            raise FitFailure(summary["status"], "Checkpoint is not a successful prefix")
                        if score is None or not np.isfinite(score):
                            raise ValueError("checkpoint validation loss is nonfinite")
                        if selection_score is None or not np.isfinite(selection_score):
                            raise ValueError("checkpoint validation selection score is nonfinite")
                        record.update(summary, has_partial_coefficient=False,
                            trajectory_checkpoint_iteration=endpoint,
                            budget_reached=(endpoint == budget or terminal_stationary))
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError, ArithmeticError, FitFailure) as error:
                        record.update(success=False, status=type(error).__name__, message=str(error),
                                      validation_mse=None, selection_score=None, budget_reached=False)
                self.selection_history_.append(record)
                budget_records.append(record)
            # Compare selected factor predictions directly. Only incumbent
            # predictions live during this pass; snapshots never retain a
            # validation-sized array. Materialize final winners only, replaying
            # comparisons if reconstruction disqualifies one of them.
            while True:
                best_budget, budget_prediction = None, None
                best_global = (self.selection_history_[self.selected_candidate_id_]
                               if self.selected_candidate_id_ is not None else None)
                next_global_prediction = global_prediction
                for record in budget_records:
                    record.pop("selection_comparison", None)
                    record.pop("budget_selection_comparison", None)
                    if not record["success"]:
                        continue
                    try:
                        prediction = self.trajectory_models_[record["grid_candidate_id"]]._checkpoint_prediction(
                            record["trajectory_checkpoint_iteration"], Xv)
                        checked_score, checked_selection = self._prediction_score(
                            prediction, Yv, preparation.validation_reference)
                        self._check_checkpoint_scores(record, checked_score, checked_selection)
                        local = self._comparison(prediction, budget_prediction, Yv,
                            best_budget["candidate_id"] if best_budget is not None else None)
                        overall = self._comparison(prediction, next_global_prediction, Yv,
                            best_global["candidate_id"] if best_global is not None else None)
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError, ArithmeticError, FitFailure) as error:
                        record.update(success=False, status=type(error).__name__, message=str(error),
                                      validation_mse=None, selection_score=None, budget_reached=False)
                        continue
                    record["budget_selection_comparison"], record["selection_comparison"] = local, overall
                    if best_budget is None or local["loss_difference"] < 0:
                        best_budget, budget_prediction = record, prediction.copy()
                    if best_global is None or overall["loss_difference"] < 0:
                        best_global, next_global_prediction = record, prediction.copy()
                if best_budget is None:
                    break
                required = [best_budget]
                if (best_global["candidate_id"] != self.selected_candidate_id_
                        and best_global is not best_budget):
                    required.append(best_global)
                materialized = {}
                for record in required:
                    try:
                        checkpoint = self.trajectory_models_[record["grid_candidate_id"]].checkpoint_model(
                            record["trajectory_checkpoint_iteration"])
                        if not checkpoint.success_:
                            raise FitFailure(checkpoint.status_, "Checkpoint is not a successful prefix")
                        checked_score, checked_selection = self._validation_mse(checkpoint, Xv, Yv,
                            selection_reference=preparation.validation_reference)
                        self._check_checkpoint_scores(record, checked_score, checked_selection)
                        materialized[record["candidate_id"]] = checkpoint
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError, ArithmeticError, FitFailure) as error:
                        record.update(success=False, status=type(error).__name__, message=str(error),
                                      validation_mse=None, selection_score=None, budget_reached=False)
                        break
                if len(materialized) != len(required):
                    continue
                self.checkpoints_[budget] = materialized[best_budget["candidate_id"]]
                budget_scores[budget], winner_ids[budget] = best_budget["validation_mse"], best_budget["candidate_id"]
                if best_global["candidate_id"] != self.selected_candidate_id_:
                    self.best_score_, self.best_params_ = best_global["validation_mse"], best_global["params"].copy()
                    self.best_selection_score_ = best_global["selection_score"]
                    self.model_ = self.estimator_ = materialized[best_global["candidate_id"]]
                    self.selected_budget_, self.selected_candidate_id_ = budget, best_global["candidate_id"]
                    global_prediction = next_global_prediction
                break
        self.diagnostics_.update(trajectory_fits=len(grid),
            successful_trajectories=sum(t["success"] for t in self.trajectory_history_),
            failed_trajectories=sum(not t["success"] for t in self.trajectory_history_))
        return budget_scores, winner_ids

    @staticmethod
    def _check_checkpoint_scores(record, score, selection_score):
        if not np.isclose(score, record["validation_mse"], rtol=1e-10, atol=1e-12):
            raise ValueError("checkpoint predictions disagree with its recorded validation loss")
        if not np.isclose(selection_score, record["selection_score"], rtol=1e-10, atol=1e-12):
            raise ValueError("checkpoint predictions disagree with its recorded selection score")

    def fit(self, X, Y, *, source: ExactSource | NoisySource, validation_data=None):
        """Fit training-only candidates and retain the successful validation winner.

        Failed candidates are not eligible, even if they have usable partial
        coefficients. If every candidate fails, return with ``success_=False``
        and status ``no_successful_candidate``; prediction then raises FitFailure.
        Equal validation scores retain the earliest budget, then grid position.
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
        self.iteration_budgets_ = _iteration_budgets(self.iteration_budgets, self.iterations)
        if not isinstance(self.checkpoint_execution, str) or self.checkpoint_execution not in ("independent", "continuous"):
            raise ValueError("checkpoint_execution must be 'independent' or 'continuous'")
        if self.checkpoint_execution == "independent" and self.checkpoint_interval is not None:
            raise ValueError("checkpoint_interval requires continuous checkpoint execution")
        self.validation_iterations_ = _validation_schedule(self.validation_iterations)
        if self.checkpoint_execution == "independent" and self.validation_iterations_:
            raise ValueError("validation_iterations requires continuous checkpoint execution")
        interval = 250 if self.checkpoint_interval is None else self.checkpoint_interval
        if isinstance(interval, (bool, np.bool_)) or not isinstance(interval, Integral) or interval < 1:
            raise ValueError("checkpoint_interval must be a positive integer or None")
        self.checkpoint_interval_ = int(interval) if self.checkpoint_execution == "continuous" else None
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
        self.best_selection_score_ = None
        self.selection_rule_ = SELECTION_RULE
        self.selected_budget_, self.selected_candidate_id_ = None, None
        self.checkpoints_ = {}
        budget_scores, budget_winner_ids = {}, {}
        self.model_, self.estimator_ = None, None
        self.diagnostics_ = {
            "theorem_certified": False, "selection_metric": "pairwise_validation_loss_difference",
            "selection_rule": SELECTION_RULE,
            "refit_on_all_data": False, "split_mode": self.split_mode_,
            "training_rows": Xtr.shape[0], "validation_rows": Xv.shape[0],
            "validation_score_is_independent_test_estimate": False,
            "source_accuracy_enforced": self.enforce_source_accuracy,
            "maximum_complement_support": maximum,
            "initialization_spectrum": self.initialization_spectrum,
            "refinement_solver": self.refinement_solver,
            "iteration_budgets": self.iteration_budgets_,
            "checkpoint_execution": "independent_fits",
        }
        grid = tuple(product(init_grid, grid_u, grid_v, support_grid))
        preparation = _FitPreparationCache()
        if self.checkpoint_execution == "continuous":
            budget_scores, budget_winner_ids = self._continuous_search(grid, Xtr, Ytr, Xv, Yv, source, preparation)
        else:
            global_prediction, budget_prediction, current_budget = None, None, None
            for candidate_id, (budget, (grid_id, values)) in enumerate(product(self.iteration_budgets_, enumerate(grid))):
                if budget != current_budget:
                    budget_prediction, current_budget = None, budget
                init, pu, pv, limits = values
                params = dict(init_penalty=init, penalty_u=pu, penalty_v=pv,
                              support_limits=limits, step_size_inverse=self.step_size_inverse)
                record = dict(candidate_id=candidate_id, grid_candidate_id=grid_id,
                              iteration_budget=budget, params=params, success=False,
                              status=None, message=None, validation_mse=None, selection_score=None,
                              selection_rule=SELECTION_RULE,
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
                        margins=self.margins, calibration=calibration, iterations=budget,
                        max_backtracks=self.max_backtracks, lasso_tol=self.lasso_tol,
                        lasso_max_iter=self.lasso_max_iter, spectral_step=self.spectral_step,
                        stationarity_tol=self.stationarity_tol,
                        initialization_spectrum=self.initialization_spectrum,
                        refinement_solver=self.refinement_solver,
                    )
                    model._fit_cache = preparation
                    preparation.validation_context = dict(grid_candidate_id=grid_id, iteration_budget=budget)
                    model.fit(Xtr, Ytr, source=source, validation_data=(Xv, Yv))
                    record.update(success=bool(model.success_), status=model.status_,
                                  message=model.message_, n_iter=int(model.n_iter_),
                                  selected_iteration=getattr(model, "selected_iteration_", None),
                                  termination_reason=getattr(model, "termination_reason_", model.status_),
                                  validation_history=getattr(model, "validation_history_", []))
                    record["diagnostics"] = getattr(model, "diagnostics_", {}).copy()
                    if model.success_:
                        score, checked_selection, prediction = self._validation_mse(model, Xv, Yv,
                            selection_reference=preparation.validation_reference, return_prediction=True)
                        selection_score = getattr(model, "best_selection_score_", checked_selection)
                        if selection_score is None or not np.isfinite(selection_score):
                            raise ValueError("candidate validation selection score is nonfinite")
                        if not np.isclose(checked_selection, selection_score, rtol=1e-10, atol=1e-12):
                            raise ValueError("candidate predictions disagree with its recorded selection score")
                        record["validation_mse"] = score
                        record["selection_score"] = selection_score
                        local_comparison = self._comparison(prediction, budget_prediction, Yv,
                                                            budget_winner_ids.get(budget))
                        global_comparison = self._comparison(prediction, global_prediction, Yv,
                                                             self.selected_candidate_id_)
                        record["budget_selection_comparison"] = local_comparison
                        record["selection_comparison"] = global_comparison
                        if budget_prediction is None or local_comparison["loss_difference"] < 0:
                            self.checkpoints_[budget] = model
                            budget_scores[budget], budget_winner_ids[budget] = score, candidate_id
                            budget_prediction = prediction.copy()
                        if global_prediction is None or global_comparison["loss_difference"] < 0:
                            self.best_score_, self.best_params_ = score, params.copy()
                            self.best_selection_score_ = selection_score
                            self.model_ = self.estimator_ = model
                            self.selected_budget_, self.selected_candidate_id_ = budget, candidate_id
                            global_prediction = prediction.copy()
                    elif hasattr(model, "coefficient_"):
                        record["has_partial_coefficient"] = True
                        record["partial_validation_mse"] = self._validation_mse(model, Xv, Yv, allow_partial=True)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError, FitFailure) as error:
                    record.update(success=False, status=type(error).__name__, message=str(error),
                                  validation_mse=None, selection_score=None)
                    if model is not None:
                        record["has_partial_coefficient"] = hasattr(model, "coefficient_")
                record["elapsed_time_sec"] = time.perf_counter() - started
                self.selection_history_.append(record)
        self.n_candidates_ = len(self.selection_history_)
        self.diagnostics_["selection_reference"] = preparation.validation_reference.metadata
        if preparation.validation_reference.prediction is not None:
            self.validation_reference_prediction_ = preparation.validation_reference.prediction.copy()
        winner = (self.selection_history_[self.selected_candidate_id_]
                  if self.selected_candidate_id_ is not None else None)
        continuations = [dict(candidate_id=record["candidate_id"], iteration_budget=record["iteration_budget"],
            success=record["success"], status=record["status"], termination_reason=record["termination_reason"])
            for record in self.selection_history_ if winner is not None
            and record["grid_candidate_id"] == winner["grid_candidate_id"]
            and record["iteration_budget"] > self.selected_budget_]
        budget_statuses = []
        for budget in self.iteration_budgets_:
            records = [record for record in self.selection_history_ if record["iteration_budget"] == budget]
            successful = sum(record["success"] for record in records)
            budget_statuses.append(dict(iteration_budget=budget, success=bool(successful),
                successful_candidates=successful, failed_candidates=len(records)-successful,
                best_candidate_id=budget_winner_ids.get(budget), best_validation_mse=budget_scores.get(budget)))
            budget_statuses[-1]["best_selection_score"] = (
                self.selection_history_[budget_winner_ids[budget]]["selection_score"]
                if budget in budget_winner_ids else None)
            if self.checkpoint_execution == "continuous":
                reached = sum(record["budget_reached"] for record in records)
                budget_statuses[-1].update(budget_reached_candidates=reached,
                    unreached_candidates=len(records)-reached, budget_fully_covered=(reached == len(records)))
        n_successful = sum(record["success"] for record in self.selection_history_)
        self.diagnostics_.update(selected_budget=self.selected_budget_,
            best_selection_score=self.best_selection_score_,
            selected_candidate_id=self.selected_candidate_id_,
            selected_checkpoint_status=winner["status"] if winner is not None else None,
            selected_checkpoint_termination_reason=winner["termination_reason"] if winner is not None else None,
            retained_budget_checkpoints=len(self.checkpoints_), successful_candidate_fits=n_successful,
            failed_candidate_fits=self.n_candidates_-n_successful, budget_statuses=budget_statuses,
            retained_earlier_checkpoint=bool(winner is not None and self.selected_budget_ < self.iteration_budgets_[-1]),
            selected_checkpoint_continuations=continuations,
            selected_checkpoint_retained_after_failure=any(not record["success"] for record in continuations))
        if self.checkpoint_execution == "continuous":
            self.selected_checkpoint_iteration_ = winner["trajectory_checkpoint_iteration"] if winner else None
            self.diagnostics_.update(selected_checkpoint_iteration=self.selected_checkpoint_iteration_,
                selected_budget_reached=winner["budget_reached"] if winner else False,
                selected_checkpoint_retained_after_failure=bool(winner and not winner["trajectory_success"]))
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
