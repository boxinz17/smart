"""Full-source-frame, partially hard-sparse two-stage regression.

Initialization rejects infeasible spectra. Refinement includes the declared
spectral constraints in its proximal step and rejects other infeasible chart
states. Unpenalized full-capacity configurations dispatch directly to target
RRR before initialization; this is an explicit branch, never a failure fallback.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np

from sparse_smart.anchors import AnchorFailure, select_anchor
from sparse_smart.chart import AnchorChart
from sparse_smart.stopping import ValidationStopping, ValidationStopRequest
from sparse_smart.validation import validation_loss_difference

from .calibration import Margins, PracticalCalibration
from .anchors import CHART_CONTINUATION_RULE, reanchor
from .initialization import InitializationFailure, reduced_lasso
from .solver import IterationRecord, _result, refine
from .source import prepare_source
from .support import FreeRows, choose_free_rows, threshold_state, validate_support_limits
from .rrr import shortcut_reason, target_rrr


REFINEMENT_SOLVERS = ("masked_chart_spectral_soft_hard", "masked_anchor_projected")


def _sum_work(values):
    """Combine numeric work counters, preserving the latest call diagnostics."""
    result = {}
    for value in values:
        for key, item in value.items():
            if key in ("last", "last_failure", "failure_threshold", "consecutive_updates", "recovery_interval"):
                result[key] = deepcopy(item)
            elif isinstance(item, dict):
                result[key] = _sum_work((result.get(key, {}), item))
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                result[key] = result.get(key, 0) + item
            else:
                result[key] = deepcopy(item)
    return result


class FitFailure(RuntimeError):
    """A reported algorithmic failure, including its machine-readable status."""

    def __init__(self, status, message):
        self.status = status
        super().__init__(f"{status}: {message}")


@dataclass(frozen=True)
class TrajectoryCheckpoint:
    """Accepted finite prefix, including its validation-selected earlier state."""

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
    validation_stopping: dict
    chart: AnchorChart
    selected_chart: AnchorChart
    anchor_switches: tuple
    chart_epoch: int
    selected_chart_epoch: int
    endpoint_record: IterationRecord


@dataclass(frozen=True)
class RRRCheckpoint:
    """A direct physical coefficient, with no fabricated chart or trajectory."""

    iteration: int
    coefficient: np.ndarray
    selected_iteration: int
    best_validation_loss: float | None
    status: str
    termination_reason: str
    message: str
    validation_stopping: dict
    certificate: dict


def _data(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite two-dimensional array")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _schedule(values, name, *, maximum=None):
    if values is None:
        return ()
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of distinct nonnegative integers") from error
    for value in values:
        _integer(value, name)
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} cannot exceed iterations")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain distinct iterations")
    return tuple(sorted(int(value) for value in values))


def _finite_record(state, record):
    return (np.isfinite(state).all() and all(
        value is not None and np.isfinite(value) for value in
        (record.objective, record.smooth_loss, record.penalty_value, record.raw_gradient_norm)))


def _freeze_source_arrays(source):
    # deepcopy deliberately isolates public arrays, but also removes NumPy's
    # read-only flags. Reinstate the source-frame contract after copying.
    for name in ("left", "right", "leading_left", "leading_right", "source_singular_values"):
        value = getattr(source, name)
        if value is not None:
            value.setflags(write=False)


def _chart_boundary_rejection(reason):
    return reason in (
        "left anchor singular value is below anchor_min",
        "right anchor singular value is below anchor_min",
        "left Cayley coordinate has operator norm above 1/2",
        "right Cayley coordinate has operator norm above 1/2",
    )


def _transition_counts(events):
    """Legacy total plus separate counts of affected sides, not proposals."""
    anchors = rotations = 0
    for event in events:
        for detail in event.get("sides", {}).values():
            anchors += detail.get("old_anchors") != detail.get("new_anchors")
            rotations += detail.get("action") in ("rotation_recenter", "anchor_switch_and_recenter")
    return dict(anchor_switches=len(events), chart_transitions=len(events),
                anchor_row_switches=anchors, rotation_recenters=rotations)


class SparseSMARTv2:
    """Lasso--SVD initialization and masked soft--hard chart refinement.

    ``coefficient_`` is validation-selected when validation data are supplied;
    ``last_coefficient_`` is the terminal accepted state. Prediction after a
    failed refinement requires ``allow_partial=True``. Validation patience
    counts accepted iterations, not validation checks. Neither budget
    completion nor a validation stop certifies optimization convergence.
    Optional chart continuation can replace weak anchors inside fixed free
    rows or recenter rotations, without counting either as an accepted update.
    By default, effective zero penalties with full outside capacities return
    target-only RRR directly, without initialization, chart constraints or
    validation checkpoint selection. Set ``rrr_shortcut=False`` only to retain
    the legacy chart path for diagnostics or reproducibility.
    """

    def __init__(self, *, rank, source_rank, margins, calibration,
                 free_directions=None, iterations=500, max_backtracks=60,
                 stationarity_tol=None, checkpoint_iterations=None,
                 validation_interval=25, validation_iterations=None,
                 validation_patience=None, validation_min_iterations=100,
                 validation_min_relative_improvement=0.0, raise_on_failure=False,
                 adaptive_anchors=False, max_anchor_switches=16,
                 refinement_solver="masked_chart_spectral_soft_hard", rrr_shortcut=True):
        self.rank, self.source_rank = rank, source_rank
        self.margins, self.calibration = margins, calibration
        self.free_directions = free_directions
        self.iterations, self.max_backtracks = iterations, max_backtracks
        self.stationarity_tol = stationarity_tol
        self.checkpoint_iterations = checkpoint_iterations
        self.validation_interval, self.validation_iterations = validation_interval, validation_iterations
        self.validation_patience = validation_patience
        self.validation_min_iterations = validation_min_iterations
        self.validation_min_relative_improvement = validation_min_relative_improvement
        self.raise_on_failure = raise_on_failure
        self.adaptive_anchors, self.max_anchor_switches = adaptive_anchors, max_anchor_switches
        self.refinement_solver = refinement_solver
        self.rrr_shortcut = rrr_shortcut

    def _failure(self, status, message):
        self.status_, self.success_, self.message_ = status, False, str(message)
        self.termination_reason_ = self.termination_reason_ or status
        self.converged_, self.optimization_converged_ = False, False
        self.metadata_.update(status=status, termination_reason=self.termination_reason_)
        if self.raise_on_failure:
            raise FitFailure(status, message)
        return self

    def fit(self, X, Y, *, source, validation_data=None, _initialization_cache=None):
        """Fit without centering, rescaling, sample splitting or spectral repair.

        Invalid inputs raise ``ValueError``. Numerical/algorithmic failures
        are inspectable fitted statuses unless ``raise_on_failure`` is set.
        The initial and terminal successful states are always checkpointed;
        additional checkpoints are specified by ``checkpoint_iterations``.
        """
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        self.status_, self.success_, self.message_ = "not_fitted", False, "Fit has not completed."
        self.n_iter_, self.history_, self.validation_history_ = 0, [], []
        self.history_chart_epochs_ = []
        self.termination_reason_, self.converged_, self.optimization_converged_ = None, False, False
        self.checkpoints_, self.checkpoint_iterations_ = {}, ()
        self.best_validation_loss_, self.selected_iteration_ = None, None
        self.anchor_switches_ = []
        self.method_ = "sparse_smart_v2"
        self.metadata_ = {"theorem_certified": False, "arithmetic": "float64",
                          "same_target_training_data": True, "full_source_frames": True,
                          "initialization_spectrum_repaired": False,
                          "selection_rule": "paired_validation_loss_difference_earliest_tie",
                          "fit_method": self.method_, "rrr_shortcut_applied": False}
        self.diagnostics_ = self.metadata_
        X, Y = _data(X, "X"), _data(Y, "Y")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have matching observation counts")
        if _initialization_cache is not None:
            if not isinstance(_initialization_cache, dict):
                raise ValueError("_initialization_cache must be a tuner-owned dictionary")
            context = (id(X), id(Y))
            if _initialization_cache.setdefault("context", context) != context:
                raise ValueError("initialization cache cannot be reused with different training observations")
        rank = _integer(self.rank, "rank", 1)
        r0 = _integer(self.source_rank, "source_rank", 1)
        if max(rank, r0) > min(X.shape[1], Y.shape[1]):
            raise ValueError("require rank and source_rank <= min(p, q)")
        for name, minimum in (("iterations", 0), ("max_backtracks", 0), ("validation_interval", 1),
                              ("max_anchor_switches", 0)):
            _integer(getattr(self, name), name, minimum)
        if not isinstance(self.adaptive_anchors, (bool, np.bool_)):
            raise ValueError("adaptive_anchors must be a boolean")
        if not isinstance(self.rrr_shortcut, (bool, np.bool_)):
            raise ValueError("rrr_shortcut must be a boolean")
        self.metadata_.update(adaptive_anchors=bool(self.adaptive_anchors),
                              max_anchor_switches=int(self.max_anchor_switches),
                              chart_continuation_rule=CHART_CONTINUATION_RULE,
                              chart_transition_count_units="transitions; anchor_row_switches and rotation_recenters count sides")
        if self.stationarity_tol is not None and (
                isinstance(self.stationarity_tol, (bool, np.bool_))
                or not isinstance(self.stationarity_tol, Real)
                or not np.isfinite(self.stationarity_tol) or self.stationarity_tol <= 0):
            raise ValueError("stationarity_tol must be finite and positive or None")
        if not isinstance(self.margins, Margins):
            raise ValueError("margins must be Margins")
        if not isinstance(self.calibration, PracticalCalibration):
            raise ValueError("calibration must be sparse_smart_v2.PracticalCalibration")
        if self.refinement_solver not in REFINEMENT_SOLVERS:
            raise ValueError(f"refinement_solver must be one of {REFINEMENT_SOLVERS}")
        direct_reason = shortcut_reason(rank=rank, p=X.shape[1], q=Y.shape[1],
            free_directions=self.free_directions, penalties=self.calibration.penalties,
            support_limits=self.calibration.support_limits) if self.rrr_shortcut else None
        if direct_reason is None and rank > r0:
            raise ValueError("require rank <= source_rank <= min(p, q) for chart refinement")
        refiner = refine
        if self.refinement_solver == "masked_anchor_projected":
            from .anchor_solver import refine as refiner
        self.metadata_.update(refinement_solver=self.refinement_solver,
            optimization_coordinates="omega_d_H" if self.refinement_solver == "masked_anchor_projected" else "omega_d_Z")
        requested = set(_schedule(self.checkpoint_iterations, "checkpoint_iterations", maximum=self.iterations))
        extra_validation = set(_schedule(self.validation_iterations, "validation_iterations"))
        stopping = ValidationStopping(self.validation_patience, self.validation_min_iterations,
                                      self.validation_min_relative_improvement)
        if stopping.enabled and validation_data is None and direct_reason is None:
            raise ValueError("validation_data is required for validation_patience")
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
                raise ValueError("validation_data must be (X_validation, Y_validation)")
            XV, YV = _data(validation_data[0], "X_validation"), _data(validation_data[1], "Y_validation")
            if XV.shape[0] != YV.shape[0] or XV.shape[1] != X.shape[1] or YV.shape[1] != Y.shape[1]:
                raise ValueError("validation_data must have matching rows, predictors and responses")
        self.n_features_in_, self.n_responses_ = X.shape[1], Y.shape[1]
        self.calibration_ = deepcopy(self.calibration)
        self.validation_stopping_ = stopping.snapshot()
        self.metadata_["rrr_shortcut_enabled"] = bool(self.rrr_shortcut)
        if direct_reason is not None:
            return self._fit_target_rrr(X, Y, rank, direct_reason,
                None if validation_data is None else (XV, YV), _initialization_cache)
        try:
            source_key = ("source", id(source), r0, X.shape[1], Y.shape[1])
            if _initialization_cache is not None and source_key in _initialization_cache:
                self.source_ = deepcopy(_initialization_cache[source_key])
            else:
                prepared = prepare_source(source, p=X.shape[1], q=Y.shape[1], source_rank=r0)
                if _initialization_cache is not None:
                    _initialization_cache[source_key] = prepared
                self.source_ = deepcopy(prepared)
            _freeze_source_arrays(self.source_)
            key = ("initialization", id(source), rank, r0, self.calibration_.init_penalty)
            if _initialization_cache is not None and key in _initialization_cache:
                initial = deepcopy(_initialization_cache[key])
            else:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    initial = reduced_lasso(X @ self.source_.leading_left, Y @ self.source_.leading_right,
                                            rank, self.calibration_.init_penalty)
                if _initialization_cache is not None:
                    _initialization_cache[key] = deepcopy(initial)
        except InitializationFailure as error:
            return self._failure("initialization_failed", error)
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            return self._failure("numerical_failure", error)
        self.initialization_ = initial
        self.metadata_["lasso"] = {"converged": bool(initial.converged),
                                    "kkt_residual": float(initial.kkt_residual),
                                    "n_iter": np.asarray(initial.n_iter).tolist()}
        if not initial.converged:
            return self._failure("lasso_not_converged", "Initializer did not meet its numerical optimality tolerances.")
        d = initial.d.copy()
        self.metadata_["initialization_singular_values"] = d.tolist()
        if (d.shape != (rank,) or not np.isfinite(d).all()
                or np.any(d < self.margins.d_lower) or np.any(d > self.margins.d_upper)
                or rank > 1 and np.any(d[:-1] - d[1:] < self.margins.gap)):
            return self._failure("initialization_spectrum_failed", "Initializer violates declared spectral margins; no repair is applied.")
        try:
            au, av = (select_anchor(factor, anchor_min=self.margins.anchor_min,
                                   qr_threshold=self.margins.qr_threshold)
                      for factor in (initial.P, initial.Q))
        except AnchorFailure as error:
            return self._failure("anchor_selection_failed", error)
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            return self._failure("numerical_failure", error)
        self.anchors_ = {"u": au, "v": av}
        self.chart_ = AnchorChart(X.shape[1], Y.shape[1], au.indices, av.indices, au.center, av.center)
        free_directions = (rank, rank) if self.free_directions is None else self.free_directions
        self.free_rows_ = choose_free_rows(self.chart_, free_directions)
        validate_support_limits(self.free_rows_, self.calibration_.support_limits)
        if (self.refinement_solver == "masked_anchor_projected"
                and tuple(self.calibration_.support_limits) != self.free_rows_.capacities):
            raise ValueError("masked_anchor_projected requires full outside support capacities; restrictive hard caps are unsupported")
        self.free_directions_ = (len(self.free_rows_.rows_u), len(self.free_rows_.rows_v))
        self.metadata_.update(free_directions=self.free_directions_,
            free_rows={"u": self.free_rows_.rows_u.tolist(), "v": self.free_rows_.rows_v.tolist()},
            anchors={"u": au.indices.tolist(), "v": av.indices.tolist()},
            support_limits=tuple(self.calibration_.support_limits),
            penalties=tuple(self.calibration_.penalties),
            free_dimension=rank * (sum(self.free_directions_) - rank),
            source_mode=self.source_.mode, fitted_rank=rank, initializer_source_rank=r0,
            initializer_penalty=self.calibration_.init_penalty,
            source_frame_dimensions=(self.source_.left.shape[1], self.source_.right.shape[1]),
            initializer_solver="reduced_lasso_then_rank_svd", refinement_solver=self.refinement_solver)
        P, Q = np.zeros((X.shape[1], rank)), np.zeros((Y.shape[1], rank))
        P[:r0], Q[:r0] = initial.P, initial.Q
        try:
            x0 = self.chart_.initial_state(P, d, Q)
            self.untruncated_initial_state_ = x0.copy()
            x0 = threshold_state(self.chart_, x0, self.free_rows_, penalties=(0., 0.),
                                 support_limits=self.calibration_.support_limits, inverse_step=1.)
            self.initial_state_ = x0.copy()
            self.initial_chart_ = self.chart_
            reason = self.chart_.domain_reason(x0, d_lower=self.margins.d_lower,
                d_upper=self.margins.d_upper, gap=self.margins.gap, anchor_min=self.margins.anchor_min)
            if reason:
                return self._failure("handoff_failed", reason)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return self._failure("handoff_failed", error)

        selected_state = None
        selected_iteration = None
        selected_prediction = None
        selected_chart = None
        selected_chart_epoch = 0
        last_finite_state = x0.copy()

        def evaluate_validation(iteration, state, *, allow_stop):
            nonlocal selected_state, selected_iteration, selected_prediction, selected_chart, selected_chart_epoch
            if self.validation_history_ and self.validation_history_[-1]["iteration"] == iteration:
                return
            with np.errstate(over="raise", invalid="raise"):
                prediction = XV @ self._coefficient(state)
                loss = float(np.mean((YV - prediction) ** 2))
            if not np.isfinite(loss):
                raise FloatingPointError("Validation MSE is not finite")
            difference = (None if selected_prediction is None else
                          validation_loss_difference(prediction, selected_prediction, YV))
            incumbent_iteration = selected_iteration
            if selected_prediction is None or difference < 0:
                self.best_validation_loss_ = loss
                selected_state, selected_iteration = state.copy(), int(iteration)
                selected_prediction = prediction.copy()
                selected_chart, selected_chart_epoch = self.chart_, len(self.anchor_switches_)
            detail = stopping.observe(iteration, loss, allow_stop=allow_stop)
            self.validation_history_.append({"iteration": int(iteration), "loss": loss,
                                            "chart_epoch": len(self.anchor_switches_),
                                            "selection_loss_difference": difference,
                                            "incumbent_iteration": incumbent_iteration,
                                            "validation_stopping": detail})
            self.validation_stopping_ = stopping.snapshot()

        def capture(iteration, state, record, terminal=None):
            if not _finite_record(state, record):
                return
            endpoint = state.copy()
            chosen = endpoint.copy() if selected_state is None else selected_state.copy()
            endpoint.setflags(write=False)
            chosen.setflags(write=False)
            self.checkpoints_[int(iteration)] = TrajectoryCheckpoint(
                int(iteration), endpoint, chosen,
                int(iteration) if selected_iteration is None else selected_iteration,
                self.best_validation_loss_, len(self.history_), len(self.validation_history_),
                "completed" if terminal is None else terminal.status,
                "max_iterations" if terminal is None else terminal.termination_reason,
                f"Retained {iteration} accepted updates." if terminal is None else terminal.message,
                stopping.snapshot(), self.chart_, self.chart_ if selected_chart is None else selected_chart,
                tuple(deepcopy(self.anchor_switches_)), len(self.anchor_switches_),
                len(self.anchor_switches_) if selected_chart is None else selected_chart_epoch,
                deepcopy(record))
            self.checkpoint_iterations_ = tuple(sorted(self.checkpoints_))

        def observe(iteration, state, record):
            nonlocal last_finite_state
            self.history_.append(record)
            self.history_chart_epochs_.append(len(self.anchor_switches_))
            self.n_iter_ = int(iteration)
            if not _finite_record(state, record):
                return
            last_finite_state = state.copy()
            eligible = (iteration == 0 or iteration % self.validation_interval == 0
                        or iteration in extra_validation or iteration in requested)
            mapping = record.projected_gradient_norm
            stationary = (self.iterations > 0 and record.mapping_domain_reason is None
                and mapping is not None and (mapping == 0 or
                    self.stationarity_tol is not None and mapping <= self.stationarity_tol))
            if validation_data is not None and eligible:
                evaluate_validation(iteration, state, allow_stop=not stationary)
            if iteration == 0 or iteration in requested:
                capture(iteration, state, record)
            if stopping.stopped:
                return ValidationStopRequest()

        try:
            with np.errstate(over="raise", invalid="raise"):
                design, response = X @ self.source_.left, Y @ self.source_.right
            offset, segment_state, segment_number = 0, x0, 0
            segment_work = []
            while True:
                def segment_observe(iteration, state, record):
                    if segment_number and iteration == 0:
                        # A coordinate change is not an accepted optimization
                        # update or another validation observation.
                        return None
                    return observe(offset + iteration, state, replace(record, iteration=offset + iteration))

                segment = refiner(self.chart_, segment_state, design, response, calibration=self.calibration_,
                    margins=self.margins, free_rows=self.free_rows_, iterations=self.iterations - offset,
                    max_backtracks=self.max_backtracks, stationarity_tol=self.stationarity_tol,
                    iterate_callback=segment_observe)
                total = offset + segment.n_iter
                work = deepcopy(segment.numerical_work)
                if self.refinement_solver == "masked_anchor_projected":
                    segment_work.append(dict(start_iteration=offset, n_iter=segment.n_iter,
                        chart_epoch=len(self.anchor_switches_), work=deepcopy(work)))
                    for key in ("totals", "search"):
                        work[key] = _sum_work(row["work"].get(key, {}) for row in segment_work)
                    work["segments"] = deepcopy(segment_work)
                    work["inner_proximal_solves"] = sum(work.get("totals", {}).get(side, {}).get("calls", 0)
                                                        for side in ("u", "v"))
                work.update(accepted_updates=total,
                            accepted_update_backtracks=sum(record.backtracks for record in self.history_),
                            **_transition_counts(self.anchor_switches_))
                result = replace(segment, n_iter=total, history=self.history_, numerical_work=work)
                if result.termination_reason == "max_iterations":
                    result.message = f"Completed {total} refinement updates."
                chart_blocked = (not result.success
                    and result.status in ("numerical_stagnation", "line_search_failed")
                    and _chart_boundary_rejection(result.last_rejection))
                if (not self.adaptive_anchors or not chart_blocked or total >= self.iterations
                        or len(self.anchor_switches_) >= self.max_anchor_switches):
                    break
                switched = reanchor(self.chart_, result.state, self.free_rows_,
                    anchor_min=self.margins.anchor_min, d_lower=self.margins.d_lower,
                    d_upper=self.margins.d_upper, gap=self.margins.gap)
                if switched is None:
                    break
                event = deepcopy(switched.event)
                event.update(iteration=total, switch_index=len(self.anchor_switches_) + 1,
                             previous_status=result.status, previous_rejection=result.last_rejection)
                self.anchor_switches_.append(event)
                self.chart_, self.free_rows_ = switched.chart, switched.free_rows
                segment_state = switched.state.copy()
                last_finite_state = segment_state.copy()
                offset, segment_number = total, segment_number + 1
            self.terminal_record_ = (replace(segment.history[-1], iteration=total)
                                     if segment.history else None)
            if result.success and self.terminal_record_ is not None and _finite_record(result.state, self.terminal_record_):
                if validation_data is not None:
                    evaluate_validation(result.n_iter, result.state, allow_stop=False)
                capture(result.n_iter, result.state, self.terminal_record_, terminal=result)
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            result = _result(last_finite_state, "numerical_failure", str(error), self.n_iter_,
                             self.history_, termination_reason="numerical_failure")
        return self._finalize(result, selected_state, selected_iteration, selected_chart)

    def _fit_target_rrr(self, X, Y, rank, reason, validation_data, cache):
        """Explicit target-only endpoint, never a response to chart failure."""
        self.method_ = "target_rrr"
        self.metadata_.update(fit_method=self.method_, rrr_shortcut_applied=True,
            rrr_shortcut_reason=reason, refinement_solver="target_rrr_closed_form",
            requested_refinement_solver=self.refinement_solver,
            optimization_coordinates="physical_coefficient",
            selection_rule="single_target_rrr_fit", source_used=False,
            full_source_frames=False, initialization_skipped="target_rrr_shortcut",
            chart_constraints_applied=False, fitted_rank=rank,
            initializer_source_rank=int(self.source_rank), initializer_penalty=self.calibration_.init_penalty,
            penalties=tuple(self.calibration_.penalties), support_limits=tuple(self.calibration_.support_limits))
        self.free_directions_ = tuple((rank, rank) if self.free_directions is None else self.free_directions)
        self.metadata_.update(free_directions=self.free_directions_, free_rows=None, anchors=None,
            fitted_support_counts=None, factor_coordinates="physical")
        self.validation_stopping_.update(applied=False, reason="target_rrr_closed_form")
        key = ("target_rrr", rank)
        reused = cache is not None and key in cache
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                direct = deepcopy(cache[key]) if reused else target_rrr(X, Y, rank)
                if not direct.certificate.get("certified", False):
                    raise FloatingPointError("RRR did not attain its numerical training-loss certificate")
                if cache is not None and not reused:
                    cache[key] = deepcopy(direct)
                score = (None if validation_data is None else
                    float(np.mean((validation_data[0] @ direct.coefficient - validation_data[1]) ** 2)))
                if score is not None and not np.isfinite(score):
                    raise FloatingPointError("RRR validation MSE is not finite")
        except (np.linalg.LinAlgError, FloatingPointError, ArithmeticError) as error:
            return self._failure("numerical_failure", error)
        self.rrr_result_ = self.result_ = direct
        self.rrr_certificate_ = deepcopy(direct.certificate)
        self.coefficient_ = direct.coefficient.copy()
        self.last_coefficient_ = direct.coefficient.copy()
        self.left_factors_, self.right_factors_ = direct.left_factors.copy(), direct.right_factors.copy()
        self.singular_values_ = direct.singular_values.copy()
        self.factors_ = dict(P=self.left_factors_.copy(), d=self.singular_values_.copy(), Q=self.right_factors_.copy())
        self.n_iter_ = self.selected_iteration_ = 0
        self.best_validation_loss_ = score
        self.status_, self.success_ = "converged", True
        self.termination_reason_ = "target_rrr_closed_form"
        self.message_ = "Returned target-only reduced-rank least squares on the numerical design range."
        self.optimization_converged_ = self.converged_ = True
        self.numerical_work_ = dict(solver="target_rrr_closed_form", rrr_solves=int(not reused),
            rrr_cache_hits=int(reused), accepted_updates=0, accepted_update_backtracks=0,
            inner_proximal_solves=0, **_transition_counts(()))
        if score is not None:
            self.validation_history_ = [dict(iteration=0, loss=score, selection_loss_difference=None,
                incumbent_iteration=None, validation_stopping=deepcopy(self.validation_stopping_))]
        self.metadata_.update(status=self.status_, termination_reason=self.termination_reason_,
            n_iter=0, selected_iteration=0, effective_rank=direct.effective_rank,
            design_rank=direct.design_rank, validation_mse=score,
            rrr_certificate=deepcopy(self.rrr_certificate_), rrr_cache_reused=bool(reused),
            validation_stopping=deepcopy(self.validation_stopping_),
            training_loss=direct.training_loss, optimal_training_loss=direct.optimal_training_loss,
            objective_gap=direct.objective_gap)
        coefficient = self.coefficient_.copy()
        coefficient.setflags(write=False)
        self.checkpoints_[0] = RRRCheckpoint(0, coefficient, 0, score, self.status_,
            self.termination_reason_, self.message_, deepcopy(self.validation_stopping_),
            deepcopy(self.rrr_certificate_))
        self.checkpoint_iterations_ = (0,)
        return self

    def _coefficient(self, state, chart=None):
        P, d, Q = (self.chart_ if chart is None else chart).reconstruct(state)
        return ((self.source_.left @ P) * d) @ (self.source_.right @ Q).T

    def _free_rows_for_chart(self, chart):
        rows, masks = [], []
        for side in ("u", "v"):
            fixed = getattr(self.free_rows_, f"rows_{side}")
            complement = getattr(chart, f"complement_{side}")
            rows.append(fixed)
            masks.append(np.repeat((~np.isin(complement, fixed))[:, None], chart.rank, axis=1))
        return FreeRows(rows[0], rows[1], masks[0], masks[1])

    def _finalize(self, result, selected_state=None, selected_iteration=None, selected_chart=None):
        work = deepcopy(result.numerical_work)
        work.update(accepted_updates=result.n_iter, **_transition_counts(self.anchor_switches_),
                    accepted_update_backtracks=sum(record.backtracks for record in result.history))
        result = replace(result, numerical_work=work)
        self.result_, self.last_state_ = result, result.state.copy()
        self.state_ = self.last_state_.copy() if selected_state is None else selected_state.copy()
        self.n_iter_, self.history_ = result.n_iter, result.history
        self.selected_iteration_ = self.n_iter_ if selected_iteration is None else int(selected_iteration)
        self.termination_reason_ = result.termination_reason
        self.numerical_work_ = deepcopy(result.numerical_work)
        self.selected_chart_ = self.chart_ if selected_chart is None else selected_chart
        self.selected_free_rows_ = self._free_rows_for_chart(self.selected_chart_)
        self.last_coefficient_ = self._coefficient(self.last_state_)
        self.coefficient_ = self._coefficient(self.state_, self.selected_chart_)
        P, d, Q = self.selected_chart_.reconstruct(self.state_)
        self.factors_ = {"P": P, "d": d, "Q": Q}
        self.left_factors_, self.right_factors_ = self.source_.left @ P, self.source_.right @ Q
        self.singular_values_ = d.copy()
        blocks = self.selected_chart_.unpack(self.state_)[-2:]
        self.supports_ = {side: np.flatnonzero(block[mask]) for side, block, mask in
                         zip(("u", "v"), blocks, (self.selected_free_rows_.penalized_u, self.selected_free_rows_.penalized_v))}
        self.metadata_.update(status=result.status, termination_reason=result.termination_reason,
            n_iter=self.n_iter_, selected_iteration=self.selected_iteration_,
            anchor_switches=deepcopy(self.anchor_switches_),
            terminal_anchors={"u": self.chart_.anchors_u.tolist(), "v": self.chart_.anchors_v.tolist()},
            selected_anchors={"u": self.selected_chart_.anchors_u.tolist(), "v": self.selected_chart_.anchors_v.tolist()},
            validation_mse=self.best_validation_loss_, validation_stopping=deepcopy(self.validation_stopping_),
            fitted_support_counts=tuple(len(self.supports_[side]) for side in ("u", "v")))
        if not result.success:
            return self._failure(result.status, result.message)
        self.status_, self.success_, self.message_ = result.status, True, result.message
        self.optimization_converged_ = result.termination_reason == "stationarity"
        self.converged_ = self.optimization_converged_ and self.selected_iteration_ == self.n_iter_
        return self

    def checkpoint_model(self, iteration):
        """Return an independent successful prefix without refitting any data."""
        iteration = _integer(iteration, "checkpoint iteration")
        checkpoint = getattr(self, "checkpoints_", {}).get(iteration)
        if checkpoint is None:
            raise FitFailure("checkpoint_unavailable", f"No successful checkpoint at iteration {iteration}")
        if self.method_ == "target_rrr":
            view = deepcopy(self)
            view.iterations = 0
            view.checkpoints_[0].coefficient.setflags(write=False)
            return view
        view = deepcopy(self)
        _freeze_source_arrays(view.source_)
        view.iterations = iteration
        view.checkpoints_ = {t: c for t, c in view.checkpoints_.items() if t <= iteration}
        view.checkpoint_iterations_ = tuple(sorted(view.checkpoints_))
        view.validation_history_ = view.validation_history_[:checkpoint.validation_history_length]
        view.history_chart_epochs_ = view.history_chart_epochs_[:checkpoint.history_length]
        view.validation_stopping_ = deepcopy(checkpoint.validation_stopping)
        view.best_validation_loss_ = checkpoint.best_validation_loss
        view.chart_ = deepcopy(checkpoint.chart)
        view.free_rows_ = view._free_rows_for_chart(view.chart_)
        view.anchor_switches_ = list(deepcopy(checkpoint.anchor_switches))
        result = _result(checkpoint.state.copy(), checkpoint.status, checkpoint.message, iteration,
            deepcopy(self.history_[:checkpoint.history_length]), termination_reason=checkpoint.termination_reason)
        if self.refinement_solver == "masked_anchor_projected":
            # Compact aggregate proximal counters cannot be apportioned to an
            # arbitrary earlier prefix. Never report the hard solver's zero.
            work = (deepcopy(self.numerical_work_) if iteration == self.n_iter_ else
                dict(solver=self.refinement_solver, coordinate_metric="H=Z/D",
                     inner_proximal_solves=None, scope="checkpoint_prefix_work_not_retained"))
            result = replace(result, numerical_work=work)
        endpoint = deepcopy(checkpoint.endpoint_record)
        view.terminal_record_ = endpoint
        result = replace(result, projected_gradient_norm=endpoint.projected_gradient_norm,
                         raw_gradient_norm=endpoint.raw_gradient_norm, mapping_displacement=endpoint.mapping_displacement,
                         proximal_uncertainty=endpoint.proximal_uncertainty,
                         mapping_refinements=endpoint.mapping_refinements,
                         mapping_precision_limited=endpoint.mapping_precision_limited)
        return view._finalize(result, checkpoint.selected_state, checkpoint.selected_iteration,
                              deepcopy(checkpoint.selected_chart))

    def predict(self, X, *, allow_partial=False):
        if not hasattr(self, "coefficient_"):
            raise FitFailure(getattr(self, "status_", "not_fitted"), "No coefficient estimate is available.")
        if not self.success_ and not allow_partial:
            raise FitFailure(self.status_, "Fit failed; allow_partial=True uses its retained accepted estimate.")
        X = _data(X, "X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of predictors")
        return X @ self.coefficient_
