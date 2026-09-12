"""Held-out target tuning of independently fitted hard-sparse candidates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from numbers import Integral, Real
import time

import numpy as np

from sparse_smart.validation import SELECTION_RULE, validation_loss_difference

from .estimator import FitFailure
from .source import prepare_source


def _matrix(value, name):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be a finite nonempty real matrix")
    try:
        array = np.array(array, dtype=float, copy=True)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a finite nonempty real matrix") from error
    if array.ndim != 2 or min(array.shape) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty real matrix")
    return array


def _settings(candidate):
    """Record configuration, without retaining a fitted trajectory or source."""
    names = (
        "rank", "source_rank", "free_directions", "iterations", "max_backtracks",
        "stationarity_tol", "checkpoint_iterations", "validation_interval",
        "validation_iterations", "validation_patience", "validation_min_iterations",
        "validation_min_relative_improvement", "margins", "calibration",
    )
    result = {}
    for name in names:
        if hasattr(candidate, name):
            value = getattr(candidate, name)
            result[name] = asdict(value) if is_dataclass(value) else deepcopy(value)
    return result


class SparseSMARTv2Tuner:
    """Choose a candidate's selected iterate by held-out mean squared error.

    Parameters
    ----------
    candidates : sequence of SparseSMARTv2
        Estimator templates. Each is deep-copied before fitting; the supplied
        objects are never fitted or mutated. All candidates must use the same
        ``source_rank`` so one source frame preparation is shared. Any other
        configuration, including iteration budgets and anchor margins, may vary.
    validation_fraction : float, default=1/3
        Fraction reserved by a reproducible random holdout when no explicit
        ``validation_data`` is supplied. The count is rounded up.
    random_state : nonnegative integer, default=0
        Seed used only for the holdout split.
    raise_on_failure : bool, default=False
        Raise ``FitFailure`` if every candidate is ineligible. Candidate fit
        exceptions are always recorded, allowing other candidates to finish.

    Only successful fits with finite validation predictions can win. Stable
    pairwise loss differences rank candidates; an exact difference of zero
    chooses the earliest candidate. The winner remains fitted on the
    training split; there is no implicit refit. Repeated validation selection
    does not yield an independent test-set estimate or a theorem certificate.
    ``results_`` (also ``candidate_results_``) contains compact audits, not
    copies of all candidate trajectories. ``best_estimator_`` retains the winner.
    """

    def __init__(self, candidates, *, validation_fraction=1 / 3, random_state=0,
                 raise_on_failure=False):
        self.candidates = candidates
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.raise_on_failure = raise_on_failure

    def _clear_fit(self):
        # Do not let a failed repeat fit expose the previous winning coefficient.
        for name in tuple(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        self.success_ = False
        self.status_ = "not_fitted"
        self.best_estimator_ = self.estimator_ = self.model_ = None
        self.best_index_ = self.best_score_ = self.best_params_ = None
        self.selected_iteration_ = None
        self.results_ = []
        self.candidate_results_ = self.selection_history_ = self.results_

    def _split(self, X, Y, validation_data):
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
                raise ValueError("validation_data must be (X_validation, Y_validation)")
            Xv = _matrix(validation_data[0], "X_validation")
            Yv = _matrix(validation_data[1], "Y_validation")
            if (len(Xv) != len(Yv) or Xv.shape[1] != X.shape[1]
                    or Yv.shape[1] != Y.shape[1]):
                raise ValueError("validation data must match rows and predictor/response dimensions")
            self.train_indices_ = self.validation_indices_ = None
            self.split_mode_ = "explicit_validation"
            return X, Y, Xv, Yv
        fraction = self.validation_fraction
        if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real)
                or not np.isfinite(fraction) or not 0 < fraction < 1):
            raise ValueError("validation_fraction must lie strictly between zero and one")
        if (isinstance(self.random_state, (bool, np.bool_))
                or not isinstance(self.random_state, Integral) or self.random_state < 0):
            raise ValueError("random_state must be a nonnegative integer")
        count = max(1, int(np.ceil(len(X) * fraction)))
        if count >= len(X):
            raise ValueError("holdout splitting requires at least one training and one validation row")
        order = np.random.default_rng(int(self.random_state)).permutation(len(X))
        self.validation_indices_ = np.sort(order[:count])
        self.train_indices_ = np.sort(order[count:])
        self.split_mode_ = "deterministic_holdout"
        return (X[self.train_indices_], Y[self.train_indices_],
                X[self.validation_indices_], Y[self.validation_indices_])

    def fit(self, X, Y, *, source, validation_data=None):
        """Fit candidate copies using training rows, then select by validation."""
        self._clear_fit()
        if not isinstance(self.raise_on_failure, bool):
            raise ValueError("raise_on_failure must be a boolean")
        try:
            candidates = tuple(self.candidates)
        except TypeError as error:
            raise ValueError("candidates must be a nonempty sequence of estimators") from error
        if not candidates:
            raise ValueError("candidates must be a nonempty sequence of estimators")
        ranks = []
        for candidate in candidates:
            if not callable(getattr(candidate, "fit", None)) or not callable(getattr(candidate, "predict", None)):
                raise ValueError("each candidate must provide fit and predict methods")
            value = getattr(candidate, "source_rank", None)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                    or value < 1):
                raise ValueError("each candidate must have a positive integer source_rank")
            ranks.append(int(value))
        if len(set(ranks)) != 1:
            raise ValueError("all candidates must use the same source_rank")

        X, Y = _matrix(X, "X"), _matrix(Y, "Y")
        if len(X) != len(Y):
            raise ValueError("X and Y must have matching rows")
        Xtr, Ytr, Xv, Yv = self._split(X, Y, validation_data)
        prepared = prepare_source(source, p=X.shape[1], q=Y.shape[1], source_rank=ranks[0])
        # A fit-scoped cache may reuse training initialization, never validation
        # selection. Read-only shared arrays keep identities stable and prevent
        # a candidate from modifying the next candidate's training observations.
        for array in (Xtr, Ytr, Xv, Yv):
            array.setflags(write=False)
        initialization_cache = {}
        self.n_candidates_ = len(candidates)
        self.metadata_ = {
            "theorem_certified": False,
            "refit_on_all_data": False,
            "validation_score_is_independent_test_estimate": False,
            "selection_rule": SELECTION_RULE,
            "source_preparations": 1,
            "training_rows": len(Xtr), "validation_rows": len(Xv),
            "retains_all_candidate_trajectories": False,
            "initialization_reuse": "same_training_data_source_rank_target_rank_and_init_penalty",
        }
        self.diagnostics_ = self.metadata_
        incumbent_prediction = None

        for index, template in enumerate(candidates):
            started = time.perf_counter()
            model, error, score, difference = None, None, None, None
            params = _settings(template)
            try:
                model = deepcopy(template)
                model.fit(Xtr, Ytr, source=prepared, validation_data=(Xv, Yv),
                          _initialization_cache=initialization_cache)
                if bool(getattr(model, "success_", False)):
                    prediction = np.asarray(model.predict(Xv.copy()))
                    if (np.iscomplexobj(prediction) or prediction.shape != Yv.shape
                            or not np.isfinite(prediction).all()):
                        raise ValueError("candidate validation predictions must be finite and match responses")
                    with np.errstate(over="raise", invalid="raise"):
                        score = float(np.mean(np.square(prediction - Yv)))
                    if not np.isfinite(score):
                        raise ValueError("candidate validation MSE must be finite")
                    if incumbent_prediction is not None:
                        difference = validation_loss_difference(prediction, incumbent_prediction, Yv)
            except Exception as exception:
                # Keep a malformed or failed candidate from aborting the search.
                # KeyboardInterrupt and SystemExit intentionally propagate.
                error = exception
            eligible = error is None and bool(getattr(model, "success_", False)) and score is not None
            status = (getattr(error, "status", type(error).__name__) if error is not None
                      else getattr(model, "status_", "fit_failed"))
            record = {
                "candidate_index": index, "candidate_id": index, "params": params,
                "success": eligible, "eligible": eligible, "status": status,
                "message": str(error) if error is not None else str(getattr(model, "message_", status)),
                "validation_mse": score if eligible else None,
                "incumbent_candidate_index": self.best_index_,
                "loss_difference": difference if eligible else None,
                "selected_iteration": getattr(model, "selected_iteration_", None),
                "n_iter": getattr(model, "n_iter_", None),
                "validation_history": deepcopy(getattr(model, "validation_history_", [])),
                "elapsed_time_sec": time.perf_counter() - started,
                "theorem_certified": False,
            }
            self.results_.append(record)
            if eligible and (incumbent_prediction is None or difference < 0):
                self.best_estimator_ = model
                self.best_index_, self.best_score_, self.best_params_ = index, score, deepcopy(params)
                incumbent_prediction = np.array(prediction, copy=True)

        if self.best_estimator_ is None:
            self.status_ = "no_successful_candidate"
            self.message_ = "No successful candidate with finite validation predictions"
            if self.raise_on_failure:
                raise FitFailure(self.status_, self.message_)
            return self
        self.success_, self.status_ = True, "selected"
        self.message_ = f"Selected candidate {self.best_index_} by validation MSE"
        self.estimator_ = self.model_ = self.best_estimator_
        self.selected_iteration_ = self.best_estimator_.selected_iteration_
        self.coefficient_ = self.best_estimator_.coefficient_.copy()
        return self

    def predict(self, X):
        """Predict with the successful selected estimator."""
        if not getattr(self, "success_", False) or self.best_estimator_ is None:
            raise FitFailure("no_successful_candidate", "No successful tuned estimator is available")
        return self.best_estimator_.predict(X)


# Familiar name for callers migrating from the original package. The v2
# constructor deliberately accepts candidates, rather than a prescribed grid.
SparseSMARTTuner = SparseSMARTv2Tuner
