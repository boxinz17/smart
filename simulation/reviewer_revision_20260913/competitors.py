"""Auditable matrix-regression competitors for the 2026-09-13 revision.

All coefficients use the convention ``Y = X @ C + noise``. Data must already
be centered using training-only quantities; no intercept or standardization is
estimated here. Rank is an upper bound, not a forced number of components.

Only the explicitly named oracle methods accept clean source frames. Every
other source-aware method receives the same fitted source coefficient matrix.
``nuclear_contrast`` is a generic convex contrast baseline, not an implementation
of any particular published transfer-learning algorithm.

Losses are divided by ``2*n``. Ridge means ``ridge * ||C-center||_F**2 / 2``;
nuclear penalty means ``nuclear_penalty * ||C-C0||_*``. RRR truncates in the
design metric. Ordinary coefficient-SVD truncation is generally incorrect.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np


IMPLEMENTATION_VERSION = "reviewer-competitors-v1"
METHODS = (
    "target_rrr", "target_ridge_rrr", "source_subspace_rrr",
    "source_subspace_ridge_rrr", "ridge_to_source", "source_target_mixture",
    "nuclear_contrast", "oracle_subspace_rrr", "oracle_subspace_ridge_rrr",
)


@dataclass(frozen=True)
class CompetitorFit:
    coefficient: np.ndarray
    method: str
    parameters: dict[str, Any]
    diagnostics: dict[str, Any]

    @property
    def coef_(self):
        """Matrix-regression convention, shape (predictors, responses)."""
        return self.coefficient

    def predict(self, design):
        x = _matrix(design, "design")
        if x.shape[1] != self.coefficient.shape[0]:
            raise ValueError("prediction design has the wrong number of columns")
        return x @ self.coefficient


@dataclass(frozen=True)
class SelectionResult:
    fit: CompetitorFit
    selected_index: int
    validation_loss: float
    candidate_results: tuple[dict[str, Any], ...]
    elapsed_seconds: float
    refit: bool

    @property
    def coefficient(self):
        return self.fit.coefficient

    def predict(self, design):
        return self.fit.predict(design)


def _matrix(value, name, *, allow_empty_columns=False):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=float)
    if (array.ndim != 2 or array.shape[0] == 0
            or (array.shape[1] == 0 and not allow_empty_columns)
            or not np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be a finite nonempty real matrix")
    return array


def _integer(value, name, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return int(value)


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _freeze(value):
    value = np.array(value, dtype=float, copy=True)
    value.flags.writeable = False
    return value


def _design_svd(x):
    u, values, vt = np.linalg.svd(x, full_matrices=False)
    tolerance = float(np.finfo(float).eps * max(x.shape) * values[0])
    count = int(np.count_nonzero(values > tolerance))
    return u[:, :count], values[:count], vt[:count].T, tolerance


def _canonical_frame(frame):
    """Choose a reproducible basis for a tied right-singular subspace."""
    basis = np.empty_like(frame)
    count = 0
    tolerance = 32 * np.finfo(float).eps * max(frame.shape)
    for row in range(frame.shape[0]):
        candidate = frame @ frame[row]
        for _ in range(2):
            candidate -= basis[:, :count] @ (basis[:, :count].T @ candidate)
        length = np.linalg.norm(candidate)
        if length > tolerance:
            basis[:, count] = candidate / length
            count += 1
        if count == frame.shape[1]:
            return basis
    raise np.linalg.LinAlgError("could not canonicalize a tied response subspace")


def _truncate(matrix, rank):
    """Truncate fitted-response coordinates, canonicalizing a tied cutoff."""
    if min(matrix.shape) == 0:
        return np.zeros_like(matrix), np.empty(0), False
    _, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    count = min(rank, len(singular))
    if count == 0:
        return np.zeros_like(matrix), singular, False
    right = right_t[:count].T
    tied = False
    tolerance = 8 * np.finfo(float).eps * max(matrix.shape) * singular[0]
    if count < len(singular) and abs(singular[count - 1] - singular[count]) <= tolerance:
        start, stop = count - 1, count + 1
        while start > 0 and singular[start - 1] - singular[count] <= tolerance:
            start -= 1
        while stop < len(singular) and singular[count - 1] - singular[stop] <= tolerance:
            stop += 1
        canonical = _canonical_frame(right_t[start:stop].T)
        right = np.column_stack((right_t[:start].T, canonical[:, :count - start]))
        tied = True
    return (matrix @ right) @ right.T, singular, tied


def _rrr(x, y, rank, ridge):
    """Global optimum of loss + ridge*||C||²/2 subject to rank(C)<=r."""
    n, p = x.shape
    u, singular, v, tolerance = _design_svd(x)
    scale = np.sqrt(singular**2 + n * ridge)
    coordinates = (singular / scale)[:, None] * (u.T @ y)
    truncated, response_singular, tied = _truncate(coordinates, rank)
    coefficient = (v / scale) @ truncated
    fitted = x @ coefficient
    objective = float(np.sum((y - fitted)**2) / (2 * n) + ridge * np.sum(coefficient**2) / 2)
    optimum = float((np.sum(y**2) - np.sum(response_singular[:rank]**2)) / (2 * n))
    # Tiny negative values can arise in the subtraction for exact interpolation.
    objective_tolerance = float(512 * np.finfo(float).eps * max(*x.shape, y.shape[1])
                                * max(np.sum(y**2) / (2 * n), np.finfo(float).tiny))
    certificate = {
        "algorithm": "design_svd_whitened_response_truncation",
        "design_rank": int(len(singular)), "design_rank_tolerance": tolerance,
        "objective": objective, "optimal_objective": max(0., optimum),
        "objective_gap": float(objective - max(0., optimum)),
        "objective_tolerance": objective_tolerance,
        "certified": bool(abs(objective - optimum) <= objective_tolerance),
        "converged": True, "rank_cutoff_tied": tied,
        "coefficient_convention": "minimum_norm_lift_in_numerical_design_range",
        "nullspace_residual_norm": float(np.linalg.norm(coefficient - v @ (v.T @ coefficient))),
    }
    return coefficient, certificate


def _source_frames(source, source_rank):
    left, singular, right_t = np.linalg.svd(source, full_matrices=False)
    tolerance = float(np.finfo(float).eps * max(source.shape) * singular[0])
    numerical_rank = int(np.count_nonzero(singular > tolerance))
    if source_rank is None:
        ranks = (numerical_rank, numerical_rank)
    elif isinstance(source_rank, (int, np.integer)) and not isinstance(source_rank, (bool, np.bool_)):
        ranks = (_integer(source_rank, "source_rank", len(singular)),) * 2
    else:
        try:
            if len(source_rank) != 2:
                raise ValueError("source_rank must be an integer or a pair")
            ranks = tuple(_integer(r, "source_rank", len(singular)) for r in source_rank)
        except TypeError as error:
            raise ValueError("source_rank must be an integer or a pair") from error
    if max(ranks) > numerical_rank:
        raise ValueError("source_rank exceeds the numerical rank of the supplied source coefficient")
    return left[:, :ranks[0]], right_t[:ranks[1]].T, {
        "source_dimensions": list(ranks), "source_numerical_rank": numerical_rank,
        "source_rank_tolerance": tolerance,
        "source_frame_convention": "ordered_svd_of_supplied_fitted_coefficient",
    }


def _oracle_frame(value, size, name):
    frame = _matrix(value, name, allow_empty_columns=True)
    if frame.shape[0] != size or frame.shape[1] > size:
        raise ValueError(f"{name} has incompatible dimensions")
    if not np.allclose(frame.T @ frame, np.eye(frame.shape[1]), rtol=1e-10, atol=1e-10):
        raise ValueError(f"{name} must have orthonormal columns")
    return frame


def _restricted_rrr(x, y, left, right, rank, ridge):
    if left.shape[1] == 0 or right.shape[1] == 0:
        return np.zeros((x.shape[1], y.shape[1])), {
            "algorithm": "empty_source_subspace", "design_rank": 0,
            "objective": float(np.sum(y**2) / (2 * len(x))),
            "objective_gap": 0., "certified": True, "converged": True,
        }
    reduced, diagnostics = _rrr(x @ left, y @ right, rank, ridge)
    coefficient = left @ reduced @ right.T
    outside_loss = float(np.sum((y - (y @ right) @ right.T)**2) / (2 * len(x)))
    diagnostics["orthogonal_response_loss"] = outside_loss
    diagnostics["objective"] += outside_loss
    diagnostics["optimal_objective"] += outside_loss
    diagnostics["source_dimensions"] = [left.shape[1], right.shape[1]]
    return coefficient, diagnostics


def _ridge_center(x, y, center, ridge):
    u, singular, v, tolerance = _design_svd(x)
    residual = y - x @ center
    correction = (v * (singular / (singular**2 + len(x) * ridge))) @ (u.T @ residual)
    coefficient = center + correction
    objective = float(np.sum((y - x @ coefficient)**2) / (2 * len(x))
                      + ridge * np.sum(correction**2) / 2)
    return coefficient, {
        "algorithm": "svd_ridge_correction_to_source", "objective": objective,
        "design_rank": len(singular), "design_rank_tolerance": tolerance,
        "coefficient_convention": "minimum_norm_correction_preserving_source_nullspace",
        "zero_ridge_convention": "continuous_limit_as_ridge_decreases_to_zero",
        "converged": True, "certified": True,
    }


def _soft_threshold(matrix, threshold):
    left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    retained = np.maximum(singular - threshold, 0.)
    return (left * retained) @ right_t, float(np.sum(retained))


def _nuclear_contrast(x, y, source, penalty, max_iter, tolerance):
    """Monotone restarted FISTA, with a feasible convex dual certificate."""
    if penalty == 0:
        coefficient, diagnostics = _ridge_center(x, y, source, 0.)
        diagnostics.update(algorithm="minimum_norm_unpenalized_source_correction", iterations=0,
                           duality_gap=0., dual_objective=diagnostics["objective"])
        return coefficient, diagnostics
    z = y - x @ source
    n = len(x)
    spectral = float(np.linalg.norm(x, ord=2))
    lipschitz = spectral**2 / n
    if lipschitz == 0:
        return source.copy(), {"algorithm": "zero_design_nuclear_contrast", "iterations": 0,
                               "objective": float(np.sum(z**2) / (2 * n)),
                               "duality_gap": 0., "converged": True, "certified": True}
    current = np.zeros_like(source)
    extrapolated = current.copy()
    momentum = 1.
    objective = float(np.sum(z**2) / (2 * n))
    gap = np.inf
    dual_objective = -np.inf
    converged = False
    for iteration in range(1, max_iter + 1):
        gradient = x.T @ (x @ extrapolated - z) / n
        updated, nuclear_norm = _soft_threshold(extrapolated - gradient / lipschitz, penalty / lipschitz)
        residual = z - x @ updated
        updated_objective = float(np.sum(residual**2) / (2 * n) + penalty * nuclear_norm)
        if updated_objective > objective:
            # Restart an acceleration step that increases the primal objective.
            extrapolated = current
            momentum = 1.
            gradient = x.T @ (x @ extrapolated - z) / n
            updated, nuclear_norm = _soft_threshold(extrapolated - gradient / lipschitz, penalty / lipschitz)
            residual = z - x @ updated
            updated_objective = float(np.sum(residual**2) / (2 * n) + penalty * nuclear_norm)
        dual = residual / n
        dual_norm = float(np.linalg.norm(x.T @ dual, ord=2))
        dual *= min(1., penalty / dual_norm) if dual_norm > 0 else 1.
        dual_objective = float(np.sum(z * dual) - n * np.sum(dual**2) / 2)
        gap = max(0., updated_objective - dual_objective)
        next_momentum = (1. + np.sqrt(1. + 4. * momentum**2)) / 2.
        extrapolated = updated + ((momentum - 1.) / next_momentum) * (updated - current)
        current, momentum, objective = updated, next_momentum, updated_objective
        if gap <= tolerance * max(1., abs(objective)):
            converged = True
            break
    return source + current, {
        "algorithm": "monotone_restarted_fista_nuclear_contrast",
        "objective": objective, "dual_objective": dual_objective, "duality_gap": gap,
        "duality_gap_threshold": tolerance * max(1., abs(objective)),
        "iterations": iteration, "max_iter": max_iter, "converged": converged,
        "certified": converged, "lipschitz_constant": lipschitz,
        "termination_reason": "duality_gap" if converged else "iteration_cap",
        "publication_status": "generic_convex_baseline_not_a_published_method_replication",
    }


def fit_competitor(design, response, method, *, rank=None, ridge=0.,
                   source_coefficient=None, source_rank=None, alpha=.5,
                   nuclear_penalty=1., oracle_left=None, oracle_right=None,
                   observed_left=None, observed_right=None,
                   max_iter=2000, tolerance=1e-7):
    """Fit one fully specified candidate; never inspect validation or truth.

    Rank constrains RRR candidates and the target component of a mixture only.
    ``ridge_to_source`` and ``nuclear_contrast`` are unrestricted-rank estimates;
    supplying rank for those methods is recorded as unused. At zero ridge,
    source-centered fits preserve source coefficients in target-design null
    directions. They therefore need not equal target-only OLS when p > n.
    Source-subspace methods may receive a common observed-source decomposition
    through observed_left/right, including its deterministic null completion.
    These frames must come from the supplied fitted source, never from truth.
    """
    started = perf_counter()
    x, y = _matrix(design, "design"), _matrix(response, "response")
    if x.shape[0] != y.shape[0]:
        raise ValueError("design and response must have the same row count")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if rank is None:
        rank = min(x.shape[1], y.shape[1])
    rank = _integer(rank, "rank", min(x.shape[1], y.shape[1]))
    ridge = _nonnegative(ridge, "ridge")
    if method == "nuclear_contrast" and ridge != 0:
        raise ValueError("nuclear_contrast does not use a ridge penalty")
    if method in ("target_rrr", "source_subspace_rrr", "oracle_subspace_rrr") and ridge != 0:
        raise ValueError("use the explicitly named ridge method when ridge is nonzero")
    oracle = method.startswith("oracle_")
    if not oracle and (oracle_left is not None or oracle_right is not None):
        raise ValueError("clean source frames can only be supplied to an oracle method")
    if (observed_left is not None or observed_right is not None) and method not in (
            "source_subspace_rrr", "source_subspace_ridge_rrr"):
        raise ValueError("observed source frames require a source-subspace method")
    source = None
    if method not in ("target_rrr", "target_ridge_rrr") and not oracle:
        source = _matrix(source_coefficient, "source_coefficient")
        if source.shape != (x.shape[1], y.shape[1]):
            raise ValueError("source_coefficient has the wrong shape")
    parameters: dict[str, Any] = {"rank": rank, "ridge": ridge}
    if method in ("target_rrr", "target_ridge_rrr"):
        coefficient, diagnostics = _rrr(x, y, rank, ridge)
    elif method in ("source_subspace_rrr", "source_subspace_ridge_rrr") or oracle:
        if oracle:
            left = _oracle_frame(oracle_left, x.shape[1], "oracle_left")
            right = _oracle_frame(oracle_right, y.shape[1], "oracle_right")
            frame_diagnostics = {"source_dimensions": [left.shape[1], right.shape[1]],
                                 "source_frame_convention": "provided_clean_source_subspaces"}
        elif observed_left is not None or observed_right is not None:
            left = _oracle_frame(observed_left, x.shape[1], "observed_left")
            right = _oracle_frame(observed_right, y.shape[1], "observed_right")
            if source_rank is None:
                dimensions = (left.shape[1], right.shape[1])
            elif isinstance(source_rank, (int, np.integer)) and not isinstance(source_rank, (bool, np.bool_)):
                dimensions = (_integer(source_rank, "source_rank"),) * 2
            else:
                try:
                    if len(source_rank) != 2:
                        raise ValueError("source_rank must be an integer or a pair")
                    dimensions = tuple(_integer(r, "source_rank") for r in source_rank)
                except TypeError as error:
                    raise ValueError("source_rank must be an integer or a pair") from error
            if dimensions[0] > left.shape[1] or dimensions[1] > right.shape[1]:
                raise ValueError("source_rank exceeds supplied observed frame dimensions")
            left, right = left[:, :dimensions[0]], right[:, :dimensions[1]]
            frame_diagnostics = {
                "source_dimensions": list(dimensions),
                "source_numerical_rank": int(np.linalg.matrix_rank(source)),
                "source_frame_convention": "supplied_common_observed_source_decomposition",
                "includes_observed_null_completion": bool(max(dimensions) > np.linalg.matrix_rank(source)),
            }
        else:
            left, right, frame_diagnostics = _source_frames(source, source_rank)
        coefficient, diagnostics = _restricted_rrr(x, y, left, right, rank, ridge)
        diagnostics.update(frame_diagnostics)
        parameters["source_rank"] = frame_diagnostics["source_dimensions"]
    elif method == "ridge_to_source":
        coefficient, diagnostics = _ridge_center(x, y, source, ridge)
        diagnostics["rank_parameter_used"] = False
    elif method == "source_target_mixture":
        alpha = _nonnegative(alpha, "alpha")
        if alpha > 1:
            raise ValueError("alpha must lie in [0, 1]")
        target, diagnostics = _rrr(x, y, rank, ridge)
        # Objective certificates belong to the fitted target component only.
        diagnostics = {"target_component_certificate": diagnostics,
                       "algorithm": "convex_source_target_coefficient_mixture",
                       "converged": True, "certified": diagnostics["certified"],
                       "rank_constraint_scope": "target_component_only"}
        coefficient = (1. - alpha) * target + alpha * source
        parameters["alpha"] = alpha
    else:
        penalty = _nonnegative(nuclear_penalty, "nuclear_penalty")
        iterations = _integer(max_iter, "max_iter")
        tolerance = _nonnegative(tolerance, "tolerance")
        if iterations == 0 or tolerance == 0:
            raise ValueError("max_iter and tolerance must be positive")
        coefficient, diagnostics = _nuclear_contrast(x, y, source, penalty, iterations, tolerance)
        parameters.update(nuclear_penalty=penalty, max_iter=iterations, tolerance=tolerance)
        diagnostics["rank_parameter_used"] = False
    if not np.all(np.isfinite(coefficient)):
        raise np.linalg.LinAlgError("competitor produced a nonfinite coefficient")
    diagnostics.update(
        implementation_version=IMPLEMENTATION_VERSION,
        information_regime="clean_source_subspace_oracle" if oracle else (
            "target_only" if source is None else "same_fitted_source_coefficient"),
        oracle=oracle, effective_rank=int(np.linalg.matrix_rank(coefficient)),
        requested_rank=rank, n_train=len(x),
        training_loss=float(np.sum((y - x @ coefficient)**2) / (2 * len(x))),
        elapsed_seconds=perf_counter() - started,
    )
    return CompetitorFit(_freeze(coefficient), method, parameters, diagnostics)


def select_competitor(design_train, response_train, design_validation, response_validation,
                      candidates: Iterable[Mapping[str, Any]], *, source_coefficient=None,
                      oracle_left=None, oracle_right=None, observed_left=None,
                      observed_right=None, refit=False,
                      require_certified=True):
    """Select by validation residual/(2*n_validation); first candidate wins ties.

    Every attempted fit has a status, fit time and validation loss in the audit.
    Failed or uncertified fits are excluded, never silently replaced. Validation
    observations enter estimation only when the caller explicitly sets refit;
    the pre-refit validation loss remains the selection score in that case.
    """
    started = perf_counter()
    x, y = _matrix(design_train, "design_train"), _matrix(response_train, "response_train")
    xv, yv = _matrix(design_validation, "design_validation"), _matrix(response_validation, "response_validation")
    if x.shape[0] != y.shape[0] or xv.shape[0] != yv.shape[0] or x.shape[1] != xv.shape[1] or y.shape[1] != yv.shape[1]:
        raise ValueError("training and validation shapes are incompatible")
    candidate_list = [dict(candidate) for candidate in candidates]
    if not candidate_list:
        raise ValueError("candidate library must be nonempty")
    forbidden = {"source_coefficient", "oracle_left", "oracle_right", "observed_left", "observed_right", "design", "response"}
    if any(forbidden.intersection(candidate) for candidate in candidate_list):
        raise ValueError("data and source objects must not be overridden inside a candidate")
    audit, fitted = [], []
    for index, candidate in enumerate(candidate_list):
        fit_started = perf_counter()
        record = {"candidate_index": index, "candidate": candidate.copy()}
        try:
            candidate_options = dict(candidate)
            candidate_options["source_coefficient"] = source_coefficient
            if str(candidate_options.get("method", "")).startswith("oracle_"):
                candidate_options.update(oracle_left=oracle_left, oracle_right=oracle_right)
            elif str(candidate_options.get("method", "")).startswith("source_subspace"):
                candidate_options.update(observed_left=observed_left, observed_right=observed_right)
            result = fit_competitor(x, y, **candidate_options)
            loss = float(np.sum((yv - result.predict(xv))**2) / (2 * len(xv)))
            if not np.isfinite(loss):
                raise np.linalg.LinAlgError("nonfinite validation loss")
            record.update(status="ok", validation_loss=loss, diagnostics=result.diagnostics)
            if require_certified and not result.diagnostics.get("certified", False):
                record.update(status="uncertified", exclusion_reason="numerical_certificate_not_met")
                fitted.append(None)
            else:
                fitted.append(result)
        except (ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError) as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
            fitted.append(None)
        record["elapsed_seconds"] = perf_counter() - fit_started
        audit.append(record)
    valid = [i for i, result in enumerate(fitted) if result is not None]
    if not valid:
        error = ValueError("no eligible competitor candidate; inspect candidate_results on this exception")
        error.candidate_results = tuple(audit)
        raise error
    selected = min(valid, key=lambda i: (audit[i]["validation_loss"], i))
    result = fitted[selected]
    if refit:
        options = dict(candidate_list[selected], source_coefficient=source_coefficient)
        if options["method"].startswith("oracle_"):
            options.update(oracle_left=oracle_left, oracle_right=oracle_right)
        elif options["method"].startswith("source_subspace"):
            options.update(observed_left=observed_left, observed_right=observed_right)
        result = fit_competitor(np.vstack((x, xv)), np.vstack((y, yv)), **options)
        if require_certified and not result.diagnostics.get("certified", False):
            raise ValueError("selected candidate failed its numerical certificate after refitting")
    return SelectionResult(result, selected, float(audit[selected]["validation_loss"]),
                           tuple(audit), perf_counter() - started, bool(refit))


def candidate_grid(method, *, ranks=(1,), ridges=(0., .01, .1, 1., 10.),
                   source_ranks=(None,), alphas=(0., .25, .5, .75, 1.),
                   nuclear_penalties=(.01, .1, 1., 10.)):
    """Build a deterministic grid; the caller chooses dimension-appropriate ranks.

    Numerical values are raw penalties under this module's normalized objective.
    A runner may scale them with training data, but must save the resulting grid.
    No truth or validation data is used to construct this grid.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method == "nuclear_contrast":
        return [{"method": method, "nuclear_penalty": value} for value in nuclear_penalties]
    if method == "ridge_to_source":
        return [{"method": method, "ridge": value} for value in ridges]
    ridge_values = (0.,) if method.endswith("_rrr") and "ridge" not in method else ridges
    settings = [{"method": method, "rank": rank, "ridge": ridge}
                for rank, ridge in product(ranks, ridge_values)]
    if method.startswith("source_subspace"):
        settings = [dict(setting, source_rank=source_rank) for setting, source_rank in product(settings, source_ranks)]
    if method == "source_target_mixture":
        settings = [dict(setting, alpha=alpha) for setting, alpha in product(settings, alphas)]
    return settings


__all__ = ["CompetitorFit", "SelectionResult", "IMPLEMENTATION_VERSION", "METHODS",
           "fit_competitor", "select_competitor", "candidate_grid"]
