"""Training-only candidates with an explicit, independently held-out selector.

No function here accepts simulation truth.  The caller supplies the same observed
source matrix and full observed frames used by the paper campaign.  All selected
coefficients are training fits: there is no validation refit or silent fallback.

RRR ranks are upper bounds; smaller source prefixes are valid candidates.  The
mixture constrains only its target component.  Ridge-to-source and nuclear
contrast are unrestricted-rank controls.  The raw initializer is coefficient-SVD
truncation of reduced Lasso, not an optimum of rank-constrained Lasso or RRR.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from itertools import product
from time import perf_counter

import numpy as np

from reviewer_revision_20260913.competitors import fit_competitor
from sparse_smart_v2 import reduced_lasso


METHODS = (
    "target_rrr", "target_ridge_rrr", "source_subspace_rrr",
    "source_subspace_ridge_rrr", "ridge_to_source", "source_target_mixture",
    "nuclear_contrast", "initializer_only",
)
_TARGET = {"target_rrr", "target_ridge_rrr"}
_SUBSPACE = {"source_subspace_rrr", "source_subspace_ridge_rrr"}
_RANK_CONSTRAINED = _TARGET | _SUBSPACE | {"initializer_only"}
_DATA_KEYS = {"X", "Y", "C0", "X_validation", "Y_validation"}


def default_configuration():
    """Return a fresh JSON-serializable declaration of the finite libraries.

    Ridge multipliers scale by ||X||_op^2/n.  Nuclear fractions scale by
    ||X.T @ (Y-X@C0)/n||_op.  Initializer penalties are absolute, matching the
    public paper initializer's loss normalization.  Zero scales stay zero;
    source-centered ridge's positive-only rule makes a zero-scale grid fail
    explicitly instead of introducing an undeclared positive penalty.
    """
    return dict(
        schema_version=1,
        ridge_multipliers=[0., 1e-4, 1e-3, .01, .1, 1., 10., 100.],
        source_dimensions=[1, 3, 5, 7, 10, 11, 15, 20],
        include_full_source_dimension=True,
        mixture_alphas=[0., .1, .25, .5, .75, .9, 1.],
        nuclear_penalty_fractions=[0., .0001, .001, .01, .1, .3, 1.],
        nuclear_max_iter=10000, nuclear_tolerance=1e-7,
        initializer_penalties=[.003, .03, .1, .3, 1., 3.],
        initializer_min_dimension=10, initializer_max_iter=20000,
        initializer_tolerance=1e-9, initializer_tie_tolerance=1e-12,
        selection_rtol=1e-10, selection_atol=1e-12,
    )


def _configuration(configuration):
    result = default_configuration()
    if configuration is not None:
        if not isinstance(configuration, Mapping):
            raise ValueError("configuration must be a mapping")
        unknown = set(configuration) - set(result)
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        result.update(deepcopy(dict(configuration)))
    if result["schema_version"] != 1:
        raise ValueError("unsupported fitting configuration schema_version")
    for name in ("ridge_multipliers", "mixture_alphas", "nuclear_penalty_fractions",
                 "initializer_penalties"):
        values = result[name]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{name} must be a nonempty finite grid")
        if any(isinstance(v, (bool, np.bool_)) or not np.isfinite(v) or v < 0
               for v in values):
            raise ValueError(f"{name} must contain finite nonnegative numbers")
        if name == "mixture_alphas" and max(values) > 1:
            raise ValueError("mixture_alphas must be between zero and one")
        if name == "initializer_penalties" and min(values) <= 0:
            raise ValueError("initializer_penalties must be positive")
        result[name] = [float(v) for v in values]
    if not isinstance(result["source_dimensions"], (list, tuple)):
        raise ValueError("source_dimensions must be a sequence")
    for value in result["source_dimensions"]:
        _integer(value, "source_dimensions")
    for name in ("nuclear_max_iter", "initializer_max_iter", "initializer_min_dimension"):
        result[name] = _integer(result[name], name)
    for name in ("nuclear_tolerance", "initializer_tolerance", "initializer_tie_tolerance",
                 "selection_rtol", "selection_atol"):
        if (isinstance(result[name], (bool, np.bool_)) or not np.isfinite(result[name])
                or result[name] <= 0):
            raise ValueError(f"{name} must be finite and positive")
        result[name] = float(result[name])
    if not isinstance(result["include_full_source_dimension"], bool):
        raise ValueError("include_full_source_dimension must be Boolean")
    return result


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _matrix(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return value


def _observations(data, method):
    if not isinstance(data, Mapping):
        raise ValueError("data must be a mapping")
    extra = set(data) - _DATA_KEYS
    if extra:
        raise ValueError(f"fit inputs must not contain truth or extra fields: {sorted(extra)}")
    required = _DATA_KEYS - ({"C0"} if method in _TARGET else set())
    missing = required - set(data)
    if missing:
        raise ValueError(f"missing fit inputs: {sorted(missing)}")
    x, y = _matrix(data["X"], "X"), _matrix(data["Y"], "Y")
    xv = _matrix(data["X_validation"], "X_validation")
    yv = _matrix(data["Y_validation"], "Y_validation")
    if (x.shape[0] != y.shape[0] or xv.shape[0] != yv.shape[0]
            or xv.shape[1] != x.shape[1] or yv.shape[1] != y.shape[1]):
        raise ValueError("training/validation dimensions do not match")
    # Target-only fits deliberately do not inspect even a malformed supplied C0.
    source = None if method in _TARGET else _matrix(data["C0"], "C0")
    if source is not None and source.shape != (x.shape[1], y.shape[1]):
        raise ValueError("C0 has the wrong shape")
    return x, y, xv, yv, source


def _frames(source_arrays, p, q):
    if not isinstance(source_arrays, Mapping) or not {"left", "right"} <= set(source_arrays):
        raise ValueError("source_arrays must contain full observed left/right frames")
    if any(str(key).lower() in {"c_star", "c_true", "truth", "oracle_left", "oracle_right"}
           for key in source_arrays):
        raise ValueError("source_arrays must not contain truth or oracle frames")
    frames = []
    for key, size in (("left", p), ("right", q)):
        frame = _matrix(source_arrays[key], key)
        if frame.shape != (size, size):
            raise ValueError(f"{key} must be the full observed {size}-by-{size} frame")
        if not np.allclose(frame.T @ frame, np.eye(size), rtol=1e-10, atol=1e-10):
            raise ValueError(f"observed {key} frame must be orthonormal")
        frames.append(frame)
    return tuple(frames)


def _json(value):
    """Keep diagnostics JSON-safe, including nonfinite failure certificates."""
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _validation_mse(x, y, coefficient):
    with np.errstate(over="raise", invalid="raise"):
        result = float(np.mean((y - x @ coefficient) ** 2))
    if not np.isfinite(result):
        raise ValueError("nonfinite validation MSE")
    return result


def _specifications(method, rank, p, q, config, ridge_scale, nuclear_scale):
    base = {"rank": rank}
    if method == "target_rrr":
        return [dict(base)]
    ridges = ([dict(base, ridge_multiplier=v, ridge=v * ridge_scale)
               for v in config["ridge_multipliers"]
               if method != "ridge_to_source" or v > 0]
              if ridge_scale is not None else [])
    if method in {"target_ridge_rrr", "ridge_to_source"}:
        return ridges
    if method == "source_target_mixture":
        return [dict(ridge, alpha=alpha)
                for ridge, alpha in product(ridges, config["mixture_alphas"])]
    if method in _SUBSPACE:
        dimensions = [[int(d), int(d)] for d in config["source_dimensions"] if d <= min(p, q)]
        if config["include_full_source_dimension"] and [p, q] not in dimensions:
            dimensions.append([p, q])
        # Repeated configured values retain their declared order and attempts.
        if method == "source_subspace_rrr":
            ridges = [dict(base)]
        return [dict(ridge, source_rank=dim, source_left_dimension=dim[0],
                     source_right_dimension=dim[1], full_source_endpoint=(dim == [p, q]))
                for dim, ridge in product(dimensions, ridges)]
    if method == "nuclear_contrast":
        return [dict(base, nuclear_penalty_fraction=v, nuclear_penalty=v * nuclear_scale,
                     max_iter=config["nuclear_max_iter"], tolerance=config["nuclear_tolerance"])
                for v in config["nuclear_penalty_fractions"]]
    dim = max(config["initializer_min_dimension"], rank)
    return [dict(base, penalty=v, source_rank=[min(dim, p), min(dim, q)],
                 max_iter=config["initializer_max_iter"], tolerance=config["initializer_tolerance"],
                 tie_tolerance=config["initializer_tie_tolerance"])
            for v in config["initializer_penalties"]]


def _initializer(x, y, frames, parameters):
    dl, dr = parameters["source_rank"]
    left, right = frames[0][:, :dl], frames[1][:, :dr]
    reduced_x, reduced_y = x @ left, y @ right
    initial = reduced_lasso(reduced_x, reduced_y, parameters["rank"], parameters["penalty"],
                            tol=parameters["tolerance"], max_iter=parameters["max_iter"],
                            tie_tol=parameters["tie_tolerance"])
    truncated = (initial.P * initial.d) @ initial.Q.T
    coefficient = left @ truncated @ right.T
    gradient_scale = max(1., parameters["penalty"],
                         float(np.max(np.abs(reduced_x.T @ reduced_y / len(x)))))
    return coefficient, dict(
        algorithm="public_sparse_smart_v2_reduced_lasso_then_coefficient_svd",
        certified=bool(initial.converged), converged=bool(initial.converged),
        certificate_scope="untruncated_reduced_lasso_only; rank_truncation_is_postprocessing",
        kkt_residual=initial.kkt_residual,
        kkt_tolerance=max(100 * np.finfo(float).eps * gradient_scale,
                          np.sqrt(parameters["tolerance"]) * gradient_scale),
        dual_gaps=initial.dual_gaps, n_iter=initial.n_iter,
        source_dimensions=[dl, dr], source_frame_convention="supplied_common_observed_source_decomposition",
        reduced_lasso_rank=int(np.linalg.matrix_rank(initial.coefficient)),
        retained_singular_values=initial.d, raw_lasso_objective=float(
            np.sum((reduced_y - reduced_x @ initial.coefficient) ** 2) / (2 * len(x))
            + parameters["penalty"] * np.sum(np.abs(initial.coefficient))),
        rank_truncated_training_loss=float(np.sum((y - x @ coefficient) ** 2) / (2 * len(x))),
        zero_initializer=bool(not np.any(coefficient)),
        chart_or_spectral_floor_restrictions_applied=False,
    )


def _fit_one(x, y, source, frames, method, parameters):
    if method == "initializer_only":
        return _initializer(x, y, frames, parameters)
    kwargs = {key: parameters[key] for key in (
        "rank", "ridge", "source_rank", "alpha", "nuclear_penalty", "max_iter", "tolerance")
        if key in parameters}
    if method == "ridge_to_source" and kwargs["ridge"] <= 0:
        raise ValueError("positive-only source ridge grid has zero training scale")
    if method not in _TARGET:
        kwargs["source_coefficient"] = source
    if method in _SUBSPACE:
        kwargs.update(observed_left=frames[0], observed_right=frames[1])
    fit = fit_competitor(x, y, method, **kwargs)
    return fit.coefficient, fit.diagnostics


def _boundaries(parameters, specifications):
    result = {}
    for key in ("ridge_multiplier", "alpha", "nuclear_penalty_fraction", "penalty",
                "source_left_dimension", "source_right_dimension"):
        if key in parameters:
            values = [s[key] for s in specifications]
            low, high, value = min(values), max(values), parameters[key]
            result[key] = dict(value=value, lower=low, upper=high,
                               at_lower=value == low, at_upper=value == high)
    return result


def fit_method(data, source_arrays, method, rank, configuration=None):
    """Return ``(selection_record, selected_coefficient_or_None)``.

    Invalid inputs raise before fitting. Numerical candidate failures are retained
    and excluded, while other candidates continue. ``status='success'`` requires
    a finite, certified candidate and recomputation of its validation MSE from
    the returned coefficient. Ties retain the first candidate in declared order.
    Target-only methods do not inspect C0 or source_arrays. Source metadata
    provenance is the caller's responsibility; no clean source frames are used.
    """
    started = perf_counter()
    if method not in METHODS:
        raise ValueError(f"unknown comparison method: {method!r}")
    config = _configuration(configuration)
    x, y, xv, yv, source = _observations(data, method)
    if rank is not None or method not in {"ridge_to_source", "nuclear_contrast"}:
        rank = _integer(rank, "rank")
    p, q = x.shape[1], y.shape[1]
    if rank is not None and rank > min(p, q):
        raise ValueError("rank exceeds ambient coefficient dimensions")
    frames = _frames(source_arrays, p, q) if method in _SUBSPACE | {"initializer_only"} else None
    ridge_scale = float(np.linalg.norm(x, ord=2) ** 2 / len(x)) if method in {
        "target_ridge_rrr", "source_subspace_ridge_rrr", "ridge_to_source", "source_target_mixture"} else None
    nuclear_scale = (float(np.linalg.norm(x.T @ (y - x @ source) / len(x), ord=2))
                     if method == "nuclear_contrast" else None)
    if any(s is not None and not np.isfinite(s) for s in (ridge_scale, nuclear_scale)):
        raise ValueError("nonfinite training-derived tuning scale")
    specifications = _specifications(method, rank, p, q, config, ridge_scale, nuclear_scale)
    record = dict(
        schema_version=1, method=method, status="no_eligible_candidate", requested_rank=rank,
        selected_index=None, selected_parameters=None, validation_mse=None,
        validation_mse_recomputed=None, effective_rank=None, source_dimensions=None,
        candidate_results=[], n_candidates=len(specifications), n_eligible=0, n_failed=0,
        elapsed_seconds=None, boundary_indicators={}, configuration=config,
        training_scales=dict(ridge=ridge_scale, nuclear_contrast=nuclear_scale),
        information_regime="target_only" if method in _TARGET else "same_observed_source_coefficient",
        validation_loss_normalization="squared_residual_sum/(n_validation*q)",
        selection_rule="first_minimum_among_certified_candidates", refit=False,
        rank_constraint_scope=("coefficient" if method in _RANK_CONSTRAINED else
                               "target_component_only" if method == "source_target_mixture" else "none"),
    )
    winner = None
    for index, parameters in enumerate(specifications):
        candidate_started = perf_counter()
        attempt = dict(index=index, parameters=deepcopy(parameters), status="failed",
                       diagnostics={}, validation_mse=None, effective_rank=None)
        try:
            coefficient, diagnostics = _fit_one(x, y, source, frames, method, parameters)
            attempt["diagnostics"] = _json(diagnostics)
            coefficient = _matrix(coefficient, "candidate coefficient")
            if coefficient.shape != (p, q):
                raise ValueError("candidate coefficient has the wrong shape")
            attempt["effective_rank"] = int(np.linalg.matrix_rank(coefficient))
            if method in _RANK_CONSTRAINED and attempt["effective_rank"] > rank:
                raise ValueError("candidate violated its declared rank bound")
            attempt["validation_mse"] = _validation_mse(xv, yv, coefficient)
            if not diagnostics.get("certified", False) or not diagnostics.get("converged", False):
                attempt.update(status="uncertified", error="numerical_certificate_not_met")
            else:
                attempt["status"] = "eligible"
                record["n_eligible"] += 1
                if winner is None or attempt["validation_mse"] < record["validation_mse"]:
                    winner = np.array(coefficient, dtype=float, copy=True)
                    record.update(selected_index=index, selected_parameters=deepcopy(parameters),
                                  validation_mse=attempt["validation_mse"],
                                  effective_rank=attempt["effective_rank"],
                                  source_dimensions=deepcopy(parameters.get("source_rank")))
        except Exception as error:
            attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
        attempt["elapsed_seconds"] = perf_counter() - candidate_started
        record["candidate_results"].append(attempt)
    record["n_failed"] = len(specifications) - record["n_eligible"]
    if winner is not None:
        try:
            recomputed = _validation_mse(xv, yv, winner)
            record["validation_mse_recomputed"] = recomputed
            if not np.isclose(recomputed, record["validation_mse"],
                              rtol=config["selection_rtol"], atol=config["selection_atol"]):
                raise ValueError("winning coefficient validation score does not match its selection record")
            record.update(status="success", boundary_indicators=_boundaries(
                record["selected_parameters"], specifications))
            winner.flags.writeable = False
        except Exception as error:
            record.update(status="selection_verification_failed", error=f"{type(error).__name__}: {error}")
            winner = None
    record["elapsed_seconds"] = perf_counter() - started
    return _json(record), winner
