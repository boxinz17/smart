"""Source fit, ranks, data scales and every method, on one preprocessed fold.

SparseSMART uses the structural campaign's candidate library and fitting loop
(code/simulation/reviewer_revision_20260913/runner.py); the comparators use
the original-design comparator code (paper_source_comparators_20260913); Park
et al. uses the campaign's bridge to the authors' R code. Every function sees
training and validation cells only; test cells never enter this module.
"""

from __future__ import annotations

import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
for _path in (CODE_ROOT / "simulation", CODE_ROOT / "sparse-smart" / "src", CODE_ROOT / "sparse-smart-v2" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from smart import RankSelectorRSC  # noqa: E402
from sparse_smart_v2 import ObservedSource, prepare_source, target_rrr  # noqa: E402


def _failure(started, error):
    return dict(success=False, error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)


# --------------------------------------------------------------------------- source and ranks

def fit_source(fd, estimator, config):
    """Reduced-rank ridge (or ridge) on source cells, tuned on held-out donors, then refitted."""
    from reviewer_revision_20260913.design import _ridge_rrr_path

    X, Y = fd.X_source, fd.Y_source
    hold = np.isin(fd.source_donor, fd.fold["source_holdout_donors"])
    if hold.all() or not hold.any():
        raise ValueError("source holdout must be a proper nonempty subset")
    max_rank = min(X.shape[1], Y.shape[1], int((~hold).sum()))
    if estimator == "reduced_rank_ridge":
        ranks = sorted({min(k, max_rank) for k in config["source_fit"]["ranks"]} | {max_rank})
    elif estimator == "ridge":
        ranks = [max_rank]
    else:
        raise ValueError(f"unknown source estimator: {estimator}")
    history, best = [], None
    for alpha, rank, coefficient in _ridge_rrr_path(X[~hold], Y[~hold], config["source_fit"]["alphas"], ranks):
        loss = float(np.mean((Y[hold] - X[hold] @ coefficient) ** 2))
        history.append(dict(alpha=alpha, rank=rank, holdout_mse=loss))
        if best is None or loss < best[0]:
            best = (loss, alpha, rank)
    loss, alpha, rank = best
    coefficient = next(_ridge_rrr_path(X, Y, [alpha], [rank]))[2]
    return coefficient, dict(estimator=estimator, selected_alpha=alpha, selected_rank=int(rank),
                             holdout_mse=loss, n_fit=int((~hold).sum()), n_holdout=int(hold.sum()),
                             holdout_donors=list(fd.fold["source_holdout_donors"]), candidates=history)


def rsc(X, Y):
    """Bunea-She-Wegkamp rank with its threshold and the projected-response spectrum."""
    selector = RankSelectorRSC()
    projected = selector.compute_projection_matrix(X) @ Y
    eigenvalues = np.linalg.eigvalsh(projected.T @ projected)[::-1]
    design_rank = int(np.linalg.matrix_rank(X))
    if Y.shape[0] <= design_rank:
        raise ValueError("RSC noise estimate needs more cells than the design rank")
    sigma = selector.estimate_sigma(Y, projected, Y.shape[1], design_rank)
    mu = selector.compute_mu(sigma, Y.shape[1], design_rank)
    rank = int(np.sum(eigenvalues >= mu))
    return rank, dict(rank=rank, threshold=float(mu), sigma=float(sigma),
                      singular_values=np.sqrt(np.maximum(eigenvalues[:40], 0)).tolist())


def target_scales(fd, rank):
    """Noise SD and top singular value of the target-only RRR fit on training cells."""
    fit = target_rrr(fd.X_train, fd.Y_train, rank)
    n, p = fd.X_train.shape
    q = fd.Y_train.shape[1]
    df = rank * (p + q - rank)
    if n * q <= df:
        raise ValueError("too few training cells to estimate the noise scale")
    residual = fd.Y_train - fd.X_train @ fit.coefficient
    sigma = float(np.sqrt(np.sum(residual ** 2) / (n * q - df)))
    d1 = float(np.linalg.svd(fit.coefficient, compute_uv=False)[0])
    return dict(sigma=sigma, d1=d1, rank=rank, degrees_of_freedom=int(df))


def prepare_bases(C0, rank0):
    p, q = C0.shape
    return prepare_source(ObservedSource(C0), p=p, q=q, source_rank=rank0)


# --------------------------------------------------------------------------- SparseSMART

def sparse_smart_configuration(scales, config, mode):
    from reviewer_revision_20260913.runner import configuration

    settings = config["sparse_smart"]
    iterations = config["smoke"]["iterations"] if mode == "smoke" else settings["iterations"]
    result = configuration(iterations=iterations)
    penalty_scale = scales["sigma"] / settings["reference_sigma"]
    margin_scale = scales["d1"] / settings["reference_top_singular_value"]
    for key in ("init_penalties", "penalties_u", "penalties_v"):
        result[key] = [value * penalty_scale for value in result[key]]
    result["margins"] = {key: value * margin_scale if key in settings["scaled_margins"] else value
                         for key, value in result["margins"].items()}
    result["scales"] = dict(penalty=penalty_scale, margin=margin_scale)
    return result


def run_sparse_smart(fd, bases, rank, rank0, scales, config, mode):
    """Fit the 211-candidate library one candidate at a time and select on validation.

    Returns winners for "sparse_smart" (all candidates, including the target-RRR
    endpoint) and "sparse_smart_transfer_only" (chart candidates only), plus a
    compact record of every candidate.
    """
    from reviewer_revision_20260913.runner import make_v2, v2_specs
    from sparse_smart.validation import validation_loss_difference

    library = sparse_smart_configuration(scales, config, mode)
    p, q = fd.X_train.shape[1], fd.Y_train.shape[1]
    case = dict(fitted_rank=rank, target_rank=rank, source_rank=rank0, p=p, q=q)
    specs = list(v2_specs(case, library, "full_caps"))
    indices = (range(len(specs)) if mode == "production"
               else [i for i in config["smoke"]["sparse_smart_candidates"] if i < len(specs)])
    cache, winners, records = {}, {}, []
    for index in indices:
        spec, started, model, error = specs[index], time.perf_counter(), None, None
        try:
            model = make_v2(spec, library)
            model.fit(fd.X_train, fd.Y_train, source=bases,
                      validation_data=(fd.X_val, fd.Y_val), _initialization_cache=cache)
        except Exception as exc:  # recorded, never substituted
            error = f"{type(exc).__name__}: {exc}"
        good = error is None and bool(getattr(model, "success_", False))
        record = dict(index=index, init_penalty=spec["init_penalty"], penalty=list(spec["penalty"]),
                      free_directions=list(spec["free_directions"]), support_limits=list(spec["support_limits"]),
                      eligible=good, error=error, status=getattr(model, "status_", "exception"),
                      method=getattr(model, "method_", None),
                      termination_reason=getattr(model, "termination_reason_", None),
                      n_iter=getattr(model, "n_iter_", None),
                      selected_iteration=getattr(model, "selected_iteration_", None),
                      numerical_work=getattr(model, "numerical_work_", {}),
                      elapsed_seconds=time.perf_counter() - started)
        coefficient = np.array(model.coefficient_, dtype=float) if good else None
        if good and not np.isfinite(coefficient).all():
            good = record["eligible"] = False
            record["error"] = "successful fit returned a nonfinite coefficient"
        if good:
            prediction = fd.X_val @ coefficient
            record["validation_mse"] = float(np.mean((fd.Y_val - prediction) ** 2))
            labels = ["sparse_smart"] + (["sparse_smart_transfer_only"] if record["method"] != "target_rrr" else [])
            for label in labels:
                old = winners.get(label)
                if old is None or validation_loss_difference(prediction, old["prediction"], fd.Y_val) < 0:
                    winners[label] = dict(coefficient=coefficient, prediction=prediction, candidate_index=index,
                                          selected_method=record["method"],
                                          selected_iteration=record["selected_iteration"],
                                          parameters=dict(init_penalty=spec["init_penalty"],
                                                          penalty=list(spec["penalty"])))
        records.append(record)
    summary = dict(n_candidates=len(specs), n_fitted=len(records), n_eligible=sum(r["eligible"] for r in records),
                   status_counts=_counts(r["status"] for r in records),
                   termination_counts=_counts(r["termination_reason"] for r in records),
                   seconds=sum(r["elapsed_seconds"] for r in records),
                   grid=dict(init_penalties=library["init_penalties"], penalties_u=library["penalties_u"],
                             penalties_v=library["penalties_v"], margins=library["margins"],
                             iterations=library["iterations"], scales=library["scales"]))
    return winners, records, summary


def _counts(values):
    counts = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


# --------------------------------------------------------------------------- comparators

def run_comparators(fd, C0, bases, rank, rank0, scales, config):
    """The simulation comparators, plus full-rank ridge (target ridge RRR at rank min(p, q))."""
    from paper_source_comparators_20260913.fitting import default_configuration, fit_method

    settings = default_configuration()
    penalty_scale = scales["sigma"] / config["sparse_smart"]["reference_sigma"]
    settings["initializer_penalties"] = [v * penalty_scale for v in settings["initializer_penalties"]]
    settings["initializer_min_dimension"] = int(rank0)
    settings["source_dimensions"] = sorted(set(config["comparators"]["source_dimensions"]) | {int(rank0)})
    data = dict(X=fd.X_train, Y=fd.Y_train, C0=C0, X_validation=fd.X_val, Y_validation=fd.Y_val)
    frames = dict(left=np.array(bases.left), right=np.array(bases.right))
    p, q = fd.X_train.shape[1], fd.Y_train.shape[1]
    jobs = [(m, m, rank) for m in config["comparators"]["methods"]] + [("target_ridge", "target_ridge_rrr", min(p, q))]
    results = {}
    for label, method, method_rank in jobs:
        started = time.perf_counter()
        try:
            record, coefficient = fit_method(data, frames, method, method_rank, settings)
            results[label] = dict(success=record["status"] == "success" and coefficient is not None,
                                  coefficient=coefficient, record=_slim(record),
                                  elapsed_seconds=time.perf_counter() - started)
        except Exception as exc:
            results[label] = _failure(started, exc)
    return results


def _slim(record):
    """Drop per-candidate diagnostics but keep each candidate's score and status."""
    slim = {k: v for k, v in record.items() if k != "candidate_results"}
    slim["candidates"] = [dict(index=c["index"], status=c["status"], validation_mse=c["validation_mse"],
                               parameters=c["parameters"], error=c.get("error"))
                          for c in record.get("candidate_results", [])]
    return slim


def run_lasso(fd, config):
    """Target-only Lasso, one penalty shared across proteins, chosen on validation."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso

    started, settings = time.perf_counter(), config["lasso"]
    X, Y = fd.X_train, fd.Y_train
    alpha_max = float(np.max(np.abs(X.T @ Y)) / len(X))
    alphas = alpha_max * np.logspace(0, np.log10(settings["min_ratio"]), settings["n_alphas"])
    model = Lasso(fit_intercept=False, max_iter=settings["max_iter"], tol=settings["tol"], warm_start=True)
    candidates, best = [], None
    for alpha in alphas:
        model.set_params(alpha=float(alpha))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, Y)
        coefficient = np.array(model.coef_.T, dtype=float)
        loss = float(np.mean((fd.Y_val - fd.X_val @ coefficient) ** 2))
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        candidates.append(dict(alpha=float(alpha), validation_mse=loss, converged=converged,
                               nonzero=int(np.count_nonzero(coefficient))))
        if converged and (best is None or loss < best[0]):
            best = (loss, float(alpha), coefficient)
    if best is None:
        return dict(success=False, error="no converged Lasso fit", record=dict(candidates=candidates),
                    elapsed_seconds=time.perf_counter() - started)
    return dict(success=True, coefficient=best[2],
                record=dict(selected_alpha=best[1], validation_mse=best[0], alpha_max=alpha_max,
                            candidates=candidates),
                elapsed_seconds=time.perf_counter() - started)


def run_park(fd, config, mode):
    """Park et al. two-stage NR with raw source cells, penalties chosen on validation."""
    from reviewer_revision_20260913.published_park import fit_park_validation

    started, settings = time.perf_counter(), config["park"]
    lambdas = config["smoke"]["park_lambdas"] if mode == "smoke" else settings["lambdas"]
    try:
        selected = fit_park_validation(fd.X_train, fd.Y_train, fd.X_val, fd.Y_val, fd.X_source, fd.Y_source,
                                       lambda_w=lambdas, lambda_delta=lambdas, max_iter=settings["max_iter"],
                                       tolerance=settings["tolerance"], require_convergence=True,
                                       timeout_seconds=settings["timeout_seconds"], allow_local=True)
    except Exception as exc:
        failure = _failure(started, exc)
        failure["diagnostics"] = {k: v for k, v in getattr(exc, "diagnostics", {}).items()
                                  if k != "candidate_results"}
        return failure
    diagnostics = {k: v for k, v in selected.fit.diagnostics.items() if k != "candidate_results"}
    return dict(success=True, coefficient=np.array(selected.fit.coefficient), intercept=np.array(selected.fit.intercept),
                record=dict(selected_index=selected.selected_index, validation_loss=selected.validation_loss,
                            lambdas=list(lambdas), candidates=list(selected.candidate_results),
                            diagnostics=diagnostics),
                elapsed_seconds=time.perf_counter() - started)
