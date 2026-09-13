"""Paired, one-factor simulation designs for the September 2026 revision.

Every estimator must receive only ``generate_case(...)["fit_data"]``. Clean
objects, population loss, and test data live in ``evaluation``; the clean-basis
benchmark is the sole explicitly labelled oracle exception. Stream-specific
seeds make target covariates/noise identical across stress levels and methods.
Source sample size means the COMPLETE available raw source sample, including
the source-only tuning split, which is subsequently used in source refitting.
"""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np


_DEFAULT = dict(p=100, q=50, target_rank=5, latent_source_rank=10,
                source_rank=10, n_train=80, n_validation=200, n_test=1000,
                noise_sd=0.5, source_noise_sd=0.01, rho=0.5,
                left_angle_deg=0.0, right_angle_deg=0.0,
                target_specific=0, diffuse_strength=0.0,
                source_internal_gap=1.0, target_internal_gap=1.0,
                source_boundary_gap=1.0, target_boundary_gap=1.0,
                source_mode="perturbed", n_source=300,
                source_fit_rank=None, source_response_noise_sd=0.5,
                relationship="historical", coefficient_close_scale=0.1)


def case_manifest(n_train_values=(80, 200), n_validation=200,
                  n_test=1000) -> list[dict[str, Any]]:
    """Return a compact manifest; replications are deliberately not prescribed.

    Gap factors reduce within-pair gaps, with exactly conserved spectral energy
    and smallest singular value. Separate boundary sweeps hold energy fixed.
    Operational source truncation is distinct from latent/fitted source rank.
    """
    cases = []
    def add(n, family, level, suffix, **updates):
        case = dict(_DEFAULT, n_train=int(n), n_validation=int(n_validation),
                    n_test=int(n_test), case_id=f"{family}_{suffix}_n{n}",
                    family=family, level=level)
        case.update(updates)
        cases.append(case)
    for n in n_train_values:
        add(n, "reference", 0, "exact")
        for noise in (0., .05, .2, 1.):
            add(n, "source_noise", noise, str(noise), source_noise_sd=noise)
        for angle in (15, 30, 60, 90):
            add(n, "containment", angle, f"both{angle}",
                left_angle_deg=angle, right_angle_deg=angle)
        add(n, "containment_side", "left30", "left30", left_angle_deg=30)
        add(n, "containment_side", "right30", "right30", right_angle_deg=30)
        for k in (1, 3):
            add(n, "target_specific", k, f"k{k}", target_specific=k)
        for strength in (0.05, 0.2, 1.0):
            add(n, "diffuse_alignment", strength, str(strength),
                diffuse_strength=strength)
        for factor in (0.2, 0.02, 0.0):
            add(n, "source_internal_gap", factor, str(factor),
                source_internal_gap=factor)
            add(n, "target_internal_gap", factor, str(factor),
                target_internal_gap=factor)
        for factor in (0.2, 0.02):
            add(n, "source_boundary_gap", factor, str(factor),
                source_boundary_gap=factor)
            add(n, "target_boundary_gap", factor, str(factor),
                target_boundary_gap=factor)
        for retained in (3, 5, 15, 20):
            add(n, "source_truncation", retained, f"rank{retained}",
                source_rank=retained)
        for n_source in (100, 300, 1000):
            add(n, "fitted_source", n_source, f"n0{n_source}",
                source_mode="fitted", n_source=n_source)
        add(n, "coefficient_close", 0.1, "delta0.1",
            relationship="coefficient_close")
        add(n, "spectral_shift", "shared_directions", "shared_directions",
            relationship="spectral_shift")
        add(n, "fitted_relationship", "coefficient_close", "coefficient_close",
            relationship="coefficient_close",source_mode="fitted",n_source=1000)
        add(n, "fitted_relationship", "spectral_shift", "spectral_shift",
            relationship="spectral_shift",source_mode="fitted",n_source=1000)
    add(40, "sparse_stress", "exact", "exact", target_rank=3,
        latent_source_rank=20, source_rank=20)
    add(40, "sparse_stress", "diffuse", "diffuse", target_rank=3,
        latent_source_rank=20, source_rank=20, diffuse_strength=1.0)
    return cases


def runtime_manifest():
    """Fixed-rank dimensional timing study, identical tuning library at each size."""
    base=next(c for c in case_manifest() if c['family']=='reference' and c['n_train']==200)
    return [dict(base,case_id=f'runtime_p{p}_q{q}_n200',family='runtime',
                 level=f'{p}x{q}',p=p,q=q) for p,q in ((100,50),(150,100),(300,200))]


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream]))


def _frame(rng, dimension, columns):
    q, r = np.linalg.qr(rng.standard_normal((dimension, columns)))
    return q * np.where(np.diag(r) < 0, -1.0, 1.0)


def _rotate_coordinates(rng, dimension, strength):
    """Continuous orthogonal path, identity at zero and diffuse at one."""
    result = np.eye(dimension)
    pairs = [(i, j) for i in range(dimension) for j in range(i + 1, dimension)]
    rng.shuffle(pairs)
    angles = rng.uniform(-np.pi, np.pi, len(pairs)) * strength
    for (i, j), angle in zip(pairs, angles):
        c, s = np.cos(angle), np.sin(angle)
        a, b = result[:, i].copy(), result[:, j].copy()
        result[:, i], result[:, j] = c * a + s * b, -s * a + c * b
    return result


def _spectrum(base, internal_factor=1.0, boundary_factor=1.0):
    """Preserve energy; internal ties leave the boundary singular value fixed.

    Adjacent pairs among all but the last coordinate form fixed clusters.
    Internal factor zero ties each pair. This is NOT a claim that every gap
    vanishes: between-cluster gaps remain positive and are reported explicitly.
    """
    values = np.asarray(base, dtype=float).copy()
    energy = float(values @ values)
    for i in range(0, len(values) - 2, 2):
        a, b = values[i:i + 2]
        relative = internal_factor * (a - b) / (a + b)
        center = np.sqrt((a * a + b * b) / (2 * (1 + relative**2)))
        values[i:i + 2] = center * np.array([1 + relative, 1 - relative])
    if boundary_factor != 1:
        values[-1] *= boundary_factor
        values[:-1] *= np.sqrt((energy - values[-1]**2) /
                              np.sum(values[:-1]**2))
    return values


def _ridge_rrr_path(X, Y, alphas, ranks):
    """Exact reduced-rank ridge minimizers, penalty = n * alpha * ||C||_F^2."""
    u, singular, vt = np.linalg.svd(X, full_matrices=False)
    projected_y = u.T @ Y
    tolerance = np.finfo(float).eps * max(X.shape) * singular.max(initial=0)
    for alpha in alphas:
        denominator = singular**2 + len(X) * alpha
        inverse = np.divide(singular, denominator, out=np.zeros_like(singular),
                            where=denominator > tolerance**2)
        ridge = (vt.T * inverse) @ projected_y
        whitened = (np.sqrt(np.maximum(singular * inverse, 0))[:, None] *
                    projected_y)
        _, _, right = np.linalg.svd(whitened, full_matrices=False)
        for rank in ranks:
            directions = right[:min(rank, len(right))].T
            yield float(alpha), int(rank), ridge @ directions @ directions.T


def _fit_source(X, Y, case):
    """Source-only tuning then refit on all available source observations."""
    n_val = max(10, int(round(0.2 * len(X))))
    n_fit = len(X) - n_val
    if n_fit < 5:
        raise ValueError("A fitted source requires at least 20 observations")
    max_rank = min(X.shape[1], Y.shape[1], n_fit)
    if case["source_fit_rank"] is None:
        ranks = sorted(set(min(k, max_rank) for k in (1, 3, 5, 10, 15, 20,
                                                       max_rank)))
    else:
        ranks = [int(case["source_fit_rank"])]
        if not 1 <= ranks[0] <= max_rank:
            raise ValueError("Fixed source fitted rank exceeds training rank")
    alphas = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    best, history = None, []
    for alpha, rank, coefficient in _ridge_rrr_path(X[:n_fit], Y[:n_fit],
                                                    alphas, ranks):
        loss = float(np.mean((Y[n_fit:] - X[n_fit:] @ coefficient)**2))
        history.append(dict(alpha=alpha, rank=rank, validation_mse=loss))
        if best is None or loss < best[0]:
            best = loss, alpha, rank
    _, coefficient_alpha, coefficient_rank = best
    fitted = next(_ridge_rrr_path(X, Y, [coefficient_alpha],
                                  [coefficient_rank]))[2]
    return fitted, dict(estimator="source-validated reduced-rank ridge",
                        selected_alpha=coefficient_alpha,
                        selected_rank=coefficient_rank,
                        validation_mse=best[0], n_fit=n_fit,
                        n_validation=n_val, n_refit=len(X), candidates=history)


def _alignment_metrics(coordinates):
    squares = np.sort(coordinates**2, axis=0)[::-1]
    return dict(tail_after_top1=float(np.sqrt(np.sum(squares[1:]))),
                tail_after_top2=float(np.sqrt(np.sum(squares[2:]))),
                effective_support_mean=float(np.mean(
                    np.sum(squares, axis=0)**2 /
                    np.maximum(np.sum(squares**2, axis=0), 1e-30))))


def generate_case(case: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Generate independent source, target fitting, validation and test samples.

    ``source_rank`` is a declared operational truncation setting. It is never
    silently replaced with the clean rank, selected source-fit rank, or empirical
    numerical rank of a perturbed matrix. All three are recorded separately.
    """
    case = dict(_DEFAULT, **dict(case))
    p, q = int(case["p"]), int(case["q"])
    r, r0 = int(case["target_rank"]), int(case["latent_source_rank"])
    if not (1 <= r <= r0 and r0 + r <= min(p, q)):
        raise ValueError("Require 1 <= target_rank <= latent_source_rank and r0+r <= min(p,q)")
    if not 1 <= int(case["source_rank"]) <= min(p, q):
        raise ValueError("Operational source_rank must lie in [1,min(p,q)]")
    if not 0 <= int(case["target_specific"]) <= r:
        raise ValueError("target_specific must lie between zero and target rank")
    for key in ("source_internal_gap", "target_internal_gap"):
        if not 0 <= case[key] <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    for key in ("source_boundary_gap", "target_boundary_gap"):
        if not 0 < case[key] <= 1:
            raise ValueError(f"{key} must lie in (0,1]")
    source_left_full = _frame(_rng(seed, 1), p, r0 + r)
    source_right_full = _frame(_rng(seed, 2), q, r0 + r)
    U0, V0 = source_left_full[:, :r0], source_right_full[:, :r0]
    target_s = _spectrum(np.linspace(5, 3, r), case["target_internal_gap"],
                         case["target_boundary_gap"])
    source_s = _spectrum(np.linspace(10, 1, r0), case["source_internal_gap"],
                         case["source_boundary_gap"])
    if case["relationship"] in ("coefficient_close", "spectral_shift"):
        indices_left = indices_right = np.arange(r)
    elif case["relationship"] == "historical":
        indices_left = _rng(seed, 3).choice(r0, r, replace=False)
        indices_right = _rng(seed, 4).choice(r0, r, replace=False)
    else:
        raise ValueError("Unknown source-target relationship")
    if case["relationship"] == "coefficient_close":
        delta = float(case["coefficient_close_scale"])
        if not 0 < delta <= 0.5:
            raise ValueError("Coefficient-closeness scale must lie in (0,0.5]")
        source_s = np.r_[target_s * (1 + delta),
                         delta * np.linspace(2, 1, r0 - r)]
    left_coordinates = _rotate_coordinates(_rng(seed, 5), r0,
                    case["diffuse_strength"])[:, indices_left]
    right_coordinates = _rotate_coordinates(_rng(seed, 6), r0,
                    case["diffuse_strength"])[:, indices_right]
    left_angles = np.full(r, np.deg2rad(case["left_angle_deg"]))
    right_angles = np.full(r, np.deg2rad(case["right_angle_deg"]))
    k = int(case["target_specific"])
    if k:
        left_angles[-k:] = right_angles[-k:] = np.pi / 2
    U_star = (U0 @ left_coordinates) * np.cos(left_angles) + \
             source_left_full[:, r0:] * np.sin(left_angles)
    V_star = (V0 @ right_coordinates) * np.cos(right_angles) + \
             source_right_full[:, r0:] * np.sin(right_angles)
    C_star = (U_star * target_s) @ V_star.T
    clean_source = (U0 * source_s) @ V0.T
    Sigma_x = case["rho"] ** np.abs(np.arange(p)[:, None] - np.arange(p))
    root_covariance = np.linalg.cholesky(Sigma_x)
    def observations(n, stream, coefficient, noise):
        X = _rng(seed, stream).standard_normal((int(n), p)) @ root_covariance.T
        Y = X @ coefficient + noise * _rng(seed, stream + 1).standard_normal((int(n), q))
        return X, Y
    X, Y = observations(case["n_train"], 10, C_star, case["noise_sd"])
    X_val, Y_val = observations(case["n_validation"], 20, C_star, case["noise_sd"])
    X_test, Y_test = observations(case["n_test"], 30, C_star, case["noise_sd"])
    source_X = source_Y = None
    if case["source_mode"] == "perturbed":
        C0 = clean_source + case["source_noise_sd"] * _rng(seed, 7).standard_normal((p, q))
        source_fit = dict(estimator="entrywise Gaussian perturbation",
                          selected_rank=None, n_refit=0)
    elif case["source_mode"] == "fitted":
        source_X, source_Y = observations(case["n_source"], 40, clean_source,
                                           case["source_response_noise_sd"])
        C0, source_fit = _fit_source(source_X, source_Y, case)
    else:
        raise ValueError("source_mode must be 'perturbed' or 'fitted'")
    source_values = np.linalg.svd(C0, compute_uv=False)
    source_error = C0 - clean_source
    left_residual = C_star - U0 @ (U0.T @ C_star)
    right_residual = C_star - (C_star @ V0) @ V0.T
    joint_residual = C_star - U0 @ (U0.T @ C_star @ V0) @ V0.T
    metadata = dict(case_id=case.get("case_id", "custom"), seed=int(seed),
        family=case.get("family", "custom"), level=case.get("level", "custom"),
        case=case, latent_source_rank=r0, operational_source_rank=int(case["source_rank"]),
        fitted_source_rank=source_fit["selected_rank"],
        source_numerical_rank=int(np.linalg.matrix_rank(C0)),
        source_fit=source_fit, source_singular_values=source_s.tolist(),
        observed_source_singular_values=source_values.tolist(),
        target_singular_values=target_s.tolist(),
        source_internal_gaps=(-np.diff(source_s)).tolist(),
        target_internal_gaps=(-np.diff(target_s)).tolist(),
        source_boundary_gap=float(source_s[-1]), target_boundary_gap=float(target_s[-1]),
        target_coefficient_energy=float(np.sum(C_star**2)),
        source_coefficient_energy=float(np.sum(clean_source**2)),
        target_population_signal=float(np.sum(C_star * (Sigma_x @ C_star)) / q),
        source_error_frobenius=float(np.linalg.norm(source_error)),
        source_error_operator=float(np.linalg.norm(source_error, ord=2)),
        source_target_distance=float(np.linalg.norm(clean_source - C_star)),
        containment_left=float(np.linalg.norm(left_residual)),
        containment_right=float(np.linalg.norm(right_residual)),
        containment_joint=float(np.linalg.norm(joint_residual)),
        left_angles_deg=np.rad2deg(left_angles).tolist(),
        right_angles_deg=np.rad2deg(right_angles).tolist(),
        alignment_left=_alignment_metrics(left_coordinates),
        alignment_right=_alignment_metrics(right_coordinates),
        pairing="named independent streams; common target X and Gaussian noise across levels",
        source_access="raw source observations plus fitted matrix" if source_X is not None else "fitted matrix only")
    return dict(fit_data=dict(X=X, Y=Y, X_validation=X_val, Y_validation=Y_val,
                             C0=C0, source_rank=int(case["source_rank"]),
                             source_X=source_X, source_Y=source_Y),
                evaluation=dict(C_star=C_star, Sigma_x=Sigma_x, U0=U0, V0=V0,
                                clean_source=clean_source, U_star=U_star, V_star=V_star,
                                X_test=X_test, Y_test=Y_test), metadata=metadata)


def error_metrics(coefficient, evaluation):
    """Population prediction excess excludes the irreducible response variance."""
    error = np.asarray(coefficient) - evaluation["C_star"]
    p, q = error.shape
    result = dict(coefficient_rmse=float(np.linalg.norm(error) / np.sqrt(p * q)),
                  coefficient_mse=float(np.sum(error**2) / (p * q)),
                  prediction_excess=float(np.sum(error * (evaluation["Sigma_x"] @ error)) / q))
    if len(evaluation["X_test"]):
        result["test_prediction_excess"] = float(np.mean((evaluation["X_test"] @ error)**2))
        result["test_response_mse"] = float(np.mean((evaluation["Y_test"] -
                                                   evaluation["X_test"] @ coefficient)**2))
    return result
