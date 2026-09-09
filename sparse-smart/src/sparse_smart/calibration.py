"""Declared numerical calibration for the sparse two-stage procedure.

The appendix constants are deliberately conservative.  Their evaluation does
not establish assumptions about an unknown signal or a realized design.  In
particular, ``primitive_checks`` are diagnostics, not a theorem certificate.

Automatic support enlargement uses an upper bound obtained by evaluating all
support counts at saturation.  It bounds the eventual inverse step size
without the circular dependence of that step size on the support enlargement.
This is a stronger numerical prescription than the appendix's conditional
bounded-conditioning shortcut.  Enlargement is computed in logarithms, and
support sizes are capped *before* converting to integers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
import sys
from typing import Any


class CalibrationError(ValueError):
    """Invalid inputs or constants outside floating-point representation."""


class SourceAccuracyError(CalibrationError):
    """The declared source noise exceeds the declared gap allowance."""


def _real(name: str, value: Any, *, minimum: float = 0.0,
          strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CalibrationError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise CalibrationError(f"{name} is not representable") from exc
    if not math.isfinite(result) or result < minimum or (strict and result == minimum):
        relation = ">" if strict else ">="
        raise CalibrationError(f"{name} must be finite and {relation} {minimum}")
    return result


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise CalibrationError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _pair(name: str, values: Any, *, minimum: int = 0) -> tuple[int, int]:
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise CalibrationError(f"{name} must contain exactly two integer bounds")
    return tuple(_integer(f"{name}[{a}]", value, minimum=minimum)
                 for a, value in enumerate(values))


def _confidence(delta: Any) -> float:
    value = _real("delta", delta, strict=True)
    if value >= 0.25:
        raise CalibrationError("delta must lie strictly between 0 and 0.25")
    return value


def penalty_pair(value: Any) -> tuple[float, float]:
    """Broadcast a scalar penalty or validate separate left/right penalties."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise CalibrationError("penalty must be a scalar or a pair of two penalties")
        return tuple(_real(f"penalty[{side}]", item) for side, item in enumerate(value))
    scalar = _real("penalty", value)
    return scalar, scalar


@dataclass(frozen=True)
class Margins:
    d_lower: float
    d_upper: float
    gap: float
    anchor_min: float = 0.01
    trial_radius: float = 1.0
    qr_threshold: float = 2.0

    def __post_init__(self) -> None:
        lower = _real("d_lower", self.d_lower, strict=True)
        upper = _real("d_upper", self.d_upper, strict=True)
        if lower >= upper / 4:
            raise CalibrationError("d_upper must exceed 4 * d_lower")
        _real("gap", self.gap, strict=True)
        anchor = _real("anchor_min", self.anchor_min, strict=True)
        if anchor >= 1 / 16:
            raise CalibrationError("anchor_min must be less than 1/16")
        _real("trial_radius", self.trial_radius, strict=True)
        _real("qr_threshold", self.qr_threshold, minimum=1.0, strict=True)


@dataclass(frozen=True)
class PrescribedCalibration:
    sigma: float
    delta: float = 0.05
    support_enlargement: float | None = None
    design_lower: float = 0.5

    def __post_init__(self) -> None:
        _real("sigma", self.sigma, strict=True)
        _confidence(self.delta)
        if self.support_enlargement is not None:
            _real("support_enlargement", self.support_enlargement,
                  minimum=1.0, strict=True)
        _real("design_lower", self.design_lower, strict=True)


@dataclass(frozen=True)
class PracticalCalibration:
    init_penalty: float
    penalty: float | tuple[float, float]
    step_size_inverse: float
    support_limits: tuple[int, int]
    delta: float = 0.05
    enforce_source_accuracy: bool = True

    def __post_init__(self) -> None:
        _real("init_penalty", self.init_penalty, strict=True)
        penalties = penalty_pair(self.penalty)
        if isinstance(self.penalty, (tuple, list)):
            object.__setattr__(self, "penalty", penalties)
        _real("step_size_inverse", self.step_size_inverse, strict=True)
        _pair("support_limits", self.support_limits)
        _confidence(self.delta)
        if not isinstance(self.enforce_source_accuracy, bool):
            raise CalibrationError("enforce_source_accuracy must be a boolean")


@dataclass(frozen=True)
class ResolvedCalibration:
    init_penalty: float
    penalty: float | tuple[float, float]
    step_size_inverse: float
    support_limits: tuple[int, int]
    mode: str
    diagnostics: dict[str, Any]

    @property
    def penalties(self) -> tuple[float, float]:
        return penalty_pair(self.penalty)


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    if not math.isfinite(value) or (positive and value <= 0):
        raise CalibrationError(
            f"{name} is outside safe floating-point representation; "
            "revise the declared bounds or use explicit practical calibration"
        )
    return value


def _exp(name: str, value: float) -> float:
    if not math.isfinite(value) or value > math.log(sys.float_info.max):
        raise CalibrationError(f"{name} overflows floating-point representation")
    return _finite(name, math.exp(value), positive=True)


def _psi(ambient: int, count: int) -> float:
    if count == 0:
        return 0.0
    return count * (1 + math.log(ambient) - math.log(count))


def _support_cap(ambient: int, reference: int, log_enlargement: float) -> int:
    if reference == 0:
        return 0
    log_count = log_enlargement + math.log(reference)
    if log_count >= math.log(ambient):
        return ambient
    # K>1 implies ceil(K*s)>s mathematically, even when a rounded product
    # would land exactly on the integer s.
    return min(ambient, max(reference + 1, math.ceil(math.exp(log_count))))


def _explicit_support_cap(ambient: int, reference: int, enlargement: float) -> int:
    """Compute ceil(K*s) exactly for the supplied floating-point value K.

    A logarithmic round trip can move an integer product just above that
    integer. Integer-ratio arithmetic preserves the supplied K, including
    adjacent floating-point values on either side of an integer boundary.
    """
    if reference == 0:
        return 0
    numerator, denominator = enlargement.as_integer_ratio()
    product = numerator * reference
    if product >= ambient * denominator:
        return ambient
    return (product + denominator - 1) // denominator


def _log1pexp(value: float) -> float:
    return max(0.0, value) + math.log1p(math.exp(-abs(value)))


def resolve_calibration(
    config: PrescribedCalibration | PracticalCalibration,
    *, n: int, p: int, q: int, rank: int, source_rank: int,
    sparsity: tuple[int, int], margins: Margins,
    source_noise: float = 0.0, source_gap: float | None = None,
    cluster_size: int = 1,
) -> ResolvedCalibration:
    """Resolve penalties, inverse step size, and fixed complement budgets.

    ``source_noise == 0`` selects exact-source working dimensions.  Positive
    source noise selects full ambient working frames and requires a positive
    source gap. The condition ``eta_0 <= source_gap / 8`` is enforced by
    default; practical calibration can explicitly disable enforcement for
    empirical experiments, retaining the condition's result in diagnostics.
    Cluster inflation applies to the declared sparsity and row bounds before
    every downstream count.  Practical tuning does not evaluate the huge
    geometry and curvature constants.
    """
    if not isinstance(config, (PrescribedCalibration, PracticalCalibration)):
        raise CalibrationError("config must be prescribed or practical calibration")
    if not isinstance(margins, Margins):
        raise CalibrationError("margins must be a Margins instance")
    n = _integer("n", n, minimum=1)
    p = _integer("p", p, minimum=1)
    q = _integer("q", q, minimum=1)
    rank = _integer("rank", rank, minimum=1)
    source_rank = _integer("source_rank", source_rank, minimum=1)
    if not rank <= source_rank <= min(p, q):
        raise CalibrationError("require rank <= source_rank <= min(p, q)")
    sparsity = _pair("sparsity", sparsity, minimum=rank)
    if any(s > source_rank * rank for s in sparsity):
        raise CalibrationError("sparsity cannot exceed source_rank * rank")
    cluster_size = _integer("cluster_size", cluster_size, minimum=1)
    if cluster_size > source_rank:
        raise CalibrationError("cluster_size cannot exceed source_rank")
    noise = _real("source_noise", source_noise)
    if source_gap is not None:
        _real("source_gap", source_gap, strict=True)
    if noise > 0 and source_gap is None:
        raise CalibrationError("positive source_noise requires a positive source_gap")

    try:
        return _resolve(config, n=n, p=p, q=q, rank=rank,
                        source_rank=source_rank, sparsity=sparsity,
                        margins=margins, noise=noise, source_gap=source_gap,
                        cluster_size=cluster_size)
    except CalibrationError:
        raise
    except (OverflowError, ZeroDivisionError, ValueError) as exc:
        raise CalibrationError(f"calibration arithmetic is not representable: {exc}") from exc


def _resolve(config, *, n, p, q, rank, source_rank, sparsity, margins,
             noise, source_gap, cluster_size):
    delta_w = _finite("delta_w", float(config.delta) / 32, positive=True)
    log64 = math.log(64) - math.log(delta_w)
    nu_dim, nv_dim = (source_rank, source_rank) if noise == 0 else (p, q)
    ambient = ((nu_dim - rank) * rank, (nv_dim - rank) * rank)
    effective = tuple(min(source_rank * rank, cluster_size * s) for s in sparsity)
    row_bounds = tuple(min(source_rank, cluster_size * min(source_rank, s))
                       for s in sparsity)
    clean = tuple(min((source_rank - rank) * rank, s - rank) for s in effective)
    ell_init = math.log(16) + 2 * math.log(source_rank) - math.log(delta_w)
    # Never form the number of anchor pairs or exp(H_anc).
    h_anchor = 2 * (math.lgamma(source_rank + 1) - math.lgamma(rank + 1)
                    - math.lgamma(source_rank - rank + 1)) + rank * math.log(2)
    h_anchor = _finite("H_anc", h_anchor)
    ell = _finite("ell", log64 + h_anchor + math.log(sum(ambient) + 1))
    source_accuracy_enforced = (not isinstance(config, PracticalCalibration)
                                or config.enforce_source_accuracy)
    source_accuracy_passed = True
    if noise == 0:
        eta0 = 0.0
        eps = errors = tails = (0.0, 0.0)
    else:
        eta0 = _finite("eta_0", 4 * noise * math.sqrt((p + q) * math.log(9) + log64))
        source_accuracy_passed = eta0 <= float(source_gap) / 8
        if source_accuracy_enforced and not source_accuracy_passed:
            raise SourceAccuracyError(
                f"declared source accuracy fails: eta_0={eta0:.6g} exceeds "
                f"source_gap/8={float(source_gap) / 8:.6g}"
            )
        eps = tuple(_finite("epsilon", 8 * math.sqrt(s) * eta0 / source_gap)
                    for s in row_bounds)
        errors = tuple(_finite("source Euclidean error", margins.d_upper * e)
                       for e in eps)
        tails = tuple(_finite("source tail", math.sqrt(m) * e)
                      for m, e in zip(ambient, errors))
    beta_screen = min(1 / math.hypot(1.0, margins.qr_threshold * math.sqrt(
        rank * (s - rank))) for s in row_bounds)
    diagnostics = {
        "theorem_certified": False,
        "source_mode": "exact" if noise == 0 else "noisy",
        "working_dimensions": (nu_dim, nv_dim), "coordinate_counts": ambient,
        "effective_sparsity": effective, "row_bounds": row_bounds,
        "clean_complement_budgets": clean, "cluster_size": cluster_size,
        "delta_w": delta_w, "ell_I": ell_init, "H_anc": h_anchor, "ell": ell,
        "source_noise": noise, "source_gap": source_gap,
        "eta_0": eta0, "source_accuracy_passed": source_accuracy_passed,
        "source_accuracy_enforced": source_accuracy_enforced,
        "source_accuracy_bypassed": (not source_accuracy_enforced
                                     and not source_accuracy_passed),
        "epsilon": eps, "epsilon_active": _finite("epsilon_active", sum(eps)),
        "E": errors, "T": tails, "T_tail": _finite("T_tail", sum(tails)),
        "beta_screen": beta_screen,
        "universal_anchor_condition": 8 * margins.anchor_min <= beta_screen / 2,
        "anchor_assumption_note": (
            "The numerical universal-anchor inequality is evaluated; favorable "
            "selection stability and latent-signal assumptions are not certified."
        ),
    }
    if isinstance(config, PracticalCalibration):
        support = _pair("support_limits", config.support_limits)
        if any(k > m for k, m in zip(support, ambient)):
            raise CalibrationError("support_limits cannot exceed working complement sizes")
        diagnostics.update({
            "prescribed_calibration": False,
            "penalties": penalty_pair(config.penalty),
            "support_saturated": tuple(k == m for k, m in zip(support, ambient)),
            "calibration_note": (
                "Explicit practical tuning; no theorem calibration claim. "
                "The source-accuracy condition failed and was explicitly bypassed; "
                "source error/tail expressions are not certified bounds."
                if not source_accuracy_passed else
                "Explicit practical tuning; no theorem calibration claim."
            ),
        })
        penalty = (penalty_pair(config.penalty) if isinstance(config.penalty, (tuple, list))
                   else float(config.penalty))
        return ResolvedCalibration(float(config.init_penalty), penalty,
                                   float(config.step_size_inverse), support,
                                   "practical", diagnostics)

    sigma = float(config.sigma)
    lambda_b = _finite("lambda_b", sigma * math.sqrt(ell / n), positive=True)
    init_penalty = _finite("lambda_I", 8 * sigma * math.sqrt(ell_init / n), positive=True)
    corrections = []
    for m, s, tail in zip(ambient, clean, tails):
        remaining = m - s
        if tail == 0 or remaining == 0:
            correction = 0
        elif lambda_b == 0 or math.log(tail) - math.log(lambda_b) >= math.log(remaining):
            correction = remaining
        else:
            correction = min(remaining, math.ceil(tail / lambda_b))
        corrections.append(correction)
    corrections = tuple(corrections)
    reference = tuple(min(m, s + j) for m, s, j in zip(ambient, clean, corrections))
    beta_blocks = tuple(
        0.0 if s == m or tail == 0 else error if j == 0
        else min(error, tail / math.sqrt(j))
        for m, s, tail, error, j in zip(ambient, reference, tails, errors, corrections)
    )
    beta = math.hypot(*beta_blocks)
    a0 = _finite("A_0", 2 + math.sqrt(rank) * (1 + margins.d_upper))
    geometry_base = _finite("geometry base", 2 + rank + margins.d_upper
                            + 1 / margins.d_lower + 1 / margins.anchor_min
                            + 1 / margins.gap, positive=True)
    log_g = 30 * math.log(2) + 30 * math.log(geometry_base)
    geometry = _exp("G", log_g)
    geometry_sq = _exp("G squared", 2 * log_g)
    alpha = _finite("alpha", min(1.0, margins.d_lower, margins.gap / math.sqrt(2))
                    / (32 * (1 + margins.d_upper) * (1 + 1 / margins.anchor_min)),
                    positive=True)
    m0 = _finite("m_0", config.design_lower * alpha * alpha, positive=True)
    mu = _finite("mu", m0 / 2, positive=True)
    b_alg = _finite("b_alg", 2 * geometry * beta)

    def constants(support):
        t = tuple(min(m, k + s) for m, k, s in zip(ambient, support, reference))
        unions = tuple(min(m, 4 * k + 2 * s)
                       for m, k, s in zip(ambient, support, reference))
        rows = min(nu_dim, 2 * rank + 8 * unions[0])
        hm = rows * (math.log(5) + 1 + math.log(nu_dim) - math.log(rows)) \
            + math.log(128) - math.log(delta_w)
        kappa = _finite("kappa_upper", 4 * (1 + math.sqrt(2 * hm / n)) ** 2)
        d_sc = rank * rank + h_anchor + sum(_psi(m, s) for m, s in zip(ambient, t)) + log64
        # log(2 + G*A0) in a form avoiding multiplication overflow.
        log_ga0 = log_g + math.log(a0)
        log_geom = log_ga0 + math.log1p(2 * math.exp(-log_ga0))
        xi = h_anchor + 6 * sum(_psi(m, s) for m, s in zip(ambient, unions)) + log64 \
            + 64 * (rank * rank + sum(unions) + 1) * log_geom
        zeta = _finite("zeta", 16 * sigma * geometry * math.sqrt(kappa * d_sc / n))
        noise_hessian = _finite("nu", 2 ** 12 * sigma * geometry * math.sqrt(kappa * xi / n))
        penalty = _finite("lambda", 8 * sigma * geometry * math.sqrt(kappa * ell / n))
        approximation = _finite("approximation perturbation", math.sqrt(kappa) * geometry * b_alg)
        inverse_step = _finite("L_alg", 2 * (kappa * geometry_sq
            + 2 * kappa * math.sqrt(rank) * margins.d_upper * geometry
            + noise_hessian + approximation + 1), positive=True)
        return dict(t=t, l=unions, m=rows, h_m=hm, kappa_upper=kappa, D_sc=d_sc,
                    Xi=xi, zeta=zeta, nu=noise_hessian, penalty=penalty,
                    approximation_perturbation=approximation, L_alg=inverse_step)

    if config.support_enlargement is None:
        saturated = constants(ambient)
        upper_l = saturated["L_alg"]
        log_enlargement = _log1pexp(2 * (math.log(4) + math.log(upper_l) - math.log(mu)))
        enlargement_rule = "saturated-support upper bound on L_alg"
    else:
        upper_l = None
        log_enlargement = math.log(float(config.support_enlargement))
        enlargement_rule = "user-declared support enlargement"
    if config.support_enlargement is None:
        support = tuple(_support_cap(m, s, log_enlargement)
                        for m, s in zip(ambient, reference))
    else:
        support = tuple(_explicit_support_cap(m, s, float(config.support_enlargement))
                        for m, s in zip(ambient, reference))
    values = constants(support)
    gamma_blocks = tuple(
        0.0 if k == m or (k == s == 0) else math.sqrt(s / (k - s)) if k > s
        else math.inf for m, k, s in zip(ambient, support, reference)
    )
    gamma = max(gamma_blocks)
    _finite("thresholding gamma", gamma)
    rg = _finite("R_g", min(1 / 32, margins.anchor_min / (32 * geometry),
        margins.d_lower / (32 * geometry), margins.gap / (32 * geometry),
        alpha / (8 * geometry)), positive=True)
    # Division order avoids an otherwise unnecessary overflow in 4*kappa*G^2.
    radius = _finite("R", min(rg, margins.trial_radius / 4,
        (m0 / geometry_sq) / (4 * values["kappa_upper"])), positive=True)
    rx = _finite("r_x", (2 * (1 + gamma) / mu) * (
        values["zeta"] + values["approximation_perturbation"]
        + values["penalty"] * math.sqrt(sum(values["t"]))))
    primitive = {
        "approximation": beta <= radius / 16,
        "curvature": values["nu"] + values["approximation_perturbation"] <= m0 / 4,
        "thresholding": gamma <= (mu / values["L_alg"]) / 4,
        "final_neighborhood": rx <= radius / 4,
    }
    if config.support_enlargement is not None:
        enlargement = float(config.support_enlargement)
    else:
        enlargement = (math.exp(log_enlargement)
                       if log_enlargement < math.log(sys.float_info.max) else None)
    diagnostics.update(values)
    diagnostics.update({
        "prescribed_calibration": True, "lambda_b": lambda_b,
        "correction_budgets": corrections, "reference_support": reference,
        "beta_blocks": beta_blocks, "beta": beta, "A_0": a0, "G": geometry,
        "alpha": alpha, "m_0": m0, "mu": mu, "b_alg": b_alg,
        "R_g": rg, "R": radius, "gamma": gamma, "r_x": rx,
        "support_enlargement": enlargement, "log_support_enlargement": log_enlargement,
        "support_enlargement_rule": enlargement_rule, "L_saturated_upper": upper_l,
        "support_saturated": tuple(k == m for k, m in zip(support, ambient)),
        "primitive_checks": primitive, "all_primitive_checks": all(primitive.values()),
        "calibration_note": (
            "Numerical sufficient-constant prescription only. Primitive checks "
            "and source-gap comparisons do not certify unknown model assumptions."
        ),
    })
    return ResolvedCalibration(init_penalty, values["penalty"], values["L_alg"],
                               support, "prescribed", diagnostics)
