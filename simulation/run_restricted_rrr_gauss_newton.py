"""Run full-sample restricted RRR followed by quotient Gauss--Newton paths.

This pilot is intentionally narrower than the complete BI-SMART procedure.  It
uses the same full target sample for the exact restricted-RRR initialization
and every refinement step.  For each requested source weight ``omega`` it
runs an independent safeguarded Gauss--Newton path and saves every accepted
iterate, from ``t=0`` through convergence or the hard cap ``T``.  The
simulation truth is used only to report Frobenius errors; it never selects an
omega or an iteration.

There is no source screening, Wedin gate, validation split, candidate
selection, target-only fallback, or cross-weight safeguard in this runner.
Consequently a failed omega path is recorded locally and does not remove the
restricted-RRR ``t=0`` estimate or affect another omega.

Examples
--------
Run the default five-weight grid for every Model-I sample-size setting::

    python run_restricted_rrr_gauss_newton.py 0 0 0

Run two explicit weights for the first setting only::

    python run_restricted_rrr_gauss_newton.py 0 0 0 \
        --setting-index 0 --omega 1 --omega 10
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np

from run_restricted_rrr import (
    DEFAULT_OUTPUT_ROOT,
    EXPERIMENT_NAMES,
    MODEL_NAMES,
    SimulationSetting,
    _atomic_pickle_dump,
    _plain_value,
    experiment_settings,
    load_experiment_seeds,
)


DEFAULT_SOURCE_WEIGHTS = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True, slots=True)
class GaussNewtonPilotControls:
    """Explicit numerical controls for the deliberately small pilot path.

    These are process choices, not theoretically selected tuning parameters.
    The trust radius is the observable initialization scale

    ``sqrt(||C_rrr||_F^2 + omega ||C0_rank-source_rank||_F^2)``.

    The matrix-free backend is the default because the dense correctness
    reference is not practical for the larger legacy simulation models.
    """

    iteration_cap: int = 10
    initial_step_size: float = 1.0
    contraction: float = 0.5
    armijo_constant: float = 1e-4
    max_backtracking_trials: int = 20
    trust_radius_multiplier: float = 1.0
    solver_backend: str = "matrix_free"
    matrix_free_max_iterations: int | None = None
    atol: float = 1e-12
    rtol: float = 1e-10
    convergence_stopping: bool = True
    stopping_atol: float = 1e-12
    stopping_rtol: float = 1e-10

    def __post_init__(self) -> None:
        integer_fields = (
            ("iteration_cap", self.iteration_cap),
            ("max_backtracking_trials", self.max_backtracking_trials),
        )
        for name, value in integer_fields:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.initial_step_size) or self.initial_step_size <= 0:
            raise ValueError("initial_step_size must be finite and positive")
        if not math.isfinite(self.contraction) or not 0 < self.contraction < 1:
            raise ValueError("contraction must lie strictly between zero and one")
        if not math.isfinite(self.armijo_constant) or not 0 < self.armijo_constant < 1:
            raise ValueError("armijo_constant must lie strictly between zero and one")
        if (
            not math.isfinite(self.trust_radius_multiplier)
            or self.trust_radius_multiplier <= 0
        ):
            raise ValueError("trust_radius_multiplier must be finite and positive")
        backend = self.solver_backend.strip().lower().replace("-", "_")
        if backend not in {"dense", "matrix_free", "auto"}:
            raise ValueError("solver_backend must be dense, matrix_free, or auto")
        iterative_cap = self.matrix_free_max_iterations
        if iterative_cap is not None and (
            isinstance(iterative_cap, (bool, np.bool_))
            or not isinstance(iterative_cap, (int, np.integer))
            or int(iterative_cap) < 1
        ):
            raise ValueError("matrix_free_max_iterations must be positive or None")
        for name, value in (("atol", self.atol), ("rtol", self.rtol)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.convergence_stopping, (bool, np.bool_)):
            raise ValueError("convergence_stopping must be boolean")
        for name, value in (
            ("stopping_atol", self.stopping_atol),
            ("stopping_rtol", self.stopping_rtol),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        object.__setattr__(self, "iteration_cap", int(self.iteration_cap))
        object.__setattr__(
            self, "max_backtracking_trials", int(self.max_backtracking_trials)
        )
        object.__setattr__(self, "solver_backend", backend)
        object.__setattr__(
            self, "convergence_stopping", bool(self.convergence_stopping)
        )
        if iterative_cap is not None:
            object.__setattr__(self, "matrix_free_max_iterations", int(iterative_cap))


def _positive_source_weight(value: Any) -> float:
    """Normalize one source weight without silently accepting booleans/NaNs."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("omega must be a positive finite real number")
    try:
        omega = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("omega must be a positive finite real number") from error
    if not math.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be a positive finite real number")
    return omega


def _gauss_newton_stopping_test(
    *,
    objective: float,
    gauss_newton_norm_squared: float,
    controls: GaussNewtonPilotControls,
) -> tuple[bool, float, float]:
    """Return the observable model-decrement convergence decision.

    For the least-squares objective ``F = ||R||^2 / 2``, a certified
    Gauss--Newton direction ``xi`` has full-step quadratic-model reduction

    ``Delta_GN = <xi, xi>_GN / 2 = ||J xi||^2 / 2``.

    Comparing this gauge-invariant quantity with an absolute-plus-relative
    objective tolerance tests stationarity before line search.  In particular,
    a tiny accepted step cannot masquerade as convergence, and an Armijo test
    need not distinguish a numerically zero reduction from floating-point
    roundoff.  The decision uses only the fitted objective, never simulation
    truth.
    """

    objective = float(objective)
    norm_squared = float(gauss_newton_norm_squared)
    if not math.isfinite(objective) or objective < 0.0:
        raise RuntimeError("the current joint objective must be finite and nonnegative")
    if not math.isfinite(norm_squared) or norm_squared < 0.0:
        raise RuntimeError(
            "the Gauss--Newton squared norm must be finite and nonnegative"
        )
    predicted_reduction = 0.5 * norm_squared
    threshold = controls.stopping_atol + controls.stopping_rtol * objective
    converged = bool(
        controls.convergence_stopping and predicted_reduction <= threshold
    )
    return converged, predicted_reduction, threshold


def result_path(
    output_root: Path,
    *,
    model: str,
    experiment: str,
    setting: SimulationSetting,
    seed_id: int,
    omega: float | None,
    omega_rule: str = "explicit",
) -> Path:
    """Return a distinct path that cannot overwrite restricted-RRR results."""

    if omega is None:
        omega_label = "infinite"
    else:
        omega_label = format(_positive_source_weight(omega), ".15g")
    rule_label = "" if omega_rule == "explicit" else f"_omega-rule={omega_rule}"
    filename = (
        f"RestrictedRRRGN_result_{model}_{experiment}_{setting.suffix}_"
        f"omega={omega_label}{rule_label}_rd_seed_id={seed_id}.pkl"
    )
    return output_root / model / experiment / filename


def _load_initial_implementations() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Load data generation and restricted RRR lazily for dry-run support."""

    from smart import generate_data
    from bi_smart import restricted_rrr

    return generate_data, restricted_rrr


def _run_gauss_newton_path(
    *,
    initial_state: Any,
    initial_coefficient: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    observed_source: np.ndarray,
    omega: float,
    controls: GaussNewtonPilotControls,
) -> dict[str, Any]:
    """Run one fixed-weight path and convert recoverable failures to data.

    PSEUDOCODE
    ----------
    1. Compute the observable scale from the exact RRR target and truncated
       source fits; use it as the trust radius for this omega.
    2. Store restricted RRR as iteration zero and freeze its core thresholds.
    3. At each iteration solve the quotient Gauss--Newton least-squares
       problem with the configured backend.
    4. Before line search, stop successfully when half the squared
       Gauss--Newton norm is negligible relative to the current objective.
    5. Otherwise try ``eta, eta*beta, ...`` until the finite-path certificate
       and
       Armijo decrease hold, then retract all Stiefel factors by polar maps.
    6. Stop only this omega on a certified numerical/line-search failure;
       retain iteration zero and every already accepted iterate for audit.
    """

    # Imports stay local so listing a dry run does not require either package
    # to have been installed into the active interpreter.
    from bi_smart.refinement import (
        RefinementError,
        default_core_thresholds,
        fitted_source,
        fitted_target,
        joint_objective,
        safeguarded_backtracking,
        solve_quotient_gauss_newton,
    )
    from bi_smart.types import FailureReason, NumericalFailure

    omega = _positive_source_weight(omega)
    initial_target = np.asarray(initial_coefficient, dtype=float)
    source_approximation = fitted_source(initial_state)
    radius = controls.trust_radius_multiplier * float(
        np.sqrt(
            np.sum(initial_target * initial_target)
            + omega * np.sum(source_approximation * source_approximation)
        )
    )
    result: dict[str, Any] = {
        "omega": omega,
        "status": "pending",
        "success": False,
        "failure_reason": None,
        "failure_message": None,
        "converged": False,
        "termination_reason": None,
        "last_convergence_check": None,
        "trust_radius": radius,
        "completed_iterations": 0,
        "requested_iterations": controls.iteration_cap,
        "iterates": [
            {
                "iteration": 0,
                "C_hat": initial_target.copy(),
                "objective": joint_objective(
                    initial_state, X, Y, observed_source, omega
                ),
                "accepted_step_size": None,
                "backtracking_trials": 0,
                "gauss_newton_diagnostics": None,
            }
        ],
    }

    try:
        thresholds = default_core_thresholds(
            initial_state,
            radius=radius,
            atol=controls.atol,
            rtol=controls.rtol,
        )
    except RefinementError as failure:
        result.update(
            status="failed",
            termination_reason="failure",
            failure_reason=FailureReason.INVALID_INITIAL_THRESHOLDS.value,
            failure_message=str(failure),
        )
        return result

    current_state = initial_state
    for iteration in range(1, controls.iteration_cap + 1):
        try:
            gauss_newton = solve_quotient_gauss_newton(
                current_state,
                X,
                Y,
                observed_source,
                omega,
                atol=controls.atol,
                rtol=controls.rtol,
                solver_backend=controls.solver_backend,
                matrix_free_max_iterations=controls.matrix_free_max_iterations,
            )
        except NumericalFailure as failure:
            result.update(
                status="failed",
                termination_reason="failure",
                failure_reason=_plain_value(failure.reason),
                failure_message=str(failure),
            )
            return result
        except np.linalg.LinAlgError as failure:
            result.update(
                status="failed",
                termination_reason="failure",
                failure_reason=FailureReason.SINGULAR_NORMAL_EQUATIONS.value,
                failure_message=str(failure),
            )
            return result
        except RefinementError as failure:
            result.update(
                status="failed",
                termination_reason="failure",
                failure_reason=FailureReason.IRREGULAR_ITERATE.value,
                failure_message=str(failure),
            )
            return result

        # The GN decrement is evaluated at the current state and before any
        # step-size contraction.  It is therefore a stationarity diagnostic,
        # unlike a small accepted displacement that might merely reflect a
        # path-safety or line-search obstruction.
        current_objective = float(result["iterates"][-1]["objective"])
        converged, predicted_reduction, stopping_threshold = (
            _gauss_newton_stopping_test(
                objective=current_objective,
                gauss_newton_norm_squared=(
                    gauss_newton.diagnostics.gauss_newton_norm_squared
                ),
                controls=controls,
            )
        )
        result["last_convergence_check"] = {
            "state_iteration": iteration - 1,
            "objective": current_objective,
            "gauss_newton_norm_squared": (
                gauss_newton.diagnostics.gauss_newton_norm_squared
            ),
            "predicted_reduction": predicted_reduction,
            "threshold": stopping_threshold,
            "stopping_enabled": controls.convergence_stopping,
            "criterion_met": predicted_reduction <= stopping_threshold,
        }
        if converged:
            result.update(
                status="successful",
                success=True,
                converged=True,
                termination_reason="gauss_newton_decrement",
            )
            return result

        try:
            backtracking = safeguarded_backtracking(
                current_state,
                gauss_newton.direction,
                initial_state,
                X,
                Y,
                observed_source,
                omega=omega,
                thresholds=thresholds,
                initial_step_size=controls.initial_step_size,
                contraction=controls.contraction,
                armijo_constant=controls.armijo_constant,
                max_trials=controls.max_backtracking_trials,
                polar_atol=controls.atol,
                polar_rtol=controls.rtol,
            )
        except NumericalFailure as failure:
            result.update(
                status="failed",
                termination_reason="failure",
                failure_reason=_plain_value(failure.reason),
                failure_message=str(failure),
            )
            return result
        except np.linalg.LinAlgError as failure:
            result.update(
                status="failed",
                termination_reason="failure",
                failure_reason=FailureReason.SINGULAR_NORMAL_EQUATIONS.value,
                failure_message=str(failure),
            )
            return result
        except RefinementError as failure:
            result.update(
                status="failed",
                termination_reason="failure",
                failure_reason=FailureReason.IRREGULAR_ITERATE.value,
                failure_message=str(failure),
            )
            return result

        if not backtracking.accepted or backtracking.state is None:
            result.update(
                status="failed",
                termination_reason="failure",
                failure_reason=FailureReason.LINE_SEARCH.value,
                failure_message=(
                    backtracking.failure_reason
                    or "No safeguarded Armijo trial was accepted."
                ),
            )
            return result

        current_state = backtracking.state
        coefficient = fitted_target(current_state)
        result["iterates"].append(
            {
                "iteration": iteration,
                "C_hat": coefficient,
                "objective": joint_objective(
                    current_state, X, Y, observed_source, omega
                ),
                "accepted_step_size": backtracking.step_size,
                "backtracking_trials": backtracking.trials,
                "gauss_newton_diagnostics": (
                    gauss_newton.diagnostics.as_metadata()
                ),
            }
        )
        result["completed_iterations"] = iteration

    result.update(
        status="successful",
        success=True,
        converged=False,
        termination_reason="iteration_cap",
    )
    return result


def _base_result(
    *,
    setting: SimulationSetting,
    model: str,
    experiment: str,
    seed_id: int,
    random_seed: int,
    omega: float | None,
    omega_rule: str,
    controls: GaussNewtonPilotControls,
    coefficient_true: np.ndarray,
) -> dict[str, Any]:
    """Create the auditable result record before either fitting stage."""

    return {
        "result_schema_version": 3,
        "C_true": coefficient_true,
        "C_hat": None,
        "avg_err": None,
        "elapsed_time_sec": None,
        "method": "full_sample_restricted_rrr_gauss_newton",
        "status": "pending",
        "success": False,
        "refinement_success": None,
        "refinement_converged": None,
        "refinement_termination_reason": None,
        "used_t0_fallback": False,
        "applicable": True,
        "failure_reason": None,
        "failure_message": None,
        "target_rank": setting.target_rank,
        "source_rank": setting.source_rank,
        "omega": omega,
        "omega_rule": omega_rule,
        "omega_is_simulation_oracle": omega_rule == "gaussian-noise",
        "controls": asdict(controls),
        "initialization_time_sec": 0.0,
        "refinement_time_sec": 0.0,
        "restricted_rrr_C_hat": None,
        "restricted_rrr_avg_err": None,
        "restricted_rrr_diagnostics": {},
        "path": None,
        "refinement_skipped_reason": None,
        "model": model,
        "experiment": experiment,
        "setting_suffix": setting.suffix,
        "setting": asdict(setting),
        "rd_seed_id": seed_id,
        "random_seed": random_seed,
        "generator_arguments": {
            "n": setting.n,
            "p": setting.p,
            "q": setting.q,
            "sigma0": setting.sigma0,
            "sigma": 0.5,
            "r0_star": 10,
            "r_star": 5,
            "random_seed": random_seed,
        },
    }


def _coefficient_error(
    coefficient: Any,
    coefficient_true: np.ndarray,
    *,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Validate one fitted matrix and compute the paper's reported metric."""

    array = np.asarray(coefficient)
    if array.shape != expected_shape:
        raise RuntimeError(
            f"Fitted coefficient has shape {array.shape}; expected {expected_shape}."
        )
    if np.iscomplexobj(array) or not np.all(np.isfinite(array)):
        raise RuntimeError("Fitted coefficient must be finite and real-valued.")
    error = float(
        np.linalg.norm(array - coefficient_true, ord="fro")
        / np.sqrt(expected_shape[0] * expected_shape[1])
    )
    return np.asarray(array, dtype=float), error


def _check_resume_compatibility(
    destination: Path,
    *,
    setting: SimulationSetting,
    model: str,
    experiment: str,
    seed_id: int,
    random_seed: int,
    omega: float | None,
    omega_rule: str,
    controls: GaussNewtonPilotControls,
) -> None:
    """Reject an existing result produced with different hidden controls.

    Numerical controls are intentionally absent from the compact filename.
    Silently treating a stale file as the requested run would therefore mix
    incomparable paths in a resumed sweep.  Compare every identity/control
    field stored in the pickle and require ``--force`` for replacement.
    """

    try:
        with destination.open("rb") as handle:
            existing = pickle.load(handle)
    except Exception as error:
        raise RuntimeError(
            f"Existing GN result {destination} cannot be read; pass --force "
            "to replace it."
        ) from error
    if not isinstance(existing, Mapping):
        raise RuntimeError(
            f"Existing GN result {destination} is not a result mapping; pass "
            "--force to replace it."
        )

    expected = {
        "result_schema_version": 3,
        "method": "full_sample_restricted_rrr_gauss_newton",
        "model": model,
        "experiment": experiment,
        "setting": asdict(setting),
        "rd_seed_id": seed_id,
        "random_seed": random_seed,
        "omega": omega,
        "omega_rule": omega_rule,
        "controls": asdict(controls),
    }
    mismatches = tuple(
        name for name, value in expected.items() if existing.get(name) != value
    )
    if mismatches:
        fields = ", ".join(mismatches)
        raise RuntimeError(
            f"Existing GN result {destination} does not match the requested "
            f"run ({fields}); pass --force to replace it."
        )


def run_setting(
    *,
    setting: SimulationSetting,
    model: str,
    experiment: str,
    seed_id: int,
    random_seed: int,
    omega: float | None,
    omega_rule: str = "explicit",
    controls: GaussNewtonPilotControls,
    destination: Path,
    force: bool = False,
    generate_data_fn: Callable[..., Mapping[str, Any]] | None = None,
    restricted_rrr_fn: Callable[..., Any] | None = None,
    refinement_path_fn: Callable[..., dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Generate and save one setting/omega path without oracle selection."""

    if omega is None:
        if omega_rule != "gaussian-noise" or setting.sigma0 != 0.0:
            raise ValueError(
                "omega=None is reserved for the exact-source gaussian-noise rule"
            )
    else:
        omega = _positive_source_weight(omega)
    if destination.exists() and not force:
        _check_resume_compatibility(
            destination,
            setting=setting,
            model=model,
            experiment=experiment,
            seed_id=seed_id,
            random_seed=random_seed,
            omega=omega,
            omega_rule=omega_rule,
            controls=controls,
        )
        return "skipped", None
    if generate_data_fn is None or restricted_rrr_fn is None:
        default_generator, default_initializer = _load_initial_implementations()
        if generate_data_fn is None:
            generate_data_fn = default_generator
        if restricted_rrr_fn is None:
            restricted_rrr_fn = default_initializer
    if refinement_path_fn is None:
        refinement_path_fn = _run_gauss_newton_path

    started = time.perf_counter()
    data = generate_data_fn(
        n=setting.n,
        p=setting.p,
        q=setting.q,
        sigma0=setting.sigma0,
        random_seed=random_seed,
    )
    X = np.asarray(data["X"])
    Y = np.asarray(data["Y"])
    coefficient_true = np.asarray(data["C_star"])
    observed_source = np.asarray(data["C0"])
    result = _base_result(
        setting=setting,
        model=model,
        experiment=experiment,
        seed_id=seed_id,
        random_seed=random_seed,
        omega=omega,
        omega_rule=omega_rule,
        controls=controls,
        coefficient_true=coefficient_true,
    )

    inapplicability = setting.inapplicability_reason()
    if inapplicability is not None:
        result.update(
            status="inapplicable",
            applicable=False,
            failure_reason=inapplicability,
            failure_message=(
                "Restricted-RRR initialization requires 1 <= target_rank <= "
                "source_rank <= min(p, q); ranks are never clamped."
            ),
        )
    else:
        initialization_started = time.perf_counter()
        initialization = restricted_rrr_fn(
            X,
            Y,
            observed_source,
            target_rank=setting.target_rank,
            source_rank=setting.source_rank,
            atol=controls.atol,
            rtol=controls.rtol,
        )
        result["initialization_time_sec"] = (
            time.perf_counter() - initialization_started
        )
        result["restricted_rrr_diagnostics"] = _plain_value(
            initialization.candidate.metadata
        )
        if not initialization.successful:
            result.update(
                status="initialization_failed",
                failure_reason=_plain_value(
                    initialization.candidate.failure_reason
                ),
                failure_message=initialization.candidate.message,
            )
        else:
            initial_coefficient = initialization.coefficient
            if initial_coefficient is None or initialization.state is None:
                raise RuntimeError(
                    "restricted_rrr reported success without coefficient/state."
                )
            initial_array, initial_error = _coefficient_error(
                initial_coefficient,
                coefficient_true,
                expected_shape=(setting.p, setting.q),
            )
            result["restricted_rrr_C_hat"] = initial_array
            result["restricted_rrr_avg_err"] = initial_error

            # Under the Gaussian noise-ratio rule, sigma0=0 corresponds to an
            # infinite source weight.  A finite GN solve is neither necessary
            # nor a faithful approximation: the source basis is pinned.  Save
            # the exact RRR anchor as the only path entry and label the skip.
            if omega is None:
                result["path"] = {
                    "omega": None,
                    "status": "skipped_exact_source",
                    "success": True,
                    "failure_reason": None,
                    "failure_message": None,
                    "converged": False,
                    "termination_reason": "source_noise_is_zero",
                    "last_convergence_check": None,
                    "trust_radius": None,
                    "completed_iterations": 0,
                    "requested_iterations": controls.iteration_cap,
                    "iterates": [
                        {
                            "iteration": 0,
                            "C_hat": initial_array,
                            "avg_err": initial_error,
                            "objective": None,
                            "accepted_step_size": None,
                            "backtracking_trials": 0,
                            "gauss_newton_diagnostics": None,
                        }
                    ],
                }
                result.update(
                    C_hat=initial_array,
                    avg_err=initial_error,
                    status="successful_t0_only",
                    success=True,
                    refinement_success=None,
                    refinement_converged=None,
                    refinement_termination_reason="source_noise_is_zero",
                    used_t0_fallback=False,
                    refinement_skipped_reason="source_noise_is_zero",
                )
                result["elapsed_time_sec"] = time.perf_counter() - started
                _atomic_pickle_dump(result, destination)
                return "written", result

            refinement_started = time.perf_counter()
            path = refinement_path_fn(
                initial_state=initialization.state,
                initial_coefficient=initial_array,
                X=X,
                Y=Y,
                observed_source=observed_source,
                omega=omega,
                controls=controls,
            )
            result["refinement_time_sec"] = time.perf_counter() - refinement_started

            # PSEUDOCODE: score every accepted iterate against C_star for the
            # simulation table, while preserving chronological order.  Never
            # minimize these errors to choose the reported coefficient.
            plain_path = _plain_value(path)
            iterates = plain_path.get("iterates")
            if not isinstance(iterates, list) or not iterates:
                raise RuntimeError("refinement path must retain iteration t=0")
            for expected_iteration, iterate in enumerate(iterates):
                if iterate.get("iteration") != expected_iteration:
                    raise RuntimeError(
                        "refinement path iterations must be consecutive from zero"
                    )
                coefficient, error = _coefficient_error(
                    iterate.get("C_hat"),
                    coefficient_true,
                    expected_shape=(setting.p, setting.q),
                )
                iterate["C_hat"] = coefficient
                iterate["avg_err"] = error
            result["path"] = plain_path

            if bool(plain_path.get("success")):
                completed_iterations = plain_path.get("completed_iterations")
                if (
                    isinstance(completed_iterations, (bool, np.bool_))
                    or not isinstance(completed_iterations, (int, np.integer))
                    or not 0 <= int(completed_iterations) <= controls.iteration_cap
                ):
                    raise RuntimeError(
                        "successful path has an invalid completed-iteration count"
                    )
                completed_iterations = int(completed_iterations)
                if len(iterates) != completed_iterations + 1:
                    raise RuntimeError(
                        "successful path length does not match its completed iterations"
                    )
                converged = plain_path.get("converged")
                if not isinstance(converged, (bool, np.bool_)):
                    raise RuntimeError("successful path must record convergence status")
                converged = bool(converged)
                termination_reason = plain_path.get("termination_reason")
                if converged:
                    if (
                        termination_reason != "gauss_newton_decrement"
                        or completed_iterations >= controls.iteration_cap
                    ):
                        raise RuntimeError(
                            "early successful path has inconsistent "
                            "convergence metadata"
                        )
                    last_check = plain_path.get("last_convergence_check")
                    if (
                        not isinstance(last_check, Mapping)
                        or last_check.get("state_iteration") != completed_iterations
                        or not bool(last_check.get("criterion_met"))
                        or not bool(last_check.get("stopping_enabled"))
                    ):
                        raise RuntimeError(
                            "converged path lacks its terminal decrement certificate"
                        )
                elif (
                    termination_reason != "iteration_cap"
                    or completed_iterations != controls.iteration_cap
                ):
                    raise RuntimeError(
                        "nonconverged successful path did not reach the iteration cap"
                    )
                endpoint = iterates[-1]
                result.update(
                    C_hat=endpoint["C_hat"],
                    avg_err=endpoint["avg_err"],
                    status="successful",
                    success=True,
                    refinement_success=True,
                    refinement_converged=converged,
                    refinement_termination_reason=termination_reason,
                    used_t0_fallback=False,
                )
            else:
                # A failed refinement call does not erase its exact RRR anchor.
                # Publish t=0 as the usable estimator while retaining both the
                # failed/partial path and its precise reason for diagnosis.
                result.update(
                    C_hat=initial_array,
                    avg_err=initial_error,
                    status="successful_t0_fallback",
                    success=True,
                    refinement_success=False,
                    refinement_converged=False,
                    refinement_termination_reason=(
                        plain_path.get("termination_reason") or "failure"
                    ),
                    used_t0_fallback=True,
                    failure_reason=plain_path.get("failure_reason"),
                    failure_message=plain_path.get("failure_message"),
                )

    result["elapsed_time_sec"] = time.perf_counter() - started
    _atomic_pickle_dump(result, destination)
    return "written", result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", type=int, choices=range(3))
    parser.add_argument("exp_id", type=int, choices=range(4))
    parser.add_argument("rd_seed_id", type=int, choices=range(100))
    parser.add_argument("--setting-index", type=int)
    parser.add_argument(
        "--omega",
        dest="omegas",
        type=_positive_source_weight,
        action="append",
        help=(
            "Positive source weight; repeat for a grid. The default is "
            "0.01, 0.1, 1, 10, 100."
        ),
    )
    parser.add_argument(
        "--omega-rule",
        choices=("gaussian-noise",),
        help=(
            "Use sigma^2/(n*sigma0^2), with known simulation sigma=0.5. "
            "This is simulation-oracle noise calibration, not truth selection."
        ),
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--initial-step-size", type=float, default=1.0)
    parser.add_argument("--contraction", type=float, default=0.5)
    parser.add_argument("--armijo-constant", type=float, default=1e-4)
    parser.add_argument("--max-backtracking-trials", type=int, default=20)
    parser.add_argument("--trust-radius-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--stopping-atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for the Gauss--Newton model decrement.",
    )
    parser.add_argument(
        "--stopping-rtol",
        type=float,
        default=1e-10,
        help="Current-objective relative tolerance for the GN model decrement.",
    )
    parser.add_argument(
        "--no-convergence-stopping",
        dest="convergence_stopping",
        action="store_false",
        help="Disable early stationarity stopping and require the iteration cap.",
    )
    parser.set_defaults(convergence_stopping=True)
    parser.add_argument(
        "--solver-backend",
        choices=("matrix_free", "dense", "auto"),
        default="matrix_free",
    )
    parser.add_argument("--matrix-free-max-iterations", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.omega_rule is not None and args.omegas is not None:
        raise SystemExit("use either repeated --omega values or --omega-rule, not both")
    controls = GaussNewtonPilotControls(
        iteration_cap=args.iterations,
        initial_step_size=args.initial_step_size,
        contraction=args.contraction,
        armijo_constant=args.armijo_constant,
        max_backtracking_trials=args.max_backtracking_trials,
        trust_radius_multiplier=args.trust_radius_multiplier,
        solver_backend=args.solver_backend,
        matrix_free_max_iterations=args.matrix_free_max_iterations,
        convergence_stopping=args.convergence_stopping,
        stopping_atol=args.stopping_atol,
        stopping_rtol=args.stopping_rtol,
    )
    explicit_omegas = tuple(
        DEFAULT_SOURCE_WEIGHTS if args.omegas is None else args.omegas
    )
    if args.omega_rule is None and len(set(explicit_omegas)) != len(explicit_omegas):
        raise SystemExit("--omega values must not contain duplicates")

    settings = experiment_settings(args.model_id, args.exp_id)
    if args.setting_index is None:
        indexed_settings = tuple(enumerate(settings))
    else:
        if not 0 <= args.setting_index < len(settings):
            raise SystemExit(
                f"--setting-index must be between 0 and {len(settings) - 1}."
            )
        indexed_settings = ((args.setting_index, settings[args.setting_index]),)

    seeds = load_experiment_seeds()
    if args.rd_seed_id >= len(seeds):
        raise SystemExit(
            f"rd_seed_id={args.rd_seed_id} is unavailable; seed file has "
            f"{len(seeds)} entries."
        )
    random_seed = int(seeds[args.rd_seed_id])
    model = MODEL_NAMES[args.model_id]
    experiment = EXPERIMENT_NAMES[args.exp_id]
    print(
        f"Restricted RRR + GN: model={model}, experiment={experiment}, "
        f"rd_seed_id={args.rd_seed_id}, random_seed={random_seed}"
    )

    for setting_index, setting in indexed_settings:
        if args.omega_rule == "gaussian-noise":
            # The DGP supplies target noise SD sigma=0.5 and source coefficient
            # noise SD sigma0.  This likelihood-scale ratio uses known
            # simulation noise parameters, so it is explicitly oracle-calibrated
            # even though it never inspects C_star.
            setting_omegas: tuple[float | None, ...] = (
                None
                if setting.sigma0 == 0.0
                else 0.5**2 / (setting.n * setting.sigma0**2),
            )
            omega_rule = "gaussian-noise"
        else:
            setting_omegas = explicit_omegas
            omega_rule = "explicit"
        for omega in setting_omegas:
            destination = result_path(
                args.output_root,
                model=model,
                experiment=experiment,
                setting=setting,
                seed_id=args.rd_seed_id,
                omega=omega,
                omega_rule=omega_rule,
            )
            applicability = setting.inapplicability_reason() or "applicable"
            print(
                f"[{setting_index}] {setting.suffix}, "
                f"omega={'infinite' if omega is None else format(omega, 'g')}: "
                f"r={setting.target_rank}, r_s={setting.source_rank} "
                f"({applicability}) -> {destination}"
            )
            if args.dry_run:
                continue
            outcome, result = run_setting(
                setting=setting,
                model=model,
                experiment=experiment,
                seed_id=args.rd_seed_id,
                random_seed=random_seed,
                omega=omega,
                omega_rule=omega_rule,
                controls=controls,
                destination=destination,
                force=args.force,
            )
            if outcome == "skipped":
                print("    skipped (result exists; pass --force to replace it)")
            else:
                assert result is not None
                endpoint = result["avg_err"]
                suffix = (
                    f", avg_err={endpoint:.6g}"
                    if endpoint is not None and np.isfinite(endpoint)
                    else ""
                )
                print(f"    wrote status={result['status']}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
