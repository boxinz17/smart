"""Run validation-tuned SparseSMART alone on the existing V1 simulation seeds.

Candidate selection uses only target training/validation data and the observed
source. Coefficient truth is read after fitting for evaluation and provenance.
The selected fit remains a training-subset fit; there is no full-data refit.
Paper baselines must be read separately from the existing figure references.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np

import run_sparse_smart as fixed_runner
import run_restricted_rrr as simulation_grid
from run_sparse_smart import MARGINS, _atomic_json_dump, _coefficient_error, _digest_json, _json_value
from run_restricted_rrr import (
    DEFAULT_OUTPUT_ROOT, DEFAULT_SEED_FILE, EXPERIMENT_NAMES, MODEL_NAMES,
    SimulationSetting, experiment_settings, load_experiment_seeds,
)


@dataclass(frozen=True)
class RunnerConfig:
    iterations: int = 500
    init_penalties: tuple[float, ...] = (.03,)
    penalties_u: tuple[float, ...] = (.0025, .01, .04)
    penalties_v: tuple[float, ...] = (.0025, .01, .04)
    support_limits: tuple[tuple[int, int], ...] | None = None
    inverse_step: float = 20.
    validation_fraction: float = .2
    split_seed: int = 0
    strict_source_check: bool = False
    spectral_step: str = "projected"
    stationarity_tol: float | None = 1e-6
    iteration_budgets: tuple[int, ...] | None = None

    def validate(self):
        for name, value in (("iterations", self.iterations), ("split_seed", self.split_seed)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.iteration_budgets is not None:
            budgets = self.iteration_budgets
            if (not isinstance(budgets, (tuple, list)) or not budgets
                    or any(type(b) is not int or b <= 0 for b in budgets)
                    or any(left >= right for left, right in zip(budgets, budgets[1:]))
                    or budgets[-1] != self.iterations):
                raise ValueError("iteration_budgets must be positive, strictly increasing, and end at iterations")
        for name in ("init_penalties", "penalties_u", "penalties_v"):
            grid = getattr(self, name)
            if not grid:
                raise ValueError(f"{name} must be a nonempty grid")
            if any(not math.isfinite(v) or v < 0 or (name == "init_penalties" and v == 0)
                   for v in grid):
                raise ValueError(f"{name} must contain finite {'positive' if name == 'init_penalties' else 'nonnegative'} values")
            if len(set(grid)) != len(grid):
                raise ValueError(f"{name} contains duplicate values")
        if not math.isfinite(self.inverse_step) or self.inverse_step <= 0:
            raise ValueError("inverse_step must be finite and positive")
        if not math.isfinite(self.validation_fraction) or not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie strictly between zero and one")
        if self.stationarity_tol is not None and (
            not math.isfinite(self.stationarity_tol) or self.stationarity_tol <= 0
        ):
            raise ValueError("stationarity_tol must be positive or None")
        if self.spectral_step not in ("projected", "reject"):
            raise ValueError("spectral_step must be projected or reject")
        if not isinstance(self.strict_source_check, bool):
            raise ValueError("strict_source_check must be boolean")
        if self.support_limits is not None:
            if not self.support_limits:
                raise ValueError("support_limits must be a nonempty grid or None")
            for pair in self.support_limits:
                if len(pair) != 2 or any(isinstance(k, bool) or not isinstance(k, int) or k < 0 for k in pair):
                    raise ValueError("support_limits must contain pairs of nonnegative integers")
            if len(set(self.support_limits)) != len(self.support_limits):
                raise ValueError("support_limits contains duplicate pairs")


def resolved_configuration(setting, config):
    config.validate()
    rank, source_rank = setting.target_rank, setting.source_rank
    nu, nv = ((source_rank, source_rank) if setting.sigma0 == 0 else (setting.p, setting.q))
    maximum = (max(0, (nu - rank) * rank), max(0, (nv - rank) * rank))
    supports = config.support_limits if config.support_limits is not None else (maximum,)
    if setting.inapplicability_reason() is None:
        if any(ku > maximum[0] or kv > maximum[1] for ku, kv in supports):
            raise ValueError(f"support_limits exceed actual complement dimensions {maximum}")
    sparsity = min(source_rank * rank, max(rank, 5))
    return dict(
        runner=asdict(config), margins=MARGINS.copy(), rank=rank, source_rank=source_rank,
        sparsity=[sparsity, sparsity], support_limits=[list(pair) for pair in supports],
        actual_complement_counts=list(maximum),
        candidate_count=len(config.iteration_budgets or (config.iterations,)) * len(config.init_penalties) * len(config.penalties_u) * len(config.penalties_v) * len(supports),
        selection_metric="mean((Y_validation - X_validation @ C_hat)**2)",
        selection_inputs=["X", "Y", "observed_source"], tuning_uses_truth=False,
        fit_sample="training_subset", refit_on_all_data=False,
        source_mode="exact_observed_svd" if setting.sigma0 == 0 else "noisy_observed_coefficient",
        source_gap_lower=None if setting.sigma0 == 0 else 1.,
        calibration_label="validation_tuned_practical_strict_source_check" if config.strict_source_check
            else "validation_tuned_practical_empirical_source_check",
        rank_semantics="fitted_dimensions; generator truth remains rank 5 and source rank 10",
    )


def _array_fingerprint(data, names):
    digest = hashlib.sha256()
    for name in names:
        value = np.ascontiguousarray(data[name])
        digest.update(json.dumps((name, value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _implementation_fingerprint(api, generator):
    paths = {Path(__file__), Path(fixed_runner.__file__), Path(simulation_grid.__file__)}
    if getattr(api, "__file__", None):
        paths.update(Path(api.__file__).resolve().parent.rglob("*.py"))
    generator_file = inspect.getsourcefile(generator)
    if generator_file is not None:
        paths.add(Path(generator_file))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.resolve()).encode())
        digest.update(path.read_bytes())
    if not getattr(api, "__file__", None):
        digest.update(b"injected_test_api")
    return digest.hexdigest()


def _load_generator():
    return fixed_runner._load_generator()


def _load_sparse_api():
    import sparse_smart
    return sparse_smart


def result_path(output_root, *, model, experiment, setting, seed_id):
    return Path(output_root) / model / experiment / (
        f"SparseSMARTTuned_result_{model}_{experiment}_{setting.suffix}_rd_seed_id={seed_id}.json"
    )


def _split_metadata(tuner, n):
    train = getattr(tuner, "train_indices_", None)
    validation = getattr(tuner, "validation_indices_", None)
    if train is None or validation is None:
        raise RuntimeError("Internal-split tuner did not expose its training/validation indices")
    train, validation = np.asarray(train), np.asarray(validation)
    if (train.ndim != 1 or validation.ndim != 1 or not train.size or not validation.size
            or not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(validation.dtype, np.integer)
            or sorted(np.concatenate((train, validation)).tolist()) != list(range(n))):
        raise RuntimeError("Tuner returned invalid, overlapping, or incomplete split indices")
    indices = dict(train_indices=train.tolist(), validation_indices=validation.tolist())
    return dict(**indices, n_train=int(train.size), n_validation=int(validation.size),
                fingerprint=_digest_json(indices), refit_on_all_data=False)


def run_setting(*, setting: SimulationSetting, model: str, experiment: str, seed_id: int,
                random_seed: int, destination: Path, config: RunnerConfig = RunnerConfig(),
                force: bool = False, generate_data_fn=None, sparse_api=None):
    """Write one tuned fit or safely resume an identical data/configuration/code cell."""
    started = time.perf_counter()
    resolved = resolved_configuration(setting, config)
    arguments = dict(n=setting.n, p=setting.p, q=setting.q, sigma0=setting.sigma0,
                     sigma=.5, r_star=5, r0_star=10, random_seed=int(random_seed))
    identity = dict(schema_version=1, method="SparseSMARTTuned", model=model,
                    experiment=experiment, rd_seed_id=seed_id, random_seed=int(random_seed),
                    setting=asdict(setting), configuration=resolved, generator_arguments=arguments)
    config_hash = _digest_json(identity)
    destination = Path(destination)
    existing = None
    if destination.exists() and not force:
        try:
            existing = json.loads(destination.read_text())
        except (ValueError, OSError) as error:
            raise ValueError(f"Cannot validate existing result {destination}; use --force") from error
        if existing.get("configuration_fingerprint") != config_hash:
            raise ValueError(f"Existing result configuration differs at {destination}; use --force")
    generator = generate_data_fn if generate_data_fn is not None else _load_generator()
    data = generator(**arguments)
    observed_hash = _array_fingerprint(data, ("X", "Y", "C0"))
    reason = setting.inapplicability_reason()
    if sparse_api is None and reason is None:
        sparse_api = _load_sparse_api()
    implementation_hash = _implementation_fingerprint(sparse_api, generator)
    if existing is not None:
        if existing.get("observed_input_fingerprint") != observed_hash:
            raise ValueError(f"Generated inputs differ from {destination}; use --force")
        if existing.get("evaluation_truth_fingerprint") != _array_fingerprint(data, ("C_star",)):
            raise ValueError(f"Evaluation truth differs from {destination}; use --force")
        if existing.get("implementation_fingerprint") != implementation_hash:
            raise ValueError(f"Implementation differs from {destination}; use --force")
        return "skipped", existing
    result = dict(identity, configuration_fingerprint=config_hash,
                  observed_input_fingerprint=observed_hash, implementation_fingerprint=implementation_hash,
                  input_fingerprint=None, evaluation_truth_fingerprint=None,
                  status="inapplicable" if reason else "failed", success=False, applicable=reason is None,
                  estimator_status=None, failure_reason=reason, failure_message=None,
                  all_candidates_failed=False, avg_err=None, initial_avg_err=None, C_hat=None,
                  best_params=None, validation_loss=None, selected_iteration=None, selected_budget=None, selected_candidate_id=None, n_iter=0,
                  selected_supports=None, termination_reason=None, split=None,
                  selection_history=[], fit_errors=[], history=[], validation_history=[], diagnostics={}, tuning_diagnostics={},
                  fit_time_sec=0., elapsed_time_sec=None, theorem_certified=False,
                  source_check_mode="strict" if config.strict_source_check else "empirical",
                  error_metric="norm(C_hat-C_star, fro) / sqrt(p*q)",
                  evaluation_scope="one_replicate_of_requested_pilot; not a 100-replicate paper aggregate")
    if reason is not None:
        result["failure_message"] = "Invalid fitted dimensions are recorded without clamping or fitting."
    else:
        X, Y, C0 = (np.asarray(data[key]) for key in ("X", "Y", "C0"))
        fit_started = time.perf_counter()
        try:
            if setting.sigma0 == 0:
                U, _, Vt = np.linalg.svd(C0, full_matrices=False)
                source = sparse_api.ExactSource(U[:, :setting.source_rank], Vt[:setting.source_rank].T)
            else:
                source = sparse_api.NoisySource(C0, setting.sigma0, gap_lower=1.)
            tuner = sparse_api.SparseSMARTTuner(
                rank=setting.target_rank, source_rank=setting.source_rank,
                sparsity=tuple(resolved["sparsity"]), margins=sparse_api.Margins(**MARGINS),
                init_penalties=config.init_penalties, penalties_u=config.penalties_u,
                penalties_v=config.penalties_v, support_limits=config.support_limits,
                iterations=config.iterations, iteration_budgets=config.iteration_budgets,
                step_size_inverse=config.inverse_step,
                validation_fraction=config.validation_fraction, random_state=config.split_seed,
                enforce_source_accuracy=config.strict_source_check, spectral_step=config.spectral_step,
                stationarity_tol=config.stationarity_tol,
            )
            tuner.fit(X, Y, source=source)
            candidates = _json_value(tuner.selection_history_)
            result.update(estimator_status=tuner.status_, success=bool(tuner.success_),
                          status="complete" if tuner.success_ else "all_candidates_failed",
                          all_candidates_failed=not bool(tuner.success_),
                          failure_reason=None if tuner.success_ else tuner.status_,
                          failure_message=None if tuner.success_ else tuner.message_,
                          best_params=_json_value(getattr(tuner, "best_params_", None)),
                          validation_loss=_json_value(getattr(tuner, "best_score_", None)),
                          selected_iteration=getattr(tuner, "selected_iteration_", None),
                          selected_budget=getattr(tuner, "selected_budget_", config.iterations if tuner.success_ else None),
                          selected_candidate_id=getattr(tuner, "selected_candidate_id_", None),
                          tuning_diagnostics=_json_value(getattr(tuner, "diagnostics_", {})),
                          split=_split_metadata(tuner, X.shape[0]), selection_history=candidates,
                          fit_errors=[entry for entry in candidates if not entry.get("success", False)])
            if tuner.success_:
                if not hasattr(tuner, "coefficient_"):
                    raise RuntimeError("Tuner reported success without a coefficient")
                # Evaluation truth is deliberately first accessed after candidate selection.
                truth = np.asarray(data["C_star"])
                coefficient = np.asarray(tuner.coefficient_)
                result.update(C_hat=coefficient, avg_err=_coefficient_error(coefficient, truth))
                winner = tuner.estimator_
                result.update(n_iter=int(winner.n_iter_), history=_json_value(winner.history_),
                              validation_history=_json_value(getattr(winner, "validation_history_", [])),
                              diagnostics=_json_value(winner.diagnostics_),
                              selected_supports=_json_value(getattr(winner, "supports_", None)),
                              termination_reason=getattr(winner, "termination_reason_", winner.status_))
                if hasattr(winner, "initial_state_"):
                    P, d, Q = winner.chart_.reconstruct(winner.initial_state_)
                    initial = ((winner.source_.left @ P) * d) @ (winner.source_.right @ Q).T
                    result["initial_avg_err"] = _coefficient_error(initial, truth)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            result.update(status="failed", success=False, avg_err=None,
                          failure_reason=type(error).__name__, failure_message=str(error))
        result["fit_time_sec"] = time.perf_counter() - fit_started
    truth_hash = _array_fingerprint(data, ("C_star",))
    result["evaluation_truth_fingerprint"] = truth_hash
    result["input_fingerprint"] = _digest_json(dict(observed=observed_hash, evaluation_truth=truth_hash))
    result["elapsed_time_sec"] = time.perf_counter() - started
    result = _json_value(result)
    _atomic_json_dump(result, destination)
    return "written", result


def _float_grid(value):
    try:
        return tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers") from error


def _support_grid(value):
    try:
        result = tuple(tuple(int(k) for k in pair.split(":")) for pair in value.split(","))
        if any(len(pair) != 2 for pair in result):
            raise ValueError
        return result
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated ku:kv pairs") from error


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", type=int, choices=range(3))
    parser.add_argument("exp_id", type=int, choices=range(4))
    parser.add_argument("rd_seed_id", type=int, choices=range(100))
    parser.add_argument("--setting-index", type=int)
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "sparse_smart_tuned")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--iteration-budgets", type=int, nargs="+",
                        help="Increasing checkpoint budgets ending at --iterations; default uses one budget")
    parser.add_argument("--init-penalties", type=_float_grid, default=(.03,))
    parser.add_argument("--penalties-u", type=_float_grid, default=(.0025, .01, .04))
    parser.add_argument("--penalties-v", type=_float_grid, default=(.0025, .01, .04))
    parser.add_argument("--support-grid", type=_support_grid, help="ku:kv pairs; default is full complement counts")
    parser.add_argument("--inverse-step", type=float, default=20.)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--spectral-step", choices=("projected", "reject"), default="projected")
    parser.add_argument("--stationarity-tol", type=float, default=1e-6)
    parser.add_argument("--strict-source-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None):
    args = _parser().parse_args(argv)
    config = RunnerConfig(iterations=args.iterations,
                          iteration_budgets=tuple(args.iteration_budgets) if args.iteration_budgets else None,
                          init_penalties=args.init_penalties,
                          penalties_u=args.penalties_u, penalties_v=args.penalties_v,
                          support_limits=args.support_grid, inverse_step=args.inverse_step,
                          validation_fraction=args.validation_fraction, split_seed=args.split_seed,
                          strict_source_check=args.strict_source_check, spectral_step=args.spectral_step,
                          stationarity_tol=args.stationarity_tol)
    config.validate()
    settings = experiment_settings(args.model_id, args.exp_id)
    if args.setting_index is not None:
        if not 0 <= args.setting_index < len(settings):
            raise SystemExit(f"--setting-index must be between 0 and {len(settings)-1}")
        settings = (settings[args.setting_index],)
    seeds = load_experiment_seeds(args.seed_file)
    if args.rd_seed_id >= len(seeds):
        raise SystemExit(f"Seed {args.rd_seed_id} is unavailable")
    model, experiment = MODEL_NAMES[args.model_id], EXPERIMENT_NAMES[args.exp_id]
    for setting in settings:
        destination = result_path(args.output_root, model=model, experiment=experiment,
                                  setting=setting, seed_id=args.rd_seed_id)
        if args.dry_run:
            print(json.dumps(dict(setting=asdict(setting), random_seed=int(seeds[args.rd_seed_id]),
                                  configuration=resolved_configuration(setting, config),
                                  destination=str(destination),
                                  inapplicability=setting.inapplicability_reason()), sort_keys=True))
            continue
        outcome, result = run_setting(setting=setting, model=model, experiment=experiment,
            seed_id=args.rd_seed_id, random_seed=int(seeds[args.rd_seed_id]),
            destination=destination, config=config, force=args.force)
        print(json.dumps(dict(outcome=outcome, setting=setting.suffix, status=result["status"],
                              avg_err=result["avg_err"], validation_loss=result["validation_loss"],
                              best_params=result["best_params"], selected_iteration=result["selected_iteration"],
                              path=str(destination)), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
