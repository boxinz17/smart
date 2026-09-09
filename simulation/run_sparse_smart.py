"""Run SparseSMART on existing V1 simulation seeds, without running baselines.

The default is an explicitly empirical noisy-source experiment: declared source
uncertainty is recorded, while its theorem sufficiency gate is not enforced.
Use --strict-source-check to retain that gate. All tuning is fixed in advance;
simulation truth is used only to evaluate coefficient errors.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from run_restricted_rrr import (
    DEFAULT_OUTPUT_ROOT, EXPERIMENT_NAMES, MODEL_NAMES, SimulationSetting,
    experiment_settings, load_experiment_seeds,
)


@dataclass(frozen=True)
class RunnerConfig:
    iterations: int = 50
    init_penalty: float = .03
    penalty: float = .01
    inverse_step: float = 20.
    support_limit: int | None = None
    strict_source_check: bool = False

    def validate(self):
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int) or self.iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        for key, value, positive in (("init_penalty", self.init_penalty, True),
                                     ("penalty", self.penalty, False),
                                     ("inverse_step", self.inverse_step, True)):
            if not math.isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError(f"{key} must be finite and {'positive' if positive else 'nonnegative'}")
        if self.support_limit is not None and (
            isinstance(self.support_limit, bool) or not isinstance(self.support_limit, int)
            or self.support_limit < 0
        ):
            raise ValueError("support_limit must be a nonnegative integer")
        if not isinstance(self.strict_source_check, bool):
            raise ValueError("strict_source_check must be boolean")


MARGINS = dict(d_lower=.05, d_upper=12., gap=.01, anchor_min=.005, trial_radius=1.)


def _json_value(value):
    """Convert diagnostics to strict JSON, retaining nonfinite values as null."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _digest_json(value):
    return hashlib.sha256(json.dumps(_json_value(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def input_fingerprint(data):
    """Hash actual generated inputs and evaluation truth, not merely the seed."""
    digest = hashlib.sha256()
    for name in ("X", "Y", "C0", "C_star"):
        array = np.ascontiguousarray(data[name])
        digest.update(json.dumps((name, array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _implementation_fingerprint(api):
    digest = hashlib.sha256(Path(__file__).read_bytes())
    module_file = getattr(api, "__file__", None)
    if module_file:
        for path in sorted(Path(module_file).parent.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    else:
        digest.update(b"injected_test_api")
    return digest.hexdigest()


def _atomic_json_dump(value, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(_json_value(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def result_path(output_root, *, model, experiment, setting, seed_id):
    return Path(output_root) / model / experiment / (
        f"SparseSMART_result_{model}_{experiment}_{setting.suffix}_rd_seed_id={seed_id}.json"
    )


def _load_generator():
    from smart import generate_data
    return generate_data


def _load_sparse_api():
    import sparse_smart
    return sparse_smart


def resolved_configuration(setting, config):
    config.validate()
    r, r0 = setting.target_rank, setting.source_rank
    nu, nv = (r0, r0) if setting.sigma0 == 0 else (setting.p, setting.q)
    maximum = (max(0, (nu - r) * r), max(0, (nv - r) * r))
    if config.support_limit is None:
        limits = tuple(min(size, 5 * r) for size in maximum)
    else:
        limits = (config.support_limit, config.support_limit)
        if setting.inapplicability_reason() is None and any(k > size for k, size in zip(limits, maximum)):
            raise ValueError(f"support_limit exceeds the complement dimensions {maximum}")
    budget = min(r0 * r, max(r, 5))
    return {
        "runner": asdict(config), "margins": MARGINS.copy(), "delta": .05,
        "sparsity": [budget, budget], "support_limits": list(limits),
        "rank": r, "source_rank": r0,
        "source_mode": "exact_observed_svd" if setting.sigma0 == 0 else "noisy_observed_coefficient",
        "source_gap_lower": None if setting.sigma0 == 0 else 1.,
        "calibration_label": "fixed_practical_oracle_dimensions_strict_source_check" if config.strict_source_check
            else "fixed_practical_oracle_dimensions_empirical_source_check",
        "rank_semantics": "fitted_dimensions; generator truth remains rank 5 and source rank 10",
        "tuning_uses_truth": False,
    }


def _coefficient_error(coefficient, truth):
    coefficient = np.asarray(coefficient)
    if coefficient.shape != truth.shape or not np.all(np.isfinite(coefficient)):
        raise ValueError("Estimator returned a nonfinite coefficient or an incorrect shape")
    return float(np.linalg.norm(coefficient - truth, ord="fro") / np.sqrt(truth.size))


def run_setting(*, setting: SimulationSetting, model: str, experiment: str, seed_id: int,
                random_seed: int, destination: Path, config: RunnerConfig = RunnerConfig(),
                force: bool = False, generate_data_fn=None, sparse_api=None):
    """Write one cell, or resume it after verifying configuration and actual data.

    Returns ("written" | "skipped", result). Failed and inapplicable cells remain
    explicit records; avg_err is populated only after a successful complete fit.
    Unexpected programming errors propagate rather than masquerading as failures.
    """
    started = time.perf_counter()
    resolved = resolved_configuration(setting, config)
    generator_arguments = dict(n=setting.n, p=setting.p, q=setting.q, sigma0=setting.sigma0,
                               sigma=.5, r_star=5, r0_star=10, random_seed=int(random_seed))
    identity = dict(schema_version=1, model=model, experiment=experiment, rd_seed_id=seed_id,
                    random_seed=int(random_seed), setting=asdict(setting), configuration=resolved,
                    generator_arguments=generator_arguments)
    fingerprint = _digest_json(identity)
    destination = Path(destination)
    existing = None
    if destination.exists() and not force:
        try:
            existing = json.loads(destination.read_text())
        except (ValueError, OSError) as error:
            raise ValueError(f"Cannot validate existing result {destination}; use --force") from error
        if existing.get("configuration_fingerprint") != fingerprint:
            raise ValueError(f"Existing result configuration differs at {destination}; use --force")
    if generate_data_fn is None:
        generate_data_fn = _load_generator()
    data = generate_data_fn(**generator_arguments)
    data_hash = input_fingerprint(data)
    reason = setting.inapplicability_reason()
    if reason is None and sparse_api is None:
        sparse_api = _load_sparse_api()
    implementation_hash = _implementation_fingerprint(sparse_api)
    if existing is not None:
        if existing.get("input_fingerprint") != data_hash:
            raise ValueError(f"Generated inputs differ from {destination}; use --force")
        if existing.get("implementation_fingerprint") != implementation_hash:
            raise ValueError(f"Implementation differs from {destination}; use --force")
        return "skipped", existing

    result = dict(identity, method="SparseSMART", configuration_fingerprint=fingerprint,
                  input_fingerprint=data_hash, implementation_fingerprint=implementation_hash,
                  status="inapplicable" if reason else "failed", success=False,
                  applicable=reason is None, estimator_status=None, failure_reason=reason,
                  failure_message=None, avg_err=None, initial_avg_err=None,
                  last_accepted_avg_err=None, C_initial=None, C_hat=None,
                  fit_time_sec=0., elapsed_time_sec=None, n_iter=0, history=[], diagnostics={},
                  source_check_mode="strict" if config.strict_source_check else "empirical",
                  error_metric="norm(C_hat-C_star, fro) / sqrt(p*q)", theorem_certified=False)
    if reason is not None:
        result["failure_message"] = "Requires 1 <= fitted rank <= fitted source rank <= min(p,q); dimensions were not clamped."
    else:
        X, Y, C0 = (np.asarray(data[name]) for name in ("X", "Y", "C0"))
        truth = np.asarray(data["C_star"])
        fit_started = time.perf_counter()
        try:
            if setting.sigma0 == 0:
                U, _, Vt = np.linalg.svd(C0, full_matrices=False)
                source = sparse_api.ExactSource(U[:, :setting.source_rank], Vt[:setting.source_rank].T)
            else:
                source = sparse_api.NoisySource(C0, setting.sigma0, gap_lower=1.)
            calibration = sparse_api.PracticalCalibration(
                init_penalty=config.init_penalty, penalty=config.penalty,
                step_size_inverse=config.inverse_step, support_limits=tuple(resolved["support_limits"]),
                delta=.05, enforce_source_accuracy=config.strict_source_check)
            estimator = sparse_api.SparseSMART(rank=setting.target_rank, source_rank=setting.source_rank,
                sparsity=tuple(resolved["sparsity"]), margins=sparse_api.Margins(**MARGINS),
                calibration=calibration, iterations=config.iterations)
            estimator.fit(X, Y, source=source)
            result.update(estimator_status=estimator.status_, success=bool(estimator.success_),
                          status="complete" if estimator.success_ else "failed",
                          failure_reason=None if estimator.success_ else estimator.status_,
                          failure_message=None if estimator.success_ else estimator.message_,
                          n_iter=int(estimator.n_iter_), history=_json_value(estimator.history_),
                          diagnostics=_json_value(estimator.diagnostics_))
            if hasattr(estimator, "initial_state_"):
                P, d, Q = estimator.chart_.reconstruct(estimator.initial_state_)
                initial = ((estimator.source_.left @ P) * d) @ (estimator.source_.right @ Q).T
                result.update(initial_avg_err=_coefficient_error(initial, truth), C_initial=initial)
            if hasattr(estimator, "coefficient_"):
                coefficient = np.asarray(estimator.coefficient_)
                error = _coefficient_error(coefficient, truth)
                result.update(C_hat=coefficient, last_accepted_avg_err=error,
                              avg_err=error if estimator.success_ else None)
            elif estimator.success_:
                raise RuntimeError("SparseSMART reported success without a coefficient")
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            result.update(status="failed", success=False, avg_err=None,
                          failure_reason=type(error).__name__, failure_message=str(error))
        result["fit_time_sec"] = time.perf_counter() - fit_started
    result["elapsed_time_sec"] = time.perf_counter() - started
    result = _json_value(result)
    _atomic_json_dump(result, destination)
    return "written", result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", type=int, choices=range(3))
    parser.add_argument("exp_id", type=int, choices=range(4))
    parser.add_argument("rd_seed_id", type=int, choices=range(100))
    parser.add_argument("--setting-index", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "sparse_smart_pilot")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--init-penalty", type=float, default=.03)
    parser.add_argument("--penalty", type=float, default=.01)
    parser.add_argument("--inverse-step", type=float, default=20.)
    parser.add_argument("--support-limit", type=int)
    parser.add_argument("--strict-source-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None):
    args = _parser().parse_args(argv)
    config = RunnerConfig(args.iterations, args.init_penalty, args.penalty,
                          args.inverse_step, args.support_limit, args.strict_source_check)
    config.validate()
    settings = experiment_settings(args.model_id, args.exp_id)
    if args.setting_index is not None:
        if not 0 <= args.setting_index < len(settings):
            raise SystemExit(f"--setting-index must be between 0 and {len(settings)-1}")
        settings = (settings[args.setting_index],)
    seeds = load_experiment_seeds()
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
                              estimator_status=result["estimator_status"], avg_err=result["avg_err"],
                              initial_avg_err=result["initial_avg_err"], path=str(destination)), sort_keys=True),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
