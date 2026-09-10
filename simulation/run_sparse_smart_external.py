"""Fit SparseSMART on all legacy training rows with fresh external tuning data.

The original training experiment is preserved exactly. Independent validation
rows share its true coefficient and observed source; they are used only for
selection. This runner never fits a competing estimator or refits on validation
rows. Its artifacts are separate from the earlier internal-holdout pilot.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from sparse_smart_selection import selection_payload
from sparse_smart_provenance import (implementation_provenance, validate_resume_identity,
                                     validate_resume_implementation)

import external_validation_data
import run_sparse_smart_tuned as old_runner
from run_restricted_rrr import (DEFAULT_OUTPUT_ROOT, DEFAULT_SEED_FILE, EXPERIMENT_NAMES,
    MODEL_NAMES, SimulationSetting, experiment_settings, load_experiment_seeds)
from run_sparse_smart import MARGINS, _atomic_json_dump, _coefficient_error, _digest_json, _json_value


@dataclass(frozen=True)
class RunnerConfig(old_runner.RunnerConfig):
    n_validation: int = 100
    validation_seed_tag: int = 1397970481
    initialization_spectrum: str = "auto"
    refinement_solver: str = "auto"

    def validate(self):
        super().validate()
        if self.initialization_spectrum not in ("auto", "projected", "reject"):
            raise ValueError("initialization_spectrum must be auto, projected, or reject")
        if self.refinement_solver not in ("auto", "chart", "anchor_projected"):
            raise ValueError("refinement_solver must be auto, chart, or anchor_projected")
        if isinstance(self.n_validation, bool) or not isinstance(self.n_validation, int) or self.n_validation < 1:
            raise ValueError("n_validation must be a positive integer")
        if (isinstance(self.validation_seed_tag, bool) or not isinstance(self.validation_seed_tag, int)
                or not 0 <= self.validation_seed_tag <= 2**32-1):
            raise ValueError("validation_seed_tag must be a 32-bit nonnegative integer")


def resolved_configuration(setting, config):
    config.validate()
    value = old_runner.resolved_configuration(setting, config)
    value.update(
        runner=asdict(config), fit_sample="all_supplied_training_rows", validation_mode="independent_external",
        n_train=setting.n, n_validation=config.n_validation, all_training_rows_used=True,
        training_matches_legacy=True, refit_on_all_data=False,
        ignored_internal_holdout_parameters=dict(validation_fraction=config.validation_fraction,
                                                 split_seed=config.split_seed),
        selection_inputs=["training_X", "training_Y", "observed_source", "validation_X", "validation_Y"],
        calibration_label="external_validation_practical_strict_source_check" if config.strict_source_check
            else "external_validation_practical_empirical_source_check",
    )
    return value


def _implementation_files():
    return (*old_runner._implementation_files(), Path(__file__), Path(external_validation_data.__file__))


def _implementation_provenance(api, generator):
    return implementation_provenance(api, generator, _implementation_files())


def _implementation_fingerprint(api, generator):
    return _implementation_provenance(api, generator)["implementation_fingerprint"]


def result_path(output_root, *, model, experiment, setting, seed_id):
    return Path(output_root)/model/experiment/(
        f"SparseSMARTExternal_result_{model}_{experiment}_{setting.suffix}_rd_seed_id={seed_id}.json")


def _split_metadata(tuner, n_train, n_validation):
    if (getattr(tuner, "split_mode_", None) != "explicit_validation"
            or getattr(tuner, "train_indices_", "missing") is not None
            or getattr(tuner, "validation_indices_", "missing") is not None
            or getattr(tuner, "training_sample_count_", None) != n_train
            or getattr(tuner, "validation_sample_count_", None) != n_validation):
        raise RuntimeError("Tuner did not retain all training rows with the explicit external validation sample")
    split = dict(mode="independent_external_validation", n_train=n_train, n_validation=n_validation,
                 train_indices=None, validation_indices=None, all_training_rows_used=True,
                 refit_on_all_data=False)
    return dict(**split, fingerprint=_digest_json(split))


def run_setting(*, setting: SimulationSetting, model: str, experiment: str, seed_id: int,
                random_seed: int, destination: Path, config: RunnerConfig = RunnerConfig(),
                force=False, generate_data_fn=None, sparse_api=None):
    """Run one declared external-validation experiment or verify its checkpoint."""
    started = time.perf_counter()
    resolved = resolved_configuration(setting, config)
    arguments = dict(n=setting.n, p=setting.p, q=setting.q, sigma0=setting.sigma0,
                     sigma=.5, r_star=5, r0_star=10, random_seed=int(random_seed))
    identity = dict(schema_version=1, method="SparseSMARTExternal", model=model, experiment=experiment,
                    rd_seed_id=seed_id, random_seed=int(random_seed), setting=asdict(setting),
                    configuration=resolved, generator_arguments=arguments)
    config_hash = _digest_json(identity)
    destination = Path(destination)
    existing = None
    if destination.exists() and not force:
        try:
            existing = json.loads(destination.read_text())
        except (ValueError, OSError) as error:
            raise ValueError(f"Cannot validate existing result {destination}; use --force") from error
        validate_resume_identity(existing, identity, destination, digest=_digest_json)
    generator = generate_data_fn if generate_data_fn is not None else old_runner._load_generator()
    data = external_validation_data.generate_external_validation(
        n_train=setting.n, p=setting.p, q=setting.q, sigma0=setting.sigma0, random_seed=int(random_seed),
        n_validation=config.n_validation, sigma=.5, r_star=5, r0_star=10,
        seed_tag=config.validation_seed_tag, generate_data_fn=generator)
    # The data-generating helper uses its fixed truth to generate Y_validation.
    # No truth object is supplied to an estimator or candidate-selection call.
    train_hash = old_runner._array_fingerprint(data, ("X", "Y", "C0"))
    validation_hash = old_runner._array_fingerprint(data, ("X_validation", "Y_validation"))
    reason = setting.inapplicability_reason()
    if sparse_api is None and reason is None:
        sparse_api = old_runner._load_sparse_api()
    provenance = _implementation_provenance(sparse_api, generator)
    implementation_hash = provenance["implementation_fingerprint"]
    if existing is not None:
        if existing.get("training_observed_input_fingerprint") != train_hash:
            raise ValueError(f"Training inputs differ from {destination}; use --force")
        if existing.get("validation_observed_input_fingerprint") != validation_hash:
            raise ValueError(f"Validation inputs differ from {destination}; use --force")
        if existing.get("evaluation_truth_fingerprint") != old_runner._array_fingerprint(data, ("C_star",)):
            raise ValueError(f"Evaluation truth differs from {destination}; use --force")
        if existing.get("validation_seed_metadata") != _json_value(data["validation_seed_metadata"]):
            raise ValueError(f"Validation seed metadata differs from {destination}; use --force")
        validate_resume_implementation(existing, provenance, destination)
        return "skipped", existing
    result = dict(identity, configuration_fingerprint=config_hash,
        training_observed_input_fingerprint=train_hash, validation_observed_input_fingerprint=validation_hash,
        implementation_fingerprint=implementation_hash, input_fingerprint=None, evaluation_truth_fingerprint=None,
        validation_seed_metadata=_json_value(data["validation_seed_metadata"]),
        status="inapplicable" if reason else "failed", success=False, applicable=reason is None,
        estimator_status=None, failure_reason=reason, failure_message=None, all_candidates_failed=False,
        avg_err=None, initial_avg_err=None, C_hat=None, best_params=None, validation_loss=None,
        selected_iteration=None, selected_budget=None, selected_candidate_id=None, n_iter=0, selected_supports=None, termination_reason=None,
        n_train=setting.n, n_validation=config.n_validation, all_training_rows_used=True,
        training_matches_legacy=True, refit_on_all_data=False, split=None,
        selection_history=[], fit_errors=[], history=[], validation_history=[], diagnostics={}, tuning_diagnostics={},
        fit_time_sec=0., elapsed_time_sec=None, theorem_certified=False,
        source_check_mode="strict" if config.strict_source_check else "empirical",
        error_metric="norm(C_hat-C_star, fro) / sqrt(p*q)",
        evaluation_scope="all_legacy_training_rows_plus_independent_validation; not a paper aggregate")
    result.update(provenance)
    if reason is not None:
        result["failure_message"] = "Invalid fitted dimensions are recorded without clamping or fitting."
    else:
        X, Y, C0, Xv, Yv = (np.asarray(data[key]) for key in ("X", "Y", "C0", "X_validation", "Y_validation"))
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
                enforce_source_accuracy=config.strict_source_check, spectral_step=config.spectral_step,
                stationarity_tol=config.stationarity_tol,
                initialization_spectrum=config.initialization_spectrum,
                refinement_solver=config.refinement_solver)
            tuner.fit(X, Y, source=source, validation_data=(Xv, Yv))
            result.update(selection_payload(tuner))
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
                split=_split_metadata(tuner, setting.n, config.n_validation), selection_history=candidates,
                fit_errors=[entry for entry in candidates if not entry.get("success", False)])
            if tuner.success_:
                if not hasattr(tuner, "coefficient_"):
                    raise RuntimeError("Tuner reported success without a coefficient")
                # Direct evaluation-truth access occurs after candidate selection.
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
                    initial = ((winner.source_.left @ P)*d) @ (winner.source_.right @ Q).T
                    result["initial_avg_err"] = _coefficient_error(initial, truth)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            result.update(status="failed", success=False, avg_err=None,
                          failure_reason=type(error).__name__, failure_message=str(error))
        result["fit_time_sec"] = time.perf_counter()-fit_started
    truth_hash = old_runner._array_fingerprint(data, ("C_star",))
    result["evaluation_truth_fingerprint"] = truth_hash
    result["input_fingerprint"] = _digest_json(dict(training_observed=train_hash,
        validation_observed=validation_hash, evaluation_truth=truth_hash))
    result["elapsed_time_sec"] = time.perf_counter()-started
    result = _json_value(result)
    _atomic_json_dump(result, destination)
    return "written", result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", type=int, choices=range(3))
    parser.add_argument("exp_id", type=int, choices=range(4))
    parser.add_argument("rd_seed_id", type=int, choices=range(100))
    parser.add_argument("--setting-index", type=int)
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT/"sparse_smart_external")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--iteration-budgets", type=int, nargs="+",
                        help="Increasing checkpoint budgets ending at --iterations; default uses one budget")
    parser.add_argument("--init-penalties", type=old_runner._float_grid, default=(.03,))
    parser.add_argument("--penalties-u", type=old_runner._float_grid, default=(.0025,.01,.04))
    parser.add_argument("--penalties-v", type=old_runner._float_grid, default=(.0025,.01,.04))
    parser.add_argument("--support-grid", type=old_runner._support_grid)
    parser.add_argument("--inverse-step", type=float, default=20.)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--validation-seed-tag", type=int, default=1397970481)
    parser.add_argument("--spectral-step", choices=("projected","reject"), default="projected")
    parser.add_argument("--initialization-spectrum", choices=("auto","projected","reject"), default="auto")
    parser.add_argument("--refinement-solver", choices=("auto","chart","anchor_projected"), default="auto")
    parser.add_argument("--stationarity-tol", type=float, default=1e-6)
    parser.add_argument("--strict-source-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    config = RunnerConfig(iterations=args.iterations,
        iteration_budgets=tuple(args.iteration_budgets) if args.iteration_budgets else None,
        init_penalties=args.init_penalties,
        penalties_u=args.penalties_u, penalties_v=args.penalties_v, support_limits=args.support_grid,
        inverse_step=args.inverse_step, strict_source_check=args.strict_source_check,
        spectral_step=args.spectral_step, stationarity_tol=args.stationarity_tol,
        initialization_spectrum=args.initialization_spectrum, refinement_solver=args.refinement_solver,
        n_validation=args.validation_size, validation_seed_tag=args.validation_seed_tag)
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
            print(json.dumps(_json_value(dict(setting=asdict(setting), random_seed=int(seeds[args.rd_seed_id]),
                configuration=resolved_configuration(setting, config), destination=str(destination))),sort_keys=True))
            continue
        outcome, result = run_setting(setting=setting, model=model, experiment=experiment,
            seed_id=args.rd_seed_id, random_seed=int(seeds[args.rd_seed_id]), destination=destination,
            config=config, force=args.force)
        print(json.dumps(dict(outcome=outcome, setting=setting.suffix, status=result["status"],
            avg_err=result["avg_err"], selected_iteration=result["selected_iteration"],
            n_train=result["n_train"],n_validation=result["n_validation"],path=str(destination)),sort_keys=True),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
