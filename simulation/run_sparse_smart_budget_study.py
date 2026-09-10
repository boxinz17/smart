"""Continuous SparseSMART budget study with independently verifiable checkpoints.

This opt-in runner has its own schema and output root. Each parameter grid point
is fitted once to the maximum budget; validation is evaluated at the declared
checkpoint interval. Earlier certified prefixes survive a later trajectory
failure, while unattained comparison caps remain explicitly unresolved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

from sparse_smart_selection import (selection_payload, validation_winner, validate_selected_score,
                                    pairwise_selection, PAIRWISE_RULE, pairwise_loss_difference)

import external_validation_data
import run_sparse_smart_external as external_runner
from batch_manifest import BatchManifest, manifest_for_resume
from run_restricted_rrr import (DEFAULT_SEED_FILE, MODEL_NAMES, EXPERIMENT_NAMES,
    SimulationSetting, experiment_settings, load_experiment_seeds)
from run_sparse_smart import MARGINS, _atomic_json_dump, _coefficient_error, _digest_json, _json_value
from sparse_smart_provenance import (implementation_provenance, validate_resume_identity,
                                     validate_resume_implementation)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = HERE / 'result' / 'sparse_smart_budget_study'
METHOD = 'SparseSMARTBudgetStudy'
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunnerConfig:
    iteration_budgets: tuple[int, ...] = (500, 1000, 2000, 4000, 8000)
    checkpoint_interval: int = 250
    init_penalties: tuple[float, ...] = (.03,)
    penalties_u: tuple[float, ...] = (.0025, .01, .04)
    penalties_v: tuple[float, ...] = (.0025, .01, .04)
    inverse_step: float = 20.
    stationarity_tol: float = 1e-6
    n_validation: int = 100
    validation_seed_tag: int = external_validation_data.VALIDATION_SEED_TAG
    checkpoint_execution: str = 'continuous'
    initialization_spectrum: str = 'projected'
    refinement_solver: str = 'anchor_projected'
    spectral_step: str = 'projected'
    strict_source_check: bool = False

    @property
    def iterations(self):
        return self.iteration_budgets[-1]

    def validate(self):
        budgets = self.iteration_budgets
        if (not isinstance(budgets, (tuple, list)) or not budgets
                or any(type(b) is not int or b <= 0 for b in budgets)
                or any(a >= b for a, b in zip(budgets, budgets[1:]))):
            raise ValueError('iteration_budgets must be positive, unique, and increasing')
        if type(self.checkpoint_interval) is not int or self.checkpoint_interval <= 0:
            raise ValueError('checkpoint_interval must be a positive integer')
        for name in ('init_penalties', 'penalties_u', 'penalties_v'):
            values = getattr(self, name)
            if (not isinstance(values, (tuple, list)) or not values
                    or any(isinstance(v, bool) or not isinstance(v, (float, int))
                           or not math.isfinite(v) or v < 0 or (name == 'init_penalties' and v == 0)
                           for v in values) or len(set(values)) != len(values)):
                raise ValueError(f'{name} must be a nonempty unique grid of finite valid penalties')
        for name in ('inverse_step', 'stationarity_tol'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if type(self.n_validation) is not int or self.n_validation != 100:
            raise ValueError('The budget study requires exactly 100 independent validation observations')
        if type(self.validation_seed_tag) is not int or not 0 <= self.validation_seed_tag < 2**32:
            raise ValueError('validation_seed_tag must be a 32-bit nonnegative integer')
        fixed = dict(checkpoint_execution='continuous', initialization_spectrum='projected',
                     refinement_solver='anchor_projected', spectral_step='projected', strict_source_check=False)
        if any(getattr(self, key) != value for key, value in fixed.items()) or type(self.strict_source_check) is not bool:
            raise ValueError('This study requires continuous empirical-source projected-anchor refinement')


def resolved_configuration(setting, config):
    config.validate()
    rank, source_rank = setting.target_rank, setting.source_rank
    nu, nv = ((source_rank, source_rank) if setting.sigma0 == 0 else (setting.p, setting.q))
    counts = [max(0, (nu-rank)*rank), max(0, (nv-rank)*rank)]
    grid = [dict(init_penalty=li, penalty_u=lu, penalty_v=lv,
                 support_limits=counts, step_size_inverse=config.inverse_step)
            for li, lu, lv in product(config.init_penalties, config.penalties_u, config.penalties_v)]
    sparsity = min(source_rank*rank, max(rank, 5))
    return dict(runner=asdict(config), iterations=config.iterations, margins=MARGINS.copy(),
        rank=rank, source_rank=source_rank, sparsity=[sparsity, sparsity],
        support_limits=[counts], actual_complement_counts=counts, candidate_grid=grid,
        trajectory_count=len(grid), candidate_count=len(grid)*len(config.iteration_budgets),
        checkpoint_execution='continuous', validation_interval=config.checkpoint_interval,
        n_train=setting.n, n_validation=config.n_validation, fit_sample='all_supplied_training_rows',
        validation_mode='independent_external', refit_on_all_data=False, tuning_uses_truth=False,
        selection_metric='mean((Y_validation-X_validation@C_hat)**2)',
        selection_inputs=['training_X', 'training_Y', 'observed_source', 'validation_X', 'validation_Y'],
        rank_semantics='fitted dimensions; generator truth remains rank 5 and source rank 10')


def result_path(output_root, *, model, experiment, setting, seed_id):
    return Path(output_root)/model/experiment/f'BudgetStudy_result_{model}_{experiment}_{setting.suffix}_rd_seed_id={seed_id}.json'


def _implementation_files():
    return (*external_runner._implementation_files(), Path(__file__),
            Path(__file__).with_name("batch_manifest.py"))


def _implementation_provenance(api, generator):
    return implementation_provenance(api, generator, _implementation_files())


def _implementation_fingerprint(api, generator):
    return _implementation_provenance(api, generator)["implementation_fingerprint"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _factor_state(model, state, iteration):
    left, singular, right = model.chart_.reconstruct(state)
    left, right = model.source_.left @ left, model.source_.right @ right
    _require(all(np.isfinite(x).all() for x in (left, singular, right)), 'Nonfinite checkpoint factors')
    return dict(iteration=int(iteration), left=left, singular_values=singular, right=right)


def _factor_prediction(factors, design):
    left, d, right = (np.asarray(factors[key]) for key in ("left", "singular_values", "right"))
    return ((design @ left)*d) @ right.T


def _factor_scores(factors, data):
    coefficient = (factors['left']*factors['singular_values']) @ factors['right'].T
    prediction = _factor_prediction(factors, data['X_validation'])
    validation = float(np.mean((data['Y_validation']-prediction)**2))
    _require(math.isfinite(validation), 'Nonfinite checkpoint validation score')
    return validation, _coefficient_error(coefficient, data['C_star'])


def _encode_trajectory(model, metadata, data):
    value = dict(_json_value(metadata), factor_states={}, checkpoints=[], history=[], validation_history=[])
    if model is None:
        value['checkpoint_iterations'] = []
        return value
    history = {row['iteration']:row for row in _json_value(getattr(model, 'history_', []))}
    snapshots = getattr(model, 'checkpoints_', {})
    keys = list(getattr(model, 'checkpoint_iterations_', []))
    _require(keys == sorted(set(keys)) and keys == sorted(snapshots), 'Inconsistent estimator checkpoint indices')
    value['checkpoint_iterations'] = keys
    value['validation_history'] = _json_value(getattr(model, 'validation_history_', []))
    score_cache = {}
    def factors(state, iteration):
        key = str(iteration)
        if key not in value['factor_states']:
            encoded = _factor_state(model, state, iteration)
            value['factor_states'][key] = encoded
            score_cache[key] = _factor_scores(encoded, data)
        return key, score_cache[key]
    previous_checkpoint = 0
    for k in keys:
        snapshot = snapshots[k]
        _require(snapshot.iteration == k and snapshot.status in ('completed', 'converged'),
                 'A failed partial endpoint cannot be an eligible checkpoint')
        selected = snapshot.selected_iteration
        _require(type(selected) is int and 0 <= selected <= k and k in history and selected in history,
                 'Checkpoint selected/terminal iteration is missing from history')
        terminal_key, (terminal_score, terminal_error) = factors(snapshot.state, k)
        selected_key, (selected_score, selected_error) = factors(snapshot.selected_state, selected)
        _require(math.isclose(selected_score, snapshot.best_validation_loss, rel_tol=1e-10, abs_tol=1e-12),
                 'Checkpoint factors disagree with selected validation score')
        interval = [history[j] for j in range(previous_checkpoint+1, k+1) if j in history]
        _require(len(interval) == k-previous_checkpoint, 'Missing accepted states between checkpoints')
        changes = [row.get('objective_change') for row in interval]
        relative_steps = [row.get('relative_step_norm') for row in interval]
        stable_change = math.fsum(changes) if all(x is not None and math.isfinite(x) for x in changes) else None
        relative_max = max(relative_steps, default=0.) if all(x is not None and math.isfinite(x) for x in relative_steps) else None
        value['checkpoints'].append(dict(checkpoint_iteration=k, status=snapshot.status, success=True,
            interval_start_iteration=previous_checkpoint, interval_accepted_steps=len(interval),
            interval_objective_change_sum=stable_change, interval_max_relative_step_norm=relative_max,
            termination_reason=snapshot.termination_reason, selected_iteration=selected,
            terminal_factor_key=terminal_key, selected_factor_key=selected_key,
            terminal_validation_mse=terminal_score, selected_validation_mse=selected_score,
            terminal_coefficient_error=terminal_error, selected_coefficient_error=selected_error,
            terminal_record=history[k], selected_record=history[selected],
            optimization_converged=snapshot.status == 'converged',
            selected_converged=snapshot.status == 'converged' and selected == k, endpoint_selected=selected == k))
        validation_rows = {row["iteration"]:row for row in value["validation_history"]}
        if validation_rows.get(selected, {}).get("selection_rule") == PAIRWISE_RULE:
            value["checkpoints"][-1]["selection_rule"] = PAIRWISE_RULE
        if "selection_score" in validation_rows.get(selected, {}):
            value["checkpoints"][-1].update(selection_score=validation_rows[selected]["selection_score"],
                terminal_selection_score=validation_rows[k]["selection_score"])
        previous_checkpoint = k
    # Keep diagnostic history once at declared checkpoints and the final endpoint.
    # A final failed partial endpoint is diagnostic-only, never in checkpoints.
    retained = set(keys) | {getattr(model, 'n_iter_', 0)}
    value['history'] = [history[k] for k in sorted(retained) if k in history]
    return value


def _cap_outcomes(candidates, trajectories, resolved, config, data=None):
    count = resolved['trajectory_count']
    _require(len(candidates) == resolved['candidate_count'], 'Incomplete continuous budget search')
    lookup = {(t['grid_candidate_id'], c['checkpoint_iteration']):c
              for t in trajectories for c in t['checkpoints']}
    for index, candidate in enumerate(candidates):
        budget = config.iteration_budgets[index//count]
        grid_id = index % count
        _require(candidate['candidate_id'] == index and candidate['grid_candidate_id'] == grid_id
                 and candidate['iteration_budget'] == budget and candidate['params'] == resolved['candidate_grid'][grid_id],
                 'Budget-major candidate grid mismatch')
        _require(type(candidate['success']) is bool and type(candidate['budget_reached']) is bool,
                 'Missing continuous checkpoint eligibility or coverage flag')
        if candidate['success']:
            k = candidate['trajectory_checkpoint_iteration']
            _require((grid_id, k) in lookup and k <= budget, 'Eligible candidate has no attained checkpoint')
            checkpoint = lookup[(grid_id, k)]
            validate_selected_score(candidate, checkpoint)
            _require(k > 0 or checkpoint['optimization_converged'], 'Unconverged initializer is not a completed positive prefix')
            _require(not candidate['budget_reached'] or k == budget or checkpoint['optimization_converged'],
                     'Unattained comparison cap incorrectly reported complete')
            _require(math.isclose(candidate['validation_mse'], checkpoint['selected_validation_mse'],
                                 rel_tol=1e-10, abs_tol=1e-12), 'Candidate score differs from checkpoint factors')
        else:
            _require(candidate['validation_mse'] is None and not candidate['budget_reached'],
                     'Failed partial candidate cannot be eligible or resolve a cap')
    pairwise = pairwise_selection(candidates)
    def prediction(outcome):
        trajectory = trajectories[outcome["winner_grid_candidate_id"]]
        factors = trajectory["factor_states"][outcome["selected_factor_key"]]
        return _factor_prediction(factors, data["X_validation"])
    outcomes = []
    for budget in config.iteration_budgets:
        rows = [c for c in candidates if c['iteration_budget'] == budget]
        unresolved = [c['grid_candidate_id'] for c in rows if not c['budget_reached']]
        eligible = [c for c in candidates if c['iteration_budget'] <= budget and c['success']]
        winner = validation_winner(eligible) if eligible else None
        outcome = dict(iteration_budget=budget, coverage_complete=not unresolved,
            budget_reached_count=len(rows)-len(unresolved), unresolved_grid_candidate_ids=unresolved,
            status='unresolved' if unresolved else 'resolved', success=winner is not None,
            eligible_candidate_count=len(eligible), eligible_grid_count=len({c['grid_candidate_id'] for c in eligible}),
            winner_candidate_id=None, winner_grid_candidate_id=None, winner_origin_budget=None,
            winner_checkpoint_iteration=None, selected_iteration=None, validation_mse=None, coefficient_error=None,
            terminal_validation_mse=None, terminal_coefficient_error=None, selected_factor_key=None,
            optimization_converged=False, selected_converged=False)
        if winner is not None:
            checkpoint = lookup[(winner['grid_candidate_id'], winner['trajectory_checkpoint_iteration'])]
            outcome.update(winner_candidate_id=winner['candidate_id'], winner_grid_candidate_id=winner['grid_candidate_id'],
                winner_origin_budget=winner['iteration_budget'], winner_checkpoint_iteration=checkpoint['checkpoint_iteration'],
                selected_iteration=checkpoint['selected_iteration'], validation_mse=winner['validation_mse'],
                coefficient_error=checkpoint['selected_coefficient_error'],
                terminal_validation_mse=checkpoint['terminal_validation_mse'],
                terminal_coefficient_error=checkpoint['terminal_coefficient_error'], selected_factor_key=checkpoint['selected_factor_key'],
                optimization_converged=checkpoint['optimization_converged'], selected_converged=checkpoint['selected_converged'])
        if any("selection_score" in c for c in candidates):
            outcome["selection_score"] = winner["selection_score"] if winner is not None else None
        if pairwise:
            _require(data is not None, "Pairwise cap comparisons require validation data")
            outcome["selection_rule"] = PAIRWISE_RULE
            outcome["validation_comparisons"] = {str(base["iteration_budget"]):
                pairwise_loss_difference(prediction(outcome), data["Y_validation"], prediction(base))
                if base["success"] and outcome["success"] else None for base in outcomes}
        outcomes.append(outcome)
    return outcomes


def run_setting(*, setting: SimulationSetting, model, experiment, seed_id, random_seed,
                destination, config=RunnerConfig(), generate_data_fn=None, sparse_api=None):
    started = time.perf_counter()
    resolved = _json_value(resolved_configuration(setting, config))
    arguments = dict(n=setting.n, p=setting.p, q=setting.q, sigma0=setting.sigma0,
                     sigma=.5, r_star=5, r0_star=10, random_seed=int(random_seed))
    identity = dict(schema_version=SCHEMA_VERSION, method=METHOD, model=model, experiment=experiment,
        rd_seed_id=int(seed_id), random_seed=int(random_seed), setting=asdict(setting),
        configuration=resolved, generator_arguments=arguments)
    fingerprint = _digest_json(identity)
    destination = Path(destination)
    existing = json.loads(destination.read_text()) if destination.exists() else None
    if existing is not None:
        validate_resume_identity(existing, identity, destination, digest=_digest_json)
    generator = generate_data_fn or external_runner.old_runner._load_generator()
    data = external_validation_data.generate_external_validation(n_train=setting.n,p=setting.p,q=setting.q,
        sigma0=setting.sigma0,random_seed=int(random_seed),n_validation=config.n_validation,
        seed_tag=config.validation_seed_tag,generate_data_fn=generator)
    reason = setting.inapplicability_reason()
    api = sparse_api if sparse_api is not None else (external_runner.old_runner._load_sparse_api() if reason is None else None)
    provenance = _implementation_provenance(api, generator)
    hashes = dict(training_observed_input_fingerprint=external_runner.old_runner._array_fingerprint(data, ('X','Y','C0')),
        validation_observed_input_fingerprint=external_runner.old_runner._array_fingerprint(data, ('X_validation','Y_validation')),
        evaluation_truth_fingerprint=external_runner.old_runner._array_fingerprint(data, ('C_star',)),
        implementation_fingerprint=provenance['implementation_fingerprint'])
    hashes['input_fingerprint'] = _digest_json(dict(training_observed=hashes['training_observed_input_fingerprint'],
        validation_observed=hashes['validation_observed_input_fingerprint'],evaluation_truth=hashes['evaluation_truth_fingerprint']))
    if existing is not None:
        _require(all(existing.get(key) == value for key,value in hashes.items()
                     if key != 'implementation_fingerprint'),
                 f'Existing budget-study data, truth, or implementation differs: {destination}')
        validate_resume_implementation(existing, provenance, destination)
        _require(existing.get('validation_seed_metadata') == _json_value(data['validation_seed_metadata']),
                 'Validation seed metadata differs')
        return 'skipped', existing
    value = dict(identity, **hashes, configuration_fingerprint=fingerprint,
        validation_seed_metadata=_json_value(data['validation_seed_metadata']),
        status='inapplicable' if reason else 'all_candidates_failed', success=False, applicable=reason is None,
        failure_reason=reason, n_train=setting.n, n_validation=config.n_validation,
        all_training_rows_used=True, training_matches_legacy=True, refit_on_all_data=False,
        source_check_mode='empirical', theorem_certified=False, selection_history=[], trajectories=[], cap_outcomes=[],
        tuning_diagnostics={}, selected_budget=None, selected_candidate_id=None, selected_iteration=None,
        validation_loss=None, avg_err=None, fit_time_sec=0., elapsed_time_sec=None,
        factor_encoding='ambient C_hat=(left*singular_values)@right.T; factor_states keyed by actual iteration',
        error_metric='norm(C_hat-C_star,fro)/sqrt(p*q)',
        evaluation_scope='targeted finite-budget study; truth errors are evaluation-only; not a paper aggregate')
    value.update(provenance)
    if reason is None:
        if setting.sigma0 == 0:
            U, _, Vt = np.linalg.svd(data['C0'],full_matrices=False)
            source = api.ExactSource(U[:,:setting.source_rank],Vt[:setting.source_rank].T)
        else:
            source = api.NoisySource(data['C0'], setting.sigma0, gap_lower=1.)
        tuner = api.SparseSMARTTuner(rank=setting.target_rank, source_rank=setting.source_rank,
            sparsity=tuple(resolved['sparsity']), margins=api.Margins(**MARGINS),
            init_penalties=config.init_penalties, penalties_u=config.penalties_u, penalties_v=config.penalties_v,
            support_limits=None, iterations=config.iterations, iteration_budgets=config.iteration_budgets,
            checkpoint_execution='continuous', checkpoint_interval=config.checkpoint_interval,
            step_size_inverse=config.inverse_step, stationarity_tol=config.stationarity_tol,
            enforce_source_accuracy=False, initialization_spectrum='projected', refinement_solver='anchor_projected',
            spectral_step='projected')
        fit_started = time.perf_counter()
        tuner.fit(data['X'], data['Y'], source=source, validation_data=(data['X_validation'],data['Y_validation']))
        value['fit_time_sec'] = time.perf_counter()-fit_started
        value.update(selection_payload(tuner, include_factors=False))
        # No coefficient truth reaches fitting or selection. All error evaluation
        # and factor-based independent score checks occur after tuner.fit returns.
        models, metadata = tuner.trajectory_models_, tuner.trajectory_history_
        _require(len(models) == len(metadata) == resolved['trajectory_count'], 'Missing continuous trajectory metadata')
        value['trajectories'] = [_encode_trajectory(fit,meta,data) for fit,meta in zip(models,metadata)]
        value['selection_history'] = [{key:item for key,item in row.items() if key != 'validation_history'}
                                       for row in _json_value(tuner.selection_history_)]
        value['cap_outcomes'] = _cap_outcomes(value['selection_history'],value['trajectories'],resolved,config,data)
        final = value['cap_outcomes'][-1]
        _require(bool(tuner.success_) == final['success'], 'Tuner and cumulative-cap eligibility disagree')
        if final['success']:
            validate_selected_score(value, final)
            _require(tuner.selected_candidate_id_ == final['winner_candidate_id']
                     and math.isclose(tuner.best_score_,final['validation_mse'],rel_tol=1e-10,abs_tol=1e-12),
                     'Tuner winner differs from cumulative checkpoint selection')
        value.update(success=final['success'],
            status='complete' if final['coverage_complete'] else 'partial' if final['success'] else 'all_candidates_failed',
            failure_reason=None if final['coverage_complete'] else 'unresolved_maximum_budget',
            tuning_diagnostics=_json_value(tuner.diagnostics_), tuner_status=tuner.status_,
            selected_budget=final['winner_origin_budget'], selected_candidate_id=final['winner_candidate_id'],
            selected_iteration=final['selected_iteration'], validation_loss=final['validation_mse'], avg_err=final['coefficient_error'])
    value['elapsed_time_sec'] = time.perf_counter()-started
    value = _json_value(value)
    _atomic_json_dump(value,destination)
    return 'written', value


def settings_for_profile(model_id, exp_id, profile='difficult'):
    settings = experiment_settings(model_id,exp_id)
    if profile == 'full':
        return settings
    if profile != 'difficult':
        raise ValueError('profile must be full or difficult')
    return tuple(s for s in settings if (exp_id == 2 and s.source_rank in (5,7)) or (exp_id == 3 and s.sigma0 == .5))


def fit_cell(task):
    model_id, exp_id, setting, seed_id, random_seed, output_root, config = task
    model, experiment = MODEL_NAMES[model_id], EXPERIMENT_NAMES[exp_id]
    destination = result_path(output_root,model=model,experiment=experiment,setting=setting,seed_id=seed_id)
    outcome, result = run_setting(setting=setting,model=model,experiment=experiment,seed_id=seed_id,
        random_seed=random_seed,destination=destination,config=config)
    return dict(model=model,experiment=experiment,setting=setting.suffix,seed_id=seed_id,outcome=outcome,
        status=result['status'],success=result['success'],path=str(destination),
        selected_budget=result['selected_budget'],selected_iteration=result['selected_iteration'],
        avg_err=result['avg_err'],validation_loss=result['validation_loss'],fit_time_sec=result['fit_time_sec'],
        cap_outcomes=result['cap_outcomes'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=int,nargs='+',choices=range(3),default=[0,1,2])
    parser.add_argument('--experiments',type=int,nargs='+',choices=range(4),default=[2,3])
    seeds_group = parser.add_mutually_exclusive_group()
    seeds_group.add_argument('--seed-ids',type=int,nargs='+')
    seeds_group.add_argument('--seed-count',type=int)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--seed-file',type=Path,default=DEFAULT_SEED_FILE)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--profile',choices=('difficult','full'),default='difficult')
    parser.add_argument('--setting-index',type=int,help='Original paper-grid index; requires one model and experiment')
    parser.add_argument('--iteration-budgets',type=int,nargs='+',default=[500,1000,2000,4000,8000])
    parser.add_argument('--checkpoint-interval',type=int,default=250)
    parser.add_argument('--stationarity-tol',type=float,default=1e-6,
        help='Positive constrained stationarity tolerance for early stopping (checked every iteration)')
    parser.add_argument('--init-penalties',type=external_runner.old_runner._float_grid,default=(.03,))
    parser.add_argument('--penalties-u',type=external_runner.old_runner._float_grid,default=(.0025,.01,.04))
    parser.add_argument('--penalties-v',type=external_runner.old_runner._float_grid,default=(.0025,.01,.04))
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args(argv)
    seed_ids = args.seed_ids if args.seed_ids is not None else list(range(args.seed_count if args.seed_count is not None else 5))
    if (not seed_ids or any(type(s) is not int or not 0 <= s < 100 for s in seed_ids)
            or len(set(seed_ids)) != len(seed_ids) or args.workers < 1
            or len(set(args.models)) != len(args.models) or len(set(args.experiments)) != len(args.experiments)):
        parser.error('Require unique valid models, experiments, seed IDs and positive workers/seed count')
    config = RunnerConfig(iteration_budgets=tuple(args.iteration_budgets),checkpoint_interval=args.checkpoint_interval,
        init_penalties=args.init_penalties,penalties_u=args.penalties_u,penalties_v=args.penalties_v,
        stationarity_tol=args.stationarity_tol)
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.setting_index is not None and (len(args.models) != 1 or len(args.experiments) != 1):
        parser.error('--setting-index requires exactly one model and experiment')
    seeds = load_experiment_seeds(args.seed_file)
    if max(seed_ids) >= len(seeds):
        parser.error('Requested seed is unavailable')
    tasks = []
    output_root = args.output_root.resolve()
    for model in args.models:
        for exp in args.experiments:
            selected = settings_for_profile(model,exp,args.profile)
            if args.setting_index is not None:
                full = experiment_settings(model,exp)
                if not 0 <= args.setting_index < len(full):
                    parser.error('--setting-index is outside the original paper grid')
                selected = tuple(s for s in selected if s == full[args.setting_index])
            tasks.extend((model,exp,s,seed,int(seeds[seed]),output_root,config) for seed in seed_ids for s in selected)
    if not tasks:
        parser.error('Selected profile/model/experiment/setting contains no cells')
    identity = dict(schema_version=SCHEMA_VERSION,method=METHOD,models=args.models,experiments=args.experiments,
        seed_ids=seed_ids,seed_file=str(args.seed_file.resolve()),seed_file_sha256=hashlib.sha256(args.seed_file.read_bytes()).hexdigest(),
        configuration=_json_value(asdict(config)),profile=args.profile,setting_index=args.setting_index,
        expected_cells=len(tasks),expected_applicable=sum(t[2].inapplicability_reason() is None for t in tasks),
        expected_inapplicable=sum(t[2].inapplicability_reason() is not None for t in tasks))
    manifest_path = output_root/'budget_study_manifest.json'
    resume_manifest = manifest_for_resume(manifest_path)
    if resume_manifest is not None:
        previous = json.loads(resume_manifest.read_text())
        if any(previous.get(key) != value for key,value in identity.items()):
            parser.error('Existing study manifest has different configuration or requested cells; use a fresh output root')
    elif output_root.exists() and any(output_root.iterdir()):
        parser.error('Use a fresh output root or an existing matching budget-study root')
    manifest = dict(identity,started=datetime.now(timezone.utc).isoformat(),workers=args.workers,
        output_root=str(output_root),cells=[],errors=[])
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('cells','errors')}),flush=True)
    if args.dry_run:
        return 0
    started = time.perf_counter()
    def collect(task, future=None):
        try:
            cell = fit_cell(task) if future is None else future.result()
            manifest['cells'].append(cell)
            print(json.dumps(dict(done=len(manifest['cells']),total=len(tasks),**cell)),flush=True)
        except Exception as error:
            failure = dict(model=MODEL_NAMES[task[0]],experiment=EXPERIMENT_NAMES[task[1]],setting=task[2].suffix,
                seed_id=task[3],exception=type(error).__name__,message=str(error))
            manifest['errors'].append(failure)
            print(json.dumps(failure),flush=True)
        manifest['status_counts'] = dict(Counter(c['status'] for c in manifest['cells']))
        attempt.update()
    with BatchManifest(manifest_path, manifest) as attempt:
        if args.workers == 1:
            for task in tasks:
                collect(task)
        else:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = {pool.submit(fit_cell,task):task for task in tasks}
                for future in as_completed(futures):
                    collect(futures[future],future)
        manifest.update(finished=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started)
        attempt.finish(not manifest['errors'] and len(manifest['cells']) == len(tasks))
    return int(bool(manifest['errors']) or len(manifest['cells']) != len(tasks))


if __name__ == '__main__':
    raise SystemExit(main())
