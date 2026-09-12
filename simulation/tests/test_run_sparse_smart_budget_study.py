"""Budget-study serialization, fallback eligibility, and launch isolation."""
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_sparse_smart_budget_study as runner
from sparse_smart import AnchorChart


CELL = runner.SimulationSetting(6,4,3,.01,1,2,'small')
CONFIG = runner.RunnerConfig(iteration_budgets=(2,4),checkpoint_interval=1,
    init_penalties=(.03,),penalties_u=(.01,.04),penalties_v=(.01,))


def data(**kwargs):
    rng = np.random.default_rng(kwargs['random_seed'])
    n,p,q = kwargs['n'],kwargs['p'],kwargs['q']
    return dict(X=rng.normal(size=(n,p)),Y=np.zeros((n,q)),C0=np.eye(p,q),C_star=np.zeros((p,q)))


def fake_api(calls, *, failure_at=None, initial_failure=False, early_convergence=False, scales=None):
    def factory(**options):
        calls['options'] = options
        tuner = SimpleNamespace()
        def fit(X,Y,*,source,validation_data):
            calls['fits'] = calls.get('fits',0)+1
            calls['training'] = (X.copy(),Y.copy())
            calls['validation'] = tuple(x.copy() for x in validation_data)
            Xv,Yv = validation_data
            grid = [dict(init_penalty=li,penalty_u=lu,penalty_v=lv,
                         support_limits=[X.shape[1]-1,Y.shape[1]-1],step_size_inverse=options['step_size_inverse'])
                    for li,lu,lv in product(options['init_penalties'],options['penalties_u'],options['penalties_v'])]
            tuner.trajectory_models_,tuner.trajectory_history_ = [],[]
            for gid,params in enumerate(grid):
                if initial_failure:
                    model = None
                    meta = dict(grid_candidate_id=gid,params=params,status='initialization_failed',success=False,
                        n_iter=0,termination_reason='initialization_failed',elapsed_time_sec=.01,diagnostics={},checkpoint_iterations=[])
                else:
                    end = 0 if early_convergence else (failure_at if gid == 0 and failure_at is not None else options['iterations'])
                    failed = gid == 0 and failure_at is not None
                    status = 'converged' if early_convergence else 'numerical_stagnation' if failed else 'completed'
                    reason = 'stationarity' if early_convergence else 'numerical_stagnation' if failed else 'max_iterations'
                    chart = AnchorChart(X.shape[1],Y.shape[1],[0],[0],np.eye(1),np.eye(1))
                    model = SimpleNamespace(chart_=chart,source_=SimpleNamespace(left=np.eye(X.shape[1]),right=np.eye(Y.shape[1])),
                        history_=[],validation_history_=[],checkpoints_={},best_validation_states_={},
                        n_iter_=end,status_=status,success_=not failed)
                    checkpoints = {0, options['iterations'], *options['iteration_budgets'],
                        *range(options['checkpoint_interval'], options['iterations']+1, options['checkpoint_interval'])}
                    evaluations = checkpoints | set(options['validation_iterations'])
                    best_score,best_state,best_iteration = float('inf'),None,None
                    for k in range(end+1):
                        d = (scales[k] if scales is not None else max(.1,1.-.2*k)) + gid*.2
                        state = chart.pack([],[],[d],np.zeros((X.shape[1]-1,1)),np.zeros((Y.shape[1]-1,1)))
                        P,singular,Q = chart.reconstruct(state)
                        coefficient = (P*singular)@Q.T
                        score = float(np.mean((Yv-Xv@coefficient)**2))
                        if k in evaluations:
                            if score < best_score:
                                best_score,best_state,best_iteration = score,state.copy(),k
                                if k not in checkpoints:
                                    model.best_validation_states_[k] = state.copy()
                            model.validation_history_.append(dict(iteration=k,loss=score))
                        model.history_.append(dict(iteration=k,objective=chart.loss(state,X,Y),
                            smooth_loss=chart.loss(state,X,Y),penalty_value=0.,step_norm=.2 if k else 0.,
                            relative_step_norm=.1 if k else 0.,projected_gradient_norm=.3,raw_gradient_norm=.3,
                            mapping_displacement=.2,proximal_uncertainty=.1,mapping_precision_limited=False,
                            mapping_refinements=0,mapping_step_size_inverse=20.,mapping_domain_reason=None,
                            step_size_inverse=20.,line_search_start_inverse=20.,objective_change=-.1 if k else None,
                            backtracks=0,rejections=[],support_u=0,support_v=0,anchor_min_u=1.,anchor_min_v=1.))
                        if k in checkpoints:
                            model.checkpoints_[k] = SimpleNamespace(iteration=k,state=state,selected_state=best_state.copy(),
                                selected_iteration=best_iteration,best_validation_loss=best_score,
                                history_length=k+1,validation_history_length=len(model.validation_history_),
                                status='converged' if early_convergence else 'completed',
                                termination_reason='stationarity' if early_convergence else 'max_iterations')
                    model.checkpoint_iterations_ = list(model.checkpoints_)
                    meta = dict(grid_candidate_id=gid,params=params,status=status,success=not failed,n_iter=end,
                        termination_reason=reason,elapsed_time_sec=.01,diagnostics={},checkpoint_iterations=list(model.checkpoints_))
                tuner.trajectory_models_.append(model)
                tuner.trajectory_history_.append(meta)
            tuner.selection_history_ = []
            for budget in options['iteration_budgets']:
                for gid,(params,model) in enumerate(zip(grid,tuner.trajectory_models_)):
                    positive = [] if model is None else [k for k in model.checkpoints_ if k <= budget and (k > 0 or early_convergence)]
                    k = max(positive) if positive else None
                    snapshot = model.checkpoints_[k] if k is not None else None
                    success = snapshot is not None
                    tuner.selection_history_.append(dict(candidate_id=len(tuner.selection_history_),grid_candidate_id=gid,
                        iteration_budget=budget,params=params,success=success,status=snapshot.status if success else 'initialization_failed',
                        n_iter=k if success else 0,selected_iteration=snapshot.selected_iteration if success else None,
                        validation_mse=snapshot.best_validation_loss if success else None,
                        trajectory_checkpoint_iteration=k,budget_reached=success and (k == budget or early_convergence),
                        trajectory_status=tuner.trajectory_history_[gid]['status'],
                        trajectory_termination_reason=tuner.trajectory_history_[gid]['termination_reason'],
                        termination_reason=snapshot.termination_reason if success else 'initialization_failed',
                        validation_history=[] if model is None or k is None else
                            [row for row in model.validation_history_ if row['iteration'] <= k],
                        diagnostics={}))
            eligible = [c for c in tuner.selection_history_ if c['success']]
            winner = min(eligible,key=lambda c:c['validation_mse']) if eligible else None
            tuner.success_,tuner.status_ = bool(winner),'selected' if winner else 'no_successful_candidate'
            tuner.selected_candidate_id_ = winner['candidate_id'] if winner else None
            tuner.best_score_ = winner['validation_mse'] if winner else None
            tuner.diagnostics_ = dict(checkpoint_execution='continuous')
            calls['tuner'] = tuner
            calls['fit_returned'] = True
            return tuner
        tuner.fit = fit
        return tuner
    return SimpleNamespace(SparseSMARTTuner=factory,Margins=lambda **kwargs:SimpleNamespace(**kwargs),
        ExactSource=lambda U,V:SimpleNamespace(left=U,right=V),
        NoisySource=lambda coefficient,noise_std,gap_lower:SimpleNamespace(coefficient=coefficient))


def run(tmp_path, *, calls=None, config=CONFIG, api=None, generator=data):
    calls = {} if calls is None else calls
    return runner.run_setting(setting=CELL,model='model1',experiment='exp3',seed_id=3,random_seed=123,
        destination=tmp_path/'record.json',config=config,generate_data_fn=generator,sparse_api=api or fake_api(calls))


def test_early_validation_roundtrip_retains_historical_bests_and_audits_coverage(tmp_path):
    import summarize_sparse_smart_budget_study as summary
    calls = {}
    config = replace(CONFIG, iteration_budgets=(4,8), checkpoint_interval=4,
                     validation_interval=4, validation_patience=None, validation_iterations=(1,2,3))
    api = fake_api(calls, scales=(1., .4, .2, .8, .5, .5, .5, .5, .6))
    _, value = run(tmp_path, calls=calls, config=config, api=api)
    assert calls['options']['validation_iterations'] == (1,2,3)
    assert value['selected_iteration'] == 2
    for trajectory in value['trajectories']:
        assert trajectory['checkpoint_iterations'] == [0,4,8]
        assert [row['iteration'] for row in trajectory['validation_history']] == [0,1,2,3,4,8]
        assert set(trajectory['factor_states']) == {'0','1','2','4','8'}
        assert [row['iteration'] for row in trajectory['history']] == [0,1,2,4,8]
        assert all(cp['selected_iteration'] == 2 for cp in trajectory['checkpoints'][1:])
    path = runner.result_path(tmp_path, model='model1', experiment='exp3', setting=CELL, seed_id=3)
    audited = summary.validate_record(value, path, setting=CELL, model_id=0, exp_id=2,
        seed_id=3, random_seed=123, data_cache={}, generate_data_fn=data)
    assert audited['validation_audit']['factor_verified_points'] == 10
    assert audited['validation_audit']['metadata_only_points'] == 2
    assert all(cap['coverage_complete'] for cap in audited['caps'])


def test_continuous_runner_fits_once_and_saves_independently_reconstructible_checkpoints(tmp_path,monkeypatch):
    calls = {}
    actual_score = runner._factor_scores
    def score(factors,generated):
        assert calls.get('fit_returned'), 'Truth error evaluated before tuning completed'
        return actual_score(factors,generated)
    monkeypatch.setattr(runner,'_factor_scores',score)
    _,value = run(tmp_path,calls=calls)
    assert calls['fits'] == 1
    assert calls['options']['iterations'] == 4 and calls['options']['iteration_budgets'] == (2,4)
    assert calls['options']['checkpoint_execution'] == 'continuous' and calls['options']['checkpoint_interval'] == 1
    assert calls['options']['stationarity_tol'] == CONFIG.stationarity_tol
    assert calls['options']['support_limits'] is None and not calls['options']['enforce_source_accuracy']
    assert calls['training'][0].shape == (6,4) and calls['validation'][0].shape == (200,4)
    np.testing.assert_array_equal(calls['training'][0],data(n=6,p=4,q=3,random_seed=123)['X'])
    assert value['status'] == 'complete' and value['success']
    assert len(value['trajectories']) == 2 and len(value['selection_history']) == 4
    assert all('validation_history' not in c for c in value['selection_history'])
    for trajectory in value['trajectories']:
        assert len(trajectory['factor_states']) == 5
        assert len(trajectory['validation_history']) == 5
        assert trajectory['checkpoints'][0]['interval_objective_change_sum'] == 0.
        assert trajectory['checkpoints'][1]['interval_objective_change_sum'] == pytest.approx(-.1)
        assert trajectory['checkpoints'][1]['interval_max_relative_step_norm'] == .1
        for checkpoint in trajectory['checkpoints']:
            factors = trajectory['factor_states'][checkpoint['selected_factor_key']]
            coefficient = (np.asarray(factors['left'])*factors['singular_values'])@np.asarray(factors['right']).T
            score = float(np.mean((calls['validation'][1]-calls['validation'][0]@coefficient)**2))
            assert score == pytest.approx(checkpoint['selected_validation_mse'],abs=1e-14)
            assert np.linalg.norm(coefficient)/np.sqrt(12) == pytest.approx(checkpoint['selected_coefficient_error'])
    assert json.loads((tmp_path/'record.json').read_text()) == value


@pytest.mark.parametrize('failure_at',[1,3])
def test_late_failure_preserves_positive_prefix_without_claiming_cap_completion(tmp_path,failure_at):
    calls = {}
    _,value = run(tmp_path,calls=calls,api=fake_api(calls,failure_at=failure_at))
    assert value['status'] == 'partial' and value['success']
    for cap in value['cap_outcomes']:
        if cap['iteration_budget'] > failure_at:
            assert not cap['coverage_complete'] and cap['status'] == 'unresolved'
            assert cap['unresolved_grid_candidate_ids'] == [0]
            row = next(c for c in value['selection_history'] if c['iteration_budget'] == cap['iteration_budget'] and c['grid_candidate_id'] == 0)
            assert row['success'] and not row['budget_reached']
            assert row['trajectory_checkpoint_iteration'] == failure_at
    assert value['trajectories'][0]['status'] == 'numerical_stagnation'


def test_stationarity_at_zero_resolves_caps_but_initial_failure_cannot(tmp_path):
    calls = {}
    _,converged = run(tmp_path/'converged',calls=calls,api=fake_api(calls,early_convergence=True))
    assert converged['status'] == 'complete'
    assert all(c['coverage_complete'] and c['optimization_converged'] for c in converged['cap_outcomes'])
    assert converged['selected_iteration'] == 0
    _,failed = run(tmp_path/'failed',api=fake_api({},initial_failure=True))
    assert failed['status'] == 'all_candidates_failed' and not failed['success']
    assert failed['avg_err'] is None and failed['selected_candidate_id'] is None
    assert all(not c['success'] and not c['coverage_complete'] for c in failed['cap_outcomes'])


def test_matching_checkpoint_skips_only_after_data_truth_and_code_verification(tmp_path,monkeypatch):
    calls = {}
    api = fake_api(calls)
    _,first = run(tmp_path,calls=calls,api=api)
    assert run(tmp_path,calls=calls,api=api)[0] == 'skipped' and calls['fits'] == 1
    original = (tmp_path/'record.json').read_bytes()
    with pytest.raises(ValueError,match='configuration differs'):
        run(tmp_path,config=replace(CONFIG,iteration_budgets=(2,5)),api=api)
    def changed_data(**kwargs):
        value = data(**kwargs)
        value['C_star'][0,0] = .1
        return value
    with pytest.raises(ValueError,match='data, truth, or implementation differs'):
        run(tmp_path,api=api,generator=changed_data)
    original_provenance = runner._implementation_provenance
    monkeypatch.setattr(runner, '_implementation_provenance', lambda *args: dict(
        original_provenance(*args), implementation_fingerprint='0'*64))
    with pytest.raises(ValueError,match='[Ii]mplementation differs'):
        run(tmp_path,api=api)
    assert (tmp_path/'record.json').read_bytes() == original
    assert first['n_train'] == 6 and first['n_validation'] == 200


@pytest.mark.parametrize('mutation',[
    lambda rows:rows[0].update(budget_reached=True,trajectory_checkpoint_iteration=1),
    lambda rows:rows[0].update(success=False,validation_mse=.00001),
    lambda rows:rows[0].update(trajectory_checkpoint_iteration=0),
    lambda rows:rows[0].update(candidate_id=3),
])
def test_ineligible_or_forged_cap_records_are_rejected(tmp_path,mutation):
    _,value = run(tmp_path)
    rows = deepcopy(value['selection_history'])
    mutation(rows)
    with pytest.raises(ValueError):
        runner._cap_outcomes(rows,value['trajectories'],value['configuration'],CONFIG)


def test_default_dry_run_declares_45_cells_without_creating_artifacts(tmp_path,capsys):
    assert runner.main(['--output-root',str(tmp_path/'new'),'--dry-run']) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest['expected_cells'] == manifest['expected_applicable'] == 45
    assert manifest['expected_inapplicable'] == 0 and manifest['seed_ids'] == list(range(5))
    assert manifest['configuration']['iteration_budgets'] == [500,1000,2000]
    assert manifest['configuration']['checkpoint_interval'] == 250
    assert manifest['configuration'] == runner._json_value(asdict(runner.RunnerConfig()))
    assert manifest['configuration']['validation_iterations'] == [1,2,5,10,15,20,25,50,100,150,200]
    resolved = runner.resolved_configuration(CELL, runner.RunnerConfig(**manifest['configuration']))
    assert resolved['trajectory_count'] == 100
    assert resolved['candidate_count'] == 300
    assert resolved['validation_schedule'] == [0,1,2,5,10,15,20,25,*range(50,2001,50)]
    assert manifest['configuration']['stationarity_tol'] == 1e-6
    assert manifest['configuration']['n_validation'] == 200
    assert manifest['configuration']['validation_interval'] == 50
    assert manifest['configuration']['validation_patience'] == 300
    assert manifest['configuration']['validation_min_iterations'] == 500
    assert manifest['configuration']['validation_min_relative_improvement'] == .001
    assert not (tmp_path/'new').exists()


def test_validation_stop_and_sample_size_cli_are_explicit_and_change_identity(tmp_path, capsys):
    configurations = []
    options = [[], ['--no-validation-stop'], ['--no-validation-stop', '--n-validation', '100']]
    for index, arguments in enumerate(options):
        output = tmp_path / str(index)
        assert runner.main(['--models', '0', '--experiments', '3', '--setting-index', '5',
            '--seed-ids', '0', '--output-root', str(output), '--dry-run', *arguments]) == 0
        manifest = json.loads(capsys.readouterr().out)
        config = manifest['configuration']
        assert config['n_validation'] == (100 if index == 2 else 200)
        assert config['validation_patience'] == (300 if index == 0 else None)
        assert config['validation_interval'] == 50
        assert config['validation_min_iterations'] == 500
        assert config['validation_min_relative_improvement'] == .001
        resolved = runner.resolved_configuration(CELL, runner.RunnerConfig(**config))
        assert resolved['n_train'] == CELL.n
        assert resolved['n_validation'] == config['n_validation']
        configurations.append(runner._digest_json(resolved))
        assert not output.exists()
    assert len(set(configurations)) == 3


def test_one_cell_smoke_selection_uses_original_grid_index(tmp_path,capsys):
    assert runner.main(['--models','2','--experiments','2','--setting-index','3',
        '--seed-ids','3','--workers','1','--output-root',str(tmp_path/'new'),'--dry-run']) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest['expected_cells'] == 1 and manifest['seed_ids'] == [3]
    assert runner.experiment_settings(2,2)[3].source_rank == 7
    assert runner.experiment_settings(0,3)[5].sigma0 == .5


def test_full_grid_dry_run_declares_all_cases_and_early_stop_tolerance(tmp_path,capsys):
    assert runner.main(['--models','0','1','2','--experiments','0','1','2','3',
        '--profile','full','--seed-count','100','--stationarity-tol','2e-6',
        '--output-root',str(tmp_path/'new'),'--dry-run']) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest['expected_cells'] == 7200
    assert manifest['expected_applicable'] == 6300 and manifest['expected_inapplicable'] == 900
    assert manifest['configuration']['iteration_budgets'][-1] == 2000
    assert manifest['configuration']['stationarity_tol'] == 2e-6
    assert manifest['configuration']['init_penalties'] == [.01, .03, .1, .3]
    assert manifest['configuration']['penalties_u'] == [.0025, .01, .04, .16, .32]
    assert manifest['configuration']['penalties_v'] == [.0025, .01, .04, .16, .32]
    assert (len(manifest['configuration']['init_penalties'])*len(manifest['configuration']['penalties_u'])
            *len(manifest['configuration']['penalties_v'])) == 100
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('args',[
    ['--seed-ids','3','3'],['--seed-count','0'],['--seed-count','101'],['--checkpoint-interval','0'],
    ['--iteration-budgets','1000','500'],['--iteration-budgets','500','500'],
    ['--models','0','0'],['--experiments','0'],['--setting-index','3'],['--workers','0'],
    ['--stationarity-tol','0'],['--stationarity-tol','-1'],
    ['--stationarity-tol','nan'],['--stationarity-tol','inf'],
    ['--validation-interval','0'],['--validation-patience','0'],
    ['--validation-min-iterations','-1'],['--validation-min-relative-improvement','nan'],
    ['--validation-min-relative-improvement','1'],['--n-validation','0'],
])
def test_invalid_cli_fails_before_writing(tmp_path,args):
    with pytest.raises(SystemExit):
        runner.main([*args,'--output-root',str(tmp_path/'new'),'--dry-run'])
    assert not (tmp_path/'new').exists()


def test_existing_other_runner_root_is_never_reused(tmp_path):
    old = tmp_path/'expanded_pilot_manifest.json'
    old.write_text('{}')
    with pytest.raises(SystemExit):
        runner.main(['--output-root',str(tmp_path),'--dry-run'])
    assert old.read_text() == '{}' and not (tmp_path/'budget_study_manifest.json').exists()


def test_inapplicable_cell_does_not_construct_tuner(tmp_path):
    bad = replace(CELL,source_rank=0)
    def forbidden(**kwargs):
        raise AssertionError('Inapplicable cell was fitted')
    api = SimpleNamespace(SparseSMARTTuner=forbidden)
    _,value = runner.run_setting(setting=bad,model='model1',experiment='exp3',seed_id=0,random_seed=123,
        destination=tmp_path/'bad.json',config=CONFIG,generate_data_fn=data,sparse_api=api)
    assert value['status'] == 'inapplicable' and not value['success']
    assert value['trajectories'] == value['selection_history'] == value['cap_outcomes'] == []


def test_late_validation_exception_endpoint_is_diagnostic_only(tmp_path):
    calls = {}
    run(tmp_path,calls=calls,api=fake_api(calls,failure_at=3))
    tuner = calls['tuner']
    model = tuner.trajectory_models_[0]
    # Mimic failure while scoring accepted iteration3: its optimization record
    # exists, but no finite validation value or eligible snapshot was captured.
    del model.checkpoints_[3]
    model.checkpoint_iterations_ = [0,1,2]
    model.validation_history_ = model.validation_history_[:3]
    generated = dict(data(n=6,p=4,q=3,random_seed=123),
        X_validation=calls['validation'][0],Y_validation=calls['validation'][1])
    encoded = runner._encode_trajectory(model,tuner.trajectory_history_[0],generated)
    assert encoded['checkpoint_iterations'] == [0,1,2]
    assert [row['iteration'] for row in encoded['history']] == [0,1,2,3]
    assert [row['iteration'] for row in encoded['validation_history']] == [0,1,2]
    assert [row['checkpoint_iteration'] for row in encoded['checkpoints']] == [0,1,2]
    assert '3' not in encoded['factor_states']
