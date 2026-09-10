"""Independent saved-record fixtures; no estimator fitting is performed."""
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import summarize_sparse_smart_budget_study as summary
import run_sparse_smart_budget_study as runner
from run_restricted_rrr import experiment_settings, load_experiment_seeds, DEFAULT_SEED_FILE, SimulationSetting
from run_sparse_smart import _digest_json, _json_value


def generator(*,n,p,q,**kwargs):
    return dict(X=np.zeros((n,p)),Y=np.zeros((n,q)),C0=np.zeros((p,q)),C_star=np.zeros((p,q)))


@lru_cache(None)
def generated(seed):
    s=experiment_settings(0,2)[2]
    c=runner.RunnerConfig(iteration_budgets=(2,4,8),checkpoint_interval=2,
        init_penalties=(.03,),penalties_u=(.0025,.01),penalties_v=(.0025,))
    return summary._data(s,int(load_experiment_seeds(DEFAULT_SEED_FILE)[seed]),c,{},generator)


def diagnostic(t):
    return dict(iteration=t,objective=10.-t/10,objective_change=None if t==0 else -.1,
        relative_step_norm=0. if t==0 else .01,projected_gradient_norm=.3,
        mapping_displacement=.2,proximal_uncertainty=.1,mapping_precision_limited=False,
        step_norm=0. if t==0 else .01)


def refresh_caps(record):
    budgets=record['configuration']['runner']['iteration_budgets']
    rows=record['selection_history']
    cp={(t['grid_candidate_id'],c['checkpoint_iteration']):c for t in record['trajectories'] for c in t['checkpoints']}
    record['cap_outcomes']=[]
    for budget in budgets:
        current=[r for r in rows if r['iteration_budget']==budget]
        eligible=[r for r in rows if r['iteration_budget']<=budget and r['success']]
        winner=min(eligible,key=lambda r:(r['validation_mse'],r['candidate_id'])) if eligible else None
        unresolved=[r['grid_candidate_id'] for r in current if not r['budget_reached']]
        out=dict(iteration_budget=budget,coverage_complete=not unresolved,budget_reached_count=len(current)-len(unresolved),
            unresolved_grid_candidate_ids=unresolved,status='unresolved' if unresolved else 'resolved',
            success=bool(eligible),eligible_candidate_count=len(eligible),eligible_grid_count=len({r['grid_candidate_id'] for r in eligible}),
            winner_candidate_id=None,winner_grid_candidate_id=None,winner_origin_budget=None,winner_checkpoint_iteration=None,
            selected_iteration=None,validation_mse=None,coefficient_error=None,terminal_validation_mse=None,
            terminal_coefficient_error=None,selected_factor_key=None,optimization_converged=False,selected_converged=False)
        if winner:
            point=cp[(winner['grid_candidate_id'],winner['trajectory_checkpoint_iteration'])]
            out.update(winner_candidate_id=winner['candidate_id'],winner_grid_candidate_id=winner['grid_candidate_id'],
                winner_origin_budget=winner['iteration_budget'],winner_checkpoint_iteration=point['checkpoint_iteration'],
                selected_iteration=point['selected_iteration'],validation_mse=winner['validation_mse'],
                coefficient_error=point['selected_coefficient_error'],terminal_validation_mse=point['terminal_validation_mse'],
                terminal_coefficient_error=point['terminal_coefficient_error'],selected_factor_key=point['selected_factor_key'],
                optimization_converged=point['optimization_converged'],selected_converged=point['selected_converged'])
        record['cap_outcomes'].append(out)
    last=record['cap_outcomes'][-1]
    record.update(success=last['success'],status='complete' if last['coverage_complete'] else 'partial' if last['success'] else 'all_candidates_failed',
        selected_budget=last['winner_origin_budget'],selected_candidate_id=last['winner_candidate_id'],
        selected_iteration=last['selected_iteration'],validation_loss=last['validation_mse'],avg_err=last['coefficient_error'])


def fixture(seed=0, *, stops=None, tied=False):
    setting=experiment_settings(0,2)[2]
    config=runner.RunnerConfig(iteration_budgets=(2,4,8),checkpoint_interval=2,
        init_penalties=(.03,),penalties_u=(.0025,.01),penalties_v=(.0025,))
    resolved=_json_value(runner.resolved_configuration(setting,config))
    random_seed=int(load_experiment_seeds(DEFAULT_SEED_FILE)[seed])
    data=generated(seed)
    value=dict(schema_version=1,method='SparseSMARTBudgetStudy',model='model1',experiment='exp3',rd_seed_id=seed,
        random_seed=random_seed,setting=asdict(setting),configuration=resolved,
        generator_arguments=dict(n=setting.n,p=setting.p,q=setting.q,sigma0=setting.sigma0,sigma=.5,r_star=5,r0_star=10,random_seed=random_seed),
        implementation_fingerprint='a'*64,n_train=setting.n,n_validation=100,all_training_rows_used=True,
        training_matches_legacy=True,refit_on_all_data=False,theorem_certified=False,source_check_mode='empirical',applicable=True,
        trajectories=[],selection_history=[])
    for key in ('training_observed_input_fingerprint','validation_observed_input_fingerprint','evaluation_truth_fingerprint','validation_seed_metadata'):
        value[key]=data[key]
    value['input_fingerprint']=_digest_json(dict(training_observed=value['training_observed_input_fingerprint'],
        validation_observed=value['validation_observed_input_fingerprint'],evaluation_truth=value['evaluation_truth_fingerprint']))
    for j,params in enumerate(resolved['candidate_grid']):
        last=(stops or {}).get(j,8)
        times=list(range(0,last+1,2))
        trajectory=dict(grid_candidate_id=j,params=params,status='completed' if last==8 else 'numerical_stagnation',
            success=last==8,n_iter=last,termination_reason='max_iterations' if last==8 else 'numerical_stagnation',
            elapsed_time_sec=.1,diagnostics={},checkpoint_iterations=times,
            history=[diagnostic(t) for t in sorted(set(times+[last]))],validation_history=[],factor_states={},checkpoints=[])
        scores,errors={},{}
        for t in times:
            scale=1. if tied else 1.-t*.04+j*.01
            left=np.eye(setting.p,setting.target_rank)
            right=np.eye(setting.q,setting.target_rank)
            d=np.arange(5,0,-1)*scale
            coefficient=(left*d)@right.T
            scores[t]=float(np.mean((data['Y']-data['X']@coefficient)**2))
            errors[t]=float(np.linalg.norm(coefficient-data['truth'])/np.sqrt(setting.p*setting.q))
            trajectory['factor_states'][str(t)]=dict(iteration=t,left=left.tolist(),singular_values=d.tolist(),right=right.tolist())
            trajectory['validation_history'].append(dict(iteration=t,loss=scores[t]))
        for t in times:
            selected=min((i for i in times if i<=t),key=lambda i:scores[i])
            trajectory['checkpoints'].append(dict(checkpoint_iteration=t,status='completed',success=True,termination_reason='checkpoint',
                interval_start_iteration=max(0,t-2),interval_accepted_steps=min(t,2),
                interval_objective_change_sum=-.1*min(t,2),interval_max_relative_step_norm=.01 if t else 0.,
                selected_iteration=selected,terminal_factor_key=str(t),selected_factor_key=str(selected),
                terminal_validation_mse=scores[t],selected_validation_mse=scores[selected],terminal_coefficient_error=errors[t],
                selected_coefficient_error=errors[selected],terminal_record=diagnostic(t),selected_record=diagnostic(selected),
                optimization_converged=False,selected_converged=False,endpoint_selected=selected==t))
        value['trajectories'].append(trajectory)
    for budget in config.iteration_budgets:
        for j,trajectory in enumerate(value['trajectories']):
            points=[c for c in trajectory['checkpoints'] if 0<c['checkpoint_iteration']<=budget]
            cp=points[-1] if points else None
            value['selection_history'].append(dict(candidate_id=len(value['selection_history']),grid_candidate_id=j,
                iteration_budget=budget,params=trajectory['params'],success=cp is not None,
                budget_reached=cp is not None and cp['checkpoint_iteration']==budget,
                trajectory_checkpoint_iteration=cp['checkpoint_iteration'] if cp else None,
                n_iter=cp['checkpoint_iteration'] if cp else 0,selected_iteration=cp['selected_iteration'] if cp else None,
                validation_mse=cp['selected_validation_mse'] if cp else None,
                status='completed' if cp else 'unreached_checkpoint',termination_reason='checkpoint' if cp else None,
                trajectory_status=trajectory['status'],trajectory_termination_reason=trajectory['termination_reason']))
    refresh_caps(value)
    identity={key:value[key] for key in ('schema_version','method','model','experiment','rd_seed_id','random_seed','setting','configuration','generator_arguments')}
    value['configuration_fingerprint']=_digest_json(identity)
    return value


def write(root,value):
    setting=SimulationSetting(**value['setting'])
    path=runner.result_path(root,model=value['model'],experiment=value['experiment'],setting=setting,seed_id=value['rd_seed_id'])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def run(root,seeds=(0,)):
    return summary.summarize(root,model_ids=(0,),seed_ids=seeds,generate_data_fn=generator)


def test_empty_default_grid_has45_missing_cells(tmp_path):
    report=summary.summarize(tmp_path)
    assert report['expected_cells']==report['missing_cells']==45
    assert report['regenerated_unique_datasets']==0
    assert report['per_model'][0]['transitions'][0]['paired_validation_gain']['n']==0


def test_complete_record_audits_factors_and_cumulative_winners(tmp_path):
    value=fixture()
    write(tmp_path,value)
    report=run(tmp_path)
    cell=report['cells'][0]
    assert cell['status']=='complete' and all(c['coverage_complete'] for c in cell['caps'])
    assert all(c['selected_near_cap'] for c in cell['caps'])
    assert report['verified_factor_states']==10 and report['regenerated_unique_datasets']==1
    assert all(t['complete_pair'] and t['validation_gain']>0 for t in cell['transitions'])
    assert report['per_setting'][0]['transitions'][0]['paired_validation_gain']['se'] is None
    # Two other requested settings are missing: do not pretend this one seed
    # is a complete across-setting model average.
    assert report['per_model'][0]['transitions'][0]['paired_validation_gain']['n']==0


def test_regenerated_data_cache_evicts_lru_arrays_and_preserves_unique_count():
    setting=SimulationSetting(5,4,3,.01,1,2,'small')
    config=runner.RunnerConfig()
    calls=[]

    def counted_generator(**kwargs):
        calls.append(kwargs['random_seed'])
        return generator(**kwargs)

    cache=summary._RegeneratedDataCache(max_entries=2)
    first=summary._data(setting,11,config,cache,counted_generator)
    second=deepcopy(summary._data(setting,22,config,cache,counted_generator))
    assert summary._data(setting,11,config,cache,counted_generator) is first
    summary._data(setting,33,config,cache,counted_generator)
    assert len(cache)==2 and {key[4] for key in cache}=={11,33}
    regenerated=summary._data(setting,22,config,cache,counted_generator)
    assert calls==[11,22,33,22] and len(cache)==2 and len(cache.seen_keys)==3
    for key,value in second.items():
        if isinstance(value,np.ndarray):
            np.testing.assert_array_equal(regenerated[key],value)
        else:
            assert regenerated[key]==value


def test_cache_eviction_preserves_complete_audit_report_byte_for_byte(tmp_path,monkeypatch):
    # The two source-rank settings share each seed's generated data but have
    # separate fitted records. Capacity one must regenerate them on the revisit.
    for seed in (0,1,2):
        base=fixture(seed)
        write(tmp_path,base)
        changed=deepcopy(base)
        setting=experiment_settings(0,2)[3]
        config=runner.RunnerConfig(iteration_budgets=(2,4,8),checkpoint_interval=2,
            init_penalties=(.03,),penalties_u=(.0025,.01),penalties_v=(.0025,))
        changed['setting']=asdict(setting)
        changed['configuration']=_json_value(runner.resolved_configuration(setting,config))
        identity={key:changed[key] for key in ('schema_version','method','model','experiment','rd_seed_id',
            'random_seed','setting','configuration','generator_arguments')}
        changed['configuration_fingerprint']=_digest_json(identity)
        write(tmp_path,changed)

    cache_type=summary._RegeneratedDataCache
    retained=[]
    calls=[]

    def counted_generator(**kwargs):
        calls.append(kwargs['random_seed'])
        return generator(**kwargs)

    def use_capacity(capacity):
        cache=cache_type(max_entries=capacity)
        retained.append(cache)
        monkeypatch.setattr(summary,'_RegeneratedDataCache',lambda:cache)
        return summary.summarize(tmp_path,model_ids=(0,),seed_ids=(0,1,2),generate_data_fn=counted_generator)

    unbounded_report=use_capacity(16)
    assert len(calls)==3
    calls.clear()
    bounded_report=use_capacity(1)
    assert len(calls)==6 and len(retained[-1])==1
    assert bounded_report['regenerated_unique_datasets']==3
    assert bounded_report['recorded_cells']==6 and bounded_report['verified_factor_states']==60
    assert json.dumps(bounded_report,sort_keys=True)==json.dumps(unbounded_report,sort_keys=True)


def test_failed_extension_retains_earlier_minimum_without_plateau_evidence(tmp_path):
    write(tmp_path,fixture(stops={0:3,1:3}))
    cell=run(tmp_path)['cells'][0]
    assert cell['failed_trajectories']==2 and cell['status']=='partial'
    assert [c['success'] for c in cell['caps']]==[True]*3
    assert [c['coverage_complete'] for c in cell['caps']]==[True,False,False]
    assert cell['caps'][-1]['selected_checkpoint']==2
    assert cell['transitions'][-1]['validation_gain']==0
    assert cell['transitions'][-1]['material_gain'] is None
    assert cell['transitions'][-1]['interpretation']=='unresolved_extension'


def test_one_failed_grid_candidate_prevents_complete_pair(tmp_path):
    write(tmp_path,fixture(stops={1:3}))
    cell=run(tmp_path)['cells'][0]
    assert cell['caps'][-1]['success'] and not cell['caps'][-1]['coverage_complete']
    assert cell['caps'][-1]['unresolved_candidate_ids']==[1]
    assert not cell['transitions'][-1]['complete_pair']


def test_initial_snapshot_alone_does_not_make_failed_prefix_eligible(tmp_path):
    write(tmp_path,fixture(stops={0:1,1:1}))
    cell=run(tmp_path)['cells'][0]
    assert cell['status']=='all_candidates_failed'
    assert all(not c['success'] and not c['coverage_complete'] for c in cell['caps'])


def test_equal_validation_uses_earliest_point_then_budget_major_grid_order(tmp_path):
    write(tmp_path,fixture(tied=True))
    cell=run(tmp_path)['cells'][0]
    assert all(c['selected_iteration']==0 and c['winner_candidate_id']==0 for c in cell['caps'])
    assert all(t['material_gain'] is False for t in cell['transitions'])


@pytest.mark.parametrize('mutation,message',[
    (lambda r:r.update(configuration_fingerprint='b'*64),'Configuration fingerprint'),
    (lambda r:r.update(training_observed_input_fingerprint='b'*64),'Combined input fingerprint'),
    (lambda r:r['trajectories'][0]['factor_states']['2']['left'][0].__setitem__(0,float('nan')),'Invalid factor'),
    (lambda r:r['trajectories'][0]['validation_history'].reverse(),'Unordered validation'),
    (lambda r:r['trajectories'][0]['validation_history'][1].update(loss=-1.),'Invalid validation loss'),
    (lambda r:r['trajectories'][0]['checkpoints'][1].update(selected_validation_mse=0.),'Checkpoint selected score'),
    (lambda r:r['cap_outcomes'][-1].update(winner_candidate_id=0),'Cumulative winner'),
    (lambda r:r['cap_outcomes'][-1].update(coefficient_error=99.),'Cumulative coefficient error'),
    (lambda r:r['trajectories'][0]['checkpoints'][1]['terminal_record'].update(proximal_uncertainty=2.),'Mapping displacement'),
    (lambda r:r.update(refit_on_all_data=True),'data-use'),
])
def test_tampering_rejected(tmp_path,mutation,message):
    value=fixture()
    mutation(value)
    write(tmp_path,value)
    with pytest.raises(ValueError,match=message):
        run(tmp_path)


def test_false_cap_coverage_rejected_even_with_eligible_fallback(tmp_path):
    value=fixture(stops={0:3,1:3})
    value['selection_history'][-1]['budget_reached']=True
    write(tmp_path,value)
    with pytest.raises(ValueError,match='Unreached cap'):
        run(tmp_path)


def test_hidden_later_checkpoint_rejected(tmp_path):
    value=fixture()
    row=value['selection_history'][-1]
    row['trajectory_checkpoint_iteration']=4
    write(tmp_path,value)
    with pytest.raises(ValueError,match='latest attained'):
        run(tmp_path)


def mark_checkpoint_stationary(trajectory,t):
    row=next(r for r in trajectory['history'] if r['iteration']==t)
    row.update(projected_gradient_norm=0.,mapping_displacement=0.,proximal_uncertainty=0.)
    for cp in trajectory['checkpoints']:
        if cp['checkpoint_iteration']==t:
            cp.update(status='converged',termination_reason='stationarity',optimization_converged=True,
                      selected_converged=cp['selected_iteration']==t,terminal_record=deepcopy(row))
        if cp['selected_iteration']==t:
            cp['selected_record']=deepcopy(row)


def test_failed_trajectory_cannot_invent_earlier_stationarity_to_cover_caps(tmp_path):
    value=fixture(stops={0:3,1:3})
    for trajectory in value['trajectories']:
        mark_checkpoint_stationary(trajectory,2)
    for row in value['selection_history']:
        row.update(budget_reached=True,status='converged',termination_reason='stationarity')
    refresh_caps(value)
    write(tmp_path,value)
    with pytest.raises(ValueError,match='False checkpoint stationarity'):
        run(tmp_path)


def test_successful_early_terminal_stationarity_covers_future_caps(tmp_path):
    value=fixture(stops={0:2,1:2})
    for trajectory in value['trajectories']:
        mark_checkpoint_stationary(trajectory,2)
        trajectory.update(success=True,status='converged',termination_reason='stationarity')
    for row in value['selection_history']:
        row.update(budget_reached=True,status='converged',termination_reason='stationarity',
                   trajectory_status='converged',trajectory_termination_reason='stationarity')
    refresh_caps(value)
    write(tmp_path,value)
    cell=run(tmp_path)['cells'][0]
    assert cell['failed_trajectories']==0
    assert all(c['coverage_complete'] and c['selected_converged'] for c in cell['caps'])


def test_omitting_required_non_analysis_checkpoint_is_rejected(tmp_path):
    value=fixture()
    for trajectory in value['trajectories']:
        trajectory['checkpoints']=[c for c in trajectory['checkpoints'] if c['checkpoint_iteration']!=6]
        trajectory['checkpoint_iterations'].remove(6)
        trajectory['validation_history']=[r for r in trajectory['validation_history'] if r['iteration']!=6]
        trajectory['history']=[r for r in trajectory['history'] if r['iteration']!=6]
        del trajectory['factor_states']['6']
        trajectory['checkpoints'][-1].update(interval_start_iteration=4,interval_accepted_steps=4,
                                            interval_objective_change_sum=-.4)
    write(tmp_path,value)
    with pytest.raises(ValueError,match='omits scheduled checkpoint'):
        run(tmp_path)


def test_checkpoint_diagnostics_must_match_trajectory_history(tmp_path):
    value=fixture()
    value['trajectories'][0]['checkpoints'][1]['terminal_record']['objective']=9.
    write(tmp_path,value)
    with pytest.raises(ValueError,match='diagnostic record differs'):
        run(tmp_path)


def test_paired_seed_statistics_exclude_failed_extensions(tmp_path):
    for seed in (0,1):
        write(tmp_path,fixture(seed))
    write(tmp_path,fixture(2,stops={1:3}))
    report=run(tmp_path,seeds=(0,1,2))
    row=report['per_setting'][0]['transitions'][1]
    assert row['complete_cell_pairs']==2 and row['unresolved_or_missing_cell_pairs']==1
    values=[c['transitions'][1]['validation_gain'] for c in report['cells'][:2]]
    assert row['paired_validation_gain']['mean']==pytest.approx(np.mean(values))
    assert row['paired_validation_gain']['se']==pytest.approx(np.std(values,ddof=1)/np.sqrt(2))


def test_threshold_rule_uses_strict_max_and_not_coefficient_error():
    a=dict(iteration_budget=2,success=True,coverage_complete=True,validation_mse=1.,coefficient_error=0.)
    b=dict(iteration_budget=4,success=True,coverage_complete=True,validation_mse=.75,coefficient_error=100.)
    result=summary.compare_caps(a,b,absolute_threshold=.25,relative_threshold=.1)
    assert result['material_gain_threshold']==.25 and result['material_gain'] is False
    result=summary.compare_caps(a,b,absolute_threshold=.1,relative_threshold=.2)
    assert result['material_gain'] is True and result['coefficient_error_change']==100.


def test_zero_base_score_has_no_division_or_false_relative_gain():
    a=dict(iteration_budget=2,success=True,coverage_complete=True,validation_mse=0.)
    b=dict(iteration_budget=4,success=True,coverage_complete=True,validation_mse=0.)
    result=summary.compare_caps(a,b)
    assert result['relative_validation_gain'] is None and result['material_gain'] is False


@pytest.mark.parametrize('threshold',[-1.,float('nan'),float('inf'),True])
def test_invalid_threshold_rejected_before_reading(tmp_path,threshold):
    with pytest.raises(ValueError,match='threshold'):
        summary.summarize(tmp_path,absolute_threshold=threshold)


def test_report_preserves_failure_caveat_and_writes_json(tmp_path):
    write(tmp_path,fixture(stops={1:3}))
    report=run(tmp_path)
    path=summary.write_outputs(report,tmp_path/'summary')
    content=path.read_text()
    assert 'unvisited later endpoint' in content and 'not significance tests' in content
    assert 'unresolved' in content.lower() and 'coefficient' in content.lower()
    saved=json.loads((path.parent/'summary.json').read_text())
    assert saved['no_fits_performed'] and saved['missing_cells']==2


def manifest(root,value):
    result=dict(schema_version=1,method='SparseSMARTBudgetStudy',models=[0],experiments=[2],seed_ids=[value['rd_seed_id']],
        seed_file_sha256=hashlib.sha256(DEFAULT_SEED_FILE.read_bytes()).hexdigest(),configuration=value['configuration']['runner'],
        profile='difficult',setting_index=2,expected_cells=1,cells=[],errors=[])
    root.mkdir(parents=True,exist_ok=True)
    (root/'budget_study_manifest.json').write_text(json.dumps(result))
    return result


def test_manifest_scope_reports_exact_single_case(tmp_path):
    value=fixture()
    write(tmp_path,value)
    manifest(tmp_path,value)
    report=summary.summarize(tmp_path,manifest_scope=True,generate_data_fn=generator)
    assert report['expected_cells']==report['recorded_cells']==1 and report['missing_cells']==0
    assert len(report['per_setting'])==1
    assert report['per_model'][0]['transitions'][0]['paired_validation_gain']['n']==1
    assert report['manifest']['sha256'] and report['manifest_scope']


def test_manifest_scope_uses_declared_cells_before_any_finish(tmp_path):
    manifest(tmp_path,fixture())
    report=summary.summarize(tmp_path,manifest_scope=True,generate_data_fn=generator)
    assert report['expected_cells']==report['missing_cells']==1
    assert report['iteration_budgets']==[2,4,8]


@pytest.mark.parametrize('field,value,message',[
    ('seed_file_sha256','b'*64,'saved-seed fingerprint'),
    ('expected_cells',2,'scope/count'),
    ('setting_index',999,'setting index'),
])
def test_manifest_identity_corruption_rejected(tmp_path,field,value,message):
    m=manifest(tmp_path,fixture())
    m[field]=value
    (tmp_path/'budget_study_manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match=message):
        summary.summarize(tmp_path,manifest_scope=True,generate_data_fn=generator)


def test_bad_stable_interval_metadata_is_rejected(tmp_path):
    value=fixture()
    value['trajectories'][0]['checkpoints'][2]['interval_start_iteration']=0
    write(tmp_path,value)
    with pytest.raises(ValueError,match='interval mismatch'):
        run(tmp_path)


def test_model_se_uses_seed_averages_not_settings_as_independent_replicates(tmp_path):
    for seed in (0,1):
        for exp,suffix in summary.DIFFICULT_SETTINGS:
            value=fixture(seed)
            setting=next(s for s in experiment_settings(0,exp) if s.suffix==suffix)
            config=runner.RunnerConfig(iteration_budgets=(2,4,8),checkpoint_interval=2,init_penalties=(.03,),penalties_u=(.0025,.01),penalties_v=(.0025,))
            value.update(experiment=f'exp{exp+1}',setting=asdict(setting),configuration=_json_value(runner.resolved_configuration(setting,config)))
            value['generator_arguments']['sigma0']=setting.sigma0
            identity={k:value[k] for k in ('schema_version','method','model','experiment','rd_seed_id','random_seed','setting','configuration','generator_arguments')}
            value['configuration_fingerprint']=_digest_json(identity)
            write(tmp_path,value)
    report=run(tmp_path,seeds=(0,1))
    t=report['per_model'][0]['transitions'][0]
    assert t['complete_cell_pairs']==6 and t['paired_validation_gain']['n']==2
    assert t['paired_validation_gain']['se']==pytest.approx(report['per_setting'][0]['transitions'][0]['paired_validation_gain']['se'])


def test_two_subthreshold_adjacent_gains_can_be_jointly_material():
    caps=[dict(iteration_budget=b,success=True,coverage_complete=True,validation_mse=v,coefficient_error=10.)
          for b,v in [(2000,1.),(4000,.99994),(8000,.99988)]]
    transitions=summary._transitions(caps,1e-4,0.)
    assert [(t['base_budget'],t['extended_budget']) for t in transitions]==[(2000,4000),(4000,8000),(2000,8000)]
    assert [t['material_gain'] for t in transitions]==[False,False,True]
    cell=dict(seed_id=0,experiment='exp3',setting='rs=5',missing=False,transitions=transitions)
    grouped=summary._aggregate([cell],(2000,4000,8000),1e-4,0.)
    assert grouped[-1]['material_gain_pairs']==1
