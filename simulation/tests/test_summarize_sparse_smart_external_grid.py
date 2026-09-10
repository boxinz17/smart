"""Full-grid auditing without fitting estimators or requiring earlier pilot files."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import summarize_sparse_smart_external_grid as summary
from run_sparse_smart import _digest_json,_json_value

SEEDS=summary.load_experiment_seeds()


def data(**kwargs):
    rng=np.random.default_rng(kwargs["random_seed"])
    n,p,q=kwargs["n"],kwargs["p"],kwargs["q"]
    X=rng.normal(size=(n,p))
    truth=rng.normal(size=(p,q))*.01
    Y=X@truth+rng.normal(size=(n,q))*.5
    C0=rng.normal(size=(p,q))*(1+kwargs["sigma0"])
    return dict(X=X,Y=Y,C_star=truth,C0=C0)


def record(model=0,experiment=0,index=0,seed=0,*,error=.1,status="complete",config=None,selected=1):
    setting=summary.experiment_settings(model,experiment)[index]
    config=config or summary.runner.RunnerConfig(iterations=2,penalties_u=(.01,.04),penalties_v=(.01,))
    resolved=_json_value(summary.runner.resolved_configuration(setting,config))
    actual=summary._data_audit(setting,SEEDS[seed],config,{},data)
    arguments=dict(n=setting.n,p=setting.p,q=setting.q,sigma0=setting.sigma0,
                   sigma=.5,r_star=5,r0_star=10,random_seed=int(SEEDS[seed]))
    identity=dict(schema_version=1,method="SparseSMARTExternal",model=f"model{model+1}",
        experiment=f"exp{experiment+1}",rd_seed_id=seed,random_seed=int(SEEDS[seed]),setting=asdict(setting),
        configuration=resolved,generator_arguments=arguments)
    split=dict(mode="independent_external_validation",n_train=setting.n,n_validation=100,
        train_indices=None,validation_indices=None,all_training_rows_used=True,refit_on_all_data=False)
    coefficient=actual["_truth"]+error
    score=float(np.mean((actual["_Y_validation"]-actual["_X_validation"]@coefficient)**2))
    candidates=[]
    for i,pu in enumerate(config.penalties_u):
        losses=[score+1.5+i]*(config.iterations+1);losses[selected]=score+i
        candidates.append(dict(candidate_id=i,params=dict(init_penalty=.03,penalty_u=pu,penalty_v=.01,
            support_limits=resolved["support_limits"][0],step_size_inverse=20.),success=True,status="completed",
            message="completed",validation_mse=score+i,partial_validation_mse=None,has_partial_coefficient=False,
            selected_iteration=selected,n_iter=config.iterations,termination_reason="max_iterations",
            validation_history=[dict(iteration=j,loss=v) for j,v in enumerate(losses)],elapsed_time_sec=.01))
    value=dict(identity,configuration_fingerprint=_digest_json(identity),implementation_fingerprint="a"*64,
        **{k:v for k,v in actual.items() if not k.startswith("_")},input_fingerprint=_digest_json(dict(
            training_observed=actual["training_observed_input_fingerprint"],
            validation_observed=actual["validation_observed_input_fingerprint"],
            evaluation_truth=actual["evaluation_truth_fingerprint"])),
        n_train=setting.n,n_validation=100,all_training_rows_used=True,training_matches_legacy=True,
        refit_on_all_data=False,split=dict(**split,fingerprint=_digest_json(split)),
        status="complete",success=True,applicable=True,all_candidates_failed=False,
        source_check_mode="empirical",theorem_certified=False,avg_err=float(np.linalg.norm(coefficient-actual["_truth"])/np.sqrt(setting.p*setting.q)),
        initial_avg_err=.4,C_hat=coefficient.tolist(),selection_history=candidates,fit_errors=[],
        best_params=deepcopy(candidates[0]["params"]),selected_iteration=selected,n_iter=config.iterations,
        validation_loss=score,validation_history=deepcopy(candidates[0]["validation_history"]),
        history=[dict(iteration=j,support_u=1,support_v=1) for j in range(config.iterations+1)],
        selected_supports=dict(u=[0],v=[0]),termination_reason="max_iterations",
        diagnostics=dict(optimization_converged=False,selected_converged=False),
        failure_reason=None,failure_message=None,fit_time_sec=.2)
    reason=setting.inapplicability_reason()
    if reason is not None:
        value.update(status="inapplicable",success=False,applicable=False,implementation_fingerprint="b"*64,
            avg_err=None,C_hat=None,best_params=None,split=None,n_iter=0,selected_iteration=None,
            selection_history=[],fit_errors=[],history=[],validation_history=[],selected_supports=None,
            diagnostics={},failure_reason=reason,termination_reason=None,validation_loss=None)
    elif status!="complete":
        for candidate in candidates:
            candidate.update(success=False,status="initialization_spectrum_failed",validation_mse=None,
                selected_iteration=None,n_iter=0,termination_reason="initialization_spectrum_failed",validation_history=[])
        value.update(status=status,success=False,all_candidates_failed=status=="all_candidates_failed",
            avg_err=None,C_hat=None,best_params=None,selected_iteration=None,n_iter=0,validation_loss=None,
            validation_history=[],history=[],selected_supports=None,diagnostics={},termination_reason=None,
            failure_reason="no_successful_candidate",fit_errors=deepcopy(candidates))
    return value


def write(root,value):
    model=int(value["model"][-1])-1;experiment=int(value["experiment"][-1])-1
    setting=next(s for s in summary.experiment_settings(model,experiment) if s.suffix==value["setting"]["suffix"])
    path=summary.runner.result_path(root,model=value["model"],experiment=value["experiment"],
        setting=setting,seed_id=value["rd_seed_id"])
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return path


def summarize(root,**kwargs):
    return summary.summarize(root,generate_data_fn=data,**kwargs)


def test_full_requested_grid_counts_unsupported_cells_without_assigning_zero_error(tmp_path):
    rows,_,audit=summarize(tmp_path)
    assert len(rows)==72 and audit["expected_runs"]==360
    assert audit["expected_applicable_runs"]==315 and audit["expected_inapplicable_runs"]==45
    assert audit["expected_unique_datasets"]==150 and audit["regenerated_unique_datasets"]==0
    invalid=[r for r in rows if not r["applicable"]]
    assert len(invalid)==9
    assert all(r["final_mean"] is None and r["failed"]==0 and r["missing"]==5 for r in invalid)
    assert audit["actual_base_training_n"]=={"model1":200,"model2":300,"model3":500}


def test_valid_different_shapes_and_invalid_hash_class_can_be_audited_together(tmp_path):
    for model in (0,1,2):write(tmp_path,record(model=model))
    write(tmp_path,record(experiment=1,index=5)) # r11 > source10
    rows,_,audit=summarize(tmp_path,seed_ids=(0,))
    assert sum(r["complete"] for r in rows)==3 and sum(r["inapplicable"] for r in rows)==1
    assert audit["regenerated_unique_datasets"]==3 # Invalid fitted rank shares default data.
    assert audit["implementation_fingerprints"]=={"True":["a"*64],"False":["b"*64]}


def test_cache_excludes_fitted_ranks_and_verifies_four_duplicate_default_cells(tmp_path):
    for experiment,index in ((0,0),(1,2),(2,4),(3,1)):
        write(tmp_path,record(experiment=experiment,index=index))
    write(tmp_path,record(experiment=1,index=3)) # A different fitted rank, same actual data.
    calls=[]
    def counted(**kwargs):
        calls.append(kwargs);return data(**kwargs)
    rows,_,audit=summary.summarize(tmp_path,model_ids=(0,),seed_ids=(0,),generate_data_fn=counted)
    assert len(calls)==1 and audit["regenerated_unique_datasets"]==1
    assert audit["verified_input_records"]==5
    assert len(audit["duplicate_default_cells_verified"])==1
    assert len(audit["duplicate_default_cells_verified"][0]["compared_cells"])==4
    assert calls[0]["r_star"]==5 and calls[0]["r0_star"]==10


def test_failures_inapplicability_and_missingness_are_distinct_and_se_is_success_only(tmp_path):
    write(tmp_path,record(experiment=1,index=4,seed=0,error=.1))
    write(tmp_path,record(experiment=1,index=4,seed=1,error=.3))
    write(tmp_path,record(experiment=1,index=4,seed=2,status="all_candidates_failed"))
    write(tmp_path,record(experiment=1,index=5,seed=0))
    rows,_,_=summarize(tmp_path,model_ids=(0,),experiments=(1,))
    row=rows[4]
    assert (row["complete"],row["failed"],row["missing"])==(2,1,2)
    assert row["final_mean"]==pytest.approx(.2) and row["final_se"]==pytest.approx(.1)
    assert row["mean_is_conditional_on_success"] and row["initialization_spectrum_failures"]==2
    assert rows[5]["inapplicable"]==1 and rows[5]["failed"]==0 and rows[5]["final_mean"] is None


@pytest.mark.parametrize("mutate,match",[
    (lambda r:r.update(random_seed=123),"random seed"),
    (lambda r:r.update(avg_err=.001),"Saved coefficient error"),
    (lambda r:r.update(C_hat=np.zeros((50,100)).tolist()),"coefficient shape"),
    (lambda r:r["split"].update(n_train=160),"split metadata"),
    (lambda r:r["validation_seed_metadata"].update(noise_std=1.),"seed metadata"),
    (lambda r:r.update(configuration_fingerprint="0"*64),"Configuration fingerprint"),
    (lambda r:r["selection_history"][0].update(selected_iteration=0),"validation minimum"),
    (lambda r:r.update(best_params=deepcopy(r["selection_history"][1]["params"])),"validation winner"),
])
def test_identity_metric_and_selection_corruption_are_rejected(tmp_path,mutate,match):
    value=record();mutate(value);write(tmp_path,value)
    with pytest.raises(ValueError,match=match):summarize(tmp_path,model_ids=(0,),experiments=(0,),seed_ids=(0,))


@pytest.mark.parametrize("field",["training_observed_input_fingerprint","validation_observed_input_fingerprint","evaluation_truth_fingerprint"])
def test_actual_regeneration_rejects_forged_but_internally_consistent_hashes(tmp_path,field):
    value=record();value[field]="e"*64
    value["input_fingerprint"]=_digest_json(dict(training_observed=value["training_observed_input_fingerprint"],
        validation_observed=value["validation_observed_input_fingerprint"],evaluation_truth=value["evaluation_truth_fingerprint"]))
    write(tmp_path,value)
    with pytest.raises(ValueError,match=f"Regenerated {field}"):
        summarize(tmp_path,model_ids=(0,),experiments=(0,),seed_ids=(0,))


def test_inapplicable_cell_cannot_masquerade_as_failed_search_or_fitted_output(tmp_path):
    value=record(experiment=2,index=0);value["status"]="all_candidates_failed"
    value["split"]=record()["split"]
    write(tmp_path,value)
    with pytest.raises(ValueError,match="marked inapplicable"):
        summarize(tmp_path,model_ids=(0,),experiments=(2,),seed_ids=(0,))


def test_different_successful_outcomes_on_duplicate_data_are_rejected(tmp_path):
    write(tmp_path,record())
    write(tmp_path,record(experiment=3,index=1,error=.2))
    with pytest.raises(ValueError,match="Duplicate default cell disagrees"):
        summarize(tmp_path,model_ids=(0,),experiments=(0,3),seed_ids=(0,))


def test_mixed_implementations_within_applicable_class_are_rejected(tmp_path):
    write(tmp_path,record(seed=0))
    value=record(seed=1);value["implementation_fingerprint"]="f"*64;write(tmp_path,value)
    with pytest.raises(ValueError,match="different implementations within an applicability class"):
        summarize(tmp_path,model_ids=(0,),experiments=(0,),seed_ids=(0,1))


def test_new_output_is_self_contained_and_labels_source_rank_and_data_budget_caveats(tmp_path):
    write(tmp_path,record())
    for index in (0,1):write(tmp_path,record(experiment=2,index=index))
    rows,paper,audit=summarize(tmp_path,model_ids=(0,),seed_ids=(0,))
    output=tmp_path/"summary"
    summary.write_outputs(rows,paper,audit,output)
    report=(output/"comparison.md").read_text()
    assert "not an equal-data-budget comparison" in report
    assert "source-rank sensitivity on the paper grid" in report
    assert "leading source directions left unpenalized" in report
    assert "Iteration-limit termination is not convergence" in report
    assert "100 repetitions" in report and "conditional on success" in report
    assert (output/"comparison.csv").is_file() and (output/"validation_audit.json").is_file()
    assert (output/"comparison_model1.png").stat().st_size>10000


def _rehash_identity(value):
    identity={key:value[key] for key in ("schema_version","method","model","experiment","rd_seed_id",
        "random_seed","setting","configuration","generator_arguments")}
    value["configuration_fingerprint"]=_digest_json(identity)


def test_historical_grid_records_keep_reject_chart_semantics_and_original_hash(tmp_path):
    value=record()
    for key in summary.runner.RunnerConfig.__dataclass_fields__:
        if key in ("initialization_spectrum","refinement_solver"):
            del value["configuration"]["runner"][key]
    _rehash_identity(value)
    original_hash=value["configuration_fingerprint"]
    path=write(tmp_path,value)
    _,_,audit=summarize(tmp_path,model_ids=(0,),experiments=(0,),seed_ids=(0,))
    assert audit["runner_config"]["initialization_spectrum"]=="reject"
    assert audit["runner_config"]["refinement_solver"]=="chart"
    assert json.loads(path.read_text())["configuration_fingerprint"]==original_hash


def test_current_grid_solver_settings_and_other_config_mismatches_are_not_normalized(tmp_path):
    value=record(config=summary.runner.RunnerConfig(iterations=2,penalties_u=(.01,.04),penalties_v=(.01,),
        initialization_spectrum="projected",refinement_solver="anchor_projected"))
    write(tmp_path,value)
    _,_,audit=summarize(tmp_path,model_ids=(0,),experiments=(0,),seed_ids=(0,))
    assert audit["runner_config"]["refinement_solver"]=="anchor_projected"
    value["configuration"]["n_train"]=199
    _rehash_identity(value);write(tmp_path,value)
    with pytest.raises(ValueError,match="Resolved configuration mismatch"):
        summarize(tmp_path,model_ids=(0,),experiments=(0,),seed_ids=(0,))


def test_removing_a_new_key_without_matching_historical_identity_hash_is_rejected(tmp_path):
    value=record()
    del value["configuration"]["runner"]["refinement_solver"]
    write(tmp_path,value)
    with pytest.raises(ValueError,match="Configuration fingerprint"):
        summarize(tmp_path,model_ids=(0,),experiments=(0,),seed_ids=(0,))


def test_coefficient_substitution_with_same_truth_error_is_rejected(tmp_path):
    value = record()
    setting = summary.experiment_settings(0, 0)[0]
    actual = summary._data_audit(setting, SEEDS[0], summary._config(value["configuration"]["runner"]), {}, data)
    coefficient = np.asarray(value["C_hat"])
    # Reflection preserves the saved Frobenius error but changes predictions.
    value["C_hat"] = (2 * actual["_truth"] - coefficient).tolist()
    write(tmp_path, value)
    with pytest.raises(ValueError, match="Selected validation MSE disagrees"):
        summarize(tmp_path, model_ids=(0,), experiments=(0,), seed_ids=(0,))


def test_scalar_coefficient_one_to_three_cannot_keep_old_validation_score():
    setting = SimpleNamespace(p=1, q=1, target_rank=1)
    actual = dict(_X_validation=np.ones((2, 1)), _Y_validation=np.ones((2, 1)),
                  _truth=np.array([[2.]]))
    value = dict(C_hat=[[1.]], avg_err=1., validation_loss=0.)
    summary._audit_selected_predictions(value, setting, actual)
    value["C_hat"] = [[3.]]
    assert np.linalg.norm(np.asarray(value["C_hat"]) - actual["_truth"]) == value["avg_err"]
    with pytest.raises(ValueError, match="Selected validation MSE disagrees"):
        summary._audit_selected_predictions(value, setting, actual)


def test_consistent_but_forged_relative_scores_are_independently_rejected(tmp_path):
    value = record()
    setting = summary.experiment_settings(0, 0)[0]
    actual = summary._data_audit(setting, SEEDS[0], summary._config(value["configuration"]["runner"]), {}, data)
    prediction = actual["_X_validation"] @ np.asarray(value["C_hat"])
    reference = np.zeros_like(prediction)
    score = summary.prediction_selection_score(prediction, actual["_Y_validation"], reference)
    value["selection_score"] = score
    value["validation_reference_prediction"] = reference.tolist()
    for candidate in value["selection_history"]:
        candidate["selection_score"] = score + candidate["candidate_id"]
        for row in candidate["validation_history"]:
            row["selection_score"] = candidate["selection_score"] + row["loss"] - candidate["validation_mse"]
    value["validation_history"] = deepcopy(value["selection_history"][0]["validation_history"])
    write(tmp_path, value)
    summarize(tmp_path, model_ids=(0,), experiments=(0,), seed_ids=(0,))

    value["selection_score"] += .25
    for candidate in value["selection_history"]:
        candidate["selection_score"] += .25
        for row in candidate["validation_history"]:
            row["selection_score"] += .25
    value["validation_history"] = deepcopy(value["selection_history"][0]["validation_history"])
    write(tmp_path, value)
    with pytest.raises(ValueError, match="Selected validation selection score disagrees"):
        summarize(tmp_path, model_ids=(0,), experiments=(0,), seed_ids=(0,))


def factor_prediction_fixture():
    rng = np.random.default_rng(107)
    setting = SimpleNamespace(p=9, q=7, target_rank=3)
    X = rng.normal(size=(12, setting.p))
    Y = 1e12 * rng.normal(size=(12, setting.q))
    left = np.linalg.qr(rng.normal(size=(setting.p, setting.target_rank)))[0]
    right = np.linalg.qr(rng.normal(size=(setting.q, setting.target_rank)))[0]
    d = np.array([3., 2., 1.])
    coefficient = (left * d) @ right.T
    prediction = ((X @ left) * d) @ right.T
    value = dict(C_hat=coefficient.tolist(), validation_loss=float(np.mean((Y-prediction)**2)),
                 selection_score=0., validation_reference_prediction=prediction.tolist(),
                 selected_factors=dict(left=left.tolist(), singular_values=d.tolist(), right=right.tolist()))
    return value, setting, dict(_X_validation=X, _Y_validation=Y)


def test_canonical_factor_prediction_and_legacy_reassociation_are_auditable():
    value, setting, actual = factor_prediction_fixture()
    dense_prediction = actual["_X_validation"] @ np.asarray(value["C_hat"])
    # A harmless grouping change is visible in relative scores under a large
    # response; new artifacts preserve the canonical factors to avoid it.
    dense_score = summary.prediction_selection_score(dense_prediction, actual["_Y_validation"],
                                                     value["validation_reference_prediction"])
    assert abs(dense_score) > 1e-8
    value["selection_rule"] = "pairwise-validation-loss-v1"
    summary._audit_selected_predictions(value, setting, actual)
    del value["selection_rule"]
    del value["selected_factors"]
    summary._audit_selected_predictions(value, setting, actual)


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r.update(selection_rule="pairwise-validation-loss-v1", selected_factors=None), "missing selected"),
    (lambda r: r["selected_factors"].update(left=[[1.]]), "factor dimensions"),
    (lambda r: r["selected_factors"]["singular_values"].__setitem__(0, -1.), "factor dimensions"),
    (lambda r: r["selected_factors"]["singular_values"].__setitem__(0, float("nan")), "factor dimensions"),
    (lambda r: r["selected_factors"]["right"][0].__setitem__(0, 2.), "not orthonormal"),
    (lambda r: r["C_hat"][0].__setitem__(0, 100.), "disagree with saved coefficient"),
])
def test_selected_factor_provenance_is_checked(mutate, match):
    value, setting, actual = factor_prediction_fixture()
    mutate(value)
    with pytest.raises(ValueError, match=match):
        summary._audit_selected_predictions(value, setting, actual)


def test_zero_design_huge_response_cannot_hide_forged_relative_score():
    setting = SimpleNamespace(p=1, q=1, target_rank=1)
    actual = dict(_X_validation=np.array([[1.], [0.]]), _Y_validation=np.array([[2.], [1e12]]))
    value = dict(C_hat=[[1.]], validation_loss=5e23, selection_score=-1.5,
                 validation_reference_prediction=[[0.], [0.]])
    summary._audit_selected_predictions(value, setting, actual)
    value["selection_score"] = -.5
    with pytest.raises(ValueError, match="selection score disagrees"):
        summary._audit_selected_predictions(value, setting, actual)
