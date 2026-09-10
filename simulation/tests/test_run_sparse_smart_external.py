"""Full training preservation, external validation isolation, and checkpoints."""
from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_sparse_smart_external as runner


def setting(sigma0=.01, n=6):
    return runner.SimulationSetting(n,20,12,sigma0,1,10,"test")


def data(**kwargs):
    n,p,q=kwargs["n"],kwargs["p"],kwargs["q"]
    return dict(X=np.arange(n*p,dtype=float).reshape(n,p)/100,Y=np.zeros((n,q)),
                C0=np.eye(p,q),C_star=np.zeros((p,q)))


def api(calls, *, success=True, error=None, wrong_split=False):
    def factory(**kwargs):
        calls["config"]=kwargs
        tuner=SimpleNamespace()
        def fit(X,Y,*,source,validation_data):
            calls["fit"]=(X.copy(),Y.copy(),source)
            calls["validation"]=tuple(v.copy() for v in validation_data)
            calls["fit_complete"]=True
            if error is not None:raise error
            tuner.split_mode_="explicit_validation"
            tuner.train_indices_=None
            tuner.validation_indices_=None
            tuner.training_sample_count_=len(X)-(1 if wrong_split else 0)
            tuner.validation_sample_count_=len(validation_data[0])
            params=dict(init_penalty=kwargs["init_penalties"][0],penalty_u=kwargs["penalties_u"][0],
                        penalty_v=kwargs["penalties_v"][0],support_limits=(19,11),step_size_inverse=20.)
            failed=dict(candidate_id=0,success=False,status="line_search_failed",params=params,
                        validation_mse=None,partial_validation_mse=.00001,has_partial_coefficient=True)
            tuner.selection_history_=[failed]
            tuner.success_=success;tuner.status_="selected" if success else "no_successful_candidate"
            tuner.message_=tuner.status_;tuner.best_params_=params if success else None
            tuner.best_score_=.3 if success else None;tuner.selected_iteration_=1 if success else None
            if success:
                tuner.selection_history_.append(dict(candidate_id=1,success=True,status="completed",params=params,validation_mse=.3))
                tuner.coefficient_=np.full((X.shape[1],Y.shape[1]),.25)
                tuner.estimator_=SimpleNamespace(n_iter_=2,history_=[dict(iteration=2)],
                    validation_history_=[dict(iteration=0,loss=.4),dict(iteration=1,loss=.3)],
                    diagnostics_={"theorem_certified":False},supports_={"u":np.array([0]),"v":np.array([0])},
                    termination_reason_="max_iterations",status_="completed")
            return tuner
        tuner.fit=fit
        return tuner
    return SimpleNamespace(SparseSMARTTuner=factory,Margins=lambda **kw:SimpleNamespace(**kw),
        ExactSource=lambda U,V:SimpleNamespace(U=U,V=V,mode="exact"),
        NoisySource=lambda coefficient,noise_std,gap_lower:SimpleNamespace(
            coefficient=coefficient,noise_std=noise_std,gap_lower=gap_lower,mode="noisy"))


def run(root, *, calls=None, config=None, fit_api=None, cell=None, force=False):
    calls={} if calls is None else calls
    return runner.run_setting(setting=setting() if cell is None else cell,model="model1",experiment="exp4",
        seed_id=0,random_seed=123,destination=root/"result.json",config=config or runner.RunnerConfig(),
        generate_data_fn=data,sparse_api=fit_api or api(calls),force=force)


def test_all_200_legacy_training_rows_and_fresh_100_validation_rows_reach_tuner(tmp_path):
    calls={}
    outcome,result=run(tmp_path,calls=calls,cell=setting(n=200))
    assert outcome == "written" and result["success"]
    original=data(n=200,p=20,q=12)
    np.testing.assert_array_equal(calls["fit"][0],original["X"])
    np.testing.assert_array_equal(calls["fit"][1],original["Y"])
    assert calls["validation"][0].shape == (100,20) and calls["validation"][1].shape == (100,12)
    assert not np.array_equal(calls["validation"][0],original["X"][:100])
    assert result["training_observed_input_fingerprint"] == runner.old_runner._array_fingerprint(original,("X","Y","C0"))
    assert result["n_train"] == 200 and result["n_validation"] == 100
    assert result["training_matches_legacy"] and result["all_training_rows_used"]
    assert result["split"]["train_indices"] is None and result["split"]["validation_indices"] is None
    assert not result["refit_on_all_data"] and not result["split"]["refit_on_all_data"]
    assert "validation_fraction" not in calls["config"] and "random_state" not in calls["config"]
    assert result["configuration"]["ignored_internal_holdout_parameters"] == {"validation_fraction":.2,"split_seed":0}
    assert json.loads((tmp_path/"result.json").read_text()) == result


class TruthGuard(Mapping):
    def __init__(self,values,calls):self.values,self.calls=values,calls
    def __getitem__(self,key):
        if key=="C_star":assert self.calls.get("fit_complete"),"Evaluation truth accessed before selection"
        return self.values[key]
    def __iter__(self):return iter(self.values)
    def __len__(self):return len(self.values)


def test_truth_is_only_read_by_runner_after_fit_not_passed_to_selection(tmp_path,monkeypatch):
    calls={}
    original=runner.external_validation_data.generate_external_validation
    def guarded(**kwargs):
        # The DGP may use truth to generate validation responses; guard begins
        # after the DGP returns, where truth is evaluation-only.
        return TruthGuard(original(**kwargs),calls)
    monkeypatch.setattr(runner.external_validation_data,"generate_external_validation",guarded)
    _,result=run(tmp_path,calls=calls)
    assert result["avg_err"] == pytest.approx(.25)
    assert result["configuration"]["tuning_uses_truth"] is False
    assert "C_star" not in calls["config"]


def test_configuration_preserves_grid_full_support_and_has_separate_root():
    cell=runner.experiment_settings(0,3)[1]
    config=runner.RunnerConfig()
    resolved=runner.resolved_configuration(cell,config)
    assert resolved["candidate_count"] == 9 and resolved["support_limits"] == [[475,225]]
    assert resolved["n_train"] == 200 and resolved["n_validation"] == 100
    assert resolved["fit_sample"] == "all_supplied_training_rows"
    assert runner._parser().parse_args(["0","3","0"]).output_root.name == "sparse_smart_external"
    exact=runner.resolved_configuration(replace(cell,sigma0=0.),config)
    assert exact["support_limits"] == [[25,25]]


def test_exact_and_noisy_sources_grid_and_gate_are_forwarded(tmp_path):
    calls={}
    config=runner.RunnerConfig(strict_source_check=True,penalties_u=(.001,.02),n_validation=37)
    _,result=run(tmp_path,calls=calls,config=config,cell=setting(0.))
    assert calls["fit"][2].mode == "exact"
    assert calls["config"]["enforce_source_accuracy"] is True
    assert calls["config"]["penalties_u"] == (.001,.02)
    assert calls["config"]["support_limits"] is None
    assert calls["config"]["iterations"] == 500
    assert result["n_validation"] == 37
    calls={}
    run(tmp_path/"noisy",calls=calls)
    assert calls["fit"][2].mode == "noisy" and calls["fit"][2].noise_std == .01
    assert calls["config"]["enforce_source_accuracy"] is False


def test_all_failed_candidates_never_supply_a_successful_partial_estimate(tmp_path):
    _,result=run(tmp_path,fit_api=api({},success=False))
    assert result["status"] == "all_candidates_failed" and not result["success"]
    assert result["all_candidates_failed"] and result["avg_err"] is None and result["C_hat"] is None
    assert result["best_params"] is None and result["validation_loss"] is None
    assert result["fit_errors"][0]["partial_validation_mse"] == .00001
    assert result["split"]["n_train"] == 6 and result["split"]["n_validation"] == 100


def test_invalid_external_split_is_not_written(tmp_path):
    with pytest.raises(RuntimeError,match="retain all training rows"):
        run(tmp_path,fit_api=api({},wrong_split=True))
    assert not (tmp_path/"result.json").exists()


def test_numerical_failure_is_recorded_but_unexpected_programming_errors_propagate(tmp_path):
    _,result=run(tmp_path,fit_api=api({},error=FloatingPointError("nonfinite validation")))
    assert result["status"] == "failed" and result["failure_reason"] == "FloatingPointError"
    assert result["avg_err"] is None
    with pytest.raises(RuntimeError,match="unexpected"):
        run(tmp_path/"other",fit_api=api({},error=RuntimeError("unexpected")))


def test_resume_checks_training_validation_truth_configuration_and_code(tmp_path,monkeypatch):
    calls={};fit_api=api(calls)
    run(tmp_path,fit_api=fit_api);calls.clear()
    outcome,_=run(tmp_path,fit_api=fit_api)
    assert outcome == "skipped" and "fit" not in calls
    with pytest.raises(ValueError,match="configuration differs"):
        run(tmp_path,fit_api=fit_api,config=runner.RunnerConfig(n_validation=99))
    original=runner.external_validation_data.generate_external_validation
    for key,match in [("X","Training inputs"),("X_validation","Validation inputs"),("C_star","Evaluation truth")]:
        def altered(key=key,**kwargs):
            values=original(**kwargs);values[key]=values[key].copy();values[key][0,0]+=1
            return values
        monkeypatch.setattr(runner.external_validation_data,"generate_external_validation",altered)
        with pytest.raises(ValueError,match=match):run(tmp_path,fit_api=fit_api)
    monkeypatch.setattr(runner.external_validation_data,"generate_external_validation",original)
    original_provenance = runner._implementation_provenance
    monkeypatch.setattr(runner, "_implementation_provenance", lambda *args: dict(
        original_provenance(*args), implementation_fingerprint="newhash"))
    with pytest.raises(ValueError,match="Implementation differs"):run(tmp_path,fit_api=fit_api)
    outcome,result=run(tmp_path,fit_api=fit_api,force=True)
    assert outcome == "written" and result["implementation_fingerprint"] == "newhash"


@pytest.mark.parametrize("config", [runner.RunnerConfig(n_validation=0),runner.RunnerConfig(n_validation=True),
    runner.RunnerConfig(validation_seed_tag=-1),runner.RunnerConfig(validation_seed_tag=2**32),
    runner.RunnerConfig(penalties_u=())])
def test_invalid_configuration_fails_before_writing(tmp_path,config):
    with pytest.raises(ValueError):run(tmp_path,config=config)
    assert not (tmp_path/"result.json").exists()
