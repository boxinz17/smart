"""Artifact validation and conditional aggregation for the tuned pilot."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summarize_sparse_smart_tuned as summary
from run_sparse_smart import _digest_json, _json_value
from run_sparse_smart_tuned import RunnerConfig, resolved_configuration, result_path


SEEDS = summary.load_experiment_seeds()


def record(seed=0, *, error=.1, status="complete", selected=1, config=None, experiment="exp1"):
    setting = summary.experiment_settings(0, 0 if experiment == "exp1" else 3)[0 if experiment == "exp1" else 1]
    config = config or RunnerConfig(iterations=2, penalties_u=(.0025, .04), penalties_v=(.01,))
    resolved = _json_value(resolved_configuration(setting, config))
    arguments = dict(n=setting.n, p=setting.p, q=setting.q, sigma0=setting.sigma0,
                     sigma=.5, r_star=5, r0_star=10, random_seed=int(SEEDS[seed]))
    identity = dict(schema_version=1, method="SparseSMARTTuned", model="model1", experiment=experiment,
                    rd_seed_id=seed, random_seed=int(SEEDS[seed]), setting=asdict(setting),
                    configuration=resolved, generator_arguments=arguments)
    order = np.random.default_rng(config.split_seed).permutation(setting.n)
    nv = int(np.ceil(setting.n*config.validation_fraction))
    indices = dict(train_indices=np.sort(order[nv:]).tolist(), validation_indices=np.sort(order[:nv]).tolist())
    candidates=[]
    for i, pu in enumerate(config.penalties_u):
        losses=[2.+i]*(config.iterations+1)
        losses[selected]=.5+i
        candidates.append(dict(candidate_id=i, params=dict(init_penalty=.03, penalty_u=pu,
            penalty_v=.01, support_limits=resolved["support_limits"][0], step_size_inverse=20.),
            success=True, status="completed", message="completed", n_iter=config.iterations,
            selected_iteration=selected, validation_mse=.5+i, partial_validation_mse=None,
            has_partial_coefficient=False, termination_reason="max_iterations",
            validation_history=[dict(iteration=j,loss=v) for j,v in enumerate(losses)],elapsed_time_sec=.01))
    value=dict(identity, configuration_fingerprint=_digest_json(identity), implementation_fingerprint="a"*64,
        observed_input_fingerprint="b"*64,evaluation_truth_fingerprint="c"*64,
        input_fingerprint=_digest_json(dict(observed="b"*64,evaluation_truth="c"*64)),
        status="complete",success=True,applicable=True,all_candidates_failed=False,
        source_check_mode="empirical",theorem_certified=False,avg_err=error,initial_avg_err=.4,
        C_hat=np.zeros((setting.p,setting.q)).tolist(),split=dict(**indices,n_train=setting.n-nv,
            n_validation=nv,fingerprint=_digest_json(indices),refit_on_all_data=False),
        selection_history=candidates,fit_errors=[],best_params=deepcopy(candidates[0]["params"]),
        selected_iteration=selected,n_iter=config.iterations,validation_loss=.5,
        validation_history=deepcopy(candidates[0]["validation_history"]),
        history=[dict(iteration=j,support_u=1,support_v=1) for j in range(config.iterations+1)],
        selected_supports=dict(u=[0],v=[0]),termination_reason="max_iterations",
        diagnostics=dict(optimization_converged=False,selected_converged=False,
                         calibration=dict(source_accuracy_bypassed=True)),failure_reason=None,
        failure_message=None,fit_time_sec=.2)
    if status != "complete":
        for c in candidates:
            c.update(success=False,status="line_search_failed",validation_mse=None,partial_validation_mse=.001,
                     has_partial_coefficient=True,termination_reason="line_search_failed")
        value.update(status=status,success=False,all_candidates_failed=status=="all_candidates_failed",
                     avg_err=None,C_hat=None,best_params=None,selected_iteration=None,n_iter=0,
                     validation_loss=None,validation_history=[],history=[],selected_supports=None,
                     diagnostics={},termination_reason=None,failure_reason="no_successful_candidate",
                     fit_errors=deepcopy(candidates))
    return value


def write(root, value):
    setting = summary.experiment_settings(0, 0 if value["experiment"] == "exp1" else 3)[0 if value["experiment"] == "exp1" else 1]
    path=result_path(root,model=value["model"],experiment=value["experiment"],setting=setting,seed_id=value["rd_seed_id"])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def summarize(root, **kwargs):
    return summary.summarize(root,experiments=(0,),**kwargs)


def test_failures_and_missing_cells_remain_explicit_and_partial_errors_are_excluded(tmp_path):
    write(tmp_path,record(0,error=.1,selected=0))
    write(tmp_path,record(1,error=.3,selected=2))
    write(tmp_path,record(2,status="all_candidates_failed"))
    rows,_,meta=summarize(tmp_path)
    row=rows[0]
    assert (row["complete"],row["failed"],row["all_candidates_failed"],row["missing"]) == (2,1,1,2)
    assert row["final_mean"] == pytest.approx(.2) and row["final_se"] == pytest.approx(.1)
    assert row["mean_is_conditional_on_success"]
    assert row["candidate_total"] == 6 and row["candidate_failed"] == 2
    assert row["chosen_initializer"] == 1 and row["refined_selections"] == 1
    assert row["selected_near_budget"] == 1 and row["selected_at_budget"] == 1
    assert row["penalty_u_lower_edge"] == 2 and row["penalty_u_refined_lower_edge"] == 1
    assert row["optimization_converged"] == row["selected_converged"] == 0
    assert row["max_iterations"] == 2 and row["n_train"] == 160 and row["n_validation"] == 40
    assert rows[1]["missing"] == 5 and rows[1]["final_mean"] is None
    assert meta["recorded_runs"] == 3


@pytest.mark.parametrize("mutation,match", [
    (lambda r:r.update(avg_err=None),"coefficient error"),
    (lambda r:r.update(success=False),"success flag"),
    (lambda r:r.update(random_seed=1),"random seed"),
    (lambda r:r["setting"].update(p=101),"setting"),
    (lambda r:r.update(configuration_fingerprint="0"*64),"fingerprint"),
    (lambda r:r["split"]["train_indices"].__setitem__(0,r["split"]["validation_indices"][0]),"split"),
    (lambda r:r["selection_history"][0].update(selected_iteration=0),"validation minimum"),
    (lambda r:r.update(best_params=deepcopy(r["selection_history"][1]["params"])),"validation winner"),
    (lambda r:r["selected_supports"].update(u=[0,0]),"Selected support"),
])
def test_corrupt_identity_split_selection_and_support_records_are_rejected(tmp_path,mutation,match):
    value=record();mutation(value);write(tmp_path,value)
    with pytest.raises(ValueError,match=match):summarize(tmp_path)


def test_failed_partial_cannot_be_promoted_to_success_or_eligible_candidate(tmp_path):
    value=record(status="all_candidates_failed")
    value["avg_err"]=.001
    write(tmp_path,value)
    with pytest.raises(ValueError,match="Failed partial error"):summarize(tmp_path)
    value=record()
    value["selection_history"][0].update(success=False,status="line_search_failed")
    value["fit_errors"]=[deepcopy(value["selection_history"][0])]
    write(tmp_path,value)
    with pytest.raises(ValueError,match="eligible validation score"):summarize(tmp_path)


@pytest.mark.parametrize("different_config", [False,True])
def test_mixed_configurations_and_implementations_cannot_be_pooled(tmp_path,different_config):
    write(tmp_path,record(0))
    if different_config:
        other=record(1,config=RunnerConfig(iterations=3,penalties_u=(.0025,.04),penalties_v=(.01,)))
    else:
        other=record(1);other["implementation_fingerprint"]="d"*64
    write(tmp_path,other)
    with pytest.raises(ValueError,match="different tuning configurations or implementations"):summarize(tmp_path)


def test_repeated_exp1_exp4_cell_is_verified_and_counted_once_as_a_dataset(tmp_path):
    write(tmp_path,record(experiment="exp1"))
    second=record(experiment="exp4")
    write(tmp_path,second)
    _,_,meta=summary.summarize(tmp_path,experiments=(0,3),seed_ids=(0,))
    assert meta["nominal_cells"] == 11 and meta["unique_requested_datasets"] == 10
    assert meta["shared_exp1_exp4_seeds_verified"] == [0]
    second["C_hat"][0][0]=.1
    write(tmp_path,second)
    with pytest.raises(ValueError,match="Shared exp1/exp4 cell disagrees"):
        summary.summarize(tmp_path,experiments=(0,3),seed_ids=(0,))


def test_output_report_keeps_selection_and_optimizer_convergence_distinct(tmp_path,monkeypatch):
    write(tmp_path,record(selected=0))
    rows,paper,meta=summarize(tmp_path,seed_ids=(0,))
    monkeypatch.setattr(summary,"_plot",lambda *args:None)
    output=tmp_path/"summary"
    summary.write_outputs(rows,paper,meta,output)
    report=(output/"comparison.md").read_text()
    assert "160 training and 40 validation" in report
    assert "no full-data refit" in report
    assert "does not identify a preference" in report
    assert "Reaching the iteration limit is not convergence" in report
    assert "Failed partial fits never become successful observations" in report
    assert (output/"comparison.csv").is_file() and (output/"validation_audit.json").is_file()
