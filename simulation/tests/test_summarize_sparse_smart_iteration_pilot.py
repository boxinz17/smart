"""Paired budget audits use saved fits and fake data, never run an estimator."""
from copy import deepcopy
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import summarize_sparse_smart_iteration_pilot as summary
from run_sparse_smart import _digest_json
from test_summarize_sparse_smart_external_grid import data, record as grid_record, write, SEEDS


@lru_cache(None)
def fixture_arrays(model,experiment,index,seed):
    setting=summary.experiment_settings(model,experiment)[index]
    generated=summary.external_validation_data.generate_external_validation(
        n_train=setting.n,p=setting.p,q=setting.q,sigma0=setting.sigma0,
        random_seed=int(SEEDS[seed]),n_validation=100,sigma=.5,r_star=5,r0_star=10,
        seed_tag=1397970481,generate_data_fn=data)
    _,_,right=np.linalg.svd(generated["X_validation"],full_matrices=False)
    # Synthetic coefficient perturbations of equal Frobenius size can have very
    # different prediction losses; this keeps truth and validation distinct.
    high=np.sqrt(setting.p)*np.repeat(right[0,:,None],setting.q,axis=1)
    low=np.sqrt(setting.p)*np.repeat(right[-1,:,None],setting.q,axis=1)
    return generated,high,low


def make_record(*, budget=2, seed=0, error=.1, base_error=None, early=False, failed=False, model=0, suffix="rs=5"):
    experiment = 3 if suffix.startswith("sigma0") else 2
    index = next(i for i,s in enumerate(summary.experiment_settings(model,experiment)) if s.suffix==suffix)
    config = summary.runner.RunnerConfig(iterations=budget,initialization_spectrum="projected",
        refinement_solver="anchor_projected",init_penalties=(.03,),
        penalties_u=(.0025,.01,.04),penalties_v=(.0025,.01,.04))
    value = grid_record(model=model,experiment=experiment,index=index,seed=seed,error=error,
                        config=config,selected=1)
    generated,high,low=fixture_arrays(model,experiment,index,seed)
    base_error=error if budget==2 or early else .1 if base_error is None else base_error
    coefficient=generated["C_star"]+error*(high if budget==2 or early else low)
    base_coefficient=generated["C_star"]+base_error*high
    score=float(np.mean((generated["Y_validation"]-generated["X_validation"]@coefficient)**2))
    base_score=float(np.mean((generated["Y_validation"]-generated["X_validation"]@base_coefficient)**2))
    value["C_hat"]=coefficient.tolist()
    value["avg_err"]=float(np.linalg.norm(coefficient-generated["C_star"])/np.sqrt(coefficient.size))
    value["selection_history"] = []
    for i,(pu,pv) in enumerate(product(config.penalties_u,config.penalties_v)):
        n_iter = 1 if early else budget
        losses = ([score+1.,score] if early else
                  [base_score+1.,base_score+.2,base_score,(base_score+score)/2,score][:n_iter+1])
        losses = [loss + .1*i for loss in losses]
        selected = min(range(len(losses)),key=losses.__getitem__)
        diagnostics = dict(refinement_solver="anchor_projected",stationarity_scope="full_chart_constraints",
            diagnostic_coordinates="omega,d,H",optimization_converged=early,selected_converged=early,
            last_projected_gradient_norm=1e-7 if early else .01)
        value["selection_history"].append(dict(candidate_id=i,
            params=dict(init_penalty=.03,penalty_u=pu,penalty_v=pv,
                        support_limits=value["configuration"]["support_limits"][0],step_size_inverse=20.),
            success=True,status="converged" if early else "completed",message="saved fixture",
            validation_mse=losses[selected],partial_validation_mse=None,has_partial_coefficient=False,
            selected_iteration=selected,n_iter=n_iter,termination_reason="stationarity" if early else "max_iterations",
            validation_history=[dict(iteration=j,loss=x) for j,x in enumerate(losses)],
            diagnostics=diagnostics,elapsed_time_sec=.1))
    select_winner(value)
    if failed:
        for c in value["selection_history"]:
            c.update(success=False,status="initialization_spectrum_failed",validation_mse=None,
                selected_iteration=None,n_iter=0,termination_reason="initialization_spectrum_failed",
                validation_history=[],diagnostics={})
        value.update(status="all_candidates_failed",success=False,all_candidates_failed=True,
            avg_err=None,initial_avg_err=None,C_hat=None,best_params=None,selected_iteration=None,n_iter=0,
            validation_loss=None,validation_history=[],history=[],selected_supports=None,diagnostics={},
            termination_reason=None,failure_reason="no_successful_candidate",
            fit_errors=deepcopy(value["selection_history"]))
    return value


def select_winner(value):
    candidates = value["selection_history"]
    winner = min((c for c in candidates if c["success"]),key=lambda c:c["validation_mse"])
    value.update(best_params=deepcopy(winner["params"]),selected_iteration=winner["selected_iteration"],
        n_iter=winner["n_iter"],validation_loss=winner["validation_mse"],
        validation_history=deepcopy(winner["validation_history"]),
        termination_reason=winner["termination_reason"],diagnostics=deepcopy(winner["diagnostics"]),
        history=[dict(iteration=j,support_u=1,support_v=1) for j in range(winner["n_iter"]+1)],
        fit_errors=[deepcopy(c) for c in candidates if not c["success"]])


def summarize(base,extended,**kwargs):
    return summary.summarize(base,extended,model_ids=(0,),base_budget=2,extended_budget=4,
                             generate_data_fn=data,**kwargs)


def rehash(value):
    identity={k:value[k] for k in ("schema_version","method","model","experiment","rd_seed_id",
        "random_seed","setting","configuration","generator_arguments")}
    value["configuration_fingerprint"]=_digest_json(identity)


def test_empty_expected_grid_has45_pairs_and_explicit_missingness(tmp_path):
    rows,audit=summary.summarize(tmp_path/"short",tmp_path/"long",generate_data_fn=data)
    assert len(rows)==9 and audit["expected_pairs"]==45
    assert audit["missing_base"]==45 and audit["missing_extended"]==45
    assert audit["successful_pairs"]==0 and not audit["all_expected_records_present"]
    assert all(r["base_error_mean"] is None and r["extended_error_se"] is None for r in rows)


def test_paired_means_se_failures_and_missingness_use_same_successful_seeds(tmp_path):
    base,extended=tmp_path/"base",tmp_path/"extended"
    for seed,old,new in ((0,.1,.2),(1,.3,.1)):
        write(base,make_record(seed=seed,error=old))
        write(extended,make_record(seed=seed,error=new,base_error=old,budget=4))
    write(base,make_record(seed=2,failed=True))
    write(extended,make_record(seed=2,budget=4,failed=True))
    write(base,make_record(seed=3,error=.9))
    rows,audit=summarize(base,extended,seed_ids=(0,1,2,3))
    row=rows[0]
    assert row["successful_pairs"]==2 and row["paired_records"]==3
    assert row["base_counts"]["successful"]==3 and row["extended_counts"]["successful"]==2
    assert row["missing_base"]==0 and row["missing_extended"]==1
    assert row["mean_is_conditional_on_joint_success"]
    assert row["base_error_mean"]==pytest.approx(.2) and row["base_error_se"]==pytest.approx(.1)
    assert row["extended_error_mean"]==pytest.approx(.15) and row["extended_error_se"]==pytest.approx(.05)
    assert row["error_change_mean"]==pytest.approx(-.05) and row["error_change_se"]==pytest.approx(.15)
    assert row["validation_change_mean"]<0
    assert row["selected_after_base_budget"]==2 and row["selected_after_base_fraction"]==1.
    assert audit["prefix_classifications"]=={"continued":18,"failure_reproduced":9}
    assert audit["regenerated_unique_datasets"]==4


def test_early_convergence_reproduces_without_requiring500_history_rows(tmp_path):
    base,extended=tmp_path/"base",tmp_path/"extended"
    write(base,make_record(early=True))
    write(extended,make_record(budget=4,early=True))
    rows,audit=summarize(base,extended,seed_ids=(0,))
    assert audit["prefix_classifications"]=={"early_convergence_reproduced":9}
    assert rows[0]["base_error_se"] is None and rows[0]["extended_error_se"] is None
    assert rows[0]["extended_counts"]["optimization_converged"]==1
    assert rows[0]["selected_after_base_budget"]==0


def test_early_convergence_cannot_silently_continue(tmp_path):
    base,extended=tmp_path/"base",tmp_path/"extended"
    shorter=make_record(early=True)
    longer=make_record(budget=4)
    for old,new in zip(shorter["selection_history"],longer["selection_history"]):
        new["validation_history"][:2]=deepcopy(old["validation_history"])
    select_winner(longer)
    write(base,shorter)
    write(extended,longer)
    with pytest.raises(ValueError,match="Early candidate stop did not reproduce"):
        summarize(base,extended,seed_ids=(0,))


def test_changed_prefix_rejected_even_when_each_winner_remains_valid(tmp_path):
    base,extended=tmp_path/"base",tmp_path/"extended"
    write(base,make_record())
    longer=make_record(budget=4)
    longer["selection_history"][7]["validation_history"][0]["loss"]+=.01
    write(extended,longer)
    with pytest.raises(ValueError,match="validation prefix differs"):
        summarize(base,extended,seed_ids=(0,))


@pytest.mark.parametrize("field",["implementation_fingerprint","training_observed_input_fingerprint"])
def test_data_or_implementation_changes_cannot_be_pooled(tmp_path,field):
    base,extended=tmp_path/"base",tmp_path/"extended"
    write(base,make_record())
    longer=make_record(budget=4)
    longer[field]="f"*64
    if field.startswith("training"):
        longer["input_fingerprint"]=_digest_json(dict(training_observed=longer[field],
            validation_observed=longer["validation_observed_input_fingerprint"],
            evaluation_truth=longer["evaluation_truth_fingerprint"]))
    write(extended,longer)
    with pytest.raises(ValueError,match="implementation fingerprints|Regenerated training"):
        summarize(base,extended,seed_ids=(0,))


def test_configuration_change_beyond_iterations_is_rejected_with_valid_identity(tmp_path):
    base,extended=tmp_path/"base",tmp_path/"extended"
    write(base,make_record())
    longer=make_record(budget=4)
    longer["configuration"]["runner"]["stationarity_tol"]=1e-5
    rehash(longer);write(extended,longer)
    with pytest.raises(ValueError,match="mixed configurations beyond iteration budget"):
        summarize(base,extended,seed_ids=(0,))


def test_saved_truth_error_cannot_select_winner_or_escape_recomputation(tmp_path):
    base,extended=tmp_path/"base",tmp_path/"extended"
    write(base,make_record(error=.1))
    longer=make_record(budget=4,error=.3)
    write(extended,longer)
    rows,audit=summarize(base,extended,seed_ids=(0,))
    assert rows[0]["error_change_mean"]>0 and rows[0]["validation_change_mean"]<0
    assert "never used to choose the iteration budget" in audit["selection_policy"]
    longer["avg_err"]=.001;write(extended,longer)
    with pytest.raises(ValueError,match="Saved coefficient error"):
        summarize(base,extended,seed_ids=(0,))


def test_consistent_history_and_prefix_cannot_forge_selected_prediction_score(tmp_path):
    base,extended=tmp_path/"base",tmp_path/"extended"
    write(base,make_record())
    longer=make_record(budget=4)
    candidate=longer["selection_history"][0]
    candidate["validation_history"][-1]["loss"]+=.01
    candidate["validation_mse"]+=.01
    select_winner(longer)
    write(extended,longer)
    with pytest.raises(ValueError,match="Selected validation MSE disagrees with saved coefficient predictions"):
        summarize(base,extended,seed_ids=(0,))


def test_report_exposes_failed_continuation_and_worsening_from_lost_winner(tmp_path):
    base,extended=tmp_path/"base",tmp_path/"extended"
    write(base,make_record())
    longer=make_record(budget=4,error=.2)
    failed=longer["selection_history"][0]
    failed.update(success=False,status="numerical_stagnation",termination_reason="numerical_stagnation",
        validation_mse=None,partial_validation_mse=failed["validation_history"][3]["loss"],has_partial_coefficient=True,n_iter=3,selected_iteration=3,
        validation_history=failed["validation_history"][:4])
    failed["diagnostics"]["last_projected_gradient_norm"]=1.2388902910912963e-6
    for c in longer["selection_history"][1:]:
        for row in c["validation_history"][3:]:row["loss"]=c["validation_history"][2]["loss"]+1.
        c["selected_iteration"]=2;c["validation_mse"]=c["validation_history"][2]["loss"]
    select_winner(longer)
    generated,high,_=fixture_arrays(0,2,2,0)
    residual=generated["Y_validation"]-generated["X_validation"]@generated["C_star"]
    direction=generated["X_validation"]@high
    # Match the new eligible winner's recorded validation MSE exactly.
    a=float(np.mean(direction**2));b=-2*float(np.mean(residual*direction))
    c=float(np.mean(residual**2))-longer["validation_loss"]
    amplitude=(-b+np.sqrt(b*b-4*a*c))/(2*a)
    coefficient=generated["C_star"]+amplitude*high
    longer["C_hat"]=coefficient.tolist();longer["avg_err"]=float(np.linalg.norm(coefficient-generated["C_star"])/np.sqrt(coefficient.size))
    write(extended,longer)
    rows,audit=summarize(base,extended,seed_ids=(0,))
    assert rows[0]["validation_worsened"]==1 and rows[0]["pairs_losing_eligible_candidates"]==1
    pair=audit["pairs"][0]
    assert pair["eligibility_lost_candidates"]==[0] and pair["base_winner_eligible_in_extended"] is False
    assert longer["best_params"]==longer["selection_history"][1]["params"]
    assert pair["base_winner_candidate_id"]==0 and audit["pairs_losing_base_winner"]==1
    assert len(audit["candidate_failures"])==1
    failure=audit["candidate_failures"][0]
    assert failure["cohort"]=="extended" and failure["budget"]==4
    assert failure["model"]=="model1" and failure["experiment"]=="exp3"
    assert failure["setting"]=="rs=5" and failure["seed_id"]==0 and failure["candidate_id"]==0
    assert failure["status"]=="numerical_stagnation" and failure["n_iter"]==3
    assert failure["terminal_residual"]==failed["diagnostics"]["last_projected_gradient_norm"]
    assert failure["partial_validation_mse"]<longer["validation_loss"]
    output=tmp_path/"failure_report"
    summary.write_outputs(rows,audit,output)
    text=(output/"report.md").read_text()
    assert text.index("## Candidate failures and selection")<text.index("## Paired outcomes")
    assert "Failed candidates: **0 at 2; 1 at 4**" in text
    assert "| extended (4) | model1 | rs=5 | 0 | 0 | numerical_stagnation | 3 | 1.23889e-06 |" in text
    assert "2-update winner lost eligibility in 1 paired cell(s)" in text
    assert "| model1 | rs=5 | 0 | 0 | numerical_stagnation |" in text
    assert "| Model | Setting | Improved | Unchanged | Worsened |" in text
    assert "| model1 | rs=5 | 0 | 0 | 1 |" in text
    assert "not automatically retained as a fallback" in text
    assert "not a stationarity certificate" in text
    assert json.loads((output/"audit.json").read_text())["candidate_failures"]==audit["candidate_failures"]


def test_report_lists_failures_without_terminal_residuals_in_both_cohorts(tmp_path):
    base,extended=tmp_path/"base",tmp_path/"extended"
    write(base,make_record(failed=True))
    write(extended,make_record(budget=4,failed=True))
    rows,audit=summarize(base,extended,seed_ids=(0,))
    assert len(audit["candidate_failures"])==18 and audit["pairs_losing_base_winner"]==0
    assert all(f["terminal_residual"] is None for f in audit["candidate_failures"])
    summary.write_outputs(rows,audit,tmp_path/"report")
    text=(tmp_path/"report"/"report.md").read_text()
    assert "Failed candidates: **9 at 2; 9 at 4**" in text
    for cohort,budget in (("base",2),("extended",4)):
        assert f"| {cohort} ({budget}) | model1 | rs=5 | 0 | 0 | initialization_spectrum_failed | 0 | — |" in text
    assert "| model1 | rs=5 | 0 | 0 | 0 |" in text
    assert "winner lost eligibility" not in text


def test_same_selected_candidate_and_iteration_require_same_coefficient(tmp_path):
    base=make_record(early=True,error=.1)
    extended=make_record(budget=4,early=True,error=.1)
    extended["C_hat"][0][0]+=.01
    with pytest.raises(ValueError,match="different coefficients"):
        summary.validate_pair(base,extended,base_budget=2,extended_budget=4)


def test_forged_selected_model_stationarity_does_not_override_candidate_status(tmp_path):
    base,extended=tmp_path/"base",tmp_path/"extended"
    write(base,make_record())
    longer=make_record(budget=4)
    longer["termination_reason"]="stationarity"
    longer["diagnostics"].update(optimization_converged=True,selected_converged=True)
    write(extended,longer)
    with pytest.raises(ValueError,match="termination differs from the winning candidate"):
        summarize(base,extended,seed_ids=(0,))


def test_unavailable_residual_does_not_become_zero_or_convergence(tmp_path):
    base,extended=tmp_path/"base",tmp_path/"extended"
    write(base,make_record())
    longer=make_record(budget=4)
    longer["selection_history"][0]["diagnostics"]["last_projected_gradient_norm"]=None
    select_winner(longer);write(extended,longer)
    rows,audit=summarize(base,extended,seed_ids=(0,))
    assert rows[0]["extended_terminal_residual_mean"] is None
    assert rows[0]["extended_terminal_residual_count"]==0
    assert rows[0]["extended_counts"]["unavailable_terminal_residual"]==1
    assert rows[0]["extended_counts"]["optimization_converged"]==0
    summary.write_outputs(rows,audit,tmp_path/"report")


def test_report_exposes_validation_policy_and_unavailable_single_seed_se(tmp_path):
    base,extended=tmp_path/"base",tmp_path/"extended"
    write(base,make_record())
    write(extended,make_record(budget=4,error=.2))
    rows,audit=summarize(base,extended,seed_ids=(0,))
    output=tmp_path/"new_report"
    summary.write_outputs(rows,audit,output)
    text=(output/"report.md").read_text()
    assert "Iteration-limit termination is not convergence" in text
    assert "single successful pair has no estimable SE" in text
    assert "Choose the iteration budget using validation" in text
    assert "No candidate failures occurred in the available records" in text
    assert "0.100000 ± —" in text and "0.200000 ± —" in text
    assert json.loads((output/"audit.json").read_text())["compared_candidate_prefixes"]==9
