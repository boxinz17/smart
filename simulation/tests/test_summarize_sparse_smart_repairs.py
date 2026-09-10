"""The repair audit rejects data drift and misleading success summaries."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SPEC = importlib.util.spec_from_file_location("repair_summary", Path(__file__).parents[1] / "summarize_sparse_smart_repairs.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def valid_pair():
    config = dict(runner=dict(iterations=3, init_penalties=[.03], penalties_u=[.01, .02],
        penalties_v=[.01], n_validation=100), candidate_count=2,
        margins=dict(anchor_min=.005))
    common = dict(schema_version=1, method="SparseSMARTExternal", model="model1", experiment="exp2", rd_seed_id=0, random_seed=12,
        setting=dict(n=4, p=2, q=2, target_rank=1, source_rank=2, sigma0=.01, suffix="r=1"),
        generator_arguments=dict(n=4, random_seed=12), n_train=4, n_validation=100,
        all_training_rows_used=True, training_matches_legacy=True, refit_on_all_data=False,
        validation_seed_metadata=dict(seed_sequence_entropy=[12, 1397970481]),
        split=dict(n_train=4, n_validation=100), input_fingerprint="a"*64,
        training_observed_input_fingerprint="b"*64, validation_observed_input_fingerprint="c"*64,
        evaluation_truth_fingerprint="d"*64, implementation_fingerprint="e"*64, applicable=True)
    common["input_fingerprint"] = summary._digest_json(dict(training_observed=common[summary.HASH_KEYS[0]],
        validation_observed=common[summary.HASH_KEYS[1]], evaluation_truth=common[summary.HASH_KEYS[2]]))
    old = dict(deepcopy(common), configuration=config, status="all_candidates_failed", success=False,
        avg_err=None, validation_loss=None, selection_history=[])
    new = dict(deepcopy(common), configuration=deepcopy(config), status="complete", success=True,
        avg_err=.1, validation_loss=2., n_iter=3, selected_iteration=1,
        termination_reason="max_iterations", selection_history=[], validation_history=[],
        C_hat=[[.2, 0.], [0., 0.]], fit_errors=[])
    new["configuration"]["runner"].update(initialization_spectrum="projected", refinement_solver="anchor_projected")
    for i, penalty in enumerate((.01, .02)):
        params = dict(init_penalty=.03, penalty_u=penalty, penalty_v=.01,
                      support_limits=[1, 1], step_size_inverse=20.)
        old["selection_history"].append(dict(candidate_id=i, params=deepcopy(params), success=False))
        loss = 3.-i
        history = [dict(iteration=j, loss=v) for j,v in enumerate((loss+1, loss, loss+1, loss+2))]
        diag = dict(initialization_spectrum="projected", refinement_solver="anchor_projected",
            initialization_singular_values_original=[0.], initialization_singular_values_projected=[.2],
            initialization_spectrum_repaired=True, initialization_spectrum_correction_norm=.2,
            stationarity_scope="full_chart_constraints", diagnostic_coordinates="omega,d,H",
            optimization_converged=False, selected_converged=False,
            projected_gradient_norm=.2, last_projected_gradient_norm=.1)
        new["selection_history"].append(dict(candidate_id=i, params=params, n_iter=3,
            status="completed", success=True, selected_iteration=1, validation_mse=loss,
            validation_history=history, termination_reason="max_iterations", diagnostics=diag))
    new["best_params"] = deepcopy(new["selection_history"][1]["params"])
    new["validation_history"] = deepcopy(new["selection_history"][1]["validation_history"])
    new["diagnostics"] = new["selection_history"][1]["diagnostics"]
    new["history"] = [dict(iteration=j, objective=10.-j, smooth_loss=9.9-j,
        penalty_value=.1, anchor_min_u=.2, anchor_min_v=.3) for j in range(4)]
    for record in (old, new):
        record["configuration_fingerprint"] = summary._digest_json({key: record[key] for key in
            ("schema_version", "method", "model", "experiment", "rd_seed_id", "random_seed", "setting",
             "configuration", "generator_arguments")})
    return old, new


def test_valid_pair_counts_projection_without_claiming_convergence():
    old, new = valid_pair()
    audited = summary.validate_pair(old, new)
    assert audited["status"] == "complete" and audited["initialization_repaired"] == 2
    assert audited["termination_reason"] == "max_iterations"
    assert not audited["optimization_converged"] and not audited["selected_converged"]
    assert audited["selected_iteration"] == 1 and audited["n_iter"] == 3


def test_relative_scores_select_iterates_and_candidates_when_absolute_losses_tie():
    old, new = valid_pair()
    for index, candidate in enumerate(new["selection_history"]):
        for row, score in zip(candidate["validation_history"], (0., -1.-index, -2.-index, -1.-index)):
            row.update(loss=1e24, selection_score=score)
        candidate.update(selected_iteration=2, validation_mse=1e24, selection_score=-2.-index)
    winner = new["selection_history"][1]
    new.update(selected_iteration=2, validation_loss=1e24, selection_score=winner["selection_score"],
               validation_history=deepcopy(winner["validation_history"]))
    audited = summary.validate_pair(old, new)
    assert audited["selected_iteration"] == 2
    new["selection_score"] = 0.
    with pytest.raises(ValueError, match="selection score mismatch"):
        summary.validate_pair(old, new)


@pytest.mark.parametrize("key", summary.HASH_KEYS)
def test_changed_training_validation_or_truth_is_rejected(key):
    old, new = valid_pair()
    new[key] = "f"*64
    with pytest.raises(ValueError, match="data protocol mismatch"):
        summary.validate_pair(old, new)


def test_changed_candidate_budget_is_rejected():
    old, new = valid_pair()
    new["configuration"]["runner"]["iterations"] = 4
    with pytest.raises(ValueError, match="budget changed"):
        summary.validate_pair(old, new)


def test_changed_penalty_grid_is_rejected():
    old, new = valid_pair()
    new["selection_history"][0]["params"]["penalty_u"] = .5
    with pytest.raises(ValueError, match="grid/order changed"):
        summary.validate_pair(old, new)


def test_nonminimum_validation_candidate_is_rejected():
    old, new = valid_pair()
    new["best_params"] = new["selection_history"][0]["params"]
    with pytest.raises(ValueError, match="minimum-validation winner"):
        summary.validate_pair(old, new)


def test_nonminimum_validation_iterate_is_rejected():
    old, new = valid_pair()
    new["selection_history"][1]["selected_iteration"] = 2
    with pytest.raises(ValueError, match="minimum validation iterate"):
        summary.validate_pair(old, new)


@pytest.mark.parametrize("field,value,message", [
    ("objective", 20., "objective increased"),
    ("anchor_min_u", .001, "anchor violates"),
])
def test_invalid_accepted_history_is_rejected(field, value, message):
    old, new = valid_pair()
    new["history"][2][field] = value
    with pytest.raises(ValueError, match=message):
        summary.validate_pair(old, new)


def test_unverified_spectral_correction_is_rejected():
    old, new = valid_pair()
    new["diagnostics"]["initialization_spectrum_correction_norm"] = 0.
    with pytest.raises(ValueError, match="Spectral correction norm mismatch"):
        summary.validate_pair(old, new)


def test_iteration_cap_cannot_be_labeled_converged():
    old, new = valid_pair()
    new["diagnostics"]["optimization_converged"] = True
    with pytest.raises(ValueError, match="optimization convergence flag mismatch"):
        summary.validate_pair(old, new)


@pytest.mark.parametrize("key,message", [
    ("configuration_fingerprint", "Configuration fingerprint mismatch"),
    ("input_fingerprint", "Combined input fingerprint mismatch"),
])
def test_corrupt_record_fingerprints_are_rejected(key, message):
    old, new = valid_pair()
    old[key] = new[key] = "f"*64
    with pytest.raises(ValueError, match=message):
        summary.validate_pair(old, new)


def test_partial_constraint_diagnostic_cannot_claim_full_projection():
    old, new = valid_pair()
    new["diagnostics"]["stationarity_scope"] = "spectral_only"
    with pytest.raises(ValueError, match="full-constraint"):
        summary.validate_pair(old, new)


@pytest.mark.parametrize("manifest,message", [
    ({"errors":[{"message":"failed worker"}]}, "worker errors"),
    ({"errors":[], "finished":None}, "not finished"),
    ({"errors":[], "finished":"now", "expected_cells":66,
      "groups":{"previous_failure":60,"control":6}, "cells":[]}, "Missing regression manifest cells"),
])
def test_incomplete_or_failed_manifest_cannot_make_success_report(tmp_path, manifest, message):
    path = tmp_path / "repair_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        summary.build_audit(path)
