"""Optional checkpoint metadata must agree with eligible candidate histories."""
from copy import deepcopy

import pytest

from test_sparse_smart_checkpoint_integration import checkpoint_record, validate


def record_with_metadata(kind):
    value, setting, config = checkpoint_record(kind)
    records = value["selection_history"]
    first, last = records[0], records[3]
    value["tuning_diagnostics"] = dict(
        iteration_budgets=[1, 2], checkpoint_execution="independent_fits",
        selected_budget=1, selected_candidate_id=0,
        selected_checkpoint_status=first["status"],
        selected_checkpoint_termination_reason=first["termination_reason"],
        retained_budget_checkpoints=2, successful_candidate_fits=3, failed_candidate_fits=1,
        budget_statuses=[
            dict(iteration_budget=1, success=True, successful_candidates=2, failed_candidates=0,
                 best_candidate_id=0, best_validation_mse=first["validation_mse"]),
            dict(iteration_budget=2, success=True, successful_candidates=1, failed_candidates=1,
                 best_candidate_id=3, best_validation_mse=last["validation_mse"])],
        retained_earlier_checkpoint=True,
        selected_checkpoint_continuations=[dict(candidate_id=2, iteration_budget=2,
            success=False, status="numerical_stagnation", termination_reason="numerical_stagnation")],
        selected_checkpoint_retained_after_failure=True,
    )
    return value, setting, config


@pytest.mark.parametrize("kind", ["tuned", "external", "grid"])
def test_consistent_checkpoint_metadata_is_accepted_without_mutation(kind):
    value, setting, config = record_with_metadata(kind)
    before = deepcopy(value)
    validate(kind, value, setting, config)
    assert value == before


@pytest.mark.parametrize("kind", ["tuned", "external", "grid"])
@pytest.mark.parametrize("invalid_id", [999, True])
def test_wrong_grid_identity_is_rejected_even_for_failed_candidate(kind, invalid_id):
    value, setting, config = record_with_metadata(kind)
    value["selection_history"][2]["grid_candidate_id"] = invalid_id
    value["fit_errors"] = [deepcopy(c) for c in value["selection_history"] if not c["success"]]
    with pytest.raises(ValueError, match="Grid candidate ID"):
        validate(kind, value, setting, config)


@pytest.mark.parametrize("kind", ["tuned", "external", "grid"])
@pytest.mark.parametrize("key,bad", [
    ("selected_budget", 2), ("selected_candidate_id", 2),
    ("selected_checkpoint_status", "numerical_stagnation"),
    ("selected_checkpoint_termination_reason", "numerical_stagnation"),
    ("selected_checkpoint_retained_after_failure", False),
    ("retained_earlier_checkpoint", False), ("retained_budget_checkpoints", 1),
    ("successful_candidate_fits", 4), ("failed_candidate_fits", True),
    ("iteration_budgets", [2]), ("checkpoint_execution", "resumed"),
    ("selected_checkpoint_continuations", []),
])
def test_checkpoint_diagnostics_cannot_contradict_saved_candidate_outcomes(kind, key, bad):
    value, setting, config = record_with_metadata(kind)
    value["tuning_diagnostics"][key] = bad
    with pytest.raises(ValueError, match=f"Checkpoint tuning diagnostics.*{key}"):
        validate(kind, value, setting, config)


@pytest.mark.parametrize("kind", ["tuned", "external", "grid"])
def test_budget_winner_cannot_be_a_failed_partial_candidate(kind):
    value, setting, config = record_with_metadata(kind)
    budget = value["tuning_diagnostics"]["budget_statuses"][1]
    budget.update(best_candidate_id=2, best_validation_mse=.00001)
    with pytest.raises(ValueError, match="Checkpoint tuning diagnostics.*budget_statuses"):
        validate(kind, value, setting, config)


@pytest.mark.parametrize("kind", ["tuned", "external", "grid"])
def test_absent_or_partial_new_metadata_remains_compatible(kind):
    value, setting, config = record_with_metadata(kind)
    value.pop("tuning_diagnostics")
    for record in value["selection_history"]:
        record.pop("grid_candidate_id")
    value["fit_errors"] = [deepcopy(c) for c in value["selection_history"] if not c["success"]]
    validate(kind, value, setting, config)
    value["tuning_diagnostics"] = {"selected_budget": 1}
    validate(kind, value, setting, config)


@pytest.mark.parametrize("kind", ["tuned", "external", "grid"])
def test_nonmapping_tuning_diagnostics_is_rejected(kind):
    value, setting, config = record_with_metadata(kind)
    value["tuning_diagnostics"] = None
    with pytest.raises(ValueError, match="Invalid checkpoint tuning diagnostics"):
        validate(kind, value, setting, config)


@pytest.mark.parametrize("kind", ["tuned", "external", "grid"])
def test_all_failed_search_has_no_selected_checkpoint_in_metadata(kind):
    value, setting, config = record_with_metadata(kind)
    for candidate in value["selection_history"]:
        candidate.update(success=False, status="numerical_stagnation", validation_mse=None,
                         termination_reason="numerical_stagnation")
    value.update(success=False, status="all_candidates_failed", all_candidates_failed=True,
                 avg_err=None, C_hat=None, best_params=None, selected_budget=None, selected_candidate_id=None)
    value["fit_errors"] = deepcopy(value["selection_history"])
    value["tuning_diagnostics"] = dict(selected_budget=None, selected_candidate_id=None,
        selected_checkpoint_status=None, retained_budget_checkpoints=0,
        successful_candidate_fits=0, failed_candidate_fits=4,
        selected_checkpoint_continuations=[], selected_checkpoint_retained_after_failure=False)
    validate(kind, value, setting, config)
    value["tuning_diagnostics"]["selected_budget"] = 1
    with pytest.raises(ValueError, match="Checkpoint tuning diagnostics.*selected_budget"):
        validate(kind, value, setting, config)
