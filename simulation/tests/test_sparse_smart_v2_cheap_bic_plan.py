"""RSC equivalence, immutable cohorts and 200-update candidate accounting."""
import copy
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
import sparse_smart_v2_bic_plan as original
import sparse_smart_v2_cheap_bic_plan as cheap


@pytest.fixture(scope="module")
def reference():
    return original.build(Path("/scratch2/test-exhaustive"), "a" * 64, seeds=list(range(10)))


def rank_records(reference, rank=5):
    records = {}
    for group in reference["groups"]:
        records[group["dataset_id"]] = dict(schema_version=1, status="complete",
            selected_rank=rank, raw_rsc_rank=rank, source_rank_cap=20,
            source_rank_tolerance=1e-10, minimum_rank=1, theta=1., sigma=.5, threshold=1.,
            n_train=group["n_train"], p=group["p"], q=group["q"], design_rank=group["p"],
            residual_df=group["q"] * (group["n_train"] - group["p"]),
            original_implementation_sha256=cheap.sha(cheap.RSC_PATH),
            training_observed_input_fingerprint="f" * 64,
            validation_used=False, generating_rank_used=False, old_grid_rounding=False)
    return records


def test_rsc_reproduces_original_rule_and_retains_even_rank():
    rng = np.random.default_rng(139)
    X = rng.normal(size=(40, 4))
    C = np.zeros((4, 3))
    C[:2, :2] = np.diag([6., 3.])
    Y = X @ C + rng.normal(scale=.02, size=(40, 3))
    C0 = np.eye(4, 3)
    result = cheap.rank_selection(X, Y, C0)
    selector = cheap._original_selector()
    assert result["raw_rsc_rank"] == int(selector.select_rank(X, Y)) == 2
    assert result["selected_rank"] == 2
    assert result["theta"] == 1.
    assert result["residual_df"] == 3 * (40 - np.linalg.matrix_rank(X))
    assert result["inputs"] == ["X", "Y", "C0"]
    assert result["original_implementation_sha256"] == cheap.sha(cheap.RSC_PATH)
    assert list(inspect.signature(cheap.rank_selection).parameters) == ["X", "Y", "C0"]
    assert not result["old_grid_rounding"] and not result["validation_used"]


def test_rsc_observed_source_cap_and_minimum_are_not_initializer_rank():
    rng = np.random.default_rng(240)
    X = rng.normal(size=(40, 4))
    C0 = np.diag([4., 3., 0., 0.])
    Y = X * 3 + rng.normal(scale=.02, size=(40, 4))
    capped = cheap.rank_selection(X, Y, C0)
    assert capped["raw_rsc_rank"] == 4
    assert capped["selected_rank"] == capped["source_rank_cap"] == 2
    projector = cheap._original_selector().compute_projection_matrix(X)
    residual_only = (np.eye(40) - projector) @ rng.normal(size=(40, 4))
    minimum = cheap.rank_selection(X, residual_only, C0)
    assert minimum["raw_rsc_rank"] == 0
    assert minimum["selected_rank"] == 1


@pytest.mark.parametrize("case, message", [
    ("zero_source", "rank-zero source"), ("saturated", "degrees of freedom"),
    ("zero_noise", "strictly positive"), ("nonfinite", "finite real matrices"),
    ("wrong_shape", "source dimensions")])
def test_undefined_rank_inputs_fail_explicitly(case, message):
    X = np.vstack((np.eye(3), np.zeros((3, 3))))
    Y = np.vstack((np.eye(3), np.ones((3, 3))))
    C0 = np.eye(3)
    if case == "zero_source": C0 *= 0
    elif case == "saturated": X, Y = np.eye(3), np.eye(3)
    elif case == "zero_noise": Y = X.copy()
    elif case == "nonfinite": Y[0, 0] = np.nan
    elif case == "wrong_shape": C0 = np.eye(2)
    with pytest.raises(ValueError, match=message):
        cheap.rank_selection(X, Y, C0)


def test_split_is_frozen_before_rank_and_has_balanced_disjoint_groups(reference, tmp_path):
    split = cheap.freeze_split(tmp_path, reference)
    assert cheap.freeze_split(tmp_path, reference) == split
    phases = split["phases"]
    assert phases["development"]["seed_ids"] == list(range(5))
    assert phases["assessment"]["seed_ids"] == list(range(5, 10))
    assert all(len(p["group_ids"]) == 165 and len(p["dataset_ids"]) == 150 for p in phases.values())
    assert set(phases["development"]["dataset_ids"]).isdisjoint(phases["assessment"]["dataset_ids"])
    assert "rank_records" not in split and "selected_init_penalties" not in split
    changed = copy.deepcopy(reference)
    changed["groups"][0]["random_seed"] += 1
    changed["plan_fingerprint"] = original.digest({k: v for k, v in changed.items() if k != "plan_fingerprint"})
    with pytest.raises(ValueError, match="frozen input differs"):
        cheap.freeze_split(tmp_path, changed)


@pytest.mark.parametrize("phase", ["development", "assessment"])
def test_candidate_count_reuse_and_preserved_protocol_aliases(reference, phase):
    before = original.digest(reference)
    plan = cheap.build_plan(Path("/scratch2/test-cheap"), "b" * 64, reference, rank_records(reference), phase)
    assert original.digest(reference) == before
    assert (plan["n_groups"], plan["n_datasets"], plan["n_tasks"], plan["n_candidates"]) == (165, 150, 990, 15015)
    assert plan["n_rrr_candidates"] == 165
    assert plan["n_iterative_candidates"] == 165 * 6 * 5 * 3
    first = [t for t in plan["tasks"] if t["group_id"] == plan["groups"][0]["group_id"]]
    assert len(first) == 6 and sum(t["include_rrr"] for t in first) == 1
    assert all(t["free_counts"] == [5, 7, 10, 15, 20] for t in first)
    assert all(t["initializer_source_rank"] == 10 for t in first)
    assert 2 * len(first[0]["free_counts"]) * 3 + 1 == 31
    config = plan["configuration"]
    assert config["penalty_pairs"] == [[.001, .001], [.0025, .0025], [.01, .01]]
    assert config["iterations"] == 200 and config["validation_patience"] is None
    assert config["validation_interval"] is None and config["validation_iterations"] == []
    assert config["refinement_solver"] == "masked_anchor_projected"
    assert set(g["protocol"] for g in plan["groups"]) == {"low_floor", "standard_floor"}
    old = {g["group_id"]: g for g in reference["groups"]}
    for group in plan["groups"]:
        assert group["margins"] == old[group["group_id"]]["margins"]
        assert group["dataset_id"] == old[group["group_id"]]["dataset_id"]
    assert plan["plan_fingerprint"] == original.digest({k: v for k, v in plan.items() if k != "plan_fingerprint"})


def test_even_and_large_ranks_are_never_rounded(reference):
    records = rank_records(reference, rank=6)
    first = reference["groups"][0]["dataset_id"]
    records[first].update(selected_rank=12, raw_rsc_rank=12)
    plan = cheap.build_plan(Path("/scratch2/test-even"), "b" * 64, reference, records, "development")
    assert plan["selection_scope"]["fitted_ranks"] == [6, 12]
    selected = [t for t in plan["tasks"] if t["rank"] == 12]
    assert selected and all(t["initializer_source_rank"] == 12 for t in selected)
    assert all(t["free_counts"] == [12, 15, 20] for t in selected)
    assert cheap.free_counts(6, 8, 12) == [6, 7]


def test_missing_or_nontraining_rank_records_are_not_silent_fallbacks(reference):
    records = rank_records(reference)
    key = reference["groups"][0]["dataset_id"]
    del records[key]
    with pytest.raises(ValueError, match="missing RSC"):
        cheap.build_plan(Path("/scratch2/test-missing"), "b" * 64, reference, records, "development")
    records = rank_records(reference)
    records[key]["validation_used"] = True
    with pytest.raises(ValueError, match="non-training-only"):
        cheap.build_plan(Path("/scratch2/test-leak"), "b" * 64, reference, records, "development")


def test_plan_freezing_requires_split_and_refuses_rank_changes(reference, tmp_path):
    records = rank_records(reference)
    with pytest.raises(ValueError, match="freeze split"):
        cheap.freeze_plan(tmp_path, "b" * 64, reference, records, "development")
    cheap.freeze_split(tmp_path, reference)
    first = cheap.freeze_plan(tmp_path, "b" * 64, reference, records, "development")
    assert cheap.freeze_plan(tmp_path, "b" * 64, reference, records, "development") == first
    assert (tmp_path / "work-items.tsv").read_text().splitlines() == [str(x) for x in range(990)]
    key = reference["groups"][0]["dataset_id"]
    records[key].update(selected_rank=6, raw_rsc_rank=6)
    with pytest.raises(ValueError, match="frozen input differs"):
        cheap.freeze_plan(tmp_path, "b" * 64, reference, records, "development")
