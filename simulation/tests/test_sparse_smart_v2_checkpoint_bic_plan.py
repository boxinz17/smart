"""Coverage, deduplication and provenance for the checkpoint-BIC campaign."""
import copy
from pathlib import Path
import sys

import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
import sparse_smart_v2_bic_plan as original
import sparse_smart_v2_checkpoint_bic_plan as checkpoint


def records_for(reference, rank=5):
    return {g["dataset_id"]: dict(schema_version=1, status="complete",
        selected_rank=rank, raw_rsc_rank=rank, source_rank_cap=20,
        source_rank_tolerance=1e-10, minimum_rank=1, theta=1., sigma=.5, threshold=1.,
        n_train=g["n_train"], p=g["p"], q=g["q"], design_rank=g["p"],
        residual_df=g["q"] * (g["n_train"] - g["p"]),
        original_implementation_sha256=checkpoint.sha(checkpoint.RSC_PATH),
        training_observed_input_fingerprint="f" * 64,
        validation_used=False, generating_rank_used=False, old_grid_rounding=False)
        for g in reference["groups"]}


@pytest.fixture(scope="module")
def reference():
    return original.build(Path("/scratch2/full-reference"), "a" * 64, seeds=[0, 1])


def make_plan(reference, records=None):
    return checkpoint.build_plan(Path("/scratch2/checkpoints"), "b" * 64,
                                 reference, records_for(reference) if records is None else records)


def test_full_campaign_candidate_counts():
    reference = original.build(Path("/scratch2/full-reference"), "a" * 64)
    plan = make_plan(reference)
    assert (plan["n_groups"], plan["n_datasets"], plan["n_display_cases"]) == (3300, 3000, 6600)
    assert plan["n_tasks"] == 9600
    assert plan["n_iterative_candidates"] == 108000
    assert plan["n_rrr_candidates"] == 4800
    assert plan["n_candidates"] == 112800
    assert plan["selection_scope"]["models"] == [0, 1, 2]
    assert plan["selection_scope"]["seed_ids"] == list(range(100))
    assert plan["selection_scope"]["fitted_ranks"] == [1, 3, 5, 7, 9, 11]


def test_all_display_requests_and_automatic_requests_are_shared_exactly(reference):
    plan = make_plan(reference)
    actual = {(t["group_id"], t["rank"], s, t["init_penalty"])
              for t in plan["tasks"] for s in t["free_counts"]}
    expected = set()
    for group in plan["groups"]:
        expected.update((group["group_id"], group["selected_target_rank"], s, initial)
                        for s in group["automatic_free_counts"] for initial in checkpoint.INITIALIZERS)
    for case in plan["display_cases"]:
        expected.update((case["group_id"], case["rank"], case["free_directions"][0], initial)
                        for initial in checkpoint.INITIALIZERS)
    assert actual == expected
    group_ranks = {(t["group_id"], t["rank"]) for t in plan["tasks"]}
    for group, rank in group_ranks:
        tasks = [t for t in plan["tasks"] if (t["group_id"], t["rank"]) == (group, rank)]
        assert len(tasks) == 2
        assert sum(t["include_rrr"] for t in tasks) == 1
        assert all(t["initializer_source_rank"] == max(10, rank) for t in tasks)
        assert tasks[0]["free_counts"] == tasks[1]["free_counts"]
    assert plan["n_rrr_candidates"] == len(group_ranks)
    rank11 = [t for t in plan["tasks"] if t["rank"] == 11]
    assert rank11 and all(t["free_counts"] == [11] and t["initializer_source_rank"] == 11 for t in rank11)


def test_rsc_even_and_new_ranks_add_only_needed_fits(reference):
    records = records_for(reference, rank=6)
    key = reference["groups"][0]["dataset_id"]
    records[key].update(selected_rank=12, raw_rsc_rank=12)
    plan = make_plan(reference, records)
    assert plan["configuration"]["automatic_ranks"] == [6, 12]
    assert plan["selection_scope"]["fitted_ranks"] == [1, 3, 5, 6, 7, 9, 11, 12]
    high = [t for t in plan["tasks"] if t["rank"] == 12]
    assert high and all(t["initializer_source_rank"] == 12 and t["free_counts"] == [12, 15, 20] for t in high)
    even = [t for t in plan["tasks"] if t["rank"] == 6]
    assert even and all(t["free_counts"] == [6, 7, 10, 15, 20] for t in even)
    # The same dataset has separate low/standard spectral protocols but one rank record.
    aliases = [g for g in plan["groups"] if g["dataset_id"] == key]
    assert len(aliases) == 2
    assert all(g["selected_target_rank"] == 12 for g in aliases)
    assert len({g["rank_selection_sha256"] for g in aliases}) == 1


def test_reference_and_training_only_configuration_preserved(reference):
    before = original.digest(reference)
    records = records_for(reference)
    before_records = original.digest(records)
    plan = make_plan(reference, records)
    assert original.digest(reference) == before
    assert original.digest(records) == before_records
    assert plan["display_cases"] == reference["display_cases"]
    old = {g["group_id"]: g for g in reference["groups"]}
    for group in plan["groups"]:
        assert all(group[k] == v for k, v in old[group["group_id"]].items())
    config = plan["configuration"]
    assert config["selection"] == config["selection_rule"] == "bic_checkpoint"
    assert config["checkpoint_iterations"] == [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200]
    assert config["iterations"] == config["checkpoint_interval"] == 200
    assert config["validation_patience"] is None and config["validation_interval"] is None
    assert config["validation_iterations"] == []
    assert config["init_penalties"] == [.003, .1]
    assert config["penalty_pairs"] == [[.001, .001], [.0025, .0025], [.01, .01]]
    for key in ("refinement_solver", "source_basis", "margins", "adaptive_anchors", "support_limits",
                "stationarity_tol", "chart_continuation_rule", "initialization_spectrum"):
        assert config[key] == reference["configuration"][key]
    assert plan["checkpoint_bic"]["rank_records"] == records
    assert plan["plan_fingerprint"] == original.digest({k: v for k, v in plan.items() if k != "plan_fingerprint"})
    assert make_plan(reference, records) == plan


@pytest.mark.parametrize("mutation, message", [
    ("missing", "missing RSC"), ("validation", "non-training-only"),
    ("source", "implementation hash"), ("undefined_noise", "invalid cap/minimum/noise"),
    ("boolean_rank", "invalid selected rank"), ("rounded_rank", "cap/minimum rule")])
def test_rank_failures_never_fallback(reference, mutation, message):
    records = records_for(reference)
    key = reference["groups"][0]["dataset_id"]
    if mutation == "missing": del records[key]
    elif mutation == "validation": records[key]["validation_used"] = True
    elif mutation == "source": records[key]["original_implementation_sha256"] = "0" * 64
    elif mutation == "undefined_noise": records[key]["sigma"] = 0
    elif mutation == "boolean_rank": records[key]["selected_rank"] = True
    elif mutation == "rounded_rank": records[key]["raw_rsc_rank"] = 6
    with pytest.raises(ValueError, match=message):
        make_plan(reference, records)


def test_corrupt_reference_and_changed_fixed_dimensions_are_rejected(reference):
    bad = copy.deepcopy(reference)
    bad["display_cases"][0]["free_directions"] = [7, 7]
    with pytest.raises(ValueError, match="fingerprint"):
        make_plan(bad)
    bad["plan_fingerprint"] = original.digest({k: v for k, v in bad.items() if k != "plan_fingerprint"})
    with pytest.raises(ValueError, match="fixed display dimensions"):
        make_plan(bad)


def test_all_free_square_endpoint_is_rrr_only(reference):
    small = copy.deepcopy(reference)
    group = small["groups"][0]
    case = next(c for c in small["display_cases"] if c["case_id"] == group["case_id"])
    group.update(p=10, q=10)
    case.update(p=10, q=10)
    small["groups"], small["display_cases"] = [group], [case]
    small["plan_fingerprint"] = original.digest({k: v for k, v in small.items() if k != "plan_fingerprint"})
    records = records_for(small)
    records[group["dataset_id"]]["source_rank_cap"] = 10
    plan = make_plan(small, records)
    assert plan["groups"][0]["automatic_free_counts"] == [5, 7, 10]
    assert all(t["free_counts"] == [5, 7] for t in plan["tasks"])
    assert plan["n_rrr_candidates"] == 1
    assert plan["n_candidates"] == 13


def test_canary_is_representative_and_not_repeated_in_production(reference, tmp_path):
    plan = checkpoint.freeze_plan(tmp_path, "b" * 64, reference, records_for(reference))
    assert checkpoint.freeze_plan(tmp_path, "b" * 64, reference, records_for(reference)) == plan
    canary = plan["canary_task_ids"]
    assert len(canary) == len(set(canary)) == 60
    selected = [t for t in plan["tasks"] if t["task_id"] in canary]
    groups = {g["group_id"]: g for g in plan["groups"]}
    chosen_groups = {t["group_id"] for t in selected}
    assert len(chosen_groups) == 15
    assert all(t["task_id"] in canary for t in plan["tasks"] if t["group_id"] in chosen_groups)
    for model in [0, 1, 2]:
        sample = [t for t in selected if groups[t["group_id"]]["model_id"] == model]
        assert {t["rank"] for t in sample} >= {1, 5, 11}
        metadata = [groups[t["group_id"]] for t in sample]
        assert {g["protocol"] for g in metadata} == {"low_floor", "standard_floor"}
        assert {g["seed_id"] for g in metadata} == {0}
        assert {g["sigma0"] for g in metadata} >= {0., .5}
        available_n = {g["n_train"] for g in groups.values() if g["model_id"] == model}
        assert {min(available_n), max(available_n)} <= {g["n_train"] for g in metadata}
    production = {int(t) for t in (tmp_path / "production-work-items.tsv").read_text().splitlines()}
    assert production.isdisjoint(canary)
    assert production | set(canary) == set(range(plan["n_tasks"]))
    changed = records_for(reference, rank=6)
    with pytest.raises(ValueError, match="frozen input differs"):
        checkpoint.freeze_plan(tmp_path, "b" * 64, reference, changed)
