"""Frozen BIC search coverage and dataset reuse, without fitting simulations."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
import sparse_smart_v2_bic_plan as planner


def test_full_plan_covers_all_fixed_displays_without_refitting_identical_data():
    plan = planner.build(Path("/scratch2/test-bic"), "a" * 64)
    assert (plan["n_groups"], plan["n_datasets"], plan["n_tasks"], plan["n_display_cases"]) == (3300, 3000, 118800, 6600)
    assert plan["n_iterative_candidates"] == 10533600
    assert plan["n_rrr_candidates"] == 19800
    assert plan["n_candidates"] == 10553400
    assert {row["seed_id"] for row in plan["display_cases"]} == set(range(100))
    assert {row["model_id"] for row in plan["display_cases"]} == {0, 1, 2}
    assert plan["plan_fingerprint"] == planner.digest({k: v for k, v in plan.items() if k != "plan_fingerprint"})


def test_aliases_preserve_different_spectral_protocols_and_rank11_fixed_policy():
    plan = planner.build(Path("/scratch2/test-bic"), "a" * 64, models=[0], seeds=[0])
    cases = {row["case_id"]: row for row in plan["display_cases"]}
    assert cases["m0_e0_k0_s0"]["group_id"] == cases["m0_e1_k2_s0"]["group_id"]
    assert len({cases[f"m0_e1_k{k}_s0"]["group_id"] for k in range(6)}) == 1
    assert cases["m0_e2_k4_s0"]["group_id"] == cases["m0_e3_k1_s0"]["group_id"]
    assert cases["m0_e0_k0_s0"]["group_id"] != cases["m0_e3_k1_s0"]["group_id"]
    groups = {row["group_id"]: row for row in plan["groups"]}
    low = groups[cases["m0_e0_k0_s0"]["group_id"]]
    standard = groups[cases["m0_e3_k1_s0"]["group_id"]]
    assert low["dataset_id"] == standard["dataset_id"]
    assert low["margins"]["d_lower"] == .001 and low["margins"]["gap"] == .0001
    assert standard["margins"]["d_lower"] == .05 and standard["margins"]["gap"] == .01
    rank11 = cases["m0_e1_k5_s0"]
    assert rank11["rank"] == rank11["source_rank"] == rank11["initializer_source_rank"] == 11
    assert rank11["free_directions"] == [11, 11]
    assert not any(row["experiment_id"] == 2 and row["source_rank"] in (0, 3) for row in cases.values())


def test_tasks_share_initializer_across_free_counts_and_rrr_across_init_penalties():
    plan = planner.build(Path("/scratch2/test-bic"), "a" * 64, models=[0], seeds=[0])
    first = [row for row in plan["tasks"] if row["group_id"] == plan["groups"][0]["group_id"]]
    assert len(first) == 36
    for rank in planner.RANKS:
        tasks = [row for row in first if row["rank"] == rank]
        assert len(tasks) == 6 and sum(row["include_rrr"] for row in tasks) == 1
        assert tasks[0]["free_counts"] == planner.free_counts(rank)
        assert all(row["initializer_source_rank"] == max(10, rank) for row in tasks)
    config = plan["configuration"]
    assert config["selection"] == "bic_terminal"
    assert config["validation_patience"] is None and config["validation_interval"] is None
    assert config["support_tolerance"] == 0
    assert config["penalties_u"] == [0, .001, .0025, .01, .04]
    assert config["penalties_v"] == [0, .001, .0025, .01]


@pytest.mark.parametrize("kwargs", [{"models": [3]}, {"seeds": [100]}, {"ranks": [2]}, {"ranks": [5, 5]}, {"models": []}])
def test_rejects_invalid_grid_selections(kwargs):
    with pytest.raises(ValueError):
        planner.build(Path("/scratch2/test-bic"), "a" * 64, **kwargs)


def test_freeze_is_idempotent_and_refuses_scientific_changes(tmp_path):
    root = tmp_path / "literal $[root]"
    (root / "source").mkdir(parents=True)
    (root / "source-manifest.json").write_text(json.dumps(dict(schema_version=1, source_root=str(root / "source"), files={})))
    plan = planner.freeze(root, models=[0], seeds=[0], ranks=[5])
    original = (root / "plan.json").read_bytes()
    assert planner.freeze(root, models=[0], seeds=[0], ranks=[5]) == plan
    assert (root / "plan.json").read_bytes() == original
    assert (root / "work-items.tsv").read_text().splitlines() == [str(i) for i in range(66)]
    with pytest.raises(ValueError, match="frozen input differs"):
        planner.freeze(root, models=[0], seeds=[0, 1], ranks=[5])
    assert (root / "plan.json").read_bytes() == original
