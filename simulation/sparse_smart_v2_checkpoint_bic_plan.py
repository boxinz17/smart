#!/usr/bin/env python3
"""Plan the cheap fixed/automatic, early-checkpoint BIC campaign.

This module only constructs immutable descriptions. Rank records must already
come from the original training-only RSC rule. The fitting library is the union
needed by the automatic procedure and the frozen fixed-parameter displays;
display aliases never create additional fits. Existing plans are never edited.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np

from sparse_smart_v2_bic_plan import METHOD, canonical, digest, immutable, sha
from sparse_smart_v2_cheap_bic_plan import RSC_PATH, _check_reference, free_counts

INITIALIZERS = (.003, .1)
POSITIVE_PENALTIES = (.001, .0025, .01)
CHECKPOINTS = (0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rank_record(record, group):
    """Check the original RSC rule and identity without reading evaluation data."""
    _require(record.get("status") == "complete" and record.get("validation_used") is False
             and record.get("generating_rank_used") is False and record.get("old_grid_rounding") is False,
             "invalid or non-training-only RSC rank record")
    for key in ("n_train", "p", "q"):
        _require(record[key] == group[key], f"RSC record dimension mismatch: {key}")
    rank = record["selected_rank"]
    # free_counts checks integer (including rejecting bool), rank and dimensions.
    counts = free_counts(rank, group["p"], group["q"])
    cap, raw = record["source_rank_cap"], record["raw_rsc_rank"]
    _require(type(cap) is int and rank <= cap <= min(group["p"], group["q"]),
             "selected rank exceeds observed source rank cap")
    _require(type(raw) is int and 0 <= raw <= group["q"] and rank == min(cap, max(1, raw)),
             "selected rank does not follow original SMART cap/minimum rule")
    _require(type(record["design_rank"]) is int
             and 0 <= record["design_rank"] <= min(group["n_train"], group["p"])
             and record["theta"] == 1. and record["residual_df"] > 0
             and record["residual_df"] == group["q"] * (group["n_train"] - record["design_rank"]),
             "RSC record violates default threshold or residual degrees of freedom")
    _require(record["source_rank_tolerance"] == 1e-10 and record["minimum_rank"] == 1
             and np.isfinite(record["sigma"]) and record["sigma"] > 0
             and np.isfinite(record["threshold"]) and record["threshold"] > 0,
             "RSC record contains invalid cap/minimum/noise settings")
    _require(record["original_implementation_sha256"] == sha(RSC_PATH),
             "RSC implementation hash differs from frozen source")
    _require(isinstance(record["training_observed_input_fingerprint"], str)
             and re.fullmatch(r"[0-9a-f]{64}", record["training_observed_input_fingerprint"]),
             "missing RSC training data identity")
    return rank, counts


def _fixed_request(case, group):
    """Validate and retain each existing fixed display's declared dimensions."""
    experiment = case["experiment_id"]
    rank = case["rank"]
    if experiment in (0, 3):
        expected_rank, count = 5, 10
    elif experiment == 1:
        _require(rank in (1, 3, 5, 7, 9, 11), "unsupported fixed target rank")
        expected_rank, count = rank, max(10, rank)
    elif experiment == 2:
        expected_rank, count = 5, case["source_rank"]
        _require(count in (5, 7, 10, 15, 20), "unsupported fixed free count")
    else:
        raise ValueError("unsupported fixed experiment")
    _require(type(rank) is int and rank == expected_rank
             and case["free_directions"] == [count, count]
             and case["source_rank"] == count
             and rank <= count <= min(group["p"], group["q"])
             and case["initializer_source_rank"] == max(10, rank),
             "fixed display dimensions differ from the authorized campaign")
    for key in ("model_id", "seed_id", "random_seed", "n_train", "p", "q", "sigma0", "protocol"):
        _require(case[key] == group[key], f"fixed display dataset/protocol mismatch: {key}")
    return rank, count


def _canary_tasks(groups, tasks):
    """Use complete production groups covering the numerical boundary cases."""
    chosen = set()
    for model in sorted({g["model_id"] for g in groups}):
        same_model = [g for g in groups if g["model_id"] == model]
        seed = min(g["seed_id"] for g in same_model)
        cohort = [g for g in same_model if g["seed_id"] == seed]
        low = sorted((g for g in cohort if g["protocol"] == "low_floor"),
                     key=lambda g: (g["n_train"], g["group_id"]))
        standard = sorted((g for g in cohort if g["protocol"] == "standard_floor"),
                          key=lambda g: (g["sigma0"], g["group_id"]))
        for candidates in (low, standard):
            if candidates:
                for group in (candidates[0], candidates[-1]):
                    chosen.add(group["group_id"])
        # Fixed rank-one/rank-eleven extremes are present only in the low-floor
        # baseline. The standard-floor sigma=.01 group exercises all free counts.
        for group in cohort:
            fixed = group["fixed_rank_free_counts"]
            for rank in (1, 11):
                if str(rank) in fixed:
                    chosen.add(group["group_id"])
            if group["protocol"] == "standard_floor" and group["sigma0"] == .01:
                chosen.add(group["group_id"])
    return [t["task_id"] for t in tasks if t["group_id"] in chosen]


def build_plan(root, source_manifest_sha256, reference_plan, rank_records):
    """Return a deterministic standard grouped-runner plan, without writes.

    ``rank_records`` is keyed by dataset ID, not display or spectral protocol.
    A source-data-identical group with different spectral bounds retains its own
    fit because it represents a different optimization protocol.
    """
    root = Path(root)
    _require(root.is_absolute(), "root must be absolute")
    _require(isinstance(source_manifest_sha256, str)
             and re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256), "invalid source manifest SHA256")
    _check_reference(reference_plan)
    groups = copy.deepcopy(reference_plan["groups"])
    displays = copy.deepcopy(reference_plan["display_cases"])
    _require(groups and displays, "reference has no groups or displays")
    by_group = {g["group_id"]: g for g in groups}
    _require(len(by_group) == len(groups), "duplicate reference group identity")
    _require(len({c["case_id"] for c in displays}) == len(displays), "duplicate fixed display identity")
    datasets = {g["dataset_id"] for g in groups}
    _require(datasets <= set(rank_records), "missing RSC rank records; no grid rounding or fallback allowed")
    fixed = {key: {} for key in by_group}
    for case in displays:
        _require(case["group_id"] in by_group, "fixed display references an unknown group")
        rank, count = _fixed_request(case, by_group[case["group_id"]])
        fixed[case["group_id"]].setdefault(rank, set()).add(count)

    config = copy.deepcopy(reference_plan["configuration"])
    config.update(init_penalties=list(INITIALIZERS), iterations=200,
        penalties_u=[0., *POSITIVE_PENALTIES], penalties_v=[0., *POSITIVE_PENALTIES],
        penalty_pairs=[[x, x] for x in POSITIVE_PENALTIES],
        selection="bic_checkpoint", selection_rule="bic_checkpoint", validation_patience=None,
        validation_interval=None, validation_iterations=[], validation_min_iterations=0,
        validation_min_relative_improvement=0., checkpoint_interval=200,
        checkpoint_iterations=list(CHECKPOINTS),
        automatic_rank_rule="one original SMART RSC estimate per dataset; no rounding",
        automatic_free_counts=[5, 7, 10, 15, 20],
        automatic_free_counts_rule="tied counts >= selected rank, including rank; bounded by min(p,q)",
        all_free_endpoint="one target RRR per dataset/protocol/rank, independent of initialization",
        stopping_rule="training stationarity or 200 accepted updates; no validation or BIC-patience stopping",
        initializer_reuse="one initializer per dataset/protocol/rank/strength across free counts and penalties")
    config.pop("cheap_bic_phase", None)
    tasks, automatic_ranks, fitted_ranks, iterative, rrr = [], set(), set(), 0, 0
    for group in groups:
        record = rank_records[group["dataset_id"]]
        rank, counts = _rank_record(record, group)
        group["selected_target_rank"] = rank
        group["automatic_free_counts"] = counts
        group["rank_selection_sha256"] = digest(record)
        group_fixed = fixed[group["group_id"]]
        _require(group_fixed, "reference group has no fixed display")
        group["fixed_rank_free_counts"] = {str(r): sorted(s) for r, s in sorted(group_fixed.items())}
        requirements = copy.deepcopy(group_fixed)
        requirements.setdefault(rank, set()).update(counts)
        automatic_ranks.add(rank)
        for fitted_rank, wanted_counts in sorted(requirements.items()):
            _require(max(10, fitted_rank) <= min(group["p"], group["q"]),
                     "initializer dimension max(10,r) exceeds ambient dimensions")
            # The all-rows-unpenalized procedure is exactly the deduplicated RRR
            # endpoint. Do not repeat it for positive penalties or initializers.
            iterative_counts = sorted(s for s in wanted_counts if not s == group["p"] == group["q"])
            fitted_ranks.add(fitted_rank)
            for index, initial in enumerate(INITIALIZERS):
                tasks.append(dict(task_id=len(tasks), group_id=group["group_id"], rank=fitted_rank,
                    initializer_source_rank=max(10, fitted_rank), init_penalty=initial,
                    free_counts=iterative_counts, include_rrr=index == 0))
                iterative += len(iterative_counts) * len(POSITIVE_PENALTIES)
            rrr += 1
    config["automatic_ranks"] = sorted(automatic_ranks)
    models = sorted({g["model_id"] for g in groups})
    seeds = sorted({g["seed_id"] for g in groups})
    canary = _canary_tasks(groups, tasks)
    plan = dict(schema_version=1, method=METHOD, root=str(root),
        reference_root=reference_plan["reference_root"], source_manifest_sha256=source_manifest_sha256,
        seed_file_sha256=reference_plan["seed_file_sha256"], configuration=config,
        groups=groups, tasks=tasks, display_cases=displays,
        n_groups=len(groups), n_datasets=len(datasets), n_tasks=len(tasks),
        n_display_cases=len(displays), n_iterative_candidates=iterative,
        n_rrr_candidates=rrr, n_candidates=iterative + rrr,
        dataset_alias_rule=reference_plan["dataset_alias_rule"],
        selection_scope=dict(models=models, seed_ids=seeds, fitted_ranks=sorted(fitted_ranks),
            automatic_ranks=sorted(automatic_ranks), experiments=[0, 1, 2, 3],
            source_count_zero_and_three="not in fixed comparison; automatic reference is source-count independent"),
        canary_task_ids=canary,
        checkpoint_bic=dict(reference_plan_fingerprint=reference_plan["plan_fingerprint"],
            rank_records={key: copy.deepcopy(rank_records[key]) for key in sorted(datasets)},
            initializer_pair=list(INITIALIZERS), initializer_pair_rule="global pair frozen on development training BIC",
            checkpoint_iterations=list(CHECKPOINTS), include_earlier_terminal=True,
            selected_state_rule="minimum training BIC over configurations and saved checkpoints",
            terminal_control="minimum terminal training BIC from the same trajectories",
            automatic_rule="selected_target_rank and automatic_free_counts within each group",
            fixed_rule="display rank and tied free_directions; shared underlying candidate library",
            canary_rule="complete production groups spanning all models, protocols, sample-size/noise extremes and fixed ranks 1 and 11",
            reuse_policy="identical terminal records do not reconstruct early states; no raw-data copies or per-fit archive"))
    plan["plan_fingerprint"] = digest(plan)
    return plan


def freeze_plan(root, source_manifest_sha256, reference_plan, rank_records):
    """Write plan and task list once, refusing to replace differing inputs."""
    root = Path(root)
    plan = build_plan(root, source_manifest_sha256, reference_plan, rank_records)
    root.mkdir(parents=True, exist_ok=True)
    immutable(root / "plan.json", canonical(plan) + b"\n")
    immutable(root / "work-items.tsv", "".join(f"{t['task_id']}\n" for t in plan["tasks"]).encode())
    immutable(root / "canary-work-items.tsv", "".join(f"{i}\n" for i in plan["canary_task_ids"]).encode())
    chosen = set(plan["canary_task_ids"])
    immutable(root / "production-work-items.tsv",
              "".join(f"{t['task_id']}\n" for t in plan["tasks"] if t["task_id"] not in chosen).encode())
    return plan
