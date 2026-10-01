#!/usr/bin/env python3
"""Frozen development/assessment plans for the 200-update RSC/BIC pilot.

Planning never generates data or fits SparseSMART. ``rank_selection`` receives
only observed training X, Y and source C0, and reproduces the original SMART
RSC implementation's default theta=1 and source-rank cap. Its explicit guards
reject undefined residual variance and a rank-zero source rather than inventing
a replacement estimator. Assessment is a six-initializer control bank; the
eventual two-initializer procedure is a subset chosen on development BIC only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

from sparse_smart_v2_bic_plan import METHOD, canonical, digest, immutable, sha

INITIALIZERS = (.003, .03, .1, .3, 1., 3.)
POSITIVE_PENALTIES = (.001, .0025, .01)
FREE_COUNTS = (5, 7, 10, 15, 20)
PHASE_SEEDS = {"development": tuple(range(5)), "assessment": tuple(range(5, 10))}
RSC_PATH = Path(__file__).resolve().parents[1] / "smart/smart/rank_selector.py"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _training_fingerprint(X, Y, C0):
    """Use the existing BIC summary's observed-training fingerprint convention."""
    value = hashlib.sha256()
    for name, array in (("X", X), ("Y", Y), ("C0", C0)):
        array = np.ascontiguousarray(array)
        value.update(json.dumps((name, array.shape, array.dtype.str)).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def _original_selector():
    # Loading this file directly avoids importing SMART's unrelated Optuna and
    # cross-validation machinery. The frozen source manifest includes the file.
    spec = importlib.util.spec_from_file_location("_cheap_bic_original_rsc", RSC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RankSelectorRSC(theta=1.)


def rank_selection(X, Y, C0):
    """Return training-only RSC diagnostics; never round to an old rank grid.

    The original rule projects using X pinv(X.T X) X.T, estimates sigma squared
    using q * (n - rank(X)) residual degrees of freedom, and counts eigenvalues
    of (P Y).T (P Y) at least 4 sigma squared (sqrt(q)+sqrt(rank(X))) squared.
    SMART then returns min(rank_1e-10(C0), max(1, raw RSC rank)).
    """
    arrays = [np.asarray(array) for array in (X, Y, C0)]
    _require(all(a.ndim == 2 and a.dtype.kind in "fiu" and np.isfinite(a).all()
                 for a in arrays), "RSC inputs must be finite real matrices")
    X, Y, C0 = arrays
    n, p = X.shape
    _require(n > 0 and p > 0 and Y.shape[0] == n and Y.shape[1] > 0,
             "RSC training dimensions are invalid")
    q = Y.shape[1]
    _require(C0.shape == (p, q), "RSC source dimensions disagree with training data")
    design_rank = int(np.linalg.matrix_rank(X))
    residual_df = q * (n - design_rank)
    _require(residual_df > 0, "RSC residual noise estimate has nonpositive degrees of freedom")
    source_rank = int(np.count_nonzero(np.linalg.svd(C0, compute_uv=False) > 1e-10))
    _require(source_rank > 0, "rank-zero source is unsupported: original SMART caps target rank at zero")
    selector = _original_selector()
    projected = selector.compute_projection_matrix(X) @ Y
    sigma = float(selector.estimate_sigma(Y, projected, q, design_rank))
    _require(np.isfinite(sigma) and sigma > 0,
             "RSC residual noise estimate must be finite and strictly positive")
    mu = float(selector.compute_mu(sigma, q, design_rank))
    _require(np.isfinite(mu) and mu > 0, "RSC threshold must be finite and strictly positive")
    eigenvalues = np.linalg.eigvalsh(projected.T @ projected)[::-1]
    _require(np.isfinite(eigenvalues).all(), "RSC projected-response spectrum is nonfinite")
    raw_rank = int(np.count_nonzero(eigenvalues >= mu))
    selected = min(source_rank, max(1, raw_rank))
    return dict(schema_version=1, status="complete", selected_rank=selected,
                raw_rsc_rank=raw_rank, source_rank_cap=source_rank,
                source_rank_tolerance=1e-10, minimum_rank=1, theta=1.,
                design_rank=design_rank, residual_df=residual_df, sigma=sigma,
                threshold=mu, n_train=n, p=p, q=q,
                training_observed_input_fingerprint=_training_fingerprint(X, Y, C0),
                original_implementation="smart/smart/rank_selector.py",
                original_implementation_sha256=sha(RSC_PATH),
                source_cap_rule="min(rank_1e-10(C0), max(1, raw_rsc_rank))",
                source_zero_policy="unsupported; no silent rank promotion",
                inputs=["X", "Y", "C0"], validation_used=False,
                generating_rank_used=False, old_grid_rounding=False)


def free_counts(rank, p, q):
    _require(type(rank) is int and 1 <= rank <= min(p, q), "invalid selected rank")
    return sorted({rank, *(s for s in FREE_COUNTS if rank <= s <= min(p, q))})


def _check_reference(reference_plan):
    _require(reference_plan.get("method") == METHOD and reference_plan.get("schema_version") == 1,
             "unsupported reference BIC plan")
    _require(reference_plan.get("plan_fingerprint") == digest(
        {k: v for k, v in reference_plan.items() if k != "plan_fingerprint"}),
        "reference plan fingerprint mismatch")
    _require(reference_plan["configuration"]["selection_rule"] == "bic_terminal"
             and reference_plan["configuration"]["validation_patience"] is None,
             "reference is not training-only terminal BIC")
    config = reference_plan["configuration"]
    _require(config["refinement_solver"] == "masked_anchor_projected"
             and config["source_basis"] == "full_observed_svd"
             and config["support_limits"] == "full_outside_entry_capacity"
             and config["adaptive_anchors"] is True
             and config["margins"]["anchor_min"] == .04,
             "reference estimator differs from the authorized solver/basis/anchor/support policy")


def split_manifest(reference_plan):
    """Declare immutable seeds, group weights and pair criterion before scores."""
    _check_reference(reference_plan)
    phases = {}
    for phase, seeds in PHASE_SEEDS.items():
        groups = [g for g in reference_plan["groups"] if g["seed_id"] in seeds]
        displays = [c for c in reference_plan["display_cases"] if c["seed_id"] in seeds]
        _require(len(groups) == 165 and len({g["group_id"] for g in groups}) == 165,
                 f"{phase} requires all 165 canonical dataset/protocol groups")
        _require(len({g["dataset_id"] for g in groups}) == 150,
                 f"{phase} requires 150 unique datasets")
        for seed in seeds:
            same = [g for g in groups if g["seed_id"] == seed]
            _require(len(same) == 33 and {g["model_id"] for g in same} == {0, 1, 2},
                     f"incomplete three-model coverage for seed {seed}")
        phases[phase] = dict(seed_ids=list(seeds), models=[0, 1, 2],
            group_ids=[g["group_id"] for g in groups],
            dataset_ids=sorted({g["dataset_id"] for g in groups}),
            display_case_ids=[c["case_id"] for c in displays],
            group_metadata_sha256=digest(groups), display_metadata_sha256=digest(displays))
    result = dict(schema_version=1, experiment="SparseSMARTv2CheapRSCT200",
        reference_plan_fingerprint=reference_plan["plan_fingerprint"], phases=phases,
        initializers=list(INITIALIZERS), positive_tied_penalties=list(POSITIVE_PENALTIES),
        accepted_update_cap=200, selection="terminal training BIC",
        pair_selection="one global pair minimizing mean (pair minimum BIC - six-initializer minimum BIC)/(n*q)",
        pair_tie_break="ascending initializer pair in lexicographic order",
        group_weighting="equal weights over unique dataset/protocol groups",
        development_coverage="all pairs use the same complete six-initializer development groups; missing executions block pair freeze",
        assessment_guard="freeze the global pair before inspecting assessment performance",
        assessment_seed_caveat="held out from shortcut design, not claimed independent of earlier analyses",
        exclusions="initializer exclusions are outcomes; missing/execution-failed records are not poor fits")
    result["split_fingerprint"] = digest(result)
    return result


def freeze_split(root, reference_plan):
    """Write the data-independent split without reading rank or result records."""
    root = Path(root)
    _require(root.is_absolute(), "root must be absolute")
    root.mkdir(parents=True, exist_ok=True)
    split = split_manifest(reference_plan)
    immutable(root / "split-manifest.json", canonical(split) + b"\n")
    return split


def build_plan(root, source_manifest_sha256, reference_plan, rank_records, phase):
    """Build a standard grouped runner plan for either frozen pilot phase.

    ``rank_records`` maps dataset_id to ``rank_selection`` diagnostics. Repeated
    protocol groups reuse the same rank record. Both phases fit six initializers
    as controls; two-initializer selection is retrospective within this bank.
    """
    root = Path(root)
    _require(root.is_absolute(), "root must be absolute")
    _require(isinstance(source_manifest_sha256, str) and
             re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256), "invalid source manifest SHA256")
    _require(phase in PHASE_SEEDS, "phase must be development or assessment")
    split = split_manifest(reference_plan)
    wanted = set(split["phases"][phase]["group_ids"])
    groups = copy.deepcopy([g for g in reference_plan["groups"] if g["group_id"] in wanted])
    datasets = {g["dataset_id"] for g in groups}
    _require(datasets <= set(rank_records), "missing RSC rank records; no grid rounding or fallback allowed")
    config = copy.deepcopy(reference_plan["configuration"])
    config.update(init_penalties=list(INITIALIZERS), iterations=200,
        penalties_u=[0., *POSITIVE_PENALTIES], penalties_v=[0., *POSITIVE_PENALTIES],
        penalty_pairs=[[x, x] for x in POSITIVE_PENALTIES],
        selection="bic_terminal", selection_rule="bic_terminal", validation_patience=None,
        validation_interval=None, validation_iterations=[], validation_min_iterations=0,
        validation_min_relative_improvement=0., checkpoint_interval=200,
        automatic_rank_rule="one original SMART RSC estimate per dataset; no rounding",
        automatic_free_counts=list(FREE_COUNTS),
        automatic_free_counts_rule="tied counts >= selected rank, including rank; bounded by min(p,q)",
        all_free_endpoint="one target RRR per dataset/protocol/selected rank, independent of initialization",
        stopping_rule="training stationarity or 200 accepted updates; no validation input",
        cheap_bic_phase=phase, initializer_reuse="one cached initializer per dataset/protocol/rank/strength across free counts and penalties")
    tasks, iterative, ranks = [], 0, set()
    for group in groups:
        record = rank_records[group["dataset_id"]]
        _require(record.get("status") == "complete" and record.get("validation_used") is False
                 and record.get("generating_rank_used") is False and record.get("old_grid_rounding") is False,
                 "invalid or non-training-only RSC rank record")
        for key in ("n_train", "p", "q"):
            _require(record[key] == group[key], f"RSC record dimension mismatch: {key}")
        rank = record["selected_rank"]
        counts = [s for s in free_counts(rank, group["p"], group["q"])
                  if not s == group["p"] == group["q"]]
        _require(type(record["source_rank_cap"]) is int and rank <= record["source_rank_cap"] <= min(group["p"], group["q"]),
                 "selected rank exceeds observed source rank cap")
        _require(type(record["raw_rsc_rank"]) is int and 0 <= record["raw_rsc_rank"] <= group["q"]
                 and rank == min(record["source_rank_cap"], max(1, record["raw_rsc_rank"])),
                 "selected rank does not follow original SMART cap/minimum rule")
        _require(record["theta"] == 1. and record["residual_df"] > 0
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
        _require(max(10, rank) <= min(group["p"], group["q"]),
                 "initializer dimension max(10,r) exceeds ambient dimensions")
        ranks.add(rank)
        group["rank_selection_sha256"] = digest(record)
        group["selected_target_rank"] = rank
        for index, initial in enumerate(INITIALIZERS):
            tasks.append(dict(task_id=len(tasks), group_id=group["group_id"], rank=rank,
                initializer_source_rank=max(10, rank), init_penalty=initial,
                free_counts=counts, include_rrr=index == 0))
            iterative += len(counts) * len(POSITIVE_PENALTIES)
    config["automatic_ranks"] = sorted(ranks)
    displays = copy.deepcopy([c for c in reference_plan["display_cases"] if c["group_id"] in wanted])
    plan = dict(schema_version=1, method=METHOD, root=str(root),
        reference_root=reference_plan["reference_root"], source_manifest_sha256=source_manifest_sha256,
        seed_file_sha256=reference_plan["seed_file_sha256"], configuration=config,
        groups=groups, tasks=tasks, display_cases=displays,
        n_groups=len(groups), n_datasets=len(datasets), n_tasks=len(tasks),
        n_display_cases=len(displays), n_iterative_candidates=iterative,
        n_rrr_candidates=len(groups), n_candidates=iterative + len(groups),
        dataset_alias_rule=reference_plan["dataset_alias_rule"],
        selection_scope=dict(models=[0, 1, 2], seed_ids=list(PHASE_SEEDS[phase]),
                             fitted_ranks=sorted(ranks), phase=phase),
        cheap_bic=dict(phase=phase, split_fingerprint=split["split_fingerprint"],
            reference_plan_fingerprint=reference_plan["plan_fingerprint"],
            rank_records={key: copy.deepcopy(rank_records[key]) for key in sorted(datasets)},
            historical_2000_terminal_reuse="unavailable at T=200 unless an identical saved T=200 state exists",
            automatic_rank_display="reuse selected rank across aliases; display labels retain original sensitivity setting"))
    plan["plan_fingerprint"] = digest(plan)
    return plan


def freeze_plan(root, source_manifest_sha256, reference_plan, rank_records, phase):
    root = Path(root)
    _require((root / "split-manifest.json").exists(), "freeze split before building any rank-dependent plan")
    _require(json.loads((root / "split-manifest.json").read_text()) == split_manifest(reference_plan),
             "frozen split does not match the reference plan")
    plan = build_plan(root, source_manifest_sha256, reference_plan, rank_records, phase)
    immutable(root / "rank-records.json", canonical(plan["cheap_bic"]["rank_records"]) + b"\n")
    immutable(root / "plan.json", canonical(plan) + b"\n")
    immutable(root / "work-items.tsv", "".join(f"{t['task_id']}\n" for t in plan["tasks"]).encode())
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze-split", "build"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reference-plan", required=True, type=Path)
    parser.add_argument("--rank-records", type=Path)
    parser.add_argument("--phase", choices=tuple(PHASE_SEEDS))
    args = parser.parse_args(argv)
    reference = json.loads(args.reference_plan.read_text())
    if args.action == "freeze-split":
        result = freeze_split(args.root, reference)
    else:
        if args.rank_records is None or args.phase is None:
            parser.error("build requires --rank-records and --phase")
        result = freeze_plan(args.root, sha(args.root / "source-manifest.json"), reference,
                             json.loads(args.rank_records.read_text()), args.phase)
    print(json.dumps({k: v for k, v in result.items() if k in
                     ("split_fingerprint", "plan_fingerprint", "n_groups", "n_datasets", "n_tasks", "n_candidates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
