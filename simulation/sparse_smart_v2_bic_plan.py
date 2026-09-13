#!/usr/bin/env python3
"""Freeze training-only BIC fits and aliases for all fixed/automatic displays.

Planning reads the canonical simulation settings and seed table; it never
generates observations, fits an estimator, or contacts a scheduler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from run_sparse_smart_v2_pilot import _case_specs, _configuration, _configure_cases

METHOD = "SparseSMARTv2BIC"
RANKS = (1, 3, 5, 7, 9, 11)
FREE_COUNTS = (5, 7, 10, 15, 20)
REFERENCE_ROOT = "/scratch2/mkolar/smart/runs/sparse-smart-v2-paper100-all-20260913T073736Z"
PROTOCOLS = {
    "low_floor": {"d_lower": .001, "gap": .0001},
    "standard_floor": {"d_lower": .05, "gap": .01},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection(values, defaults, allowed, name):
    values = list(defaults if values is None else values)
    if (not values or any(type(value) is not int or value not in allowed for value in values)
            or len(values) != len(set(values))):
        raise ValueError(f"{name} must contain distinct allowed integer values")
    return sorted(values)


def free_counts(rank):
    return sorted({rank, *(value for value in FREE_COUNTS if value >= rank)})


def dataset_key(case):
    return (case["model_id"], case["n_train"], case["sigma0"], case["seed_id"])


def protocol_for(experiment):
    return "low_floor" if experiment in (0, 1) else "standard_floor"


def build(root, source_manifest_sha256, models=None, seeds=None, ranks=None, reference_root=REFERENCE_ROOT):
    """Return a deterministic plan, with no writes and no scientific fitting."""
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("root must be absolute")
    if not Path(reference_root).is_absolute():
        raise ValueError("reference_root must be absolute")
    if not isinstance(source_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256):
        raise ValueError("source_manifest_sha256 must be a SHA256 hex digest")
    models = selection(models, range(3), range(3), "models")
    seeds = selection(seeds, range(100), range(100), "seeds")
    ranks = selection(ranks, RANKS, RANKS, "ranks")
    bases, seeds_hash = _case_specs(models=models, experiments=[0, 3], seed_ids=seeds)
    groups, lookup = [], {}
    for original in bases:
        group = dict(original, group_id=original["case_id"],
                     protocol=protocol_for(original["experiment_id"]))
        group["dataset_id"] = "data_" + digest(dataset_key(group))[:24]
        group["margins"] = dict(_configuration()["margins"], **PROTOCOLS[group["protocol"]])
        key = (*dataset_key(group), group["protocol"])
        if key in lookup:
            raise ValueError("duplicate dataset/protocol group")
        lookup[key] = group["group_id"]
        groups.append(group)

    config = dict(_configuration(), selection="bic_terminal", selection_rule="bic_terminal",
                  refinement_solver="masked_anchor_projected", validation_patience=None,
                  validation_min_iterations=0, validation_min_relative_improvement=0.,
                  validation_interval=None, validation_iterations=[], support_tolerance=0.,
                  automatic_ranks=ranks, automatic_free_counts=list(FREE_COUNTS),
                  automatic_free_counts_rule="tied counts >= fitted rank, including fitted rank",
                  all_free_endpoint="one target RRR per dataset/protocol/fitted rank",
                  stopping_rule="training stationarity or declared iteration cap; no validation input",
                  protocols=PROTOCOLS)
    tasks = []
    iterative_count = 0
    for group in groups:
        for rank in ranks:
            counts = free_counts(rank)
            for index, initial in enumerate(config["init_penalties"]):
                tasks.append(dict(task_id=len(tasks), group_id=group["group_id"], rank=rank,
                                  initializer_source_rank=max(10, rank), init_penalty=initial,
                                  free_counts=counts, include_rrr=index == 0))
                iterative_count += len(counts) * 19

    originals, _ = _case_specs(models=models, experiments=[0, 1, 2, 3], seed_ids=seeds)
    supported = [case for case in originals if case["experiment_id"] != 2 or case["source_rank"] >= 5]
    fixed, _ = _configure_cases(supported, 10, "expand")
    displays = []
    for case in fixed:
        if case["rank"] not in ranks:
            continue
        protocol = protocol_for(case["experiment_id"])
        displays.append(dict(case, group_id=lookup[(*dataset_key(case), protocol)], protocol=protocol))
    rrr_count = len(groups) * len(ranks)
    plan = dict(schema_version=1, method=METHOD, root=str(root), reference_root=str(reference_root),
                source_manifest_sha256=source_manifest_sha256, seed_file_sha256=seeds_hash,
                configuration=config, groups=groups, tasks=tasks, display_cases=displays,
                n_groups=len(groups), n_datasets=len({g["dataset_id"] for g in groups}),
                n_tasks=len(tasks), n_display_cases=len(displays),
                n_iterative_candidates=iterative_count, n_rrr_candidates=rrr_count,
                n_candidates=iterative_count + rrr_count,
                dataset_alias_rule="same generated data across fitted-dimension experiments; keep spectral protocols distinct",
                selection_scope=dict(models=models, seed_ids=seeds, fitted_ranks=ranks,
                                     source_count_zero_and_three="not in fixed comparison; automatic reference is source-count independent"))
    plan["plan_fingerprint"] = digest(plan)
    return plan


def immutable(path, contents):
    if path.exists():
        if path.is_symlink() or path.read_bytes() != contents:
            raise ValueError(f"existing frozen input differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(contents)


def freeze(root, *, models=None, seeds=None, ranks=None, reference_root=REFERENCE_ROOT):
    root = Path(root).resolve()
    manifest_path = root / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != 1 or
            Path(manifest["source_root"]).resolve() != (root / "source").resolve()):
        raise ValueError("source manifest must identify the frozen ROOT/source")
    plan = build(root, sha(manifest_path), models, seeds, ranks, reference_root)
    immutable(root / "plan.json", canonical(plan) + b"\n")
    immutable(root / "work-items.tsv", "".join(f"{task['task_id']}\n" for task in plan["tasks"]).encode())
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--models", nargs="+", type=int)
    parser.add_argument("--seed-ids", nargs="+", type=int)
    parser.add_argument("--ranks", nargs="+", type=int)
    parser.add_argument("--reference-root", type=Path, default=Path(REFERENCE_ROOT))
    args = parser.parse_args(argv)
    try:
        plan = freeze(args.root, models=args.models, seeds=args.seed_ids, ranks=args.ranks, reference_root=args.reference_root)
        print(json.dumps({key: plan[key] for key in ("root", "plan_fingerprint", "n_groups", "n_datasets",
                         "n_tasks", "n_candidates", "n_display_cases")}, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
